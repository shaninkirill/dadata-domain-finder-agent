"""
Провайдер веб-поиска: Yandex Search API v2 (Yandex Cloud).

Документация: https://yandex.cloud/ru/docs/search-api/

Синхронный эндпоинт /v2/web/search принимает JSON-запрос (авторизация по
API-ключу сервисного аккаунта) и возвращает конверт {"rawData": "<base64 XML>"}
с результатами поиска в классической XML-схеме Яндекс.XML.
"""

import base64
import binascii
import logging
import xml.etree.ElementTree as ET

import requests

import config

logger = logging.getLogger(__name__)

#: Переиспользуемая HTTP-сессия (keep-alive, пул соединений).
_SESSION = requests.Session()


def _normalize_count(count: int) -> int:
    """
    Возвращает корректное количество результатов (groupsOnPage) для Yandex Search API.

    :param count: Запрошенное количество результатов; значение вне диапазона
                  [1, YANDEX_MAX_RESULTS_COUNT] заменяется значением по умолчанию
    :return: Целое число результатов в допустимом диапазоне
    """
    if 1 <= count <= config.YANDEX_MAX_RESULTS_COUNT:
        return count
    return config.SEARCH_RESULTS_COUNT


def parse_yandex_xml(xml_bytes: bytes) -> list[dict[str, str]]:
    """
    Разбирает XML-ответ Yandex Search API и извлекает результаты поиска.

    :param xml_bytes: XML-документ ответа (классическая схема Яндекс.XML)
    :return: Список словарей {"title", "url", "snippet"}; пустой список, если ничего не найдено
    :raises ValueError: Если XML содержит ошибку поиска (кроме кода 15 «ничего не найдено»)
    :raises ET.ParseError: Если XML не удалось разобрать
    """
    root = ET.fromstring(xml_bytes)

    error = root.find(".//response/error")
    if error is not None:
        code = error.get("code", "")
        if code == "15":  # 15 — «ничего не найдено», ошибкой не считается
            return []
        raise ValueError(f"Yandex Search API error {code}: {error.text}")

    results: list[dict[str, str]] = []
    for doc in root.iterfind(".//doc"):
        url = (doc.findtext("url") or "").strip()
        title = (doc.findtext("title") or "").strip()
        if not url or not title:
            continue
        snippet = (doc.findtext("headline") or "").strip()
        if not snippet:
            snippet = " ".join(
                (passage.text or "").strip() for passage in doc.findall("./passages/passage")
            ).strip()
        results.append({"title": title, "url": url, "snippet": snippet})
    return results


def _extract_error_message(response: requests.Response) -> str:
    """
    Извлекает текст ошибки из JSON-ответа Yandex Cloud.

    :param response: HTTP-ответ с ненулевым кодом состояния
    :return: Текст ошибки или пустая строка, если извлечь его не удалось
    """
    try:
        payload = response.json()
    except ValueError:
        return ""
    if isinstance(payload, dict) and payload.get("message"):
        return str(payload["message"])
    return ""


def yandex_web_search(query: str, count: int) -> str:
    """
    Выполняет веб-поиск через Yandex Search API v2 и возвращает результаты в текстовом виде.

    :param query: Поисковый запрос (до 400 символов и 40 слов)
    :param count: Количество результатов; некорректное значение заменяется
                  настройкой по умолчанию
    :return: Строка с результатами (Title/URL/Snippet), либо сообщение об ошибке
             или об отсутствии результатов
    """
    if not config.YANDEX_SEARCH_API_KEY or not config.YANDEX_SEARCH_FOLDER_ID:
        logger.warning("Вызов yandex_web_search без заданных ключей Yandex Cloud")
        return (
            "Ошибка: не заданы YANDEX_SEARCH_API_KEY и YANDEX_SEARCH_FOLDER_ID в .env. "
            "Поиск недоступен, используй другие инструменты."
        )
    if len(query) > 400:
        return "Ошибка поиска: запрос длиннее 400 символов, сократи и переформулируй его."

    body = {
        "query": {
            "searchType": "SEARCH_TYPE_RU",
            "queryText": query,
            "familyMode": "FAMILY_MODE_MODERATE",
            "page": "0",  # индексация страниц результата с нуля
        },
        "groupSpec": {
            "groupMode": "GROUP_MODE_FLAT",
            "groupsOnPage": str(_normalize_count(count)),
            "docsInGroup": "1",
        },
        "l10n": "LOCALIZATION_RU",
        "folderId": config.YANDEX_SEARCH_FOLDER_ID,
        "responseFormat": "FORMAT_XML",
    }
    if config.YANDEX_SEARCH_REGION:
        body["region"] = config.YANDEX_SEARCH_REGION

    headers = {
        "Authorization": f"Api-Key {config.YANDEX_SEARCH_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = _SESSION.post(
            config.YANDEX_SEARCH_API_URL,
            headers=headers,
            json=body,
            timeout=config.HTTP_TIMEOUT,
        )
    except requests.RequestException as exc:
        logger.warning("Сетевая ошибка Yandex Search API («%s»): %s", query, exc)
        return f"Ошибка поиска: сетевая ошибка ({exc.__class__.__name__})."

    if response.status_code != 200:
        message = _extract_error_message(response)
        logger.warning(
            "Yandex Search API вернул HTTP %s для запроса «%s»: %s",
            response.status_code,
            query,
            message,
        )
        detail = f" ({message})" if message else ""
        return f"Ошибка поиска: HTTP {response.status_code}{detail}."

    try:
        raw_data = response.json().get("rawData")
    except ValueError:
        raw_data = None
    if not raw_data:
        return "Ошибка поиска: некорректный ответ Yandex Search API (нет rawData)."

    try:
        results = parse_yandex_xml(base64.b64decode(raw_data))
    except ValueError as exc:
        logger.warning("Yandex Search API вернул ошибку в XML: %s", exc)
        return f"Ошибка поиска: {exc}."
    except (ET.ParseError, binascii.Error, TypeError):
        logger.warning("Yandex Search API вернул повреждённый ответ для запроса «%s»", query)
        return "Ошибка поиска: некорректный XML в ответе."

    if not results:
        logger.info("Yandex: по запросу «%s» ничего не найдено", query)
        return "Ничего не найдено."

    logger.info("Yandex: по запросу «%s» найдено результатов: %s", query, len(results))
    return "\n---\n".join(
        f"Title: {item['title']}\nURL: {item['url']}\nSnippet: {item['snippet']}"
        for item in results
    )
