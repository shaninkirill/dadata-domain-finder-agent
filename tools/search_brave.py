"""
Провайдер веб-поиска: Brave Search API.

Документация: https://brave.com/search/api/
Используется при SEARCH_PROVIDER=brave.
"""

import logging

import requests

import config

logger = logging.getLogger(__name__)

#: Endpoint веб-поиска Brave Search API.
_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

#: Переиспользуемая HTTP-сессия (keep-alive, пул соединений).
_SESSION = requests.Session()


def _normalize_count(count: int) -> int:
    """
    Возвращает корректное количество результатов для Brave Search API.

    :param count: Запрошенное количество результатов; значение вне диапазона
                  [1, BRAVE_MAX_RESULTS_COUNT] заменяется значением по умолчанию
    :return: Целое число результатов в допустимом диапазоне
    """
    if 1 <= count <= config.BRAVE_MAX_RESULTS_COUNT:
        return count
    return config.SEARCH_RESULTS_COUNT


def brave_web_search(query: str, count: int) -> str:
    """
    Выполняет веб-поиск через Brave Search API и возвращает результаты в текстовом виде.

    :param query: Поисковый запрос
    :param count: Количество результатов; некорректное значение заменяется
                  настройкой по умолчанию
    :return: Строка с результатами (Title/URL/Snippet), либо сообщение об ошибке
             или об отсутствии результатов
    """
    if not config.BRAVE_API_KEY:
        logger.warning("Вызов brave_web_search без заданного BRAVE_API_KEY")
        return (
            "Ошибка: не задан API-ключ Brave Search (BRAVE_API_KEY в .env). "
            "Поиск недоступен, используй другие инструменты."
        )

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": config.BRAVE_API_KEY,
    }
    params = {"q": query, "count": _normalize_count(count)}

    try:
        response = _SESSION.get(
            _SEARCH_URL,
            headers=headers,
            params=params,
            timeout=config.HTTP_TIMEOUT,
        )
    except requests.RequestException as exc:
        logger.warning("Сетевая ошибка при поиске «%s»: %s", query, exc)
        return f"Ошибка поиска: сетевая ошибка ({exc.__class__.__name__})."

    if response.status_code != 200:
        logger.warning("Brave Search вернул HTTP %s для запроса «%s»", response.status_code, query)
        return f"Ошибка поиска: HTTP {response.status_code}."

    try:
        data = response.json()
    except ValueError:
        logger.warning("Brave Search вернул некорректный JSON для запроса «%s»", query)
        return "Ошибка поиска: некорректный JSON в ответе."

    items = data.get("web", {}).get("results", [])
    results = []
    for item in items:
        results.append(
            "Title: {title}\nURL: {url}\nSnippet: {snippet}".format(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
            )
        )

    if not results:
        logger.info("Brave: по запросу «%s» ничего не найдено", query)
        return "Ничего не найдено."

    logger.info("Brave: по запросу «%s» найдено результатов: %s", query, len(results))
    return "\n---\n".join(results)
