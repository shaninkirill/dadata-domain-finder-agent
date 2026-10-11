"""
Инструмент веб-поиска: единая точка для агента с переключаемым провайдером.

Активный провайдер задаётся переменной окружения SEARCH_PROVIDER:
- «yandex» — Yandex Search API v2 (Yandex Cloud), лучше подходит для
  русскоязычного поиска (используется по умолчанию);
- «brave» — Brave Search API.

Оба провайдера возвращают результаты в одинаковом текстовом формате
(Title/URL/Snippet), поэтому агенту не важно, какой из них активен.
"""

import logging

from langchain_core.tools import tool

import config
from tools.search_brave import brave_web_search
from tools.search_yandex import yandex_web_search

logger = logging.getLogger(__name__)

#: Реестр провайдеров: имя → функция (query, count) → строка с результатами.
_PROVIDERS = {
    "yandex": yandex_web_search,
    "brave": brave_web_search,
}


@tool
def web_search(query: str, count: int = 0) -> str:
    """
    Выполняет поиск в интернете по переданному запросу и возвращает текстовые результаты (сниппеты и ссылки).

    Используй инструмент, чтобы найти информацию об организации по ИНН:
    название компании, официальный сайт, упоминания доменов в каталогах.

    :param query: Поисковый запрос, например: «ИНН 7721581040 официальный сайт»
    :param count: Количество результатов (от 1 до 100); если не указано —
                  используется значение по умолчанию из конфигурации
    :return: Строка с результатами поиска (Title, URL, Snippet для каждой находки),
             либо сообщение об ошибке или об отсутствии результатов
    """
    provider = _PROVIDERS.get(config.SEARCH_PROVIDER)
    if provider is None:
        available = ", ".join(sorted(_PROVIDERS))
        return (
            f"Ошибка поиска: неизвестный провайдер «{config.SEARCH_PROVIDER}» "
            f"(доступны: {available})."
        )
    return provider(query, count)
