"""
Фабрика LLM-клиента: Caila через OpenAI-совместимый интерфейс.

Caila (https://caila.io) — LLM-шлюз Just AI с единым OpenAI-совместимым API
(300+ моделей), поэтому используется ChatOpenAI из langchain_openai
с кастомным base_url.
"""

import logging

from langchain_openai import ChatOpenAI

import config

logger = logging.getLogger(__name__)


def create_llm(**overrides) -> ChatOpenAI:
    """
    Создаёт клиент LLM через OpenAI-совместимый API Caila.

    :param overrides: Дополнительные именованные параметры для ChatOpenAI,
                      переопределяющие значения из конфигурации
    :return: Настроенный экземпляр ChatOpenAI, готовый к bind_tools и invoke
    :raises RuntimeError: Если не задан OPENAI_API_KEY
    """
    if not config.OPENAI_API_KEY:
        raise RuntimeError("Не задан OPENAI_API_KEY — проверьте файл .env")

    params = dict(
        model=config.LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
        timeout=config.LLM_TIMEOUT,
        max_retries=config.LLM_MAX_RETRIES,
    )
    params.update(overrides)

    logger.info(
        "Создан LLM-клиент: модель=%s, base_url=%s", config.LLM_MODEL, config.OPENAI_BASE_URL
    )
    return ChatOpenAI(**params)
