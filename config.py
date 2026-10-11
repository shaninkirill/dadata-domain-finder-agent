"""
Конфигурация проекта: загрузка переменных окружения из .env и глобальные константы.

Все настройки берутся из файла .env (шаблон — .env.example) с откатом
на разумные значения по умолчанию. Настройка логирования — в core/logger.py.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

#: Корневая директория проекта.
BASE_DIR: Path = Path(__file__).resolve().parent

# Загружаем переменные окружения из .env, не перезаписывая уже установленные.
load_dotenv(BASE_DIR / ".env", override=False)

# --- Настройки LLM (Caila, OpenAI-совместимый API) ---

#: API-ключ Caila (https://caila.io), формат «caila-...».
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

#: Базовый URL OpenAI-совместимого API Caila.
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://caila.io/api/openai/v1")

#: Название модели из каталога Caila (https://caila.io, 300+ моделей).
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")

#: Температура генерации (низкая — для детерминированных ответов).
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

#: Таймаут одного запроса к LLM, секунды.
LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))

#: Количество автоматических повторов при ошибках запроса к LLM.
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))

# --- Web Search (общие настройки) ---

#: Активный провайдер веб-поиска: «yandex» (Yandex Search API) или «brave» (Brave Search API).
SEARCH_PROVIDER: str = os.getenv("SEARCH_PROVIDER", "yandex").strip().lower()

#: Количество результатов на один поисковый запрос по умолчанию.
SEARCH_RESULTS_COUNT: int = int(os.getenv("SEARCH_RESULTS_COUNT", "7"))

# --- Yandex Search API v2 (https://yandex.cloud/services/search-api) ---

#: API-ключ сервисного аккаунта Yandex Cloud (заголовок Authorization: Api-Key).
YANDEX_SEARCH_API_KEY: str = os.getenv("YANDEX_SEARCH_API_KEY", "")

#: Идентификатор каталога Yandex Cloud (folderId), в котором активирован Search API.
YANDEX_SEARCH_FOLDER_ID: str = os.getenv("YANDEX_SEARCH_FOLDER_ID", "")

#: Необязательный ID региона поиска (влияет на ранжирование; только для SEARCH_TYPE_RU).
YANDEX_SEARCH_REGION: str = os.getenv("YANDEX_SEARCH_REGION", "")

#: Endpoint синхронного веб-поиска Yandex Search API v2.
YANDEX_SEARCH_API_URL: str = os.getenv(
    "YANDEX_SEARCH_API_URL",
    "https://searchapi.api.cloud.yandex.net/v2/web/search",
)

#: Максимально допустимое количество результатов одного запроса (groupsOnPage).
YANDEX_MAX_RESULTS_COUNT: int = 100

# --- Brave Search API (https://brave.com/search/api) ---

#: API-ключ Brave Search (используется при SEARCH_PROVIDER=brave).
BRAVE_API_KEY: str = os.getenv("BRAVE_API_KEY", "")

#: Максимально допустимое количество результатов одного запроса в Brave Search API.
BRAVE_MAX_RESULTS_COUNT: int = 20

# --- Общие HTTP-настройки ---

#: Таймаут внешних HTTP-запросов, секунды.
HTTP_TIMEOUT: int = int(os.getenv("HTTP_TIMEOUT", "15"))

#: User-Agent для исходящих запросов.
USER_AGENT: str = os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; DomainFinderAgent/1.0)")

# --- Настройки агента ---

#: Максимальное число шагов графа агента (вызовы LLM + инструменты).
MAX_AGENT_STEPS: int = int(os.getenv("MAX_AGENT_STEPS", "24"))

#: Обрезка текста страницы для контекста LLM, символов.
FETCH_PAGE_MAX_CHARS: int = int(os.getenv("FETCH_PAGE_MAX_CHARS", "4000"))

#: Обрезка истории переписки для принудительного извлечения ответа, символов.
MAX_TRANSCRIPT_CHARS: int = int(os.getenv("MAX_TRANSCRIPT_CHARS", "12000"))

#: Сколько раз узел validate может вернуть агента на доработку ответа.
MAX_VALIDATION_RETRIES: int = int(os.getenv("MAX_VALIDATION_RETRIES", "2"))

# --- Пути ---

#: Папка с входными JSON-файлами.
INPUT_DIR: Path = Path(os.getenv("INPUT_DIR", str(BASE_DIR / "input")))

#: Папка с результатами.
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "output")))

# --- Логирование ---

#: Уровень логирования (DEBUG, INFO, WARNING, ERROR).
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


def validate_config() -> None:
    """
    Проверяет обязательные параметры конфигурации перед запуском пайплайна.

    :return: None
    :raises RuntimeError: Если не задан обязательный API-ключ LLM (OPENAI_API_KEY)
                          или выбран неизвестный провайдер веб-поиска
    """
    log = logging.getLogger(__name__)
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Не задан OPENAI_API_KEY. Скопируйте .env.example в .env и укажите ключ Caila."
        )
    if SEARCH_PROVIDER not in ("yandex", "brave"):
        raise RuntimeError(
            f"Неизвестный SEARCH_PROVIDER: {SEARCH_PROVIDER!r} (поддерживаются yandex и brave)."
        )
    if SEARCH_PROVIDER == "yandex" and (
        not YANDEX_SEARCH_API_KEY or not YANDEX_SEARCH_FOLDER_ID
    ):
        log.warning(
            "SEARCH_PROVIDER=yandex, но не заданы YANDEX_SEARCH_API_KEY и/или "
            "YANDEX_SEARCH_FOLDER_ID — инструмент web_search будет сообщать об ошибке. "
            "Активируйте Search API в консоли Yandex Cloud: https://yandex.cloud/services/search-api"
        )
    if SEARCH_PROVIDER == "brave" and not BRAVE_API_KEY:
        log.warning(
            "SEARCH_PROVIDER=brave, но не задан BRAVE_API_KEY — инструмент web_search "
            "будет сообщать об ошибке. Получите ключ на https://brave.com/search/api"
        )
