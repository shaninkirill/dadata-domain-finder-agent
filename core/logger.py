"""
Настройка логирования приложения.

Уровень логирования задаётся переменной окружения LOG_LEVEL (см. config.py),
формат единый для всех модулей пайплайна.
"""

import logging
import sys

import config

#: Формат записей лога.
_LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging() -> None:
    """
    Настраивает консольное логирование для всего приложения.

    :return: None
    """
    try:
        # Защита от UnicodeEncodeError в консолях с не-UTF8 кодировкой.
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format=_LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Подавляем шумные логи HTTP-библиотек.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
