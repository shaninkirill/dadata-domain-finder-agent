"""
Единая точка входа агентного пайплайна.

Считывает JSON-файл с ИНН организации из папки input/, запускает агента,
который ищет основной сайт организации в интернете, и сохраняет результат
в папку output/ в формате {"domain": "<домен>"} или {"domain": null}.

Пример запуска:
    python pipeline.py input/example.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import agents
import config
import core

logger = logging.getLogger("pipeline")


def load_inn(input_path: Path) -> str:
    """
    Читает ИНН организации из входного JSON-файла.

    :param input_path: Путь к входному файлу, например input/example.json
    :return: Строка с ИНН организации
    :raises FileNotFoundError: Если входной файл не существует
    :raises ValueError: Если файл не является корректным JSON, не содержит поле inn
                        или значение inn не проходит валидацию
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Входной файл не найден: {input_path}")

    with open(input_path, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Файл {input_path} не является корректным JSON: {exc}") from exc

    raw_inn = data.get("inn") if isinstance(data, dict) else None
    if isinstance(raw_inn, int) and not isinstance(raw_inn, bool):
        raw_inn = str(raw_inn)  # допускаем запись числа вместо строки

    if not isinstance(raw_inn, str) or not core.validate_inn(raw_inn):
        raise ValueError(
            f'В файле {input_path} отсутствует корректное поле "inn" '
            "(ожидается строка из 10 или 12 цифр)"
        )
    return raw_inn.strip()


def save_result(output_path: Path, domain: str | None) -> None:
    """
    Сохраняет результат пайплайна в JSON-файл.

    :param output_path: Путь к выходному файлу, например output/result.json
    :param domain: Найденный домен или None, если сайт не найден
    :return: None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump({"domain": domain}, file, ensure_ascii=False, indent=2)
        file.write("\n")
    logger.info("Результат сохранён: %s", output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Разбирает аргументы командной строки.

    :param argv: Список аргументов (по умолчанию — аргументы запуска скрипта)
    :return: Namespace с полями input и output
    """
    parser = argparse.ArgumentParser(
        description="Поиск основного сайта организации по ИНН (агентный пайплайн)"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Путь к входному JSON-файлу (по умолчанию — первый *.json из папки input/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Путь к выходному JSON-файлу (по умолчанию — output/result.json)",
    )
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """
    Определяет входной и выходной пути на основе аргументов командной строки.

    :param args: Распарсенные аргументы командной строки
    :return: Кортеж (путь к входному файлу, путь к выходному файлу)
    :raises FileNotFoundError: Если путь ввода не указан и в папке input/ нет JSON-файлов
    """
    if args.input:
        input_path = Path(args.input)
    else:
        candidates = sorted(config.INPUT_DIR.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"В папке {config.INPUT_DIR} нет JSON-файлов. "
                "Укажите путь явно: python pipeline.py input/example.json"
            )
        input_path = candidates[0]
        logger.info("Входной файл не указан — используется %s", input_path)

    output_path = Path(args.output) if args.output else config.OUTPUT_DIR / "result.json"
    return input_path, output_path


def main(argv: list[str] | None = None) -> int:
    """
    Основная функция пайплайна: чтение входа, запуск агента, запись результата.

    :param argv: Список аргументов командной строки
    :return: Код возврата: 0 — успешное завершение (включая domain=null),
             1 — ошибка входных данных, конфигурации или выполнения
    """
    core.setup_logging()
    logger.info("=== Запуск пайплайна поиска домена по ИНН ===")

    try:
        config.validate_config()

        input_path, output_path = resolve_paths(parse_args(argv))
        inn = load_inn(input_path)
        logger.info("Входной файл: %s, ИНН: %s", input_path, inn)

        agent = agents.DomainFinderAgent()
        result = agent.run(inn)

        save_result(output_path, result["domain"])
        logger.info(
            "Итоговый ответ: %s",
            json.dumps({"domain": result["domain"]}, ensure_ascii=False),
        )
        return 0

    except (ValueError, FileNotFoundError) as exc:
        logger.error("Ошибка входных данных: %s", exc)
    except RuntimeError as exc:
        logger.error("Ошибка выполнения пайплайна: %s", exc)
    except Exception:
        logger.exception("Непредвиденная ошибка пайплайна")
    return 1


if __name__ == "__main__":
    sys.exit(main())
