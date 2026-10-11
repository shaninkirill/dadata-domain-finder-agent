"""
Демонстрация работоспособности пайплайна на наборе кейсов.

В отличие от pipeline.py, входные данные берутся не из JSON-файлов в папке
input/, а из списка DEMO_CASES ниже. Для каждого кейса запускается агентный
поиск домена по ИНН, результат сравнивается с ожидаемым, печатается сводная
таблица и сохраняется отчёт в output/demo_results.json.

Кейсы:
1. базовый — организация с известным сайтом;
2. проблемный — сайт с TLS-сертификатом национального УЦ (SSL-fallback);
3. негативный — ликвидированная организация без сайта (ожидается null);
4. действующая организация без сайта (ожидается null);
5. действующая организация, представленная только в каталогах и соцсетях (ожидается null).

Пример запуска:
    python demo.py        # все кейсы
    python demo.py 2      # только кейс №2
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime

import agents
import config
import core

logger = logging.getLogger("demo")

#: Демонстрационные кейсы: ИНН, ожидаемый домен (None — сайт не должен быть найден).
DEMO_CASES: list[dict] = [
    {
        "inn": "7721581040",
        "expected": "dadata.ru",
        "description": "Базовый кейс: ООО «Дейта Кью» (DaData)",
    },
    {
        "inn": "7707083893",
        "expected": "sberbank.ru",
        "description": "Проблемный кейс: ПАО Сбербанк (TLS-сертификат национального УЦ)",
    },
    {
        "inn": "7714270020",
        "expected": None,
        "description": "Негативный кейс: ликвидированное ООО «Пример Ком» без сайта",
    },
    {
        "inn": "1831203731",
        "expected": None,
        "description": "Действующее ООО «Кольцо» (Ижевск) без сайта",
    },
    {
        "inn": "2725016474",
        "expected": None,
        "description": "ГСК №57 (Хабаровск): действующий кооператив, только каталоги/соцсети",
    },
]


def run_case(agent: agents.DomainFinderAgent, case: dict) -> dict:
    """
    Запускает один демонстрационный кейс и сравнивает результат с ожиданием.

    :param agent: Экземпляр агента поиска домена
    :param case: Словарь кейса с ключами inn, expected, description
    :return: Словарь результата: inn, expected, got, passed, seconds, answer, error
    """
    started = time.monotonic()
    error: str | None = None
    answer = ""
    got: str | None = None
    try:
        result = agent.run(case["inn"])
        got = result["domain"]
        answer = (result["answer"] or "")[:200]
    except RuntimeError as exc:
        error = str(exc)

    seconds = round(time.monotonic() - started, 1)
    expected = case["expected"]
    expected_set = expected if isinstance(expected, set) else {expected}
    expected_label = " / ".join(sorted(str(item) for item in expected_set))
    return {
        "inn": case["inn"],
        "description": case["description"],
        "expected": expected_label,
        "got": got,
        "passed": error is None and got in expected_set,
        "seconds": seconds,
        "answer": answer,
        "error": error,
    }


def print_report(results: list[dict]) -> None:
    """
    Печатает сводную таблицу результатов по кейсам.

    :param results: Список результатов, возвращённых run_case
    :return: None
    """
    width = 84
    print("\n" + "=" * width)
    print(f"{'№':<3} {'ИНН':<13} {'Ожидание':<13} {'Получено':<13} {'Статус':<6} {'Время'}")
    print("-" * width)
    for index, res in enumerate(results, start=1):
        status = "OK" if res["passed"] else "FAIL"
        print(
            f"{index:<3} {res['inn']:<13} {res['expected']:<13} "
            f"{str(res['got']):<13} {status:<6} {res['seconds']}с"
        )
        if res["error"]:
            print(f"    Ошибка: {res['error']}")
    print("=" * width)
    passed = sum(1 for res in results if res["passed"])
    print(f"Пройдено кейсов: {passed} из {len(results)}")


def save_report(results: list[dict]) -> None:
    """
    Сохраняет отчёт по кейсам в output/demo_results.json.

    :param results: Список результатов, возвращённых run_case
    :return: None
    """
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.OUTPUT_DIR / "demo_results.json"
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    logger.info("Отчёт сохранён: %s", report_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Разбирает аргументы командной строки.

    :param argv: Список аргументов (по умолчанию — аргументы запуска скрипта)
    :return: Namespace с полем case — номером кейса или None
    """
    parser = argparse.ArgumentParser(
        description="Демонстрация пайплайна поиска домена по ИНН на наборе кейсов"
    )
    parser.add_argument(
        "case",
        nargs="?",
        type=int,
        default=None,
        help="Номер кейса (начиная с 1); если не указан — выполняются все кейсы",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Запускает демонстрационные кейсы, печатает отчёт и сохраняет его в JSON.

    :param argv: Аргументы командной строки
    :return: Код возврата: 0 — все кейсы пройдены, 1 — есть непройденные
             или ошибка конфигурации
    """
    core.setup_logging()
    try:
        config.validate_config()
    except RuntimeError as exc:
        logger.error("Ошибка конфигурации: %s", exc)
        return 1

    args = parse_args(argv)
    if args.case is not None:
        if not 1 <= args.case <= len(DEMO_CASES):
            logger.error("Кейс №%s не существует (доступны 1–%s)", args.case, len(DEMO_CASES))
            return 1
        cases = [DEMO_CASES[args.case - 1]]
    else:
        cases = DEMO_CASES

    logger.info("=== Демонстрация пайплайна: %s кейс(ов) ===", len(cases))
    agent = agents.DomainFinderAgent()
    results = [run_case(agent, case) for case in cases]

    print_report(results)
    save_report(results)

    return 0 if all(res["passed"] for res in results) else 1


if __name__ == "__main__":
    sys.exit(main())
