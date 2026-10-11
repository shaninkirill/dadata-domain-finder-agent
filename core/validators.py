"""
Валидация и нормализация базовых сущностей пайплайна: ИНН и доменов.

Функции используются и точкой входа (pipeline.py), и агентом
(agents/domain_finder.py), поэтому вынесены в общий модуль core.
"""

import re

#: ИНН: 10 цифр для юридических лиц, 12 для ИП (проверяется контрольная сумма).
_INN_LENGTHS: tuple[int, ...] = (10, 12)

#: Веса контрольной суммы 10-значного ИНН (первые 9 разрядов).
_INN_WEIGHTS_10: tuple[int, ...] = (2, 4, 10, 3, 5, 9, 4, 6, 8)

#: Веса первой контрольной цифры 12-значного ИНН (первые 10 разрядов).
_INN_WEIGHTS_12_FIRST: tuple[int, ...] = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)

#: Веса второй контрольной цифры 12-значного ИНН (первые 11 разрядов).
_INN_WEIGHTS_12_SECOND: tuple[int, ...] = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)

#: Приблизительная форма домена (поддерживает кириллицу, например «сайт.рф»).
_DOMAIN_SHAPE_RE = re.compile(
    r"^[a-z0-9а-яё][a-z0-9а-яё-]*(?:\.[a-z0-9а-яё-]+)+$", re.IGNORECASE
)

#: Домены, не являющиеся ответом по своей природе: каталоги и справочники компаний,
#: агрегаторы, соцсети и госреестры. LLM нередко выдаёт их вместо сайта организации —
#: это домены-источники, на которых модель нашла данные.
BLOCKED_DOMAINS: frozenset[str] = frozenset(
    {
        # Каталоги и справочники компаний
        "rusprofile.ru",
        "checko.ru",
        "list-org.com",
        "companies.ru",
        "egrul.ru",
        "zachestnyibiznes.ru",
        "audit-it.ru",
        "sbis.ru",
        "casebook.ru",
        "ruscompany.com",
        "org-info.com",
        # Госреестры и госуслуги
        "nalog.ru",
        "nalog.gov.ru",
        "gosuslugi.ru",
        "zakupki.gov.ru",
        # Соцсети
        "vk.com",
        "ok.ru",
        "t.me",
        "dzen.ru",
        "youtube.com",
    }
)


def _inn_control_digit(digits: str, weights: tuple[int, ...]) -> int:
    """
    Вычисляет контрольную цифру ИНН по весам разрядов.

    :param digits: Разряды ИНН, участвующие в расчёте
    :param weights: Веса разрядов (длина равна len(digits))
    :return: Контрольная цифра (0–9)
    """
    control = sum(int(digit) * weight for digit, weight in zip(digits, weights)) % 11
    return control % 10


def validate_inn(inn: str) -> bool:
    """
    Проверяет формат и контрольную сумму ИНН (10 цифр для ЮЛ, 12 для ИП).

    :param inn: Строка с ИНН
    :return: True, если формат и контрольная сумма корректны, иначе False
    """
    digits = inn.strip()
    if not digits.isdigit() or len(digits) not in _INN_LENGTHS:
        return False

    if len(digits) == 10:
        return _inn_control_digit(digits[:9], _INN_WEIGHTS_10) == int(digits[9])

    if _inn_control_digit(digits[:10], _INN_WEIGHTS_12_FIRST) != int(digits[10]):
        return False
    return _inn_control_digit(digits[:11], _INN_WEIGHTS_12_SECOND) == int(digits[11])


def is_domain_like(value: str) -> bool:
    """
    Проверяет, что строка похожа на домен (без схемы, пути и параметров).

    :param value: Проверяемая строка
    :return: True, если строка похожа на домен, иначе False
    """
    return bool(_DOMAIN_SHAPE_RE.match(value.strip()))


def is_allowed_domain(domain: str) -> bool:
    """
    Проверяет, что домен может быть основным сайтом организации.

    Блокируются каталоги компаний, агрегаторы, соцсети и госреестры: модель
    может выдать их вместо сайта самой организации.

    :param domain: Нормализованный домен (в нижнем регистре, без «www»)
    :return: True, если домен не входит в чёрный список, иначе False
    """
    lowered = domain.strip().lower().rstrip(".")
    return not any(
        lowered == blocked or lowered.endswith("." + blocked)
        for blocked in BLOCKED_DOMAINS
    )


#: Домены-заглушки, которые LLM склонна «конструировать» из названий организаций
#: (например, ООО «Пример Ком» → example.com). Такого ответа быть не может.
PLACEHOLDER_DOMAINS: frozenset[str] = frozenset(
    {
        "example.com",
        "example.org",
        "example.net",
        "example.ru",
        "test.com",
        "test.ru",
        "сайт.рф",
        "домен.рф",
    }
)


def is_placeholder_domain(domain: str | None) -> bool:
    """
    Проверяет, что домен является заглушкой, а не реальным сайтом организации.

    :param domain: Нормализованный домен или None
    :return: True, если домен входит в список доменов-заглушек, иначе False
    """
    if not domain:
        return False
    return domain.strip().lower().rstrip(".") in PLACEHOLDER_DOMAINS


def normalize_domain(raw_domain: str | None) -> str | None:
    """
    Приводит домен к стандартному виду: без схемы, «www», порта, пути и параметров, в нижнем регистре.

    :param raw_domain: Сырая строка с доменом, например «https://www.DaData.ru/about»
    :return: Нормализованный домен («dadata.ru») или None, если строка не похожа на домен
    """
    if not raw_domain:
        return None

    domain = raw_domain.strip().lower()
    domain = re.sub(r"^[a-z][a-z0-9+.-]*://", "", domain)  # схема (http://, https://)
    domain = domain.split("/", 1)[0]  # путь
    domain = domain.split("?", 1)[0].split("#", 1)[0]  # query и фрагмент
    domain = domain.split("@", 1)[-1]  # логин@хост
    domain = domain.split(":", 1)[0]  # порт
    domain = domain.strip(". ")
    if domain.startswith("www."):
        domain = domain[len("www.") :]

    if not is_domain_like(domain):
        return None
    return domain
