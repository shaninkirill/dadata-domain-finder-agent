"""Тесты core.validators: контрольная сумма ИНН, нормализация и блоклист доменов."""

import os
import sys
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
sys.path.insert(0, r"C:\my_projects\dadata-domain-finder-agent")

from core.validators import (  # noqa: E402
    is_allowed_domain,
    is_placeholder_domain,
    normalize_domain,
    validate_inn,
)


class ValidateInnTests(unittest.TestCase):
    """Контрольная сумма и формат ИНН."""

    def test_valid_legal_entity_inns(self):
        """Реальные действующие ИНН юридических лиц проходят проверку."""
        for inn in ("7721581040", "7707083893", "1831203731", "7714270020", "2725016474"):
            with self.subTest(inn=inn):
                self.assertTrue(validate_inn(inn))

    def test_valid_individual_entrepreneur_inn(self):
        """Известный корректный 12-значный ИНН индивидуального предпринимателя."""
        self.assertTrue(validate_inn("500100732259"))

    def test_invalid_checksum(self):
        """Изменение контрольной цифры и заведомо неверная сумма отклоняются."""
        for inn in ("7721581041", "500100732258", "1234567890"):
            with self.subTest(inn=inn):
                self.assertFalse(validate_inn(inn))

    def test_wrong_format(self):
        """Неправильная длина и нецифровые символы отклоняются."""
        for inn in ("12345", "7721581040123", "772158104a", "", "  "):
            with self.subTest(inn=inn):
                self.assertFalse(validate_inn(inn))

    def test_surrounding_whitespace_tolerated(self):
        """Пробелы по краям не ломают проверку."""
        self.assertTrue(validate_inn("  7721581040 "))


class NormalizeDomainTests(unittest.TestCase):
    """Нормализация доменов."""

    def test_variants(self):
        cases = [
            ("https://www.DaData.ru/about?utm=1", "dadata.ru"),
            ("dadata.ru", "dadata.ru"),
            ("http://dadata.ru:8080/", "dadata.ru"),
            ("www.sberbank.ru", "sberbank.ru"),
            ("сайт.рф", "сайт.рф"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_domain(raw), expected)

    def test_garbage_returns_none(self):
        for raw in ("не домен", None, "", "just text with spaces"):
            with self.subTest(raw=raw):
                self.assertIsNone(normalize_domain(raw))


class AllowedDomainTests(unittest.TestCase):
    """Блоклист каталогов, агрегаторов и соцсетей."""

    def test_blocked(self):
        for domain in ("rusprofile.ru", "checko.ru", "www.checko.ru", "kontur.checko.ru", "vk.com", "nalog.ru"):
            with self.subTest(domain=domain):
                self.assertFalse(is_allowed_domain(domain))

    def test_allowed(self):
        for domain in ("dadata.ru", "ozon.ru", "сайт.рф"):
            with self.subTest(domain=domain):
                self.assertTrue(is_allowed_domain(domain))


class PlaceholderDomainTests(unittest.TestCase):
    """Домены-заглушки."""

    def test_placeholders(self):
        for domain in ("example.com", "example.ru", "test.com", "test.ru"):
            with self.subTest(domain=domain):
                self.assertTrue(is_placeholder_domain(domain))

    def test_empty_is_not_placeholder(self):
        self.assertFalse(is_placeholder_domain(""))
        self.assertFalse(is_placeholder_domain(None))

    def test_real_domain_not_placeholder(self):
        self.assertFalse(is_placeholder_domain("dadata.ru"))


if __name__ == "__main__":
    unittest.main()
