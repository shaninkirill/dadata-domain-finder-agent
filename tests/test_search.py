"""Тесты провайдеров поиска: парсер XML Яндекса и нормализация count."""

import os
import sys
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
sys.path.insert(0, r"C:\my_projects\dadata-domain-finder-agent")

import config  # noqa: E402
from tools.search_brave import _normalize_count as brave_count  # noqa: E402
from tools.search_yandex import _normalize_count as yandex_count  # noqa: E402
from tools.search_yandex import parse_yandex_xml  # noqa: E402

SAMPLE_XML = """<?xml version="1.0" encoding="utf-8"?>
<yandexsearch version="1.0">
  <response>
    <results>
      <grouping>
        <group>
          <doc id="0">
            <url>https://dadata.ru</url>
            <domain>dadata.ru</domain>
            <title>Да&#210;ата — сервисы проверки контактных данных</title>
            <passages>
              <passage>Найдено по ИНН 7721581040.</passage>
              <passage>Официальный сайт компании.</passage>
            </passages>
          </doc>
        </group>
        <group>
          <doc id="1">
            <url>https://rusprofile.ru/org/7721581040</url>
            <domain>rusprofile.ru</domain>
            <title>ООО «Дейта Кью» — Rusprofile</title>
            <headline>Справочник: реквизиты, ОГРН, отчётность.</headline>
          </doc>
        </group>
        <group>
          <doc id="2">
            <url>https://example.com/no-title</url>
            <domain>example.com</domain>
          </doc>
        </group>
      </grouping>
    </results>
  </response>
</yandexsearch>"""


class YandexXmlParserTests(unittest.TestCase):
    """Разбор XML-ответа Yandex Search API."""

    def test_parses_docs(self):
        results = parse_yandex_xml(SAMPLE_XML.encode("utf-8"))
        self.assertEqual(len(results), 2)  # doc без title пропускается
        self.assertEqual(results[0]["url"], "https://dadata.ru")
        self.assertEqual(results[0]["snippet"], "Найдено по ИНН 7721581040. Официальный сайт компании.")
        self.assertEqual(results[1]["snippet"], "Справочник: реквизиты, ОГРН, отчётность.")
        self.assertTrue(results[0]["title"].startswith("Да"))

    def test_error_15_means_nothing_found(self):
        xml = b'<?xml version="1.0"?><yandexsearch><response><error code="15">nothing</error></response></yandexsearch>'
        self.assertEqual(parse_yandex_xml(xml), [])

    def test_other_error_raises_value_error(self):
        xml = b'<?xml version="1.0"?><yandexsearch><response><error code="48">limit</error></response></yandexsearch>'
        with self.assertRaises(ValueError):
            parse_yandex_xml(xml)


class CountNormalizationTests(unittest.TestCase):
    """Нормализация количества результатов для обоих провайдеров."""

    def test_yandex(self):
        for value, expected in [(0, config.SEARCH_RESULTS_COUNT), (1, 1), (5, 5), (100, 100), (101, config.SEARCH_RESULTS_COUNT), (-1, config.SEARCH_RESULTS_COUNT)]:
            with self.subTest(value=value):
                self.assertEqual(yandex_count(value), expected)

    def test_brave(self):
        for value, expected in [(0, config.SEARCH_RESULTS_COUNT), (20, 20), (21, config.SEARCH_RESULTS_COUNT), (-1, config.SEARCH_RESULTS_COUNT)]:
            with self.subTest(value=value):
                self.assertEqual(brave_count(value), expected)


if __name__ == "__main__":
    unittest.main()
