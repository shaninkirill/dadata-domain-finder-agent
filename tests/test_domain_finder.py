"""Тесты агента: парсинг ответов LLM, evidence-гвард, INN-проверка, дедуп ключей."""

import os
import sys
import unittest
from typing import Any

os.environ.setdefault("OPENAI_API_KEY", "test-key")
sys.path.insert(0, r"C:\my_projects\dadata-domain-finder-agent")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from agents.domain_finder import (  # noqa: E402
    DomainFinderAgent,
    iter_json_snippets,
)
from core.validators import normalize_domain  # noqa: E402


class ScriptedLLM(BaseChatModel):
    """LLM, возвращающая заранее заданные ответы (для оффлайн-тестов графа)."""

    responses: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.responses.pop(0)))])

    def bind_tools(self, tools, **kwargs: Any):
        return self


class ParseDomainTests(unittest.TestCase):
    """Разбор финальных ответов LLM через JsonOutputParser и fallback'и."""

    def setUp(self):
        self.agent = DomainFinderAgent()

    def test_cases(self):
        cases = [
            ('{"domain": "dadata.ru"}', (True, "dadata.ru")),
            ('Ответ:\n```json\n{"domain": "dadata.ru"}\n```', (True, "dadata.ru")),
            ('{"domain": null}', (True, None)),
            ('{"domain": null}\nПояснение...', (True, None)),
            ('Сначала текст {"domain": "dadata.ru"} и ещё текст', (True, "dadata.ru")),
            ('{"Domain": "X.RU"}', (True, "X.RU")),
            ('[{"domain": "dadata.ru"}]', (True, "dadata.ru")),
            ('{"domain": "нет"}', (True, None)),
            ('{"domain": 123}', (False, None)),
            ('{"url": "dadata.ru"}', (False, None)),
            ("dadata.ru", (True, "dadata.ru")),
            ('"dadata.ru"', (True, "dadata.ru")),
            ("Не нашёл сайт", (False, None)),
            ("", (False, None)),
        ]
        for text, expected in cases:
            with self.subTest(text=text[:25]):
                self.assertEqual(DomainFinderAgent._parse_domain(text), expected)


class NormalizeInAgentTests(unittest.TestCase):
    """normalize_domain доступен агенту и консистентен."""

    def test_cases(self):
        self.assertEqual(normalize_domain("https://www.DaData.ru/about"), "dadata.ru")
        self.assertIsNone(normalize_domain("не домен"))


class JsonSnippetTests(unittest.TestCase):
    """Извлечение JSON-фрагментов, окружённых текстом."""

    def test_embedded(self):
        snippets = list(iter_json_snippets('текст {"a": {"b": 1}} конец {"c": 2}'))
        self.assertEqual(len(snippets), 2)
        self.assertEqual(snippets[0], '{"a": {"b": 1}}')
        self.assertEqual(snippets[1], '{"c": 2}')


class CallKeyTests(unittest.TestCase):
    """Ключи дедупликации вызовов инструментов."""

    def test_same_args_same_key(self):
        call_a = {"name": "web_search", "args": {"query": "x", "count": 5}}
        call_b = {"name": "web_search", "args": {"count": 5, "query": "x"}}
        self.assertEqual(DomainFinderAgent._call_key(call_a), DomainFinderAgent._call_key(call_b))

    def test_different_args_different_key(self):
        call_a = {"name": "web_search", "args": {"query": "x"}}
        call_b = {"name": "web_search", "args": {"query": "y"}}
        self.assertNotEqual(DomainFinderAgent._call_key(call_a), DomainFinderAgent._call_key(call_b))


class OwnershipCheckTests(unittest.TestCase):
    """Мягкая INN-проверка принадлежности домена и проба реквизитных страниц."""

    def setUp(self):
        self.agent = DomainFinderAgent()
        self.inn = "7721581040"

    @staticmethod
    def _messages(main_page_text: str):
        human = HumanMessage(content=f"ИНН организации: {7721581040}. Найди сайт.")
        assistant = AIMessage(
            content="",
            tool_calls=[{"name": "fetch_page", "args": {"url": "https://vasilek.ru"}, "id": "call_1"}],
        )
        tool = ToolMessage(content=main_page_text, tool_call_id="call_1")
        return [human, assistant, tool]

    def test_inn_missing_on_fetched_page(self):
        messages = self._messages("Главная страница кафе «Василёк» без реквизитов")
        self.assertTrue(self.agent._inn_missing_on_page("vasilek.ru", messages))

    def test_inn_present_on_fetched_page(self):
        messages = self._messages(f"О компании. ИНН {self.inn}. Реквизиты.")
        self.assertFalse(self.agent._inn_missing_on_page("vasilek.ru", messages))

    def test_no_fetch_no_bounce(self):
        messages = [HumanMessage(content=f"ИНН организации: {self.inn}.")]
        self.assertFalse(self.agent._inn_missing_on_page("vasilek.ru", messages))

    def test_probe_finds_inn_on_contacts_page(self):
        class FakeFetch:
            @staticmethod
            def invoke(inp):
                url = inp["url"]
                text = f"Страница {url}."
                if "/contacts" in url or "/about" in url:
                    text += f" ИНН {self.inn}."
                return text

        self.agent._tool_map["fetch_page"] = FakeFetch()
        messages = self._messages("Главная страница без ИНН")
        found = self.agent._search_inn_on_domain("vasilek.ru", self.inn, messages)
        self.assertEqual(found, "/contacts")

    def test_probe_misses_everywhere(self):
        class FakeFetch:
            @staticmethod
            def invoke(inp):
                return f"Страница {inp['url']} без реквизитов."

        self.agent._tool_map["fetch_page"] = FakeFetch()
        messages = self._messages("Главная страница без ИНН")
        self.assertIsNone(self.agent._search_inn_on_domain("vasilek.ru", self.inn, messages))

    def test_probe_skips_already_fetched_paths(self):
        """Путь, который агент уже загружал, не проверяется повторно."""
        seen = []

        class FakeFetch:
            @staticmethod
            def invoke(inp):
                seen.append(inp["url"])
                return "контент"

        self.agent._tool_map["fetch_page"] = FakeFetch()
        messages = [
            HumanMessage(content=f"ИНН организации: {self.inn}."),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "fetch_page", "args": {"url": "https://vasilek.ru/contacts"}, "id": "c1"}
                ],
            ),
            ToolMessage(content="Страница контактов без ИНН.", tool_call_id="c1"),
        ]
        self.assertIsNone(self.agent._search_inn_on_domain("vasilek.ru", self.inn, messages))
        self.assertEqual(seen, ["https://vasilek.ru/about", "https://vasilek.ru/rekvizity"])


class EvidenceGuardTests(unittest.TestCase):
    """Evidence-гвард: домен должен многократно встречаться в наблюдениях."""

    def setUp(self):
        self.agent = DomainFinderAgent()

    def test_multiple_mentions_pass(self):
        messages = [
            HumanMessage(content="задача"),
            ToolMessage(
                content="Brave: результаты. URL: https://dadata.ru Title: ДаData",
                tool_call_id="e1",
            ),
            ToolMessage(content="Ещё результаты. URL: https://dadata.ru/about", tool_call_id="e2"),
        ]
        self.assertTrue(self.agent._is_evidenced("dadata.ru", messages))

    def test_single_mention_fails(self):
        messages = [
            HumanMessage(content="задача"),
            ToolMessage(
                content="Brave: URL: https://woodtec.com.ru один результат",
                tool_call_id="e1",
            ),
        ]
        self.assertFalse(self.agent._is_evidenced("woodtec.com.ru", messages))


class ValidateNodeTests(unittest.TestCase):
    """Узел validate на синтетических состояниях графа."""

    def setUp(self):
        self.agent = DomainFinderAgent()

    @staticmethod
    def _state(text, retries=0):
        """Состояние графа с поисковым наблюдением, подтверждающим dadata.ru."""
        evidence = ToolMessage(
            content="Brave: URL: https://dadata.ru Title: ДаData URL: https://dadata.ru/about",
            tool_call_id="search_1",
        )
        return {
            "messages": [
                HumanMessage(content=f"ИНН организации: {7721581040}. Найди сайт."),
                AIMessage(content=text),
                evidence,
            ],
            "steps": 1,
            "validation_retries": retries,
            "needs_agent_retry": False,
        }

    def test_catalog_domain_triggers_retry(self):
        update = self.agent._validation_node(self._state('{"domain": "rusprofile.ru"}'))
        self.assertTrue(update["needs_agent_retry"])

    def test_placeholder_triggers_retry(self):
        update = self.agent._validation_node(self._state('{"domain": "example.com"}'))
        self.assertTrue(update["needs_agent_retry"])

    def test_valid_domain_passes(self):
        update = self.agent._validation_node(self._state('{"domain": "dadata.ru"}'))
        self.assertFalse(update["needs_agent_retry"])

    def test_null_passes_without_retry(self):
        update = self.agent._validation_node(self._state('{"domain": null}'))
        self.assertFalse(update["needs_agent_retry"])

    def test_retry_limit_exhausted(self):
        """После исчерпания MAX_VALIDATION_RETRIES замечания прекращаются."""
        update = self.agent._validation_node(self._state('{"domain": "rusprofile.ru"}', retries=2))
        self.assertFalse(update["needs_agent_retry"])


class GraphStructureTests(unittest.TestCase):
    """Сборка графа и подсказки провайдера."""

    def test_nodes(self):
        agent = DomainFinderAgent()
        nodes = set(agent._graph.get_graph().nodes)
        self.assertIn("agent", nodes)
        self.assertIn("tools", nodes)
        self.assertIn("validate", nodes)

    def test_provider_hint_402(self):
        class FakeError(Exception):
            status_code = 402

        hint = DomainFinderAgent._provider_hint(FakeError()) if hasattr(DomainFinderAgent, "_provider_hint") else None
        from agents.domain_finder import _provider_hint  # noqa: PLC0415

        self.assertIn("402", _provider_hint(FakeError()))

    def test_provider_hint_empty_for_other(self):
        from agents.domain_finder import _provider_hint  # noqa: PLC0415

        self.assertEqual(_provider_hint(ValueError("x")), "")


if __name__ == "__main__":
    unittest.main()
