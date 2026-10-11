"""
Агент поиска основного сайта организации по ИНН.

Реализован как граф LangGraph с циклом «модель → инструменты → модель» и
финальным узлом validate (детерминированная проверка ответа). Итог разбирается
JsonOutputParser'ом по pydantic-схеме DomainAnswer: {"domain": ...} / {"domain": null}.
"""

import logging
import re
from collections.abc import Iterator
from typing import Annotated, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

import config
from agents import prompts
from agents.llm_factory import create_llm
from core.validators import (
    is_allowed_domain,
    is_domain_like,
    is_placeholder_domain,
    normalize_domain,
)
from tools import fetch_page, web_search


def _idna_form(domain: str) -> str:
    """
    Возвращает punycode-представление домена (для кириллических доменов).

    :param domain: Домен, возможно в кириллице (например «сайт.рф»)
    :return: Punycode-представление («xn--80asaitg.xn--p1ai») или исходный домен
    """
    try:
        return domain.encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError):
        return domain.lower()

logger = logging.getLogger(__name__)


class DomainAnswer(BaseModel):
    """Схема финального ответа агента: описывает ожидаемый JSON и используется для формат-инструкций."""

    domain: str | None = Field(
        default=None,
        description=(
            "Домен основного сайта организации без протокола и «www», например dadata.ru; "
            "null — если сайт не найден или данных недостаточно"
        ),
    )


#: Парсер финального JSON-ответа агента: извлекает JSON-объект из markdown-блоков
#: и окружающего текста, в том числе из частично обрезанных ответов.
_JSON_PARSER = JsonOutputParser(pydantic_object=DomainAnswer)


def iter_json_snippets(text: str) -> Iterator[str]:
    """
    Извлекает из текста сбалансированные по скобкам фрагменты {...} и [...].

    Используется как fallback к JsonOutputParser: тот не находит JSON-объект,
    окружённый пояснительным текстом с обеих сторон (текст до открывающей скобки).

    :param text: Текст ответа модели
    :return: Итератор строк-кандидатов, являющихся потенциально валидным JSON
    """
    stack: list[str] = []
    snippet_start: int | None = None
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            if not stack:
                snippet_start = index
            stack.append(char)
        elif char in "}]":
            if stack:
                stack.pop()
                if not stack and snippet_start is not None:
                    yield text[snippet_start : index + 1]
                    snippet_start = None
    # Незакрытые фрагменты игнорируются: их попытается спасти JsonOutputParser
    # со своим режимом частичного разбора.

#: Пути типовых страниц реквизитов, проверяемые валидатором, когда ИНН не найден
#: на главной странице домена-кандидата.
_OWNERSHIP_PROBE_PATHS: tuple[str, ...] = ("/about", "/contacts", "/rekvizity")

#: Минимальное число упоминаний домена в наблюдениях инструментов, чтобы считать
#: его подтверждённым. Настоящие официальные сайты упоминаются в выдаче многократно;
#: случайный «мусорный» домен из шума результатов, как правило, один раз.
MIN_EVIDENCE_MENTIONS: int = 2

#: Инструменты агента по умолчанию: веб-поиск и загрузка страниц.
DEFAULT_TOOLS: list[BaseTool] = [web_search, fetch_page]


def _provider_hint(exc: Exception) -> str:
    """
    Возвращает человекопонятную подсказку по ошибкам провайдера LLM.

    :param exc: Исключение, поднятое при обращении к API LLM
    :return: Текст подсказки (пустая строка, если код ошибки не распознан)
    """
    status = getattr(exc, "status_code", None)
    if status == 402:
        return (
            " | Подсказка: провайдер заблокировал модель по биллингу (HTTP 402). "
            "Проверьте баланс и лимиты в личном кабинете Caila: бонусные кредиты "
            "могут не действовать на дорогие модели. Попробуйте другую LLM_MODEL "
            "(например gpt-4.1-mini или gpt-4o-mini)."
        )
    if status == 429:
        return (
            " | Подсказка: превышен лимит запросов (HTTP 429). Подождите немного "
            "или уменьшите частоту запусков."
        )
    return ""


class AgentState(TypedDict):
    """Внутреннее состояние графа агента."""

    messages: Annotated[list[BaseMessage], add_messages]
    steps: int
    validation_retries: int
    needs_agent_retry: bool


class DomainFinderAgent:
    """
    Агент, который по ИНН организации находит её основной сайт в интернете.

    Граф LangGraph состоит из трёх узлов: «agent» вызывает LLM с привязанными
    инструментами, «tools» выполняет затребованные вызовы, «validate» детерминированно
    проверяет финальный ответ (формат домена и блоклист каталогов/агрегаторов).
    Если ответ не проходит проверку, validate возвращает агента на доработку
    (ограниченное число раз), иначе из истории сообщений извлекается
    JSON вида {"domain": ...}.
    """

    def __init__(self, llm: BaseChatModel | None = None, tools: list[BaseTool] | None = None) -> None:
        """
        Инициализирует агента: модель, инструменты и скомпилированный граф LangGraph.

        :param llm: Экземпляр языковой модели (если None — создаётся клиент Caila)
        :param tools: Список инструментов агента (если None — инструменты по умолчанию)
        :return: None
        :raises RuntimeError: Если не задан API-ключ LLM
        """
        self._llm = llm if llm is not None else create_llm()
        self._tools = list(tools) if tools is not None else list(DEFAULT_TOOLS)
        self._tool_map = {tool.name: tool for tool in self._tools}
        self._model_with_tools = self._llm.bind_tools(self._tools)
        # Инструкции формата ответа (по pydantic-схеме) — встраиваются в промпты.
        self._format_instructions = _JSON_PARSER.get_format_instructions()
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def run(self, inn: str) -> dict[str, str | None]:
        """
        Запускает агентный поиск основного домена организации по ИНН.

        :param inn: ИНН организации (10 или 12 цифр)
        :return: Словарь {"domain": <домен или None>, "answer": <финальный текст ответа агента>}
        :raises RuntimeError: При ошибке выполнения графа или обращении к LLM
        """
        logger.info("Запуск агентного поиска домена для ИНН %s", inn)
        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=prompts.build_system_prompt(self._format_instructions)),
                HumanMessage(content=prompts.build_user_prompt(inn)),
            ],
            "steps": 0,
            "validation_retries": 0,
            "needs_agent_retry": False,
        }

        try:
            final_state = self._graph.invoke(
                initial_state,
                config={"recursion_limit": config.MAX_AGENT_STEPS * 3},
            )
        except GraphRecursionError as exc:
            raise RuntimeError("Превышен лимит итераций агента") from exc
        except Exception as exc:
            raise RuntimeError(f"Ошибка выполнения графа агента: {exc}") from exc

        domain, answer = self._extract_answer(final_state["messages"])
        normalized_domain = normalize_domain(domain)
        if domain is not None and normalized_domain is None:
            logger.warning("Агент вернул некорректный домен «%s» — результат null", domain)
        if is_placeholder_domain(normalized_domain):
            logger.warning(
                "Домен «%s» — заглушка (сконструирован моделью), итог обнулён",
                normalized_domain,
            )
            normalized_domain = None
        if normalized_domain is not None and not is_allowed_domain(normalized_domain):
            logger.warning(
                "Домен «%s» — каталог/агрегатор/соцсеть, итог обнулён",
                normalized_domain,
            )
            normalized_domain = None
        if normalized_domain is not None and not self._is_evidenced(
            normalized_domain, final_state["messages"]
        ):
            logger.warning(
                "Домен «%s» не подтверждён наблюдениями инструментов — результат null",
                normalized_domain,
            )
            normalized_domain = None

        logger.info("Финальный ответ агента: %s", answer)
        return {"domain": normalized_domain, "answer": answer}

    # ------------------------------------------------------------------
    # Построение и выполнение графа
    # ------------------------------------------------------------------

    def _build_graph(self):
        """
        Строит и компилирует граф агента LangGraph.

        :return: Скомпилированный граф (Runnable), принимающий AgentState на входе
        """
        builder = StateGraph(AgentState)
        builder.add_node("agent", self._agent_node)
        builder.add_node("tools", self._tools_node)
        builder.add_node("validate", self._validation_node)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tools": "tools", "validate": "validate"},
        )
        builder.add_edge("tools", "agent")
        builder.add_conditional_edges(
            "validate",
            self._route_after_validation,
            {"agent": "agent", END: END},
        )
        return builder.compile()

    def _agent_node(self, state: AgentState) -> dict:
        """
        Узел графа: вызов LLM с привязанными инструментами.

        :param state: Текущее состояние графа (история сообщений, счётчик шагов)
        :return: Обновление состояния с ответом модели и увеличенным счётчиком шагов
        :raises RuntimeError: При ошибке обращения к LLM (после встроенных ретраев)
        """
        try:
            response = self._model_with_tools.invoke(state["messages"])
        except Exception as exc:
            logger.exception("Ошибка вызова LLM на шаге %s", state["steps"])
            raise RuntimeError(f"Ошибка обращения к LLM: {exc}{_provider_hint(exc)}") from exc
        return {"messages": [response], "steps": state["steps"] + 1}

    @staticmethod
    def _call_key(tool_call: dict) -> str:
        """
        Строит ключ вызова инструмента для дедупликации повторных запросов.

        :param tool_call: Описание вызова инструмента (name, args)
        :return: Строка вида «web_search|count=5;query=...»
        """
        args = tool_call.get("args") or {}
        rendered = ";".join(f"{key}={value}" for key, value in sorted(args.items()))
        return f"{tool_call.get('name')}|{rendered}"

    def _tools_node(self, state: AgentState) -> dict:
        """
        Узел графа: последовательное выполнение инструментальных вызовов, запрошенных LLM.

        Дословно повторные вызовы (тот же инструмент с теми же аргументами) не
        выполняются повторно — агент получает кэшированный результат с пометкой,
        что запрос уже выполнялся.

        :param state: Текущее состояние графа (последнее сообщение содержит tool_calls)
        :return: Обновление состояния с сообщениями ToolMessage и увеличенным счётчиком шагов
        """
        last_message = state["messages"][-1]
        tool_messages: list[ToolMessage] = []

        # Кэш ранее выполненных вызовов: ключ -> наблюдение.
        observations: dict[str, str] = {
            message.tool_call_id: self._message_text(message)
            for message in state["messages"]
            if isinstance(message, ToolMessage)
        }
        executed: dict[str, str] = {}
        for message in state["messages"]:
            if isinstance(message, AIMessage):
                for call in message.tool_calls or []:
                    key = self._call_key(call)
                    if key not in executed and call.get("id") in observations:
                        executed[key] = observations[call["id"]]

        for tool_call in last_message.tool_calls:
            key = self._call_key(tool_call)
            if key in executed:
                logger.info(
                    "Инструмент %s: дословно повторный вызов — возвращён кэш",
                    tool_call["name"],
                )
                tool_messages.append(
                    ToolMessage(
                        content=(
                            "Этот вызов уже выполнялся, новый запрос не отправлялся. "
                            "Предыдущий результат:\n\n" + executed[key][:2000]
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            tool = self._tool_map.get(tool_call["name"])
            if tool is None:
                tool_messages.append(
                    ToolMessage(
                        content=f"Ошибка: неизвестный инструмент «{tool_call['name']}».",
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            logger.info(
                "Вызов инструмента %s(%s)", tool_call["name"], tool_call.get("args", {})
            )
            try:
                observation = str(tool.invoke(tool_call.get("args", {})))
            except Exception as exc:
                logger.exception("Инструмент %s завершился с ошибкой", tool_call["name"])
                observation = f"Ошибка выполнения инструмента: {exc}"
            executed[key] = observation

            tool_messages.append(
                ToolMessage(content=observation, tool_call_id=tool_call["id"])
            )

        return {"messages": tool_messages, "steps": state["steps"] + 1}

    def _route_after_agent(self, state: AgentState) -> str:
        """
        Определяет следующий узел после вызова LLM.

        :param state: Текущее состояние графа
        :return: «tools», если модель запросила инструменты и лимит шагов не исчерпан;
                 иначе «validate» — проверка финального ответа
        """
        last_message = state["messages"][-1]
        has_tool_calls = isinstance(last_message, AIMessage) and bool(last_message.tool_calls)

        if has_tool_calls and state["steps"] >= config.MAX_AGENT_STEPS:
            logger.warning(
                "Достигнут лимит шагов агента (%s) — принудительное завершение цикла",
                config.MAX_AGENT_STEPS,
            )
            return "validate"
        if has_tool_calls:
            return "tools"
        return "validate"

    def _validation_node(self, state: AgentState) -> dict:
        """
        Узел графа: детерминированная проверка финального ответа агента.

        Проверяет формат домена, домены-заглушки, блоклист каталогов/агрегаторов
        и подтверждённость домена наблюдениями инструментов. При нарушении добавляет
        в историю corrective-сообщение и возвращает агента на доработку, если лимит
        перезапусков не исчерпан. Финальные гварды в run() гарантируют, что заглушки,
        каталоги и неподтверждённые домены не попадут в итоговый ответ.

        :param state: Текущее состояние графа
        :return: Обновление состояния (замечание и счётчики при доработке)
        """
        last_ai_text = ""
        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage) and self._message_text(message):
                last_ai_text = self._message_text(message)
                break

        found, domain = self._parse_domain(last_ai_text)
        if not found or domain is None:
            # null или нераспознанный ответ — финально извлекается в run().
            return {"needs_agent_retry": False}

        normalized = normalize_domain(domain)
        if normalized is None:
            logger.warning("Валидация: домен «%s» не прошёл проверку формата", domain)
            return {"needs_agent_retry": False}

        if is_placeholder_domain(normalized):
            # Заглушка (example.com и т.п.) — модель сконструировала домен из названия.
            reason_message = prompts.build_placeholder_feedback(normalized)
            logger.warning("Валидация: домен «%s» распознан как заглушка", normalized)
        elif not is_allowed_domain(normalized):
            reason_message = prompts.build_validation_feedback(normalized)
            logger.warning(
                "Валидация: домен «%s» — каталог/агрегатор", normalized
            )
        elif not self._is_evidenced(normalized, state["messages"]):
            # Домен не встречался в результатах поиска и загруженных страницах.
            reason_message = prompts.build_no_evidence_feedback(normalized)
            logger.warning(
                "Валидация: домен «%s» не подтверждён наблюдениями инструментов", normalized
            )
        elif self._inn_missing_on_page(normalized, state["messages"]):
            # Страница загружена, но ИНН организации на ней нет. Пробуем типовые
            # страницы реквизитов; если ИНН найден — домен подтверждён, если нет —
            # мягкое замечание (агент может подтвердить домен поиском или отказаться).
            inn = self._extract_inn(state["messages"])
            found_path = self._search_inn_on_domain(normalized, inn, state["messages"])
            if found_path is not None:
                logger.info(
                    "Валидация: ИНН найден на https://%s%s — домен подтверждён",
                    normalized,
                    found_path,
                )
                return {"needs_agent_retry": False}
            reason_message = prompts.build_ownership_feedback(normalized, inn)
            logger.warning(
                "Валидация: ИНН организации не найден на страницах домена «%s»",
                normalized,
            )
        else:
            logger.info("Валидация: домен «%s» прошёл проверку", normalized)
            return {"needs_agent_retry": False}

        retries = state.get("validation_retries", 0)
        if retries >= config.MAX_VALIDATION_RETRIES:
            logger.warning(
                "Валидация: домен «%s» отклонён, но агент уже использовал перезапуски — "
                "принимаем ответ как есть",
                normalized,
            )
            return {"needs_agent_retry": False}

        logger.warning(
            "Валидация: агент отправлен на доработку (попытка %s из %s)",
            retries + 1,
            config.MAX_VALIDATION_RETRIES,
        )
        return {
            "messages": [HumanMessage(content=reason_message)],
            "validation_retries": retries + 1,
            "needs_agent_retry": True,
        }

    @staticmethod
    def _is_evidenced(domain: str, messages: list[BaseMessage]) -> bool:
        """
        Проверяет, что домен многократно встречался в наблюдениях инструментов.

        Настоящие официальные сайты упоминаются в результатах поиска и на страницах
        несколько раз; случайные домены из шума выдачи — как правило, однократно.

        :param domain: Нормализованный домен-кандидат
        :param messages: История сообщений графа
        :return: True, если число упоминаний домена (или его punycode-формы) не меньше
                 MIN_EVIDENCE_MENTIONS, иначе False
        """
        variants = {domain.lower(), _idna_form(domain)}
        mentions = 0
        for message in messages:
            if isinstance(message, ToolMessage):
                text = DomainFinderAgent._message_text(message).lower()
                mentions += sum(text.count(variant) for variant in variants)
                if mentions >= MIN_EVIDENCE_MENTIONS:
                    return True
        return False

    @staticmethod
    def _extract_inn(messages: list[BaseMessage]) -> str:
        """
        Извлекает ИНН организации из первого пользовательского сообщения.

        :param messages: История сообщений графа
        :return: Строка с ИНН или пустая строка, если ИНН не найден
        """
        for message in messages:
            if isinstance(message, HumanMessage):
                match = re.search(r"\b\d{10}(?:\d{2})?\b", DomainFinderAgent._message_text(message))
                return match.group(0) if match else ""
        return ""

    def _fetch_observation_for(self, domain: str, messages: list[BaseMessage]) -> str | None:
        """
        Возвращает наблюдение fetch_page для указанного домена, если агент его загружал.

        :param domain: Нормализованный домен-кандидат
        :param messages: История сообщений графа
        :return: Текст загруженной страницы или None, если домен не загружался
        """
        wanted = {domain.lower(), _idna_form(domain)}
        fetch_urls: dict[str, str] = {}
        for message in messages:
            if isinstance(message, AIMessage):
                for call in message.tool_calls or []:
                    if call.get("name") == "fetch_page":
                        fetch_urls[call.get("id", "")] = str(
                            (call.get("args") or {}).get("url", "")
                        )
        for message in messages:
            if isinstance(message, ToolMessage):
                url = fetch_urls.get(message.tool_call_id, "")
                if not url:
                    continue
                url_domain = normalize_domain(url)
                if url_domain and url_domain.lower() in wanted:
                    return self._message_text(message)
        return None

    def _inn_missing_on_page(self, domain: str, messages: list[BaseMessage]) -> bool:
        """
        Проверяет, загружал ли агент страницу домена и отсутствует ли на ней ИНН организации.

        Замечание формируется только если страница реально загружалась, но ИНН на ней
        нет — это сигнал возможной перепутанной организации. Незагруженный домен
        проверкой не тревожится (промпт отдельно требует проверять перед ответом).

        :param domain: Нормализованный домен-кандидат
        :param messages: История сообщений графа
        :return: True, если страница домена загружена и ИНН организации на ней отсутствует
        """
        inn = self._extract_inn(messages)
        if not inn:
            return False
        observation = self._fetch_observation_for(domain, messages)
        if observation is None:
            return False
        return inn not in observation

    def _search_inn_on_domain(
        self, domain: str, inn: str, messages: list[BaseMessage]
    ) -> str | None:
        """
        Догружает типовые страницы реквизитов домена и ищет на них ИНН организации.

        Реквизиты (ИНН) обычно находятся в разделах контактов, «о компании» и оферте,
        а не на главной странице. Вызовы fetch_page внутри узла не попадают в историю
        сообщений, чтобы не раздувать контекст.

        :param domain: Нормализованный домен-кандидат
        :param inn: ИНН организации из задания
        :param messages: История сообщений графа (для исключения повторных загрузок)
        :return: Путь страницы, на которой найден ИНН, или None, если ИНН не найден
        """
        tool = self._tool_map.get("fetch_page")
        if tool is None or not inn:
            return None

        lower_domain = domain.lower()
        idna_domain = _idna_form(domain)
        fetched_paths: set[str] = set()
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            for call in message.tool_calls or []:
                if call.get("name") != "fetch_page":
                    continue
                url = str((call.get("args") or {}).get("url", "")).lower()
                if lower_domain in url or (idna_domain and idna_domain in url):
                    path = url.split(lower_domain, 1)[-1].split(idna_domain, 1)[-1]
                    fetched_paths.add(path.split("?", 1)[0].rstrip("/") or "/")

        for probe_path in _OWNERSHIP_PROBE_PATHS:
            if probe_path in fetched_paths:
                continue
            probe_url = f"https://{domain}{probe_path}"
            try:
                observation = str(tool.invoke({"url": probe_url}))
            except Exception as exc:  # noqa: BLE001 — ошибка загрузки не прерывает проверку
                logger.warning(
                    "Проба %s завершилась ошибкой: %s", probe_url, exc.__class__.__name__
                )
                continue
            if inn in observation:
                return probe_path
        return None

    def _route_after_validation(self, state: AgentState) -> str:
        """
        Определяет следующий узел после валидации.

        :param state: Текущее состояние графа
        :return: «agent» для повторного поиска с учётом замечания, иначе END
        """
        return "agent" if state.get("needs_agent_retry") else END

    # ------------------------------------------------------------------
    # Извлечение финального ответа
    # ------------------------------------------------------------------

    def _extract_answer(self, messages: list[BaseMessage]) -> tuple[str | None, str]:
        """
        Извлекает домен из истории сообщений агента.

        :param messages: История сообщений графа после завершения цикла
        :return: Кортеж (домен или None, финальный текст ответа агента)
        """
        ai_texts = [
            self._message_text(message)
            for message in messages
            if isinstance(message, AIMessage) and self._message_text(message)
        ]

        for text in reversed(ai_texts):
            found, domain = self._parse_domain(text)
            if found:
                return domain, text

        logger.info("Не удалось разобрать ответ агента — запускаю принудительное извлечение")
        forced_text = self._forced_extraction(messages)
        found, domain = self._parse_domain(forced_text)
        if found:
            return domain, forced_text
        return None, forced_text

    def _forced_extraction(self, messages: list[BaseMessage]) -> str:
        """
        Запрашивает у LLM финальный JSON-ответ по всей истории исследования.

        :param messages: История сообщений графа
        :return: Текст ответа модели (ожидается JSON вида {"domain": ...})
        :raises RuntimeError: При ошибке обращения к LLM
        """
        transcript = "\n\n".join(
            f"[{message.type}] {self._message_text(message)}"
            for message in messages
            if self._message_text(message)
        )
        transcript = transcript[-config.MAX_TRANSCRIPT_CHARS :]

        try:
            response = self._llm.invoke(
                [
                    SystemMessage(
                        content=prompts.build_extraction_system_prompt(self._format_instructions)
                    ),
                    HumanMessage(content=f"История исследования:\n\n{transcript}"),
                ]
            )
        except Exception as exc:
            logger.exception("Ошибка при принудительном извлечении ответа")
            raise RuntimeError(f"Ошибка обращения к LLM при извлечении ответа: {exc}") from exc
        return self._message_text(response)

    @staticmethod
    def _parse_domain(text: str) -> tuple[bool, str | None]:
        """
        Извлекает значение поля domain из текста ответа LLM.

        Основной разбор выполняется JsonOutputParser'ом: он находит JSON-объект
        в markdown-блоках и частично обрезанных ответах. Если JSON окружён
        пояснительным текстом с обеих сторон, добираются отдельные сбалансированные
        фрагменты. Дополнительно поддерживается ответ «голым» доменом без JSON.

        :param text: Текст ответа модели (JSON, markdown-блок или обычный текст)
        :return: Кортеж (флаг «ответ распознан», домен или None для null)
        """
        if not text:
            return False, None

        try:
            data = _JSON_PARSER.parse(text)
        except OutputParserException:
            data = None

        # JsonOutputParser не достаёт JSON с текстом до открывающей скобки —
        # добираем такие фрагменты вручную.
        snippets: list[object] = []
        if not isinstance(data, (dict, list)):
            for snippet in iter_json_snippets(text):
                try:
                    snippets.append(_JSON_PARSER.parse(snippet))
                except OutputParserException:
                    continue

        records: list[dict] = []
        for candidate in (data, *snippets):
            if isinstance(candidate, dict):
                records.append(candidate)
            elif isinstance(candidate, list):
                records.extend(item for item in candidate if isinstance(item, dict))

        for record in records:
            for key, value in record.items():
                if str(key).lower() != "domain":
                    continue
                if value is None:
                    return True, None
                if isinstance(value, str):
                    domain = value.strip().strip('"').strip()
                    if not domain or domain.lower() in ("none", "null", "нет"):
                        return True, None
                    return True, domain
                return False, None  # домен пришёл не строкой — ответ не распознан

        # Модель могла ответить JSON-строкой с доменом или просто доменом без обёртки.
        candidates = [text.strip()]
        if isinstance(data, str):
            candidates.insert(0, data.strip())
        for candidate in candidates:
            if (
                candidate
                and len(candidate) <= 100
                and " " not in candidate
                and is_domain_like(candidate)
            ):
                return True, candidate

        return False, None

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        """
        Возвращает текстовое содержимое сообщения.

        :param message: Сообщение из истории (контент может быть строкой или списком блоков)
        :return: Строка с текстом сообщения (пустая строка, если текста нет)
        """
        content = message.content
        if isinstance(content, str):
            return content
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
