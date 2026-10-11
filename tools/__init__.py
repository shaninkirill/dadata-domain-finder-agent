"""
Инструменты агента: независимые модули с внешними вызовами.

Каждый инструмент оформлен через декоратор @tool из langchain_core и может
использоваться LLM как обычная функция с текстовым результатом.
"""

from tools.site_fetch import fetch_page
from tools.web_search import web_search

__all__ = ["web_search", "fetch_page"]
