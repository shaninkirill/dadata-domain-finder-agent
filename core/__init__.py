"""
Ядро проекта: общие инфраструктурные модули — логирование и валидация.

Модули core/ не содержат бизнес-логики пайплайна и переиспользуются
агентами, инструментами и точкой входа.
"""

from core.logger import setup_logging
from core.validators import (
    is_allowed_domain,
    is_domain_like,
    is_placeholder_domain,
    normalize_domain,
    validate_inn,
)

__all__ = [
    "setup_logging",
    "validate_inn",
    "normalize_domain",
    "is_domain_like",
    "is_allowed_domain",
    "is_placeholder_domain",
]
