import structlog

from .config import load_settings
from .logging import configure_logging
from .test_trans import test_transformer


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    test_transformer(settings)
