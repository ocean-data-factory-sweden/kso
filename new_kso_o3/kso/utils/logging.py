"""Simple logging setup."""
import logging
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file) if log_file else logging.StreamHandler(),
        ],
    )
