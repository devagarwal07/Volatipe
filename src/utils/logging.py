import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_LOGGER: Optional[logging.Logger] = None

def get_logger(name: str = "volatility_india", level: str = "INFO", log_file: str = "logs/app.log") -> logging.Logger:
    global _LOGGER
    if _LOGGER:
        return _LOGGER.getChild(name)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("volatility_india")
    logger.setLevel(level.upper())
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = RotatingFileHandler(log_file, maxBytes=20 * 1024 * 1024, backupCount=5)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    _LOGGER = logger
    return logger.getChild(name)
