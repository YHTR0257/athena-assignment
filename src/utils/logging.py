import logging
import sys
import os

def setup_logging()->logging.Logger:
    logger = logging.getLogger("athena-assignment")
    logger.setLevel(os.getenv("LOG_LEVEL", "DEBUG").upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
