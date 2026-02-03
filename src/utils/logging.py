import logging
import sys
import os
from dotenv import load_dotenv

def setup_logging(log_level:str="None")->logging.Logger:
    logger = logging.getLogger("athena-assignment")

    # ハンドラが既に設定されている場合は追加しない
    if logger.handlers:
        return logger

    load_dotenv()
    if log_level != "None":
        logger.setLevel(log_level.upper())
    else:
        # Default to environment variable
        log_level = os.getenv("LOG_LEVEL", "DEBUG")
        logger.setLevel(log_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level.upper())

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
