import logging
import sys
import os
from dotenv import load_dotenv

def setup_logging(log_level:str="None")->logging.Logger:
    load_dotenv()
    logger = logging.getLogger("athena-assignment")
    if log_level != "None":
        logger.setLevel(log_level.upper())
    else:
        # Default to environment variable
        log_level = os.getenv("LOG_LEVEL", "DEBUG")
        logger.setLevel(log_level.upper())
    print(f"Logging level set to: {log_level.upper()}")

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level.upper())

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
