import logging
import sys
import os
from dotenv import load_dotenv


class _TeeStream:
    """stdoutとファイルの両方に出力するストリーム"""
    def __init__(self, stream, log_file):
        self._stream = stream
        self._file = open(log_file, "a", encoding="utf-8")

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()


def setup_logging(log_level:str="None", log_file:str|None=None)->logging.Logger:
    logger = logging.getLogger("athena-assignment")

    load_dotenv()
    if log_level != "None":
        level = log_level.upper()
    else:
        level = os.getenv("LOG_LEVEL", "DEBUG").upper()

    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # FileHandler: log_fileが指定された場合、未設定なら追加
    if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # print()等の標準出力もログファイルにキャプチャ
        if not isinstance(sys.stdout, _TeeStream):
            sys.stdout = _TeeStream(sys.__stdout__, log_file)
            sys.stderr = _TeeStream(sys.__stderr__, log_file)

    # StreamHandler: 未設定なら追加
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream:
        handler = logging.StreamHandler(sys.__stdout__)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
