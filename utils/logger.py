import os
import logging

LOG_DIR = "logs"


class SimpleLoggerWrapper:
    def __init__(self, level: int, format: str, name: str):
        self.logger = logging.getLogger()
        self.logger.setLevel(level)

        formatter = logging.Formatter(format)
        self.set_stream_handler(formatter=formatter)
        self.set_file_handler(name=name, formatter=formatter)

    def set_stream_handler(self, formatter: logging.Formatter):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def set_file_handler(self, name: str, formatter: logging.Formatter):
        num = self._get_log_num(name=name)
        file_handler = logging.FileHandler(os.path.join(LOG_DIR, f"{name}_{num}.log"))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _get_log_num(self, name: str):
        file_names = sorted(
            file_name
            for file_name in os.listdir(LOG_DIR)
            if file_name.startswith(name) and file_name.endswith(".log")
        )
        return int(file_names[-1][:-4].split("_")[-1]) + 1 if file_names else 0

    @classmethod
    def get_logger(cls, level: int, format: str, name: str):
        return cls(level, format, name).logger
