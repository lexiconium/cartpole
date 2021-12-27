import os
import logging
from typing import Union

import torch

import matplotlib
import matplotlib.pyplot as plt

LOG_DIR = "logs"

font = {"family": "DejaVu Sans", "weight": "normal", "size": 16}
matplotlib.rc("font", **font)


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


class ScoreLogger:
    def __init__(self):
        self.scores = []

    def __getitem__(self, i):
        return self.scores[i]

    def append(self, score: Union[int, float]):
        self.scores.append(score)

    def draw(self, name: str):
        mean_scores = torch.cat(
            [
                torch.zeros(99),
                torch.tensor(self.scores).float().unfold(dimension=0, size=100, step=1).mean(dim=1).view(-1),
            ]
        )
        fig, ax = plt.subplots(figsize=(15, 10))
        plt.plot(self.scores, color="#1f77b4", label="score")
        plt.plot(mean_scores.numpy(), color="#ff7f0e", label="100-episode mean score")
        plt.title("scores", fontname="Times New Roman", size=32, fontweight="bold")
        plt.ylabel("score", fontname="Times New Roman", size=24, fontweight="bold")
        plt.xlabel("episode", fontname="Times New Roman", size=24, fontweight="bold")
        plt.legend(loc="lower right")

        num = self._get_plot_num(name=name)
        plt.savefig(os.path.join(LOG_DIR, f"{name}_{num}.png"))

    def _get_plot_num(self, name: str):
        file_names = sorted(
            file_name
            for file_name in os.listdir(LOG_DIR)
            if file_name.startswith(name) and file_name.endswith(".log")
        )
        return int(file_names[-1][:-4].split("_")[-1]) if file_names else 0
