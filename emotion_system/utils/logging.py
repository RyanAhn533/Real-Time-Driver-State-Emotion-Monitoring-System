import json
import logging
from pathlib import Path


def setup_logger(name: str, log_dir: str = None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class MetricLogger:
    """Append-only JSONL metric logger."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    def read_all(self):
        if not self.path.exists():
            return []
        with open(self.path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
