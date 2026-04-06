"""
K-MER Structured Logger
========================
Python logging 모듈 기반 구조적 로깅.

기존 print() 대체용. 콘솔 (INFO) + 로테이팅 파일 (DEBUG).

Usage:
    from core.logger import get_logger
    logger = get_logger("kmer.driver.realsense")
    logger.info("Camera started: serial=%s", serial)
    logger.warning("Frame drop detected")
    logger.error("Pipeline failure: %s", error)
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


_INITIALIZED = False
_LOG_DIR = None


def setup_logging(
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_dir: str = "logs",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
    base_dir: Optional[str] = None,
):
    """
    로깅 시스템 초기화.

    Args:
        console_level: 콘솔 출력 레벨 ("DEBUG", "INFO", "WARNING", "ERROR")
        file_level: 파일 출력 레벨
        log_dir: 로그 디렉토리 (base_dir 기준 상대경로)
        max_bytes: 로그 파일 최대 크기
        backup_count: 로그 파일 보관 개수
        base_dir: 기본 디렉토리 (None이면 sensing/ 디렉토리)
    """
    global _INITIALIZED, _LOG_DIR

    if _INITIALIZED:
        return

    # 기본 디렉토리 결정
    if base_dir is None:
        base_dir = str(Path(__file__).parent.parent)

    _LOG_DIR = os.path.join(base_dir, log_dir)
    os.makedirs(_LOG_DIR, exist_ok=True)

    # Root logger for kmer namespace
    root = logging.getLogger("kmer")
    root.setLevel(logging.DEBUG)

    # 기존 핸들러 제거 (중복 방지)
    root.handlers.clear()

    # 포맷
    fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # 파일 핸들러 (rotating)
    log_file = os.path.join(_LOG_DIR, "kmer_sensing.log")
    fh = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    fh.setFormatter(fmt)
    root.addHandler(fh)

    _INITIALIZED = True
    root.info("Logging initialized: console=%s, file=%s → %s",
              console_level, file_level, log_file)


def get_logger(name: str) -> logging.Logger:
    """
    Named logger 반환.

    Args:
        name: 로거 이름 (예: "kmer.driver.realsense", "kmer.inference")

    Returns:
        logging.Logger 인스턴스
    """
    if not _INITIALIZED:
        # 자동 초기화 (기본값)
        setup_logging()

    # kmer. 접두어 보장
    if not name.startswith("kmer."):
        name = f"kmer.{name}"

    return logging.getLogger(name)
