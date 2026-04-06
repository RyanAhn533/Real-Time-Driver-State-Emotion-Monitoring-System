"""
K-MER Sensing Buffers
======================
기존 sensing_main.py의 버퍼 클래스를 re-export.

동작 보존 우선 원칙: 원본 코드를 수정하지 않고 그대로 import하여 사용.

Usage:
    from core.buffers import Latest, Ring1D, BioQueues, ModelInputs
"""

import sys
from pathlib import Path

# 원본 sensing 코드 경로 추가
_RAW_SENSING_DIR = (
    Path(__file__).parent.parent
    / "raw_sensing_code"
    / "Real-Time-Driver-State-Emotion-Monitoring-System-ysh_sensing_260305"
    / "sensing"
)

if str(_RAW_SENSING_DIR) not in sys.path:
    sys.path.insert(0, str(_RAW_SENSING_DIR))

# 원본 클래스들을 그대로 import (동작 보존)
from sensing_main import Latest, Ring1D, BioQueues, ModelInputs  # noqa: F401

# inference_loop도 원본 그대로 사용 가능
from sensing_main import inference_loop  # noqa: F401

__all__ = [
    "Latest",
    "Ring1D",
    "BioQueues",
    "ModelInputs",
    "inference_loop",
]
