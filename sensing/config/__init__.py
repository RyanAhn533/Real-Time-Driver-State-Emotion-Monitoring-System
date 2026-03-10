"""
K-MER Sensing Config Loader
============================
YAML config 파일을 로드하여 dataclass로 변환.

Usage:
    from config import load_config
    cfg = load_config()  # sensing/config/sensing_config.yaml 로드
    print(cfg.hardware.realsense_main.serial)
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


# ── Dataclass 정의 ──────────────────────────────────────────────────────

@dataclass
class RealSenseConfig:
    serial: str = ""
    width: int = 1280
    height: int = 720
    fps: int = 30
    enabled: bool = True


@dataclass
class RodeConfig:
    sample_rate: int = 48000
    blocksize: int = 8192
    latency: float = 0.5
    keywords: List[str] = field(default_factory=lambda: [
        "Wireless GO II", "RØDE", "RODE", "Wireless GO", "GO II"
    ])


@dataclass
class WatchConfig:
    vid: int = 0x0456
    pid: int = 0x2CFE
    mac: str = "F1-18-1C-93-7C-42"
    ble_timeout: int = 60
    sdk_retry: int = 5
    dongle_wait_sec: int = 10


@dataclass
class HardwareConfig:
    realsense_main: RealSenseConfig = field(default_factory=RealSenseConfig)
    realsense_sub: RealSenseConfig = field(default_factory=lambda: RealSenseConfig(enabled=False))
    rode: RodeConfig = field(default_factory=RodeConfig)
    watch: WatchConfig = field(default_factory=WatchConfig)


@dataclass
class InferenceConfig:
    hz: float = 10.0
    device: str = "cuda"
    kfer_ckpt: str = "../emotion_system/result/best.pth"
    kmer_ckpt: str = "../multimodal_dms/results_kmer/best_model.pth"
    audio_sec: float = 2.0
    audio_sr: int = 48000
    enable_audio_experts: bool = True
    enable_face_expert: bool = False


@dataclass
class ThresholdsConfig:
    ear_closed: float = 0.21
    perclos_low_attn: float = 0.2
    perclos_drowsy: float = 0.4
    arousal_stress: float = 0.6
    fatigue_arousal: float = 0.3
    fatigue_duration_sec: float = 30.0
    fatigue_drowsy_sec: float = 10.0


@dataclass
class GatewayConfig:
    enabled: bool = True
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout: float = 1.0
    retry_interval: float = 5.0


@dataclass
class TemporalSmoothingConfig:
    emotion_window: int = 7
    drowsy_window: int = 5
    compound_window: int = 7
    ema_alpha: float = 0.3


@dataclass
class ShadowModeConfig:
    enabled: bool = True
    log_comparison: bool = True


@dataclass
class LoggingConfig:
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    log_dir: str = "logs"
    max_bytes: int = 10_485_760   # 10MB
    backup_count: int = 5


@dataclass
class SensingConfig:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    temporal_smoothing: TemporalSmoothingConfig = field(default_factory=TemporalSmoothingConfig)
    shadow_mode: ShadowModeConfig = field(default_factory=ShadowModeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ── Helper: dict → dataclass ────────────────────────────────────────────

def _dict_to_dataclass(cls, d):
    """딕셔너리를 dataclass 인스턴스로 변환 (nested 지원)."""
    if d is None:
        return cls()

    import dataclasses
    fieldtypes = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}

    for key, val in d.items():
        if key not in fieldtypes:
            continue  # 알 수 없는 키 무시

        ftype = fieldtypes[key]

        # hex 문자열 → int 변환 (vid, pid 등)
        if ftype is int and isinstance(val, str):
            try:
                val = int(val, 0)  # 0x 접두어 자동 처리
            except ValueError:
                pass

        # nested dataclass 처리
        if isinstance(val, dict):
            try:
                import dataclasses as dc
                # ftype이 dataclass인지 확인
                origin = getattr(ftype, '__origin__', None)
                if origin is None and dc.is_dataclass(ftype):
                    val = _dict_to_dataclass(ftype, val)
            except (TypeError, AttributeError):
                pass

        kwargs[key] = val

    return cls(**kwargs)


# ── Main loader ──────────────────────────────────────────────────────────

_CONFIG_DIR = Path(__file__).parent
_DEFAULT_CONFIG = _CONFIG_DIR / "sensing_config.yaml"


def load_config(config_path: Optional[str] = None) -> SensingConfig:
    """
    YAML config 파일 로드.

    Args:
        config_path: YAML 파일 경로 (None이면 기본 경로 사용)

    Returns:
        SensingConfig dataclass 인스턴스
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG

    if not path.exists():
        print(f"[Config] Config file not found: {path}, using defaults")
        return SensingConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Top-level sections → nested dataclass
    cfg = SensingConfig(
        hardware=_dict_to_dataclass(HardwareConfig, raw.get("hardware")),
        inference=_dict_to_dataclass(InferenceConfig, raw.get("inference")),
        thresholds=_dict_to_dataclass(ThresholdsConfig, raw.get("thresholds")),
        gateway=_dict_to_dataclass(GatewayConfig, raw.get("gateway")),
        temporal_smoothing=_dict_to_dataclass(
            TemporalSmoothingConfig, raw.get("temporal_smoothing")
        ),
        shadow_mode=_dict_to_dataclass(ShadowModeConfig, raw.get("shadow_mode")),
        logging=_dict_to_dataclass(LoggingConfig, raw.get("logging")),
    )

    # Nested hardware configs
    hw = raw.get("hardware", {})
    if "realsense_main" in hw:
        cfg.hardware.realsense_main = _dict_to_dataclass(
            RealSenseConfig, hw["realsense_main"]
        )
    if "realsense_sub" in hw:
        cfg.hardware.realsense_sub = _dict_to_dataclass(
            RealSenseConfig, hw["realsense_sub"]
        )
    if "rode" in hw:
        cfg.hardware.rode = _dict_to_dataclass(RodeConfig, hw["rode"])
    if "watch" in hw:
        cfg.hardware.watch = _dict_to_dataclass(WatchConfig, hw["watch"])

    return cfg


# ── Self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    print("=== K-MER Sensing Config ===")
    print(f"  RealSense Main: serial={cfg.hardware.realsense_main.serial}")
    print(f"  RealSense Sub:  enabled={cfg.hardware.realsense_sub.enabled}")
    print(f"  RODE: sr={cfg.hardware.rode.sample_rate}, blocksize={cfg.hardware.rode.blocksize}")
    print(f"  Watch: MAC={cfg.hardware.watch.mac}")
    print(f"  Inference: hz={cfg.inference.hz}, device={cfg.inference.device}")
    print(f"  Gateway: port={cfg.gateway.port}, enabled={cfg.gateway.enabled}")
    print(f"  Shadow Mode: enabled={cfg.shadow_mode.enabled}")
    print(f"  Thresholds: stress_arousal={cfg.thresholds.arousal_stress}")
    print(f"  Temporal: ema_alpha={cfg.temporal_smoothing.ema_alpha}")
