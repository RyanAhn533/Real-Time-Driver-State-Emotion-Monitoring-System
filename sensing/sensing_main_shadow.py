"""
K-MER Shadow Mode Main
========================
기존 sensing_main.py의 동작을 보존하면서
새 E2E 파이프라인을 병렬로 실행하여 로그 비교.

핵심 원칙:
  1. 기존 센서 스레드 함수 (run_realsense, run_rode, run_watch) 그대로 호출
  2. 기존 inference_loop() 그대로 실행 (원본 출력 보존)
  3. shadow_inference_loop() 추가 실행 (E2E pipeline → 로그 비교)
  4. ModelInputs 버퍼를 공유 (두 루프가 동일 입력 사용)

데이터 흐름:
    Sensors → ModelInputs (shared buffers, 기존 그대로)
                ├── inference_loop()         [기존, 원본 출력]
                └── shadow_inference_loop()  [새 E2E, 로그 비교]

Usage:
    python sensing_main_shadow.py
    python sensing_main_shadow.py --shadow_only    # E2E만 (기존 루프 비활성화)
    python sensing_main_shadow.py --no_audio_experts  # 오디오 전문가 비활성화
"""

import sys
import os
import time
import argparse
import threading
import numpy as np
from pathlib import Path

# ── 프로젝트 경로 설정 ──────────────────────────────────────────────────
_SENSING_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SENSING_DIR.parent  # Jetson_thor/

# 기존 센싱 코드 경로
_RAW_SENSING_DIR = (
    _SENSING_DIR
    / "raw_sensing_code"
    / "Real-Time-Driver-State-Emotion-Monitoring-System-ysh_sensing_260305"
    / "sensing"
)
_RAW_GATEWAY_DIR = (
    _SENSING_DIR
    / "raw_sensing_code"
    / "Real-Time-Driver-State-Emotion-Monitoring-System-ysh_sensing_260305"
    / "multimodal_dms"
)

# sys.path 설정 (기존 코드 import 가능하도록)
for _p in [
    str(_RAW_SENSING_DIR),
    str(_PROJECT_ROOT / "emotion_system"),
    str(_PROJECT_ROOT / "multimodal_dms"),
    str(_SENSING_DIR),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Config & Logger ──────────────────────────────────────────────────────
from config import load_config
from core.logger import setup_logging, get_logger

# ── 기존 코드 import (동작 보존) ─────────────────────────────────────────
from sensing_main import Latest, Ring1D, BioQueues, ModelInputs, inference_loop
from realsense import run_realsense
from rode import run_rode
from watch import run_watch

# ── 새 E2E Pipeline ──────────────────────────────────────────────────────
from pipeline.e2e_pipeline import E2EPipeline

logger = get_logger("kmer.main")


# ── Shadow Inference Loop ────────────────────────────────────────────────

def shadow_inference_loop(
    shutdown: threading.Event,
    mi: ModelInputs,
    pipeline: E2EPipeline,
    hz: float = 10.0,
    log_comparison: bool = True,
    original_inferencer=None,
):
    """
    E2E Pipeline 기반 추론 루프 (shadow mode).

    기존 inference_loop()와 동일한 입력(ModelInputs)을 사용하되,
    E2E pipeline으로 처리하고 로그만 남김.

    Args:
        shutdown: 종료 이벤트
        mi: ModelInputs (공유 버퍼)
        pipeline: E2EPipeline 인스턴스
        hz: 추론 주기
        log_comparison: old vs new 비교 로그 출력
        original_inferencer: (optional) 기존 KMERInferencer (비교용)
    """
    shadow_logger = get_logger("kmer.shadow")
    dt = 1.0 / float(hz)
    last_print = 0.0
    cycle = 0

    shadow_logger.info("Shadow inference loop started (hz=%.1f)", hz)

    while not shutdown.is_set():
        t0 = time.time()

        # 기존 ModelInputs에서 데이터 가져오기 (sensing_main.py와 동일)
        fts, frame = mi.frame_main.get()
        _, audio = mi.audio.snapshot()
        ppg, eda, temp = mi.bio.snapshot()

        if frame is not None:
            try:
                # E2E Pipeline 실행
                result = pipeline.process_cycle(
                    frame=frame,
                    audio=audio if len(audio) > 0 else None,
                    ppg=ppg if ppg else None,
                    eda=eda if eda else None,
                    temp=temp if temp else None,
                )

                cycle += 1

                # 주기적 로그 출력 (1초마다)
                now = time.time()
                if now - last_print >= 1.0:
                    _log_shadow_result(shadow_logger, result, fts, ppg, eda, temp)
                    last_print = now

            except Exception as e:
                shadow_logger.error("Shadow inference error: %s", e, exc_info=True)

        # 주기 유지
        elapsed = time.time() - t0
        sleep_time = max(0.0, dt - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    shadow_logger.info("Shadow inference loop stopped (total cycles=%d)", cycle)


def _log_shadow_result(log, result, fts, ppg, eda, temp):
    """Shadow 결과 로그 출력."""
    mode = result.get("mode", "?")
    emo = result.get("emotion_name_ko", "-")
    emo_code = result.get("emotion_code", -1)
    arousal = result.get("smoothed_arousal", result.get("arousal"))
    valence = result.get("smoothed_valence", result.get("valence"))
    drowsy = result.get("drowsy_level", result.get("drowsy", 0))
    stress = result.get("stress", False)
    low_attn = result.get("low_attention", False)
    fatigue = result.get("fatigue", False)
    compound = result.get("compound_label", "-")
    kfer_emo = result.get("kfer_emotion", "-")
    kfer_cf = result.get("kfer_confidence", 0.0)
    pkt_hex = result.get("packet_hex", "-")

    av_str = f"A={arousal:.3f}" if arousal is not None else "A=None"
    if valence is not None:
        av_str += f" V={valence:.3f}"

    flags = []
    if stress: flags.append("STRESS")
    if low_attn: flags.append("LOW_ATTN")
    if drowsy: flags.append("DROWSY")
    if fatigue: flags.append("FATIGUE")
    flags_str = "|".join(flags) if flags else "NORMAL"

    log.info(
        "[shadow] fts=%d | mode=%s | kfer=%s(%.2f) → %s(code=%d) | "
        "%s | drowsy=%d | compound=%s | flags=%s | "
        "bio=%dppg/%deda/%dtemp | pkt=%s",
        fts, mode, kfer_emo, kfer_cf, emo, emo_code,
        av_str, drowsy, compound, flags_str,
        len(ppg) if ppg else 0,
        len(eda) if eda else 0,
        len(temp) if temp else 0,
        pkt_hex or "-",
    )


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser("K-MER Shadow Mode Main")
    ap.add_argument("--config", default=None, help="Config YAML 경로")
    ap.add_argument("--shadow_only", action="store_true",
                    help="E2E pipeline만 실행 (기존 inference_loop 비활성화)")
    ap.add_argument("--no_audio_experts", action="store_true",
                    help="emotion2vec/audeering 비활성화 (메모리 절약)")
    ap.add_argument("--gateway_off", action="store_true",
                    help="Gateway 전송 비활성화")
    ap.add_argument("--device", default=None, help="cuda or cpu")
    ap.add_argument("--hz", type=float, default=None, help="추론 루프 주기 (Hz)")
    args = ap.parse_args()

    # ── Config 로드 ──
    cfg = load_config(args.config)

    # CLI 오버라이드
    if args.no_audio_experts:
        cfg.inference.enable_audio_experts = False
    if args.gateway_off:
        cfg.gateway.enabled = False
    if args.device:
        cfg.inference.device = args.device
    if args.hz:
        cfg.inference.hz = args.hz

    # ── Logger 초기화 ──
    setup_logging(
        console_level=cfg.logging.console_level,
        file_level=cfg.logging.file_level,
        log_dir=cfg.logging.log_dir,
        max_bytes=cfg.logging.max_bytes,
        backup_count=cfg.logging.backup_count,
        base_dir=str(_SENSING_DIR),
    )

    logger.info("=" * 60)
    logger.info("K-MER Shadow Mode Main Starting")
    logger.info("=" * 60)
    logger.info("Config: device=%s, hz=%.1f, audio_experts=%s, gateway=%s",
                cfg.inference.device, cfg.inference.hz,
                cfg.inference.enable_audio_experts, cfg.gateway.enabled)

    # ── 공유 버퍼 (기존 그대로) ──
    mi = ModelInputs(audio_sr=cfg.inference.audio_sr, audio_sec=cfg.inference.audio_sec)

    # ── 센서 콜백 (기존 그대로) ──
    def on_frame_main(ts_ms: int, frame_bgr: np.ndarray):
        mi.frame_main.set(ts_ms, frame_bgr)

    def on_audio_chunk(ts_ms: int, mono: np.ndarray, sr: int):
        mi.audio.push(ts_ms, mono)

    def on_ppg(ts_ms: int, d1: float, d2: float):
        mi.bio.push_ppg(ts_ms, d1, d2)

    def on_eda(ts_ms: int, real: float):
        mi.bio.push_eda(ts_ms, real)

    def on_temp(ts_ms: int, skin_c: float):
        mi.bio.push_temp(ts_ms, skin_c)

    shutdown = threading.Event()
    threads = []

    # ── 센서 스레드 (기존 함수 그대로 호출) ──
    hw = cfg.hardware

    # RealSense Main
    threads.append(threading.Thread(
        target=run_realsense,
        kwargs=dict(
            shutdown_event=shutdown,
            device_serial=hw.realsense_main.serial,
            is_main_cam=True,
            on_frame=on_frame_main,
            width=hw.realsense_main.width,
            height=hw.realsense_main.height,
            fps=hw.realsense_main.fps,
        ),
        daemon=True,
        name="RS_MAIN",
    ))

    # RealSense Sub (config로 토글)
    if hw.realsense_sub.enabled:
        threads.append(threading.Thread(
            target=run_realsense,
            kwargs=dict(
                shutdown_event=shutdown,
                device_serial=hw.realsense_sub.serial,
                is_main_cam=False,
                on_frame=None,
            ),
            daemon=True,
            name="RS_SUB",
        ))

    # RODE Microphone
    threads.append(threading.Thread(
        target=_safe_run_sensor,
        args=("RODE", run_rode, shutdown),
        kwargs=dict(
            on_audio_chunk=on_audio_chunk,
            sample_rate=hw.rode.sample_rate,
            blocksize=hw.rode.blocksize,
            latency=hw.rode.latency,
        ),
        daemon=True,
        name="RODE",
    ))

    # ADI Study Watch
    threads.append(threading.Thread(
        target=_safe_run_sensor,
        args=("WATCH", run_watch, shutdown),
        kwargs=dict(
            on_ppg=on_ppg,
            on_eda=on_eda,
            on_temp=on_temp,
        ),
        daemon=True,
        name="WATCH",
    ))

    # ── 기존 Inference Loop (shadow_only가 아닌 경우) ──
    original_inferencer = None
    if not args.shadow_only:
        try:
            from kmer_inferencer import KMERInferencer
            original_inferencer = KMERInferencer(
                kfer_ckpt=cfg.inference.kfer_ckpt,
                kmer_ckpt=cfg.inference.kmer_ckpt,
                device=cfg.inference.device,
                enable_audio_experts=cfg.inference.enable_audio_experts,
            )
            threads.append(threading.Thread(
                target=inference_loop,
                args=(shutdown, mi, original_inferencer),
                kwargs=dict(hz=cfg.inference.hz),
                daemon=True,
                name="INFER_ORIG",
            ))
            logger.info("Original inference_loop enabled")
        except Exception as e:
            logger.warning("Original inferencer init failed: %s. Running shadow-only.", e)

    # ── Shadow E2E Pipeline ──
    pipeline = E2EPipeline(cfg)
    threads.append(threading.Thread(
        target=shadow_inference_loop,
        args=(shutdown, mi, pipeline),
        kwargs=dict(
            hz=cfg.inference.hz,
            log_comparison=cfg.shadow_mode.log_comparison,
            original_inferencer=original_inferencer,
        ),
        daemon=True,
        name="INFER_SHADOW",
    ))

    # ── 스레드 시작 ──
    for t in threads:
        t.start()
        logger.info("Thread started: %s", t.name)

    logger.info("All threads started (%d total)", len(threads))

    # ── Health monitoring (10초마다) ──
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
    finally:
        shutdown.set()
        logger.info("Shutdown signal sent, waiting for threads...")

        # 파이프라인 정리
        pipeline.shutdown()

        # 스레드 종료 대기
        for t in threads:
            t.join(timeout=5.0)
            if t.is_alive():
                logger.warning("Thread %s did not stop within timeout", t.name)
            else:
                logger.info("Thread %s stopped", t.name)

        logger.info("K-MER system stopped")


def _safe_run_sensor(name: str, func, shutdown: threading.Event, **kwargs):
    """
    센서 스레드를 안전하게 실행.

    센서 장애 시 다른 스레드에 영향 주지 않음 (Degraded Mode 지원).
    """
    sensor_logger = get_logger(f"kmer.driver.{name.lower()}")
    try:
        func(shutdown_event=shutdown, **kwargs)
    except Exception as e:
        sensor_logger.error("%s failed: %s (continuing in degraded mode)", name, e)
        # 크래시하지 않고 스레드만 종료 → 다른 센서는 계속 동작
        while not shutdown.is_set():
            time.sleep(1.0)


if __name__ == "__main__":
    main()
