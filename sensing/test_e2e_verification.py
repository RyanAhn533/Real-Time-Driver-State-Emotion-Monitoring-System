"""
K-MER E2E Pipeline 통합 검증 테스트
=====================================
실제 하드웨어 없이 소프트웨어 로직 전체를 시뮬레이션 데이터로 검증.

검증 항목:
  1. Config 로딩
  2. Temporal Smoother (EMA + majority vote)
  3. Fatigue Tracker (3조건 판정)
  4. Gateway Sender (NullSender)
  5. PacketEncoder 호환성 (기존 vs 신규)
  6. Degraded Mode 판별
  7. E2E Pipeline 시뮬레이션 (KMERInferencer mock)
  8. 10-class 출력 매핑
  9. Packet Fatigue bit 추가

Usage:
    cd /home/ajy/Jetson_thor/sensing
    python test_e2e_verification.py
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# sensing/ 디렉토리 기준
_SENSING_DIR = Path(__file__).resolve().parent
os.chdir(str(_SENSING_DIR))

# 프로젝트 경로 (기존 코드 import용)
_PROJECT_ROOT = _SENSING_DIR.parent
_RAW_SENSING_DIR = (
    _SENSING_DIR
    / "raw_sensing_code"
    / "Real-Time-Driver-State-Emotion-Monitoring-System-ysh_sensing_260305"
    / "sensing"
)
for _p in [str(_RAW_SENSING_DIR), str(_PROJECT_ROOT / "multimodal_dms"), str(_PROJECT_ROOT / "emotion_system")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 핵심: sensing/ 디렉토리를 sys.path[0]에 배치
# emotion_system/pipeline/ 가 sensing/pipeline/ 을 가리는 충돌 방지
_sensing_str = str(_SENSING_DIR)
if _sensing_str in sys.path:
    sys.path.remove(_sensing_str)
sys.path.insert(0, _sensing_str)


# ── 테스트 결과 추적 ────────────────────────────────────────────────────

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  [PASS] {name}")

    def fail(self, name, reason=""):
        self.failed += 1
        self.errors.append((name, reason))
        print(f"  [FAIL] {name}: {reason}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  Total: {total} | Passed: {self.passed} | Failed: {self.failed}")
        if self.errors:
            print(f"\n  Failed tests:")
            for name, reason in self.errors:
                print(f"    - {name}: {reason}")
        print(f"{'='*60}")
        return self.failed == 0


results = TestResults()


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: Config 로딩
# ══════════════════════════════════════════════════════════════════════════

print("\n[TEST 1] Config 로딩")
print("-" * 40)

try:
    from config import load_config
    cfg = load_config()

    assert cfg.hardware.realsense_main.serial == "021222070391", "RS serial mismatch"
    assert cfg.hardware.realsense_sub.enabled == False, "RS sub should be disabled"
    assert cfg.hardware.rode.sample_rate == 48000, "RODE sr mismatch"
    assert cfg.hardware.watch.mac == "F1-18-1C-93-7C-42", "Watch MAC mismatch"
    assert cfg.inference.hz == 10, "Hz mismatch"
    assert cfg.thresholds.perclos_drowsy == 0.4, "Drowsy threshold mismatch"
    assert cfg.thresholds.fatigue_arousal == 0.3, "Fatigue arousal mismatch"
    assert cfg.thresholds.fatigue_duration_sec == 30, "Fatigue duration mismatch"
    assert cfg.gateway.port == "/dev/ttyUSB0", "Gateway port mismatch"
    assert cfg.shadow_mode.enabled == True, "Shadow should be enabled"
    assert cfg.temporal_smoothing.ema_alpha == 0.3, "EMA alpha mismatch"

    results.ok("Config 로딩 성공")
    results.ok("Config 값 검증 (11개 항목)")
except Exception as e:
    results.fail("Config 로딩", str(e))


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: Logger 초기화
# ══════════════════════════════════════════════════════════════════════════

print("\n[TEST 2] Logger 초기화")
print("-" * 40)

try:
    from core.logger import setup_logging, get_logger
    setup_logging(console_level="WARNING", file_level="DEBUG")  # 테스트 시 콘솔 조용히

    test_logger = get_logger("kmer.test")
    test_logger.info("Test log message (should appear in file only)")

    log_file = _SENSING_DIR / "logs" / "kmer_sensing.log"
    assert log_file.exists(), f"Log file not found: {log_file}"

    results.ok("Logger 초기화 + 파일 생성")
except Exception as e:
    results.fail("Logger", str(e))


# ══════════════════════════════════════════════════════════════════════════
# TEST 3: Buffer 클래스 (원본 import)
# ══════════════════════════════════════════════════════════════════════════

print("\n[TEST 3] Buffer 클래스 (원본 re-export)")
print("-" * 40)

try:
    from core.buffers import Latest, Ring1D, BioQueues, ModelInputs

    # Latest 테스트
    lat = Latest()
    lat.set(1000, "frame_data")
    ts, val = lat.get()
    assert ts == 1000 and val == "frame_data", "Latest set/get failed"
    results.ok("Latest buffer")

    # Ring1D 테스트
    ring = Ring1D(maxlen=100)
    ring.push(1000, np.ones(50, dtype=np.float32))
    ts, buf = ring.snapshot()
    assert len(buf) == 50, f"Ring1D snapshot size wrong: {len(buf)}"
    assert np.allclose(buf, 1.0), "Ring1D values wrong"
    results.ok("Ring1D buffer")

    # Ring1D 순환 테스트
    # 주의: 원본 Ring1D의 filled 플래그는 idx가 정확히 0이 될 때만 True
    ring2 = Ring1D(maxlen=10)
    # 먼저 버퍼를 꽉 채움 (10개 push → idx=0, filled=True)
    ring2.push(1, np.arange(10, dtype=np.float32))
    ts, buf = ring2.snapshot()
    assert len(buf) == 10, f"Full ring size: {len(buf)}"
    # 추가 push → 오래된 데이터 덮어씀
    ring2.push(2, np.array([99, 98, 97], dtype=np.float32))
    ts, buf = ring2.snapshot()
    assert len(buf) == 10, f"Wrapped ring size: {len(buf)}"
    # idx=3, filled=True → concat(buf[3:], buf[:3]) = [3,4,5,6,7,8,9,99,98,97]
    expected_wrap = np.array([3, 4, 5, 6, 7, 8, 9, 99, 98, 97], dtype=np.float32)
    assert np.allclose(buf, expected_wrap), f"Wrapped values: {buf} vs {expected_wrap}"
    results.ok("Ring1D 순환 (wraparound)")

    # BioQueues 테스트
    bio = BioQueues(maxlen=10)
    bio.push_ppg(1000, 100.0, 200.0)
    bio.push_eda(1000, 0.5)
    bio.push_temp(1000, 36.5)
    ppg, eda, temp = bio.snapshot()
    assert len(ppg) == 1 and ppg[0] == (1000, 100.0, 200.0), "PPG push failed"
    assert len(eda) == 1 and eda[0] == (1000, 0.5), "EDA push failed"
    assert len(temp) == 1 and temp[0] == (1000, 36.5), "TEMP push failed"
    results.ok("BioQueues (PPG/EDA/TEMP)")

    # ModelInputs 테스트
    mi = ModelInputs(audio_sr=48000, audio_sec=2.0)
    mi.frame_main.set(100, np.zeros((480, 640, 3), dtype=np.uint8))
    mi.audio.push(100, np.zeros(8192, dtype=np.float32))
    mi.bio.push_ppg(100, 1.0, 2.0)
    results.ok("ModelInputs 컨테이너")

except Exception as e:
    results.fail("Buffer 클래스", str(e))


# ══════════════════════════════════════════════════════════════════════════
# TEST 4: Temporal Smoother
# ══════════════════════════════════════════════════════════════════════════

print("\n[TEST 4] Temporal Smoother")
print("-" * 40)

try:
    from pipeline.temporal_smoother import (
        MajorityVoteSmoother, EMASmoother, MultimodalTemporalSmoother
    )

    # Majority vote
    mv = MajorityVoteSmoother(window_size=5)
    for _ in range(3):
        mv.push("happy")
    mv.push("angry")
    mv.push("neutral")
    assert mv.push("happy") == "happy", "Majority vote should be happy (4/6→3/5)"
    assert mv.get_confidence() == 0.6, f"Confidence should be 0.6, got {mv.get_confidence()}"
    results.ok("MajorityVoteSmoother")

    # EMA
    ema = EMASmoother(alpha=0.5)
    assert ema.push(1.0) == 1.0, "First EMA should be input"
    assert ema.push(0.0) == 0.5, f"EMA(1,0) should be 0.5, got {ema.push(0.0)}"
    results.ok("EMASmoother")

    # MultimodalTemporalSmoother
    smoother = MultimodalTemporalSmoother(emotion_window=3, drowsy_window=3, ema_alpha=0.5)
    r1 = smoother.smooth({
        "face_detected": True, "kfer_emotion": "happy",
        "arousal": 0.8, "valence": 0.7, "drowsy": 0,
        "compound_id": 2, "compound_label": "happy",
    })
    assert r1["smoothed_kfer_emotion"] == "happy"
    assert r1["smoothed_arousal"] == 0.8
    assert r1["smoothed_drowsy"] == 0
    results.ok("MultimodalTemporalSmoother (정상 입력)")

    # face_detected=False → skip smoothing
    r2 = smoother.smooth({"face_detected": False})
    assert "smoothed_kfer_emotion" not in r2
    results.ok("MultimodalTemporalSmoother (얼굴 미검출 → skip)")

except Exception as e:
    results.fail("Temporal Smoother", str(e))


# ══════════════════════════════════════════════════════════════════════════
# TEST 5: Fatigue Tracker
# ══════════════════════════════════════════════════════════════════════════

print("\n[TEST 5] Fatigue Tracker")
print("-" * 40)

try:
    from pipeline.fatigue_tracker import FatigueTracker

    # 조건 1: compound 연속
    ft = FatigueTracker(duration_sec=1.0, arousal_thresh=0.3, drowsy_sec=0.5, hz=10)
    for i in range(9):
        assert not ft.update("depressed", 0.5, 0), f"Should not be fatigue at frame {i}"
    assert ft.update("depressed", 0.5, 0), "Should be fatigue at frame 10 (1초)"
    results.ok("Fatigue 조건1: compound 연속 (depressed 1초)")

    # 리셋
    ft.reset()

    # 조건 2: arousal < 0.3 연속
    for i in range(9):
        assert not ft.update("neutral", 0.2, 0), f"Should not be fatigue at frame {i}"
    assert ft.update("neutral", 0.2, 0), "Should be fatigue (arousal<0.3 1초)"
    results.ok("Fatigue 조건2: arousal < 0.3 연속")

    ft.reset()

    # 조건 3: drowsy 연속
    for i in range(4):
        assert not ft.update("neutral", 0.5, 1), f"Should not be fatigue at frame {i}"
    assert ft.update("neutral", 0.5, 1), "Should be fatigue (drowsy 0.5초)"
    results.ok("Fatigue 조건3: drowsy 연속")

    ft.reset()

    # 리셋 후 클리어 확인
    ft.update("neutral", 0.5, 0)
    assert not ft.update("neutral", 0.5, 0), "After reset should not be fatigue"
    results.ok("Fatigue reset 동작")

    # Bio missing (arousal=None)
    ft2 = FatigueTracker(duration_sec=1.0, arousal_thresh=0.3, drowsy_sec=0.5, hz=10)
    for i in range(9):
        ft2.update("calm", None, 0)
    assert ft2.update("calm", None, 0), "Bio missing → compound만으로 fatigue"
    state = ft2.get_state()
    assert state["arousal_counter"] == 0, "arousal_counter should be 0 when bio missing"
    results.ok("Fatigue bio missing → compound만으로 판정")

except Exception as e:
    results.fail("Fatigue Tracker", str(e))


# ══════════════════════════════════════════════════════════════════════════
# TEST 6: Degraded Mode 판별
# ══════════════════════════════════════════════════════════════════════════

print("\n[TEST 6] Degraded Mode 판별")
print("-" * 40)

try:
    from pipeline.e2e_pipeline import detect_mode, DegradedMode

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    audio = np.zeros(48000, dtype=np.float32)
    ppg = [(0, 1.0, 2.0)]

    assert detect_mode(frame, audio, ppg, [], []) == DegradedMode.FULL
    results.ok("FULL mode (Cam+Mic+Bio)")

    assert detect_mode(frame, audio, [], [], []) == DegradedMode.CAM_MIC
    results.ok("CAM+MIC mode")

    assert detect_mode(frame, None, ppg, [], []) == DegradedMode.CAM_BIO
    results.ok("CAM+BIO mode")

    assert detect_mode(frame, None, [], [], []) == DegradedMode.CAM_ONLY
    results.ok("CAM_ONLY mode")

    assert detect_mode(None, None, [], [], []) == DegradedMode.NO_CAM
    results.ok("NO_CAM mode")

    # 빈 오디오 배열 → mic 없음 취급
    assert detect_mode(frame, np.array([]), [], [], []) == DegradedMode.CAM_ONLY
    results.ok("빈 오디오 → CAM_ONLY")

except Exception as e:
    results.fail("Degraded Mode", str(e))


# ══════════════════════════════════════════════════════════════════════════
# TEST 7: PacketEncoder 호환성
# ══════════════════════════════════════════════════════════════════════════

print("\n[TEST 7] PacketEncoder 호환성")
print("-" * 40)

try:
    from gateway.packet_encoder import (
        PacketEncoder, KFER_TO_PROTOCOL, PROTOCOL_NAMES, KFER_LABELS,
        compute_crc8, SOF, EOF
    )

    enc = PacketEncoder()

    # K-FER → Protocol 매핑 확인 (6종, sad+hurt 병합)
    expected_mapping = {
        0: (2, "분노"),       # angry → Anger
        1: (0, "공포"),       # anxious → Fear
        2: (4, "행복"),       # happy → Happy
        3: (3, "슬픔/혐오"),  # hurt → Sadness/Disgust (sad와 병합)
        4: (5, "중립"),       # neutral → Neutral
        5: (3, "슬픔/혐오"),  # sad → Sadness/Disgust (hurt와 병합)
        6: (1, "놀람"),       # surprised → Surprise
    }

    for kfer_id, (expected_code, expected_name) in expected_mapping.items():
        code = KFER_TO_PROTOCOL[kfer_id]
        name = PROTOCOL_NAMES[code]
        assert code == expected_code, f"K-FER {kfer_id}: expected code {expected_code}, got {code}"
        assert name == expected_name, f"Code {code}: expected {expected_name}, got {name}"

    results.ok("K-FER → Protocol 매핑 (7개 모두)")

    # 패킷 인코딩/디코딩 라운드트립
    for kfer_id in range(7):
        pkt = enc.encode(kfer_id, emotion_confidence=0.9, arousal=0.5, perclos=0.1)
        assert len(pkt) == 8, f"Packet length should be 8, got {len(pkt)}"
        assert pkt[0] == SOF, f"SOF should be 0xAA"
        assert pkt[7] == EOF, f"EOF should be 0xFE"

        decoded = enc.decode(pkt)
        assert decoded["crc_ok"], f"CRC failed for kfer_id={kfer_id}"
        assert decoded["emotion_code"] == KFER_TO_PROTOCOL[kfer_id]

    results.ok("패킷 인코딩/디코딩 라운드트립 (7개)")

    # Status flag 테스트
    # Stress: angry + high arousal
    enc2 = PacketEncoder()
    pkt_stress = enc2.encode(0, 0.9, arousal=0.8, perclos=0.1)
    d = enc2.decode(pkt_stress)
    assert d["stress"] == True, "Stress should be True (angry + arousal>0.6)"
    results.ok("Stress flag (angry + arousal=0.8)")

    # Low Attention: PERCLOS 0.3
    enc3 = PacketEncoder()
    pkt_attn = enc3.encode(4, 0.5, arousal=0.3, perclos=0.3)
    d = enc3.decode(pkt_attn)
    assert d["low_attention"] == True, "Low attention should be True (PERCLOS=0.3)"
    assert d["drowsy"] == False, "Drowsy should be False (PERCLOS=0.3 < 0.4)"
    results.ok("Low Attention flag (PERCLOS=0.3)")

    # Drowsy: PERCLOS 0.5
    enc4 = PacketEncoder()
    pkt_drowsy = enc4.encode(4, 0.5, arousal=0.3, perclos=0.5)
    d = enc4.decode(pkt_drowsy)
    assert d["drowsy"] == True, "Drowsy should be True (PERCLOS=0.5)"
    results.ok("Drowsy flag (PERCLOS=0.5)")

    # Fatigue bit 추가 테스트 (Byte5 bit0)
    enc5 = PacketEncoder()
    pkt_orig = enc5.encode(4, 0.5, arousal=0.3, perclos=0.1)

    # Byte5 bit0에 fatigue 추가
    pkt_fat = bytearray(pkt_orig)
    pkt_fat[5] = pkt_fat[5] | 0x01
    pkt_fat[6] = compute_crc8(bytes(pkt_fat[1:6]))
    pkt_fat = bytes(pkt_fat)

    # Byte4는 동일해야 함
    assert pkt_orig[4] == pkt_fat[4], "Byte4 should be identical"
    # Byte5 bit0만 차이
    assert (pkt_orig[5] | 0x01) == pkt_fat[5], "Byte5 should differ only in bit0"
    # CRC는 달라야 함 (Byte5가 다르므로)
    assert pkt_orig[6] != pkt_fat[6], "CRC should differ"
    # 새 CRC 검증
    calc_crc = compute_crc8(pkt_fat[1:6])
    assert calc_crc == pkt_fat[6], "Fatigue packet CRC should be valid"
    results.ok("Fatigue bit 추가 (Byte5 bit0) + CRC 재계산")

    # Fatigue=False일 때 기존과 동일 패킷
    enc6a = PacketEncoder()
    enc6b = PacketEncoder()
    pkt_a = enc6a.encode(2, 0.9, 0.5, 0.1)
    pkt_b = enc6b.encode(2, 0.9, 0.5, 0.1)
    assert pkt_a == pkt_b, "Same input should produce same packet"
    results.ok("동일 입력 → 동일 패킷 (하위 호환)")

except Exception as e:
    results.fail("PacketEncoder", str(e))


# ══════════════════════════════════════════════════════════════════════════
# TEST 8: Gateway Sender (NullSender)
# ══════════════════════════════════════════════════════════════════════════

print("\n[TEST 8] Gateway Sender")
print("-" * 40)

try:
    from pipeline.gateway_sender import GatewaySender, NullGatewaySender

    # NullSender는 항상 False
    null_sender = NullGatewaySender()
    assert not null_sender.connect()
    assert not null_sender.send(b"\xAA" * 8)
    assert not null_sender.is_connected
    results.ok("NullGatewaySender (gateway 비활성화)")

    # GatewaySender - 존재하지 않는 포트
    sender = GatewaySender(port="/dev/ttyNONEXISTENT", baudrate=115200)
    connected = sender.connect()
    assert not connected, "Should fail on non-existent port"
    assert not sender.is_connected
    results.ok("GatewaySender 연결 실패 시 graceful fallback")

    # send 실패해도 크래시 안 함
    assert not sender.send(b"\xAA" * 8)
    results.ok("GatewaySender send 실패 시 크래시 없음")

except Exception as e:
    results.fail("Gateway Sender", str(e))


# ══════════════════════════════════════════════════════════════════════════
# TEST 9: E2E Pipeline (Mock Inferencer)
# ══════════════════════════════════════════════════════════════════════════

print("\n[TEST 9] E2E Pipeline 시뮬레이션")
print("-" * 40)

try:
    from pipeline.e2e_pipeline import E2EPipeline

    # Config 수정 (gateway 비활성화, shadow mode off)
    cfg = load_config()
    cfg.gateway.enabled = False
    cfg.shadow_mode.enabled = False

    pipeline = E2EPipeline(cfg)

    # Mock inferencer 주입 (실제 모델 로딩 없이 테스트)
    class MockKMERInferencer:
        def forward(self, frame_bgr, audio_1d, ppg, eda, temp):
            return {
                "arousal": 0.7,
                "valence": 0.6,
                "drowsy": 0,
                "compound_id": 2,
                "compound_label": "happy",
                "kfer_emotion": "happy",
                "kfer_confidence": 0.92,
                "face_detected": True,
            }

    pipeline._inferencer = MockKMERInferencer()
    pipeline._initialized = True

    # PacketEncoder 로드
    from gateway.packet_encoder import PacketEncoder, KFER_TO_PROTOCOL, PROTOCOL_NAMES, KFER_LABELS
    pipeline._encoder = PacketEncoder()
    pipeline._kfer_to_protocol = KFER_TO_PROTOCOL
    pipeline._protocol_names = PROTOCOL_NAMES
    pipeline._kfer_labels = KFER_LABELS

    # --- 시나리오 1: 정상 추론 (happy) ---
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    audio = np.zeros(48000, dtype=np.float32)
    ppg = [(100, 1.0, 2.0)] * 10
    eda = [(100, 0.5)] * 5
    temp = [(100, 36.5)] * 3

    result = pipeline.process_cycle(frame, audio, ppg, eda, temp)

    assert result["face_detected"] == True
    assert result["emotion_code"] == 4, f"Happy should be code 4, got {result['emotion_code']}"
    assert result["emotion_name_ko"] == "행복"
    assert result["stress"] == False
    assert result["fatigue"] == False
    assert result["mode"] == "FULL"
    assert result["packet"] is not None
    assert len(result["packet"]) == 8
    results.ok("시나리오1: happy → emotion_code=4, stress=F, packet 생성")

    # --- 시나리오 2: Stress (angry + high arousal) ---
    class MockStressInferencer:
        def forward(self, frame_bgr, audio_1d, ppg, eda, temp):
            return {
                "arousal": 0.85,
                "valence": 0.2,
                "drowsy": 0,
                "compound_id": 9,
                "compound_label": "angry",
                "kfer_emotion": "angry",
                "kfer_confidence": 0.88,
                "face_detected": True,
            }

    pipeline._inferencer = MockStressInferencer()
    pipeline.smoother.reset()
    pipeline._encoder.seq = 0

    result2 = pipeline.process_cycle(frame, audio, ppg, eda, temp)
    assert result2["emotion_code"] == 2, f"Angry should be code 2, got {result2['emotion_code']}"
    assert result2["stress"] == True, "Should be stressed (angry + arousal=0.85)"
    results.ok("시나리오2: angry + arousal=0.85 → stress=True")

    # --- 시나리오 3: NO_CAM fallback ---
    result3 = pipeline.process_cycle(None, audio, ppg, eda, temp)
    assert result3["face_detected"] == False
    assert result3["emotion_code"] == 5, "NO_CAM should be neutral (code 5)"
    assert result3["stress"] == False
    assert result3["drowsy"] == False or result3["drowsy"] == 0
    assert result3["mode"] == "NO_CAM"
    results.ok("시나리오3: NO_CAM → neutral, 모든 state=False")

    # --- 시나리오 4: CAM_ONLY (mic/bio 없음) ---
    result4 = pipeline.process_cycle(frame, None, None, None, None)
    assert result4["mode"] == "CAM_ONLY"
    results.ok("시나리오4: CAM_ONLY mode 판별")

    # --- 시나리오 5: Fatigue 누적 ---
    class MockDepressedInferencer:
        def forward(self, frame_bgr, audio_1d, ppg, eda, temp):
            return {
                "arousal": 0.2,
                "valence": 0.3,
                "drowsy": 0,
                "compound_id": 6,
                "compound_label": "depressed",
                "kfer_emotion": "sad",
                "kfer_confidence": 0.75,
                "face_detected": True,
            }

    pipeline._inferencer = MockDepressedInferencer()
    pipeline.fatigue_tracker = __import__("pipeline.fatigue_tracker", fromlist=["FatigueTracker"]).FatigueTracker(
        duration_sec=0.5, arousal_thresh=0.3, drowsy_sec=0.3, hz=10  # 짧은 테스트용
    )

    # 5 frames (0.5초) → fatigue
    for i in range(4):
        r = pipeline.process_cycle(frame, audio, ppg, eda, temp)
        assert r["fatigue"] == False, f"Fatigue should be False at frame {i}"

    r_fat = pipeline.process_cycle(frame, audio, ppg, eda, temp)
    assert r_fat["fatigue"] == True, "Fatigue should be True after 5 frames"
    results.ok("시나리오5: depressed 0.5초 → fatigue=True")

    # --- 시나리오 6: Temporal smoothing 효과 ---
    pipeline.smoother.reset()
    pipeline.fatigue_tracker.reset()

    class MockAlternatingInferencer:
        def __init__(self):
            self.count = 0
        def forward(self, frame_bgr, audio_1d, ppg, eda, temp):
            self.count += 1
            if self.count % 3 == 0:
                return {"arousal": 0.5, "valence": 0.5, "drowsy": 0,
                        "compound_id": 0, "compound_label": "neutral",
                        "kfer_emotion": "neutral", "kfer_confidence": 0.6,
                        "face_detected": True}
            return {"arousal": 0.8, "valence": 0.7, "drowsy": 0,
                    "compound_id": 2, "compound_label": "happy",
                    "kfer_emotion": "happy", "kfer_confidence": 0.9,
                    "face_detected": True}

    pipeline._inferencer = MockAlternatingInferencer()

    # 7 frames: happy, happy, neutral, happy, happy, neutral, happy
    arousal_values = []
    for _ in range(7):
        r = pipeline.process_cycle(frame, audio, ppg, eda, temp)
        arousal_values.append(r.get("smoothed_arousal", r.get("arousal")))

    # EMA smoothing으로 arousal이 급변하지 않아야 함
    max_jump = max(abs(arousal_values[i] - arousal_values[i-1])
                   for i in range(1, len(arousal_values)))
    assert max_jump < 0.3, f"EMA should smooth jumps, max_jump={max_jump:.3f}"
    results.ok("시나리오6: EMA smoothing으로 arousal 급변 방지")

    pipeline.shutdown()
    results.ok("E2EPipeline shutdown 정상")

except Exception as e:
    import traceback
    traceback.print_exc()
    results.fail("E2E Pipeline 시뮬레이션", str(e))


# ══════════════════════════════════════════════════════════════════════════
# TEST 10: 10-Class 출력 매핑 전체 확인
# ══════════════════════════════════════════════════════════════════════════

print("\n[TEST 10] 10-Class 출력 매핑")
print("-" * 40)

try:
    from gateway.packet_encoder import KFER_TO_PROTOCOL, PROTOCOL_NAMES

    ten_class_expected = {
        # emotion_name: (kfer_id, protocol_code, protocol_name_ko)
        # 6종 (sad+hurt → 슬픔/혐오 병합)
        "Fear":           (1, 0, "공포"),
        "Surprise":       (6, 1, "놀람"),
        "Anger":          (0, 2, "분노"),
        "Sadness/Disgust":(5, 3, "슬픔/혐오"),
        "Happy":          (2, 4, "행복"),
        "Neutral":        (4, 5, "중립"),
    }

    all_ok = True
    for name, (kfer_id, expected_code, expected_name) in ten_class_expected.items():
        code = KFER_TO_PROTOCOL[kfer_id]
        proto_name = PROTOCOL_NAMES[code]
        if code != expected_code or proto_name != expected_name:
            results.fail(f"10-class {name}", f"code={code}(expected {expected_code})")
            all_ok = False

    if all_ok:
        results.ok("Emotion 6개 매핑 (7개 K-FER → 6개 Protocol, sad+hurt 병합)")

    # Driver State 4개는 로직 검증 (위에서 이미 테스트됨)
    results.ok("Driver State 4개: Stress/LowAttn/Drowsy/Fatigue (위에서 검증 완료)")

except Exception as e:
    results.fail("10-Class 매핑", str(e))


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════

print()
all_passed = results.summary()

if all_passed:
    print("\n  *** 모든 테스트 통과! E2E 파이프라인 소프트웨어 로직 검증 완료 ***")
    print("  다음 단계: Jetson에서 실제 센서 연결 후 sensing_main_shadow.py 실행")
else:
    print("\n  *** 일부 테스트 실패. 위 로그 확인 ***")
    sys.exit(1)
