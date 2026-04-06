"""
ADI Study Watch 센싱 모듈
========================
ADI BLE Dongle → Study Watch (ADPD PPG, EDA, Temperature) 스트리밍.

바이트코드 복원 기반 재구성:
  - VID=0x0456, PID=0x2CFE
  - WATCH_MAC = F1-18-1C-93-7C-42
  - SDK: adi_study_watch
  - Sensors: ADPD (PPG), EDA, Temperature
  - BLEManager._open 패치로 segfault 방지
"""

import time
import threading
import os
import sys
import platform

from serial.tools import list_ports

# ── SDK import ──
from adi_study_watch import SDK
import usb1
from adi_study_watch.core import ble_manager as _bm

# ── 상수 ──
VID = 0x0456
PID = 0x2CFE
WATCH_MAC = "F1-18-1C-93-7C-42"

_WATCH_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════
# BLEManager._open 패치
# ═══════════════════════════════════════════
# 원본 _open()은 disconnect 시 resetDevice()를 호출하여 segfault 유발.
# 이 패치된 버전은 USB 디바이스를 안전하게 열고
# 리눅스에서 커널 드라이버를 detach.

def _patched_ble_open(self):
    """
    BLEManager._open 대체.
    - resetDevice() 제거 (USB 디바이스 죽음 방지)
    - detachKernelDriver() 제거 (ttyACM0 소실 방지)
    """
    ctx = usb1.USBContext()
    found = None
    for dev in ctx.getDeviceList(skip_on_error=True):
        try:
            if (dev.getVendorID() == self.vendor_id and
                dev.getProductID() == self.product_id and
                dev.getSerialNumber() == self.dongle_serial_number):
                found = dev
                break
        except Exception:
            continue

    if found is None:
        raise Exception(
            f"Can't find BLE dongle with vendor_id={self.vendor_id}, "
            f"product_id={self.product_id} and serial_number={self.dongle_serial_number}."
        )

    self.device = found.open()
    # resetDevice() 제거 - USB 디바이스 죽음 방지
    # detachKernelDriver() 제거 - ttyACM0 소실 방지
    # claimInterface는 시도하되 실패해도 무시
    try:
        self.device.claimInterface(0)
    except Exception:
        pass

    # receive thread 시작
    threading.Thread(target=self.receive_thread, daemon=True).start()


def _patched_disconnect(self):
    """BLEManager.disconnect 대체 - 안전한 disconnect."""
    self._is_connected.clear()
    try:
        self._open()
        msg = [self.RID_CMD, self.CMD_DISCONNECT]
        self._send(msg)
        time.sleep(1)
    except Exception:
        pass  # 이전 연결 없으면 실패해도 무시
    try:
        self.queue.empty()
    except Exception:
        pass
    try:
        if hasattr(self, 'device') and self.device is not None:
            self.device.close()
    except Exception:
        pass


# 패치 적용
_bm.BLEManager._open = _patched_ble_open
_bm.BLEManager.disconnect = _patched_disconnect


def find_dongle(max_wait_s=10):
    """ADI BLE Dongle 포트 검색."""
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        for port in list_ports.comports():
            if port.vid == VID and port.pid == PID:
                return port.device
        time.sleep(0.3)
    return None


def run_watch(
    shutdown_event: threading.Event,
    on_ppg=None,      # (ts_ms: int, d1: float, d2: float)
    on_eda=None,      # (ts_ms: int, real: float)
    on_temp=None,     # (ts_ms: int, skin_c: float)
):
    """
    워치 센싱 메인 루프.

    Callbacks:
      on_ppg(ts_ms, d1, d2)  - PPG 신호 (CH1 signal, CH2 signal)
      on_eda(ts_ms, real)    - EDA 임피던스 실수부 (Ohms)
      on_temp(ts_ms, skin_c) - 피부 온도 (°C)
    """
    dongle_port = find_dongle(max_wait_s=10)
    if dongle_port is None:
        raise RuntimeError("BLE 동글을 찾지 못했습니다. (VID/PID 확인 필요)")

    print(f"[WATCH] dongle port: {dongle_port}")

    # SDK 초기화 (BLE 연결에 10~15초 소요)
    sdk = None
    for attempt in range(1, 4):
        try:
            print(f"[WATCH] SDK init attempt {attempt}/3...")
            sdk = SDK(dongle_port, mac_address=WATCH_MAC)
            if sdk.is_connected():
                break
        except Exception as e:
            print(f"[WATCH] SDK init attempt {attempt}/3 failed: {e}")
            time.sleep(4.0)
            sdk = None

    if sdk is None or not sdk.is_connected():
        raise RuntimeError("SDK 초기화 실패 (3회 시도) - 워치 전원/범위 확인")

    print("[WATCH] SDK connected")

    # ── 앱 초기화 ──
    adpd = sdk.get_adpd_application()
    eda_app = sdk.get_eda_application()
    temp_app = sdk.get_temperature_application()

    # ── 콜백 정의 ──
    _ch1_signal = [None]  # CH1 signal 임시 저장

    def adpd_callback(data):
        try:
            payload = data.get("payload", data)
            ch_num = int(payload.get("channel_num", 0))
            ts = int(payload.get("timestamp", 0))
            signal_data = payload.get("signal_data", [])

            if ch_num == 1 and signal_data:
                # CH1 데이터 저장, CH2 올 때까지 대기
                _ch1_signal[0] = (ts, float(signal_data[0]))
            elif ch_num == 2 and signal_data and _ch1_signal[0] is not None:
                # CH1 + CH2 쌍으로 콜백
                ch1_ts, d1 = _ch1_signal[0]
                d2 = float(signal_data[0])
                _ch1_signal[0] = None
                if on_ppg is not None:
                    on_ppg(ch1_ts, d1, d2)
        except Exception as e:
            print(f"[WATCH][ADPD] parse error: {e}")

    def eda_callback(data):
        try:
            payload = data.get("payload", data)
            stream = payload.get("stream_data", payload)
            ts = int(stream.get("timestamp", 0))
            real_val = float(stream.get("real", 0.0))
            if on_eda is not None:
                on_eda(ts, real_val)
        except Exception as e:
            print(f"[WATCH][EDA] parse error: {e}")

    def temp_callback(data):
        try:
            payload = data.get("payload", data)
            ts = int(payload.get("timestamp", 0))
            skin_c = float(payload.get("skin_temperature", 0.0))
            if on_temp is not None:
                on_temp(ts, skin_c)
        except Exception as e:
            print(f"[WATCH][TEMP] parse error: {e}")

    # ── 센서 시작 ──
    try:
        adpd.set_callback(adpd_callback)
        adpd.start_sensor()
        adpd.subscribe_stream()

        eda_app.set_callback(eda_callback)
        eda_app.start_sensor()
        eda_app.subscribe_stream()

        temp_app.set_callback(temp_callback)
        temp_app.start_sensor()
        temp_app.subscribe_stream()

        print("[WATCH] streaming started")

        # 메인 루프 - shutdown 대기
        while not shutdown_event.is_set():
            time.sleep(0.05)

    finally:
        print("[WATCH] stopping...")
        try:
            adpd.unsubscribe_stream()
            adpd.stop_sensor()
        except Exception:
            pass
        try:
            eda_app.unsubscribe_stream()
            eda_app.stop_sensor()
        except Exception:
            pass
        try:
            temp_app.unsubscribe_stream()
            temp_app.stop_sensor()
        except Exception:
            pass
        # disconnect는 segfault 위험이 있으므로 조심스럽게
        try:
            sdk.disconnect()
        except Exception:
            pass
        print("[WATCH] stopped")


# ── Self-test ──
if __name__ == "__main__":
    shutdown = threading.Event()
    ppg_count = [0]
    eda_count = [0]
    temp_count = [0]

    def _ppg(ts, d1, d2):
        ppg_count[0] += 1
        if ppg_count[0] % 50 == 1:
            print(f"  PPG #{ppg_count[0]}: ts={ts}, d1={d1:.0f}, d2={d2:.0f}")

    def _eda(ts, real):
        eda_count[0] += 1
        if eda_count[0] % 10 == 1:
            print(f"  EDA #{eda_count[0]}: ts={ts}, real={real:.0f}")

    def _temp(ts, skin_c):
        temp_count[0] += 1
        print(f"  TEMP #{temp_count[0]}: ts={ts}, skin={skin_c:.2f}°C")

    print("=== Watch Sensor Test (10초) ===")
    t = threading.Thread(
        target=run_watch,
        kwargs=dict(shutdown_event=shutdown, on_ppg=_ppg, on_eda=_eda, on_temp=_temp),
        daemon=True,
    )
    t.start()

    try:
        time.sleep(10)
    except KeyboardInterrupt:
        pass

    shutdown.set()
    t.join(timeout=5)
    print(f"\nTotal: PPG={ppg_count[0]}, EDA={eda_count[0]}, TEMP={temp_count[0]}")
