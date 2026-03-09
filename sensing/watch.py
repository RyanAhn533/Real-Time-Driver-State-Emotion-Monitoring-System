import time
import threading
import os
from sys import platform

_WATCH_DIR = os.path.dirname(os.path.abspath(__file__))
from serial.tools import list_ports
from adi_study_watch import SDK
import usb1
import adi_study_watch.core.ble_manager as _bm

VID, PID = 0x0456, 0x2CFE
WATCH_MAC = "F1-18-1C-93-7C-42"

# BLEManager._open()은 getSerialNumber()가 타임아웃 나면 장치를 건너뜀.
# cdc_acm 드라이버가 붙어있을 때 발생하는 문제. serial number 읽기 실패 시
# VID/PID만으로 매칭하도록 패치.
def _patched_ble_open(self):
    context = usb1.USBContext()
    device = None
    for dev in context.getDeviceList(skip_on_error=True):
        try:
            if dev.getVendorID() != self.vendor_id or dev.getProductID() != self.product_id:
                continue
            try:
                s_number = dev.getSerialNumber()
                if s_number == self.dongle_serial_number:
                    device = dev
                    break
            except Exception:
                # serial number 읽기 실패 → VID/PID 일치 장치를 사용
                device = dev
                break
        except Exception:
            pass
    if device is None:
        raise Exception(f"Can't find BLE dongle with vendor_id={self.vendor_id}, "
                        f"product_id={self.product_id} and serial_number={self.dongle_serial_number}.")
    self.device = device.open()
    self.device.resetDevice()
    if platform in ("linux", "linux2"):
        if self.device.kernelDriverActive(0):
            self.device.detachKernelDriver(0)
    self.device.claimInterface(0)
    threading.Thread(target=self.receive_thread, daemon=True).start()

_bm.BLEManager._open = _patched_ble_open

def find_dongle(max_wait_s: int = 10):
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        for p in list_ports.comports():
            if p.vid == VID and p.pid == PID:
                return p.device
        time.sleep(0.3)
    return None

def run_watch(
    shutdown_event: threading.Event,
    on_ppg=None,
    on_eda=None,
    on_temp=None,
):
    port = find_dongle(max_wait_s=10)
    if not port:
        raise RuntimeError("BLE 동글을 찾지 못했습니다. (VID/PID 확인 필요)")

    print(f"[WATCH] dongle port: {port}")

    sdk = None
    for attempt in range(5):
        try:
            sdk = SDK(
                serial_port_address=port,
                mac_address=WATCH_MAC,
                ble_vendor_id=VID,
                ble_product_id=PID,
                ble_timeout=60,
                check_version=False,
            )
            break
        except Exception as e:
            print(f"[WATCH] SDK init attempt {attempt+1}/5 failed: {e}")
            if attempt < 4:
                time.sleep(2.0)
    if sdk is None:
        raise RuntimeError("SDK 초기화 실패 (5회 시도)")

    # 앱 핸들
    adpd_app = sdk.get_adpd_application()
    eda_app  = sdk.get_eda_application()
    temp_app = sdk.get_temperature_application()
    pm_app   = sdk.get_pm_application()

    # DCB 선택
    if pm_app.get_chip_id(pm_app.CHIP_ADPD4K)["payload"]["chip_id"] == 0xC0:
        adpd_dcfg = os.path.join(_WATCH_DIR, "dcb_cfg/DVT1_MV_UC2_ADPD_dcb.dcfg")
    else:
        adpd_dcfg = os.path.join(_WATCH_DIR, "dcb_cfg/DVT2_MV_UC2_ADPD_dcb.dcfg")

    def adpd_callback(data: dict):
        try:
            payload = data.get("payload", {})
            channel_num = int(payload.get("channel_num", 0))
            if channel_num != 1:
                return
            ts = int(payload.get("timestamp", 0))
            sig = payload.get("signal_data", []) or []
            d1 = float(sig[0]) if len(sig) > 0 else 0.0
            d2 = float(sig[1]) if len(sig) > 1 else 0.0
            if on_ppg is not None:
                on_ppg(ts, d1, d2)
        except Exception as e:
            print("[WATCH][ADPD] parse error:", e)

    def eda_callback(data: dict):
        try:
            stream = data["payload"]["stream_data"]
            for v in stream:
                ts = int(v["timestamp"])
                real = float(v["real"])
                if on_eda is not None:
                    on_eda(ts, real)
        except Exception as e:
            print("[WATCH][EDA] parse error:", e)

    def temp_callback(data: dict):
        try:
            payload = data.get("payload", {})
            ts = int(payload.get("timestamp", 0))
            skin = float(payload.get("skin_temperature", 0.0))
            if on_temp is not None:
                on_temp(ts, skin)
        except Exception as e:
            print("[WATCH][TEMP] parse error:", e)

    # 콜백 연결
    adpd_app.set_callback(adpd_callback)
    temp_app.set_callback(temp_callback)
    eda_app.set_callback(eda_callback)

    try:
        eda_app.write_library_configuration([[0x0, 0x1E], [0x02, 0x01]])
    except Exception:
        pass

    try:
        adpd_app.write_device_configuration_block_from_file(adpd_dcfg)

        adpd_app.start_sensor()
        temp_app.start_sensor()
        eda_app.start_sensor()

        time.sleep(2)

        adpd_app.subscribe_stream(adpd_app.STREAM_ADPD6)
        temp_app.subscribe_stream()
        eda_app.subscribe_stream()

        print("[WATCH] streaming started")

        while not shutdown_event.is_set():
            time.sleep(0.05)

    finally:
        print("[WATCH] stopping...")
        try:
            adpd_app.unsubscribe_stream(adpd_app.STREAM_ADPD6)
        except Exception:
            pass
        for app in (temp_app, eda_app):
            try:
                app.unsubscribe_stream()
            except Exception:
                pass

        try:
            adpd_app.stop_sensor()
        except Exception:
            pass
        for app in (temp_app, eda_app):
            try:
                app.stop_sensor()
            except Exception:
                pass

        print("[WATCH] stopped")
