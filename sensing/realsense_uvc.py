"""
RealSense D435 카메라 모듈
=========================
pyrealsense2 → OpenCV UVC 자동 fallback.
"""

import time
import threading
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    HAS_RS = True
except ImportError:
    HAS_RS = False


def _find_realsense_uvc():
    """UVC로 RealSense 비디오 디바이스 검색."""
    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None and frame.shape[1] >= 640:
                return idx
    return None


def run_realsense(
    shutdown_event: threading.Event,
    device_serial: str = "",
    is_main_cam: bool = True,
    on_frame=None,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
):
    tag = "MAIN" if is_main_cam else "SUB"

    # pyrealsense2 사용 가능하면 원래 방식
    if HAS_RS:
        pipe = rs.pipeline()
        cfg = rs.config()
        if device_serial:
            cfg.enable_device(device_serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            pipe.start(cfg)
        except Exception as e:
            print(f"[RealSense:{tag}] pyrealsense start failed: {e}, trying UVC...")
        else:
            print(f"[RealSense:{tag}] started (pyrealsense2): {device_serial} {width}x{height}@{fps}")
            try:
                while not shutdown_event.is_set():
                    try:
                        frameset = pipe.wait_for_frames(timeout_ms=1000)
                    except Exception:
                        continue
                    frame = frameset.get_color_frame()
                    if not frame:
                        continue
                    img = np.asanyarray(frame.get_data())
                    ts_ms = int(time.time() * 1000)
                    if on_frame is not None:
                        on_frame(ts_ms, img)
            finally:
                try:
                    pipe.stop()
                except Exception:
                    pass
                print(f"[RealSense:{tag}] stopped")
            return

    # OpenCV UVC fallback
    dev_idx = _find_realsense_uvc()
    if dev_idx is None:
        print(f"[RealSense:{tag}] no camera found")
        return

    cap = cv2.VideoCapture(dev_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[RealSense:{tag}] started (UVC): /dev/video{dev_idx} {actual_w}x{actual_h}@{fps}")

    try:
        while not shutdown_event.is_set():
            ret, img = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            ts_ms = int(time.time() * 1000)
            if on_frame is not None:
                on_frame(ts_ms, img)
    finally:
        cap.release()
        print(f"[RealSense:{tag}] stopped")


if __name__ == "__main__":
    shutdown = threading.Event()
    count = [0]

    def _on_frame(ts, frame):
        count[0] += 1
        if count[0] % 30 == 1:
            print(f"  Frame #{count[0]}: ts={ts}, shape={frame.shape}")

    print("=== RealSense Test (5s) ===")
    t = threading.Thread(
        target=run_realsense,
        kwargs=dict(shutdown_event=shutdown, on_frame=_on_frame),
        daemon=True,
    )
    t.start()
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        pass
    shutdown.set()
    t.join(timeout=3)
    print(f"\nTotal frames: {count[0]}, ~{count[0]/5:.1f} fps")
