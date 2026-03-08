import time
import threading
import numpy as np
import pyrealsense2 as rs

def run_realsense(
    shutdown_event: threading.Event,
    device_serial: str,
    is_main_cam: bool = True,
    on_frame=None,  # (ts_ms:int, frame_bgr:np.ndarray)
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
):
    tag = "MAIN" if is_main_cam else "SUB"
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(device_serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    try:
        pipe.start(cfg)
    except Exception as e:
        print(f"[RealSense:{tag}] start failed: {e}")
        return
    
    print(f"[RealSense:{tag}] started: {device_serial} {width}x{height}@{fps}")

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