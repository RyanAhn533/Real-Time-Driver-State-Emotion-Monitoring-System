#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AU Coordinate Extractor
========================
이미지 폴더(class별 하위폴더) → CSV (path, label, au center coords)

기존 2_AU_crop_csv_copy_3.py 기반이지만 단순화:
  - 새 모델은 _cx, _cy만 필요 (wx1/wy1/wx2/wy2 crop box 불필요)
  - EAR(Eye Aspect Ratio) 도 함께 추출 (졸음감지용)
  - auto-orient(0/180도) 포함
  - spawn multiprocessing

출력 CSV 스키마:
  path, label, work_w, work_h,
  forehead_cx, forehead_cy,
  eyes_left_cx, eyes_left_cy,
  eyes_right_cx, eyes_right_cy,
  nose_cx, nose_cy,
  cheek_left_cx, cheek_left_cy,
  cheek_right_cx, cheek_right_cy,
  mouth_cx, mouth_cy,
  chin_cx, chin_cy,
  ear_left, ear_right
"""

import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOGGING_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

import csv
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm

import cv2
cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


# ═══════════════ AU Region Definitions ═══════════════
# 새 모델은 6개 region, 좌우 분리 → 총 9개 포인트
AU_REGIONS = [
    # (name,  landmark_index_or_tuple)
    ("forehead",    (69, 299, 9)),       # 평균 → 1 point
    ("eyes_left",   159),                # 왼눈 → 1 point
    ("eyes_right",  386),                # 오른눈 → 1 point
    ("nose",        195),                # 코 → 1 point
    ("cheek_left",  186),                # 왼뺨 → 1 point
    ("cheek_right", 410),                # 오른뺨 → 1 point
    ("mouth",       13),                 # 입 → 1 point
    ("chin",        18),                 # 턱 → 1 point
]

# EAR landmarks
LEFT_EYE_IDX =  [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


# ═══════════════ Utilities ═══════════════

def list_images(root: Path) -> List[Tuple[Path, str]]:
    """class별 하위폴더에서 이미지 수집."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    pairs = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in exts:
                pairs.append((f, cls_name))
    return pairs


def resize_short_side(img, short_side: int):
    h, w = img.shape[:2]
    if min(h, w) == short_side:
        return img
    scale = short_side / float(min(h, w))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def compute_ear(landmarks, eye_indices, w, h):
    """Eye Aspect Ratio."""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    def dist(a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    v1 = dist(pts[1], pts[5])
    v2 = dist(pts[2], pts[4])
    hz = dist(pts[0], pts[3])
    if hz < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * hz)


def eye_distance(lms, w, h):
    lx = lms.landmark[159].x * w
    ly = lms.landmark[159].y * h
    rx = lms.landmark[386].x * w
    ry = lms.landmark[386].y * h
    return float(np.hypot(rx - lx, ry - ly))


def auto_orient_180(fm, img):
    """0도/180도 중 눈 간 거리가 큰 쪽 선택."""
    best_deg, best_lms, best_score, best_img = -1, None, -1.0, None
    for deg in (0, 180):
        test = img if deg == 0 else cv2.rotate(img, cv2.ROTATE_180)
        res = fm.process(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            continue
        lms = res.multi_face_landmarks[0]
        sc = eye_distance(lms, test.shape[1], test.shape[0])
        if sc > best_score:
            best_deg, best_lms, best_score, best_img = deg, lms, sc, test
    if best_img is None:
        return img, None, 0
    return best_img, best_lms, best_deg


def get_center(landmarks, idx_or_tuple, w, h):
    """단일 인덱스 또는 튜플 → (cx, cy) 픽셀 좌표."""
    if isinstance(idx_or_tuple, tuple):
        xs = [landmarks[i].x for i in idx_or_tuple]
        ys = [landmarks[i].y for i in idx_or_tuple]
        return float(np.mean(xs)) * w, float(np.mean(ys)) * h
    else:
        return landmarks[idx_or_tuple].x * w, landmarks[idx_or_tuple].y * h


# ═══════════════ Worker ═══════════════

_fm = None
_work_short = None

def _worker_init(work_short):
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except:
        pass

    global _fm, _work_short
    _work_short = work_short

    import mediapipe as mp
    _fm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    )


def _process_one(args):
    """단일 이미지 처리 → CSV 행 반환."""
    img_path, cls_name = args

    img0 = cv2.imread(str(img_path))
    if img0 is None:
        return None

    # 1) work resolution
    work = resize_short_side(img0, _work_short)

    # 2) auto orient + FaceMesh
    work, lms, rot_deg = auto_orient_180(_fm, work)
    if lms is None:
        return None

    h, w = work.shape[:2]
    landmarks = lms.landmark

    # 3) AU center coordinates
    row = [str(img_path), cls_name, w, h]
    for name, idx in AU_REGIONS:
        cx, cy = get_center(landmarks, idx, w, h)
        row.extend([f"{cx:.2f}", f"{cy:.2f}"])

    # 4) EAR (for drowsiness)
    ear_l = compute_ear(landmarks, LEFT_EYE_IDX, w, h)
    ear_r = compute_ear(landmarks, RIGHT_EYE_IDX, w, h)
    row.extend([f"{ear_l:.4f}", f"{ear_r:.4f}"])

    return row


# ═══════════════ Main Processing ═══════════════

def extract_au_csv(
    src_root: Path,
    out_csv: Path,
    work_short: int = 800,
    workers: Optional[int] = None,
    chunksize: int = 32,
) -> Path:
    """
    이미지 폴더 → AU 좌표 CSV 생성.

    Args:
        src_root: 클래스별 하위폴더가 있는 이미지 루트
        out_csv: 출력 CSV 경로
        work_short: 작업 해상도 (짧은 변 기준)
        workers: 병렬 워커 수
        chunksize: imap chunksize

    Returns:
        out_csv 경로
    """
    samples = list_images(src_root)
    if not samples:
        print(f"[WARN] No images found in {src_root}")
        return out_csv

    if workers is None:
        workers = min(cpu_count(), 12)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Header
    header = ["path", "label", "work_w", "work_h"]
    for name, _ in AU_REGIONS:
        header += [f"{name}_cx", f"{name}_cy"]
    header += ["ear_left", "ear_right"]

    success = 0
    fail = 0

    with Pool(processes=workers, initializer=_worker_init,
              initargs=(work_short,)) as pool, \
         open(out_csv, "w", newline="", encoding="utf-8") as f:

        wr = csv.writer(f)
        wr.writerow(header)

        for row in tqdm(pool.imap(_process_one, samples, chunksize=chunksize),
                        total=len(samples), desc=f"AU extract: {src_root.name}",
                        dynamic_ncols=True):
            if row is not None:
                wr.writerow(row)
                success += 1
            else:
                fail += 1

    print(f"[완료] {src_root.name} → {out_csv}")
    print(f"  성공: {success}, 실패(얼굴 미검출): {fail}")

    return out_csv


# ═══════════════ CLI ═══════════════

if __name__ == "__main__":
    import argparse
    set_start_method("spawn", force=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="이미지 루트 (class 폴더 포함)")
    ap.add_argument("--out", type=Path, required=True, help="출력 CSV 경로")
    ap.add_argument("--work-short", type=int, default=800)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    extract_au_csv(args.src, args.out, args.work_short, args.workers)
