import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import time, threading
from collections import deque, Counter
import numpy as np
import cv2
import torch
import torch.nn.functional as F

import mediapipe as mp

from realsense import run_realsense
from models.fer_model import AUFERModel


# ─────────────────────────────────────────────
# 감정 그룹 정의
#
# SUSTAINED  : 표정이 지속되는 감정 → 낮은 vote 기준 (빠른 전환 허용)
# TRANSIENT  : 일시적으로 튀는 감정 → 높은 vote 기준 (엄격한 검증)
# ─────────────────────────────────────────────
SUSTAINED_EMOTIONS = {"neutral", "happy", "sad"}
TRANSIENT_EMOTIONS = {"surprised", "angry", "anxious", "hurt", "fear", "disgust"}

VOTE_MIN = {
    "sustained": 6,   # vote_window 중 전부 → 지속형은 충분히 확인 후 전환
    "transient": 3,   # vote_window 중 3번 이상 → 순간형은 빠르게 포착
}

def get_emotion_group(emotion: str) -> str:
    if emotion in TRANSIENT_EMOTIONS:
        return "transient"
    return "sustained"


# ─────────────────────────────────────────────
# StableEmotionDetector
#
# 전략 1: Majority Voting (감정 그룹별 기준 차등)
# 전략 2: EMA (score 평활화)
# 전략 3: Confidence Threshold + Hysteresis OFF
# 전략 4: 최소 유지 시간 (min_hold_sec)
# ─────────────────────────────────────────────
class StableEmotionDetector:
    def __init__(
        self,
        vote_window: int      = 6,
        alpha: float          = 0.3,
        conf_threshold: float = 0.5,   # softmax 확률 기준 (0~1)
        off_hold_frames: int  = 3,
        min_hold_sec: float   = 0.5,
    ):
        self.vote_window     = vote_window
        self.alpha           = alpha
        self.conf_threshold  = conf_threshold
        self.off_hold_frames = off_hold_frames
        self.min_hold_sec    = min_hold_sec

        self._vote_buf        = deque(maxlen=vote_window)
        self._ema_scores      = {}
        self._low_conf_streak = 0

        self._alert_on        = False
        self._current_emotion = None
        self._current_score   = None
        self._emotion_set_at  = None

    def _try_set_emotion(self, emotion: str, score: float):
        now = time.time()
        if self._emotion_set_at is not None:
            if now - self._emotion_set_at < self.min_hold_sec:
                return
        self._current_emotion = emotion
        self._current_score   = score
        self._alert_on        = True
        self._emotion_set_at  = now

    def update(self, emotion: str | None, scores: dict | None):
        """
        emotion : 모델 예측 dominant emotion (str)
        scores  : {"happy": 0.82, "sad": 0.05, ...}  (softmax 확률, 0~1)
        반환    : (stable_emotion, stable_score, alert_on)
        """
        # ── 얼굴 없음 ──
        if emotion is None or scores is None:
            self._low_conf_streak += 1
            if self._low_conf_streak >= self.off_hold_frames:
                self._alert_on        = False
                self._current_emotion = None
                self._current_score   = None
                self._ema_scores      = {}
                self._emotion_set_at  = None
            return self._current_emotion, self._current_score, self._alert_on

        raw_score = float(scores.get(emotion, 0.0))

        # ── 전략 3: Confidence Threshold ──
        if raw_score < self.conf_threshold:
            self._low_conf_streak += 1
            if self._low_conf_streak >= self.off_hold_frames:
                self._alert_on        = False
                self._current_emotion = None
                self._current_score   = None
                self._ema_scores      = {}
                self._emotion_set_at  = None
            return self._current_emotion, self._current_score, self._alert_on

        self._low_conf_streak = 0

        # ── 전략 1: Majority Voting ──
        self._vote_buf.append(emotion)
        vote_counts            = Counter(self._vote_buf)
        top_emotion, top_count = vote_counts.most_common(1)[0]

        group     = get_emotion_group(top_emotion)
        min_votes = VOTE_MIN[group]

        if len(self._vote_buf) < self.vote_window or top_count < min_votes:
            return self._current_emotion, self._current_score, self._alert_on

        voted_emotion = top_emotion

        # ── 전략 2: EMA 업데이트 ──
        if not self._ema_scores:
            self._ema_scores = {k: float(v) for k, v in scores.items()}
        else:
            for k, v in scores.items():
                prev = self._ema_scores.get(k, float(v))
                self._ema_scores[k] = self.alpha * float(v) + (1.0 - self.alpha) * prev

        stable_emotion = max(self._ema_scores, key=self._ema_scores.get)
        stable_score   = self._ema_scores[stable_emotion]

        if stable_emotion != voted_emotion:
            return self._current_emotion, self._current_score, self._alert_on

        # ── 전략 4: 최소 유지 시간 ──
        self._try_set_emotion(stable_emotion, stable_score)

        return self._current_emotion, self._current_score, self._alert_on

# -----------------------------
# 설정 (best.pth 학습 기준)
# -----------------------------
IMG_SIZE = 224
NUM_AU = 8       # forehead, eyes_left, eyes_right, nose, cheek_left, cheek_right, mouth, chin
NUM_CLASSES = 7
PTH_PATH = r"/workspace/test/emotion_system/result/best.pth"

# build_label_mapping은 알파벳 순 정렬 → 학습 때와 동일하게
CLASS_NAMES = ["angry", "anxious", "happy", "hurt", "neutral", "sad", "surprised"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 모델 로드  (checkpoint 구조: {"model": state_dict, "config": ..., ...})
# pretrained=False: checkpoint로 덮어쓸 거라 다운로드 불필요
# -----------------------------
model = AUFERModel(num_au=NUM_AU, num_classes=NUM_CLASSES, img_size=IMG_SIZE,
                   pretrained=False).to(device)
ckpt = torch.load(PTH_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"[INFO] checkpoint loaded: epoch={ckpt.get('epoch')}, best_f1={ckpt.get('best_metric', '?'):.4f}")

# -----------------------------
# 최신 프레임 버퍼 (실시간: queue 대신 latest 1장)
# -----------------------------
latest_lock = threading.Lock()
latest_ts = 0
latest_bgr = None

def on_frame(ts_ms: int, frame_bgr: np.ndarray):
    global latest_ts, latest_bgr
    with latest_lock:
        latest_ts = ts_ms
        latest_bgr = frame_bgr  # 최신 프레임으로 덮어쓰기 (드랍 OK)

def get_latest():
    with latest_lock:
        return latest_ts, None if latest_bgr is None else latest_bgr.copy()

# -----------------------------
# 전처리: BGR -> (1,3,224,224) float tensor  (학습과 동일)
# 학습 trainer.py: mean/std = model.backbone.norm_mean/std 사용
# -----------------------------
NORM_MEAN = np.array(model.backbone.norm_mean, dtype=np.float32)
NORM_STD  = np.array(model.backbone.norm_std,  dtype=np.float32)
print(f"[INFO] norm_mean={NORM_MEAN}, norm_std={NORM_STD}")

def preprocess_face224(face_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    x = (x - NORM_MEAN) / NORM_STD
    x = np.transpose(x, (2,0,1))  # CHW
    return torch.from_numpy(x).unsqueeze(0).to(device)  # (1,3,224,224)

# -----------------------------
# FaceMesh → AU center coords 8개  (au_extractor.py의 AU_REGIONS와 동일)
#   forehead   : lm 69, 299, 9 평균
#   eyes_left  : lm 159
#   eyes_right : lm 386
#   nose       : lm 195
#   cheek_left : lm 186
#   cheek_right: lm 410
#   mouth      : lm 13
#   chin       : lm 18
# -----------------------------
mp_face = mp.solutions.face_mesh

# (name, landmark_index_or_tuple)  — au_extractor.AU_REGIONS 그대로
AU_REGIONS = [
    ("forehead",     (69, 299, 9)),
    ("eyes_left",    159),
    ("eyes_right",   386),
    ("nose",         195),
    ("cheek_left",   186),
    ("cheek_right",  410),
    ("mouth",        13),
    ("chin",         18),
]

def _get_center(lms, idx_or_tuple):
    """landmark list → (cx_norm, cy_norm) 0~1 범위."""
    if isinstance(idx_or_tuple, tuple):
        xs = [lms[i].x for i in idx_or_tuple]
        ys = [lms[i].y for i in idx_or_tuple]
        return float(np.mean(xs)), float(np.mean(ys))
    return lms[idx_or_tuple].x, lms[idx_or_tuple].y

WORK_SHORT = 800  # au_extractor.py work_short_side와 동일

def detect_face_and_au(frame_bgr: np.ndarray, mesh):
    """
    au_extractor.py 파이프라인과 동일하게 AU 좌표 추출.

    1. short side → WORK_SHORT 리사이즈
    2. FaceMesh로 landmark 검출
    3. 모든 landmark의 bbox로 얼굴 crop (padding 20%)
    4. crop 기준으로 AU 좌표를 IMG_SIZE(224) 스케일로 변환

    Returns:
        face_crop_bgr : (H,W,3) 얼굴 crop  (224 리사이즈 전)
        au_coords     : (8,2) float32, 224×224 기준 픽셀 좌표
        bbox_in_frame : (x1,y1,x2,y2) 원본 프레임 기준 박스 (시각화용)
    실패 시 (None, None, None) 반환.
    """
    h, w = frame_bgr.shape[:2]
    scale = WORK_SHORT / min(h, w)
    ww, wh = int(round(w * scale)), int(round(h * scale))
    work = cv2.resize(frame_bgr, (ww, wh), interpolation=cv2.INTER_LINEAR)

    work_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    res = mesh.process(work_rgb)
    if not res.multi_face_landmarks:
        return None, None, None

    lms = res.multi_face_landmarks[0].landmark

    # 얼굴 bounding box (work 해상도 기준)
    xs = [lm.x * ww for lm in lms]
    ys = [lm.y * wh for lm in lms]
    x1_w, x2_w = int(min(xs)), int(max(xs))
    y1_w, y2_w = int(min(ys)), int(max(ys))

    # padding 20% (au_extractor는 패딩 없이 전체 face 이미지를 씀 — 여기선 약간 추가)
    pad_x = int((x2_w - x1_w) * 0.20)
    pad_y = int((y2_w - y1_w) * 0.20)
    x1_w = max(0, x1_w - pad_x)
    y1_w = max(0, y1_w - pad_y)
    x2_w = min(ww, x2_w + pad_x)
    y2_w = min(wh, y2_w + pad_y)

    face_w = x2_w - x1_w
    face_h = y2_w - y1_w
    if face_w < 10 or face_h < 10:
        return None, None, None

    face_crop = work[y1_w:y2_w, x1_w:x2_w]

    # AU 좌표: work 픽셀 → crop 내 픽셀 → 224 스케일
    # (au_extractor: work 해상도 저장 후 dataset에서 * img_size/work_w 로 변환)
    coords = []
    for _, idx in AU_REGIONS:
        cx_n, cy_n = _get_center(lms, idx)
        cx_in_crop = cx_n * ww - x1_w          # crop 내 x 픽셀
        cy_in_crop = cy_n * wh - y1_w          # crop 내 y 픽셀
        cx_224 = float(np.clip(cx_in_crop * (IMG_SIZE / face_w), 0, IMG_SIZE - 1))
        cy_224 = float(np.clip(cy_in_crop * (IMG_SIZE / face_h), 0, IMG_SIZE - 1))
        coords.append([cx_224, cy_224])

    au_coords = np.array(coords, dtype=np.float32)  # (8,2)

    # 원본 프레임 기준 bbox (시각화용)
    bbox = (
        int(x1_w / scale), int(y1_w / scale),
        int(x2_w / scale), int(y2_w / scale),
    )
    return face_crop, au_coords, bbox

# -----------------------------
# 메인
# -----------------------------
def main():
    shutdown = threading.Event()

    # RealSense 스레드 시작 (네 코드 구조 그대로)
    # run_realsense()는 on_frame 콜백으로 프레임을 넘김 :contentReference[oaicite:1]{index=1}
    t = threading.Thread(
        target=run_realsense,
        kwargs=dict(
            shutdown_event=shutdown,
            device_serial="254622073310",
            is_main_cam=True,
            on_frame=on_frame,
            width=1280, height=720, fps=30,
        ),
        daemon=True
    )
    t.start()

    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0

    # ── 감정 안정화기 초기화 ──
    stabilizer = StableEmotionDetector(
        vote_window=6,          # 최근 6번의 모델 결과
        alpha=0.3,              # EMA 계수
        conf_threshold=0.5,     # softmax 확률 임계값 (0~1)
        off_hold_frames=3,      # Alert OFF 연속 저신뢰 프레임 수
        min_hold_sec=0.5,       # 감정 최소 유지 시간 (초)
    )
    last_display = {"color": (0, 255, 0), "txt": ""}

    with mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as mesh, torch.no_grad():

        while True:
            ts, frame = get_latest()
            if frame is None:
                time.sleep(0.005)
                continue

            # 1) 얼굴 검출 + AU 좌표 추출 (au_extractor.py 동일 파이프라인)
            face, au, bbox = detect_face_and_au(frame, mesh)

            if face is None:
                stabilizer.update(None, None)
                cv2.putText(frame, "FaceMesh: no face", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow("FER", frame)
                if (cv2.waitKey(1) & 0xFF) in [27, ord('q')]:
                    break
                continue

            # 2) 모델 입력 만들기
            x = preprocess_face224(face)                            # (1,3,224,224)
            au_t = torch.from_numpy(au).unsqueeze(0).to(device)    # (1,8,2)

            # 3) 추론
            logits = model(x, au_t)
            prob = F.softmax(logits, dim=1)
            conf, pred = torch.max(prob, dim=1)

            pred_i = int(pred.item())
            conf_f = float(conf.item())

            # ── 안정화기에 전달할 scores dict 생성 (softmax 확률) ──
            probs_np = prob[0].cpu().numpy()
            raw_emotion = CLASS_NAMES[pred_i]
            raw_scores  = {name: float(probs_np[i]) for i, name in enumerate(CLASS_NAMES)}

            stable_emo, stable_score, alert_on = stabilizer.update(raw_emotion, raw_scores)

            # 확정된 감정 기준으로 표시 (미확정 시 이전 값 유지)
            if stable_emo is not None:
                group     = get_emotion_group(stable_emo)
                box_color = (0, 255, 0) if group == "sustained" else (255, 120, 0)
                last_display["color"] = box_color
                last_display["txt"]   = f"{stable_emo}  {stable_score:.2f}"

            # raw 결과도 작게 표시 (디버그용)
            raw_label = f"(raw: {raw_emotion} {conf_f:.2f})"

            # 4) FPS
            fps_n += 1
            if fps_n % 10 == 0:
                now = time.time()
                fps = 10.0 / (now - fps_t0 + 1e-9)
                fps_t0 = now

            # 5) 원본 프레임에 얼굴 박스 + 안정화된 감정 표시
            bx1, by1, bx2, by2 = bbox
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), last_display["color"], 2)
            cv2.putText(frame, last_display["txt"], (bx1, by1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, last_display["color"], 2)
            cv2.putText(frame, raw_label, (bx1, by2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            cv2.putText(frame, f"FPS {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.imshow("FER", frame)

            # 6) AU 점 + 클래스별 확률 디버그 창
            dbg = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            for i, (xpt, ypt) in enumerate(au.astype(int)):
                cv2.circle(dbg, (xpt, ypt), 4, (0, 0, 255), -1)
                cv2.putText(dbg, AU_REGIONS[i][0][:3], (xpt + 4, ypt - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

            # 클래스별 확률 막대 표시 (probs_np는 위에서 이미 계산됨)
            bar_w = 224
            bar_panel = np.zeros((len(CLASS_NAMES) * 22 + 8, bar_w, 3), dtype=np.uint8)
            for ci, (cname, cp) in enumerate(zip(CLASS_NAMES, probs_np)):
                y = ci * 22 + 4
                filled = int(cp * (bar_w - 80))
                color = (0, 200, 0) if ci == pred_i else (100, 100, 100)
                cv2.rectangle(bar_panel, (80, y), (80 + filled, y + 16), color, -1)
                cv2.putText(bar_panel, f"{cname[:7]:<7} {cp:.2f}", (2, y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

            cv2.imshow("Face224+AU", dbg)
            cv2.imshow("Probs", bar_panel)

            if (cv2.waitKey(1) & 0xFF) in [27, ord('q')]:
                break

    shutdown.set()
    t.join(timeout=1.0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
