# K-MER Jetson Thor — Claude Agent 작업 지침

## 빠른 실행 (복붙용)

### 1. 실시간 데모 (카메라 + 모니터 + USB 패킷 전송)
```bash
# 0. USB 가젯 권한 (터미널 열 때마다 1회)
sudo chmod 666 /dev/ttyGS0

# 1. Docker + X11 준비
xhost +local:docker && docker start kmer_run

# 2. 데모 실행 (오디오 Expert 없이 — 빠름)
docker exec -e DISPLAY=$DISPLAY kmer_run \
  python3 /workspace/Real-Time-Driver-State-Emotion-Monitoring-System/kmer_demo_final.py --no_audio

# 2-1. 전체 실행 (오디오 Expert 포함)
docker exec -e DISPLAY=$DISPLAY kmer_run \
  python3 /workspace/Real-Time-Driver-State-Emotion-Monitoring-System/kmer_demo_final.py
```
- 종료: 모니터 창에서 Q/ESC 또는 Ctrl+C
- 녹화: `recordings/` 폴더에 자동 저장

### 2. USB 시뮬레이션 전송 (13-byte 모트렉스 패킷)
```bash
sudo chmod 666 /dev/ttyGS0
cd /home/jetson/work/kmer3/kmer1
python3 gateway_v2/demo_usb_send.py --simulate --port /dev/ttyGS0
```

### 3. USB 카메라 실시간 전송
```bash
sudo chmod 666 /dev/ttyGS0
cd /home/jetson/work/kmer3/kmer1
python3 gateway_v2/demo_usb_send.py --live --port /dev/ttyGS0
```

### 4. 노트북 수신 (Windows, COM7)
```bash
pip install pyserial
python usb_receiver.py COM7
```

### Docker 컨테이너가 없을 때 (최초 1회)
```bash
docker run -d --name kmer_run --runtime nvidia --privileged \
  --network host -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/jetson/work:/workspace \
  -v /dev:/dev \
  kmer_ready sleep infinity
```

### 트러블슈팅
- **카메라 흑백**: 이전 python 프로세스가 카메라 잡고 있음 → `docker exec kmer_run bash -c 'kill -9 $(pgrep python3)'` 후 재실행
- **Qt xcb 에러**: `xhost +local:docker` 확인, `docker exec kmer_run apt-get install -y libsm6 libice6 libxcb-xinerama0`
- **/dev/ttyGS0 없음/권한**: `sudo chmod 666 /dev/ttyGS0` (l4t 가젯에 acm.GS0 이미 활성화됨, g_serial 불필요)
- **protobuf 에러**: `docker exec kmer_run pip install protobuf==4.25.5`

### USB 통신 정보
- **포트**: Jetson `/dev/ttyGS0` ↔ 게이트웨이/노트북 (USB-C 가젯)
- **주의**: `/dev/ttyACM0`은 ADI BLE 동글(워치)이므로 절대 사용 금지
- **프로토콜**: 모트렉스 **12-byte** (2026-04-03 변경, 이전 13-byte)
  - DATA_LEN 필드: 3바이트→2바이트
  - BODY[0]: 비트패킹→감정코드 full byte
  - 상태 플래그(Stress/LowAttn/Drowsy) 제거
- **Baud**: 115200
- **패킷 예시** (공포, 감정7, 상태7): `02 01 00 00 10 03 00 F0 00 FC 1C 03`
- **자세한 보고서**: `docs/USB_통신_전체보고서_260403.md`

### USB 게이트웨이 연결 현황 (2026-04-03)
- **CDC ACM**: 노트북 OK, 게이트웨이 미수신 (setup request -95)
- **Mass Storage**: 네비에서 미디어로 인식, 실시간 통신 불가
- **Bulk Transfer**: FFS v1으로 가젯 생성 성공, write 블로킹 (호스트 미수신)
- **결론**: 게이트웨이가 기대하는 USB 디바이스 클래스/VID/PID 확인 필요 (모트렉스 문의 중)

---

## 현재 상태 (2026-04-06)

### 산업부 과제 (kmer1, 온디바이스)
- **Jetson AGX Thor**: 223.195.35.106, user: jetson, pw: jetson
- **SSH**: paramiko만 가능 (sshpass Windows 안 먹음)
  ```python
  ssh = paramiko.SSHClient()
  ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
  ssh.connect('223.195.35.106', username='jetson', password='jetson', timeout=5)
  ```
- **Python**: `/c/Users/Ryan/anaconda3/python.exe` (Windows 로컬)
- **Docker**: `kmer_run` (kmer_ready 이미지, --privileged --network host)
  - bind mount: `/home/jetson/work` → `/workspace`, `/dev` → `/dev`
  - 꺼져있으면 `docker start kmer_run`
- **프로젝트**: `/home/jetson/work/Real-Time-Driver-State-Emotion-Monitoring-System/`
- **데모**: `/workspace/Real-Time-Driver-State-Emotion-Monitoring-System/kmer_demo_final.py`

### 코드 구조 (Jetson)
```
Real-Time-Driver-State-Emotion-Monitoring-System/
├── sensing/                  # kmer1 메인 (sensing_main.py에 cutin+외부상황 통합됨)
│   ├── sensing_main.py       # 진입점 — RealSense/RODE/Watch/CutinDetector/USB 스레드
│   ├── kmer_inferencer_v2.py # 멀티모달 추론기 (FaceMesh 1회, Expert 병렬)
│   ├── fer_inferencer.py     # K-FER 단독 추론
│   ├── realsense_uvc.py      # 카메라
│   ├── rode.py               # 마이크
│   ├── watch.py              # ADI Study Watch BLE
│   └── pipeline/             # E2E pipeline, smoother, fatigue, gateway
├── cutin_detection/          # 외부상황 감지
│   ├── core/
│   │   ├── detector_yolo.py          # YOLO 끼어들기 (Temporal Slot Query 적용 예정)
│   │   ├── external_situation.py     # 3-Tier (Obstacle+Scene+Frustration+Tracker+TTC)
│   │   ├── cutin_state.py            # thread-safe 전역 상태
│   │   └── detector_base.py          # 큐 기반 워커
│   ├── usb_sender/                   # 500ms 8-byte 패킷
│   └── yolov8n.pt, yolov8n-seg.pt
├── emotion_system/           # K-FER 모델 + 체크포인트
├── multimodal_dms/           # Expert + Fusion 학습 코드
└── recordings/               # 데모 영상
```

### 데모 변경사항 (2026-04-03)
- **Temporal Smoother 적용**: majority vote (window=9) + EMA (alpha=0.2)
  - 감정이 매 프레임 바뀌는 문제 해결
  - USB 패킷도 smoothed 감정으로 전송
- **FaceMesh 시각화**: 얼굴 윤곽, 눈, 입, 코, 눈썹 라인 표시
- **산업부 공식 UI**: 네이비/골드 톤, 헤더바, 구조적 레이아웃
- **SIMULATED 표시**: 워치 미연결 시 시뮬 바이오 신호에 라벨
- **inferencer 변경**: forward에 `_kfer_probs`, `_landmarks` 반환 추가
- 백업: `kmer_demo_final_backup.py`

### 설치된 것
- PyTorch 2.10.0+cu130, CUDA 13.0, TensorRT 10.15
- ultralytics 8.4.19, mediapipe 0.10.18
- supervision 0.27.0 (ByteTrack)
- neurokit2, librosa, sounddevice, pyserial, bleak

### 로컬 코드 (Windows)
- `C:\Users\Ryan\workspace\kmer3\` — git repo (GitHub: RyanAhn533/kmer3)
  - `kmer1/`, `kmer2/`, `scripts/`, `kmer-pkg/` — 전체 코드
  - `data/` — 학습 데이터, 모델, external_situation (~4GB)

### NAS
- `\\223.195.35.115\Heartlab\진행 프로젝트\산업부_차량\`
  - `2차년도\Jetson_Thor_backup\` — 환경 덤프, 세팅 가이드, kmer1 백업
  - `Simulator_data\` — 시뮬레이터 원본 영상 (C001~C019)

---

## TODO (외부상황 고도화)

### 1. YOLOPv2 연결 (drivable area + lane detection)
- 현재: YOLOv8n detection만
- 목표: YOLOP 3-task (detection + drivable seg + lane) 동시 출력
- GitHub: https://github.com/CAIC-AD/YOLOPv2

### 2. ByteTrack 교체
- 현재: 자체 IOU Tracker (external_situation.py ObjectTracker)
- 목표: supervision ByteTrack 또는 Ultralytics 내장 track mode
- `pip install supervision` → 설치 완료

### 3. 과제 정의 5개 이벤트 모듈
| 이벤트 | 감정 매핑 | 상태 |
|--------|----------|------|
| 급 끼어들기 | Anger | 기본 구현됨 (lane overlap Δ) |
| 시야저하 | Anxiety | brightness 휴리스틱만 → contrast+glare 추가 필요 |
| 보행자/이륜차 돌발 | Fear | person bbox overlap 있음 → TTC_VRU 추가 필요 |
| 도로 위 장애물 | Fear | 미구현 → drivable mask 기반 unknown object |
| 안정 순항 | Calm | 모든 이벤트 False 판정 |

### 4. Temporal Slot Query (DETR decoder)
- 과제 PPT S7, S24-25에 정의된 구조
- YOLO 검출 → Embedding → Slot Attention → Self-Attention → Classifier (Sigmoid)
- 현재 Cut-in에 적용 → 다른 이벤트로 확장
- 구현 필요

### 5. 공통 Feature 계산
- D(t): bbox 높이 기반 상대 거리 (focal length 캘리브 or bbox 역수 근사)
- v_rel(t): 거리 변화율
- TTC(t) = D(t) / max(ε, -v_rel(t))
- lane_ratio: 차선 영역 침범률

---

## 현대차 버전 (kmer3 확장, 제한 없음)
- 클라우드 VLM (Claude Vision / GPT-4o) 사용 가능
- 온디바이스 제약 없음
- 감정 모델 전면 교체 가능 (한국형 불필요)
- 아키텍처: 실시간 YOLO + 클라우드 LLM reasoning

---

## 과제 참조 문서
- 외부상황 정의 PPT: `NAS\2차년도\외부상황\차량과제_외부상황관련자료_260303.pptx`
- USB 프로토콜: `docs/USB_통신_전체보고서_260403.md` (최신)
- USB 프로토콜 (이전): `cutin_detection/usb_sender/protocol.py` + `docs/02_JETSON_APPLICATION.md`
- 환경 세팅: `NAS\Jetson_Thor_backup\NEW_JETSON_SETUP_GUIDE.md`
- 모트렉스 확인요청: USB에 저장 (`260403_USB_패킷/모트렉스_확인요청사항.txt`)

## 규칙
- 데이터 삭제 절대 금지 (확인 없이)
- 센싱 실험 중이면 절대 안 건드림
- 코드 수정 시 Jetson에 paramiko SFTP로 업로드
- Docker 안에서 테스트: `docker exec gracious_edison python3 /workspace/test.py`
