# K-MER 새 Jetson 환경 세팅 가이드

**대상:** Jetson AGX Thor (또는 Orin 계열)  
**목적:** K-MER 온디바이스 감정인식 + USB 게이트웨이 통신 환경 구축  
**최종 업데이트:** 2026-04-06

---

## 1. 실행 구조

```
┌─────────────────────────────────────────────┐
│  Jetson Host (L4T Ubuntu)                   │
│                                             │
│  ┌───────────────────────────────────────┐  │
│  │  Docker 컨테이너 (kmer_run)           │  │
│  │  - PyTorch, CUDA, TensorRT            │  │
│  │  - mediapipe, ultralytics             │  │
│  │  - kmer_demo_final.py 실행            │  │
│  │  - /workspace ← bind mount            │  │
│  └───────────────────────────────────────┘  │
│                                             │
│  호스트에서 직접 실행:                        │
│  - USB 전송 (gateway_v2/demo_usb_send.py)   │
│  - USB 가젯 설정 (/dev/ttyGS0)             │
└─────────────────────────────────────────────┘
```

### Docker에서 돌리는 것
- `kmer_demo_final.py` (실시간 데모 — PyTorch/CUDA/mediapipe 필요)
- 모델 추론 전체 (KMERInferencer, KFERExpert, KMERFusion 등)
- OpenCV GUI 표시 (X11 포워딩)

### 호스트에서 돌리는 것
- USB 시뮬레이션 전송 (`demo_usb_send.py` — pyserial만 필요)
- USB 가젯 설정/권한 (`chmod`, `systemctl`)
- Docker 관리 (`docker start/exec`)

---

## 2. 하드웨어 연결

| 장치 | 포트 | 용도 |
|------|------|------|
| Intel RealSense D435 | USB-A (허브) | 카메라 (640x480, 30fps) |
| RODE Wireless GO II | USB-A (허브) | 마이크 |
| ADI Study Watch + BLE 동글 | USB-A → `/dev/ttyACM0` | PPG/EDA/Temp |
| 게이트웨이/노트북 | USB-C → `/dev/ttyGS0` | **USB 패킷 전송** |
| 키보드/마우스 | USB-A (허브) | 입력 |
| 모니터 | DP/HDMI | 대시보드 표시 |

> **주의:** `/dev/ttyACM0`은 워치 동글 전용. USB 통신에 절대 사용 금지!

---

## 3. 환경 세팅 (새 Jetson)

### 3.1 JetPack / L4T 확인
```bash
cat /etc/nv_tegra_release
# 현재: R38.4.0 (JetPack 6.x)
```

### 3.2 Docker 이미지 준비

현재 사용 중인 이미지: `kmer_ready` (24.8GB)

**방법 1: 기존 이미지 복사 (권장)**
```bash
# 기존 Jetson에서 이미지 저장
docker save kmer_ready | gzip > kmer_ready.tar.gz

# 새 Jetson에서 로드
docker load < kmer_ready.tar.gz
```

**방법 2: 직접 빌드**
```bash
# NVIDIA L4T PyTorch 베이스 이미지에서 시작
docker run -d --name kmer_build --runtime nvidia --privileged \
  --network host \
  nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.5-py3 \
  sleep infinity

# 필요 패키지 설치 (requirements_docker_full.txt 참고)
docker exec kmer_build pip install \
  mediapipe==0.10.18 \
  ultralytics==8.4.19 \
  librosa==0.11.0 \
  sounddevice==0.5.5 \
  pyserial \
  bleak==0.20.2 \
  neurokit2==0.2.13 \
  protobuf==4.25.5

# 이미지 커밋
docker commit kmer_build kmer_ready
```

### 3.3 Docker 컨테이너 생성
```bash
docker run -d --name kmer_run --runtime nvidia --privileged \
  --network host -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/jetson/work:/workspace \
  -v /dev:/dev \
  kmer_ready sleep infinity
```

### 3.4 모델 체크포인트 배치
```
emotion_system/
├── result/
│   └── best.pth                    # K-FER 체크포인트
└── multimodal/
    └── checkpoints/ckpt_v3/fold_1/
        └── best.pth                # KMERFusion 체크포인트
```

> 체크포인트 파일은 NAS에서 복사:  
> `\\223.195.35.115\Heartlab\진행 프로젝트\산업부_차량\2차년도\Jetson_Thor_backup\`

### 3.5 호스트 Python 패키지 (USB 전송용)
```bash
pip3 install pyserial numpy
```

### 3.6 USB 가젯 설정
```bash
# ttyGS0은 L4T에서 자동 생성됨 (g_serial 불필요)
# 매 부팅 후 실행:
sudo systemctl stop serial-getty@ttyGS0.service
sudo chmod 666 /dev/ttyGS0
```

---

## 4. 핵심 패키지 버전 (Docker 내)

| 패키지 | 버전 | 용도 |
|--------|------|------|
| torch | 2.11.0a0+nv26.02 | GPU 추론 |
| opencv-python | 4.13.0 | 영상 처리, GUI |
| mediapipe | 0.10.18 | FaceMesh |
| ultralytics | 8.4.19 | YOLOv8 (외부상황) |
| librosa | 0.11.0 | 오디오 처리 |
| sounddevice | 0.5.5 | 마이크 캡처 |
| pyserial | 3.5 | USB 시리얼 통신 |
| bleak | 0.20.2 | BLE (워치) |
| neurokit2 | 0.2.13 | 생체신호 처리 |
| protobuf | 4.25.5 | mediapipe 호환 |

> 전체 목록: `requirements_docker_full.txt` (329개 패키지)

---

## 5. 실행 명령어

### 데모 실행
```bash
# 0. USB 가젯 권한
sudo chmod 666 /dev/ttyGS0

# 1. Docker + X11
xhost +local:docker && docker start kmer_run

# 2. 데모 실행 (오디오 없이)
docker exec -e DISPLAY=$DISPLAY kmer_run \
  python3 /workspace/Real-Time-Driver-State-Emotion-Monitoring-System/kmer_demo_final.py --no_audio

# 3. 전체 실행 (오디오 포함)
docker exec -e DISPLAY=$DISPLAY kmer_run \
  python3 /workspace/Real-Time-Driver-State-Emotion-Monitoring-System/kmer_demo_final.py
```

### USB 전송 테스트
```bash
cd /home/jetson/work/kmer-ondevice-gateway

# 드라이런 (패킷 확인만)
python3 gateway_v2/demo_usb_send.py --dry-run

# 시뮬레이션 (10초 간격 전송)
python3 gateway_v2/demo_usb_send.py --simulate --port /dev/ttyGS0 --interval 10

# 카메라 실시간 전송
python3 gateway_v2/demo_usb_send.py --live --port /dev/ttyGS0
```

---

## 6. 트러블슈팅

| 문제 | 해결 |
|------|------|
| `/dev/ttyGS0` 없음 | L4T 가젯 자동 생성, 재부팅 시도 |
| Permission denied | `sudo chmod 666 /dev/ttyGS0` |
| agetty 점유 | `sudo systemctl stop serial-getty@ttyGS0.service` |
| 카메라 흑백/안뜸 | `docker exec kmer_run bash -c 'kill -9 $(pgrep python3)'` |
| Qt xcb 에러 | `xhost +local:docker` |
| protobuf 에러 | `docker exec kmer_run pip install protobuf==4.25.5` |
| mediapipe FaceMesh 에러 | tensorflow shim 확인 (kmer_demo_final.py 상단) |

---

## 7. NAS 경로

| 경로 | 내용 |
|------|------|
| `\\223.195.35.115\Heartlab\진행 프로젝트\산업부_차량\` | 프로젝트 루트 |
| `2차년도\Jetson_Thor_backup\` | 환경 덤프, 세팅 가이드, 백업 |
| `Simulator_data\` | 시뮬레이터 원본 영상 (C001~C019) |

---

## 8. SSH 접속

```python
import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('223.195.35.106', username='jetson', password='jetson', timeout=5)
```

> sshpass는 Windows에서 안 됨. paramiko만 사용.

---

**MOTIE R&D RS-2024-00487049 | Sejong Univ. HEART Lab**
