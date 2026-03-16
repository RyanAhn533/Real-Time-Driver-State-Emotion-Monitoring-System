# Jetson AGX Thor 원격 접속 가이드

> 이 문서를 Claude에게 던지면 바로 연결 작업을 수행할 수 있습니다.

---

## 디바이스 정보

| 항목 | 값 |
|------|-----|
| 디바이스 | NVIDIA Jetson AGX Thor Developer Kit |
| 호스트명 | `jetson` |
| 사용자 | `jetson` |
| 비밀번호 | `jetson` |
| OS | Ubuntu 24.04.4 LTS (aarch64) |
| 위치 | 세종대학교 연구실 (내부 네트워크) |

---

## 1. VS Code Tunnel (어디서든 접속 — 추천)

인터넷만 되면 **브라우저 하나로 바로 접속 가능**합니다.
포트포워딩, VPN 필요 없음. Jetson이 켜져 있기만 하면 됩니다.

### 접속 방법

**브라우저:**
```
https://vscode.dev/tunnel/jetson
```
→ GitHub 로그인 (RyanAhn533) → 바로 VS Code 환경 사용

**VS Code 데스크톱 앱:**
1. VS Code 열기
2. `Ctrl+Shift+P` → `Remote-Tunnels: Connect to Tunnel`
3. GitHub 로그인 → `jetson` 선택

### 터널 서비스 관리

Jetson에서 systemd 서비스로 돌고 있어서 **재부팅해도 자동 실행**됩니다.

```bash
# 상태 확인
systemctl --user status code-tunnel.service

# 재시작
systemctl --user restart code-tunnel.service

# 로그 보기
code tunnel service log

# 서비스 제거
code tunnel service uninstall

# 서비스 재설치
code tunnel service install
```

### 터널이 안 될 때

```bash
# 1. 서비스 상태 확인
systemctl --user status code-tunnel.service

# 2. 죽어 있으면 재시작
systemctl --user restart code-tunnel.service

# 3. 그래도 안 되면 재설치
code tunnel service uninstall
code tunnel service install

# 4. GitHub 인증 만료 시
code tunnel --accept-server-license-terms
# → https://github.com/login/device 에서 코드 입력
```

---

## 2. SSH 접속 (같은 네트워크에서)

세종대 내부 네트워크 또는 VPN 연결 상태에서 사용 가능합니다.

### 네트워크 정보

| 인터페이스 | IP | 용도 |
|-----------|-----|------|
| `enP2p1s0` (이더넷) | `223.195.35.106` | **세종대 내부 네트워크** |
| `l4tbr0` (USB) | `192.168.55.1` | USB 케이블 직결 시 |
| `docker0` | `172.17.0.1` | Docker 내부 (접속용 아님) |

### 접속 명령어

```bash
# 기본 SSH
ssh jetson@223.195.35.106

# GUI 앱 포워딩 (X11)
ssh -X jetson@223.195.35.106

# USB 직결 시
ssh jetson@192.168.55.1

# 특정 포트 포워딩 (예: Jupyter 8888)
ssh -L 8888:localhost:8888 jetson@223.195.35.106
```

### SSH 설정

| 항목 | 값 |
|------|-----|
| 포트 | 22 |
| X11 포워딩 | 활성화 |
| 비밀번호 인증 | 허용 |
| SFTP | 활성화 |

### VS Code Remote SSH (같은 네트워크에서)

1. VS Code → Extensions → `Remote - SSH` 설치
2. `Ctrl+Shift+P` → `Remote-SSH: Connect to Host`
3. `jetson@223.195.35.106` 입력
4. 비밀번호: `jetson`

---

## 3. 파일 전송

```bash
# PC → Jetson
scp myfile.py jetson@223.195.35.106:~/work/

# Jetson → PC
scp jetson@223.195.35.106:~/work/result.pth ./

# 폴더 통째로
scp -r jetson@223.195.35.106:~/work/Real-Time-Driver-State-Emotion-Monitoring-System ./
```

---

## 4. 프로젝트 경로

```
/home/jetson/work/Real-Time-Driver-State-Emotion-Monitoring-System/
├── emotion_system/       # K-FER 모델 + 학습
├── multimodal_dms/       # K-MER 멀티모달 퓨전
├── sensing/              # 실시간 추론 (Jetson)
├── SYSTEM_OVERVIEW.md    # 시스템 아키텍처 문서
├── JETSON_THOR_ENVIRONMENT.md  # 환경 분석서
└── PACKAGE_VERSIONS_20260316.txt  # 패키지 버전 백업
```

---

## 5. GPU 상태 확인

```bash
# GPU 상태
nvidia-smi

# PyTorch GPU 확인
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 기대 출력: True NVIDIA Thor
```

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA Thor (Compute 11.0) |
| CUDA | 13.0 |
| 드라이버 | 580.00 |
| GPU 메모리 | 122.8 GB (Unified) |
| PyTorch | 2.10.0 (CUDA 13.0 빌드) |

---

## 요약: 뭘로 접속하면 되나?

| 상황 | 방법 |
|------|------|
| **카페/집/아무데서나** | `https://vscode.dev/tunnel/jetson` (브라우저) |
| **세종대 연구실 내부** | `ssh jetson@223.195.35.106` |
| **USB 케이블 직결** | `ssh jetson@192.168.55.1` |
| **VS Code로 개발** | Remote Tunnel (`jetson`) 또는 Remote SSH |
