# Jetson AGX Thor 개발 환경 분석서

> 분석 일자: 2026-03-16
> 분석 대상: NVIDIA Jetson AGX Thor Developer Kit

---

## 1. 하드웨어 사양

### 1.1 보드 정보

| 항목 | 값 |
|------|-----|
| 모델명 | NVIDIA Jetson AGX Thor Developer Kit |
| SoC | Tegra 264 (Thor) |
| Board ID | P4071-0000 + P3834-0008 |
| Compatible | `nvidia,p4071-0000+p3834-0008 nvidia,p3834-0008 nvidia,tegra264` |
| 아키텍처 | aarch64 (ARM 64-bit) |

### 1.2 CPU

| 항목 | 값 |
|------|-----|
| 코어 수 | 14코어 |
| 코어 타입 | ARM Cortex-X4 (CPU part: 0xd83, implementer: 0x41) |
| 아키텍처 레벨 | ARMv8 |
| 최대 클럭 | 2.601 GHz |
| BogoMIPS | 2000.00 (코어당) |
| ISA 확장 | SVE2, BF16, INT8 (i8mm), SHA-512, AES, SM3/SM4, DIT, BTI, MTE(PAC) |

**주요 CPU Feature 상세:**

| Feature | 설명 | 활용 |
|---------|------|------|
| `sve2` | Scalable Vector Extension 2 | SIMD 연산 가속 (가변 벡터 길이) |
| `bf16` | BFloat16 | 추론 시 BF16 연산 지원 |
| `i8mm` | INT8 Matrix Multiply | INT8 양자화 모델 가속 |
| `sha512` / `aes` | 암호화 가속 | 보안 통신 HW 가속 |
| `paca` / `pacg` | Pointer Authentication | 보안 강화 (ROP 방지) |
| `bti` | Branch Target Identification | 보안 강화 (JOP 방지) |

### 1.3 GPU

| 항목 | 값 |
|------|-----|
| GPU 이름 | NVIDIA Thor |
| PCI Device ID | 2b00 (rev a1) |
| PCI Bus | 0000:01:00.0 (3D controller) |
| 드라이버 버전 | 580.00 |
| CUDA 버전 | 13.0 |
| MIG 모드 | Disabled |
| 분석 시점 온도 | 40C |
| 분석 시점 사용률 | 19% (데스크톱 환경) |
| 분석 시점 소비 전력 | 2W |

### 1.4 메모리

| 항목 | 값 |
|------|-----|
| 총 메모리 | 128GB (128,790,032 kB) |
| 가용 메모리 | ~122GB (분석 시점) |
| 사용 중 | ~5.7GB (분석 시점, 데스크톱 환경) |
| Swap | 미설정 (0B) |
| 메모리 구조 | CPU-GPU Unified Memory (iGPU 공유) |

### 1.5 스토리지

| 항목 | 값 |
|------|-----|
| 디바이스 | WD PC SN5000S M.2 2280 NVMe SSD (DRAM-less) |
| 총 용량 | 937GB |
| 사용량 | 65GB (8%) |
| 가용 공간 | 824GB |
| 마운트 | `/dev/nvme0n1p1` → `/` |
| EFI 파티션 | `/dev/nvme0n1p10` → `/boot/efi` (63MB) |

### 1.6 네트워크

| 인터페이스 | 칩셋 | 규격 |
|-----------|------|------|
| WiFi | Realtek RTL8852CE | PCIe 802.11ax (Wi-Fi 6) |
| Ethernet | Realtek RTL8126 | PCIe GbE+ |

### 1.7 PCIe 토폴로지

```
0000:00:00.0 PCI bridge      → NVIDIA Device 22e6
0000:01:00.0 3D controller   → NVIDIA Thor GPU (Device 2b00)
0001:00:00.0 PCI bridge      → NVIDIA Device 22d8
0001:01:00.0 Network         → RTL8852CE WiFi 6
0002:00:00.0 PCI bridge      → NVIDIA Device 22d8
0002:01:00.0 Ethernet        → RTL8126
0005:00:00.0 PCI bridge      → NVIDIA Device 22d8
0005:01:00.0 NVMe            → WD SN5000S SSD
```

---

## 2. 시스템 소프트웨어

### 2.1 OS / BSP

| 항목 | 값 |
|------|-----|
| OS | Ubuntu 24.04.4 LTS (Noble Numbat) |
| 커널 | 6.8.12-tegra (SMP PREEMPT) |
| 커널 빌드일 | 2025-12-30 |
| L4T 릴리스 | R38.4.0 (GCID: 43443517) |
| 커널 Variant | OOT (Out-of-Tree) |
| GCC | 13.2.0 (crosstool-NG 1.26.0) |
| 전력 모드 | MAXN (최대 성능) |

### 2.2 NVIDIA 드라이버 / CUDA 스택

| 패키지 | 버전 |
|--------|------|
| NVIDIA Driver | 580.00 (Open Kernel Module) |
| CUDA Toolkit | 13.0 (V13.0.48, release build) |
| cuDNN | 9.12.0.46 (cuda-13 빌드) |
| CUDA 경로 | `/usr/local/cuda-13.0` (심링크: `/usr/local/cuda`) |
| nvidia-cuda-dev | 7.1-b107 |

### 2.3 L4T 패키지 (주요)

| 패키지 | 버전 |
|--------|------|
| nvidia-l4t-core | 38.4.0 |
| nvidia-l4t-cuda | 38.4.0 |
| nvidia-l4t-3d-core | 38.4.0 |
| nvidia-l4t-camera | 38.4.0 |
| nvidia-l4t-gstreamer | 38.4.0 |
| nvidia-l4t-firmware | 38.4.0 |
| nvidia-l4t-bsp-openrm | 38.4.0 |
| nvidia-l4t-display-kernel | 6.8.12-tegra-38.4.0 |
| nvidia-l4t-jetson-io | 38.4.0 |

### 2.4 컨테이너 런타임

| 패키지 | 버전 |
|--------|------|
| Docker | 28.2.2 |
| NVIDIA Container Toolkit | 1.18.1 |
| libnvidia-container | 1.18.1 |

### 2.5 비전 라이브러리

| 패키지 | 버전 |
|--------|------|
| OpenCV | 4.6.0 (시스템 패키지, dev 포함) |
| GStreamer (NVIDIA) | L4T 38.4.0 통합 |

---

## 3. Python / AI-ML 환경

### 3.1 Python 환경

| 항목 | 값 |
|------|-----|
| 배포판 | miniforge3 |
| Python 버전 | 3.13 |
| 위치 | `/home/jetson/miniforge3/` |

### 3.2 주요 Python 패키지

| 패키지 | 버전 | 비고 |
|--------|------|------|
| PyTorch | 2.10.0 | **CUDA 13.0 빌드 (GPU 사용 가능)** |
| torchvision | 0.25.0 | CUDA 지원 |
| torchaudio | 2.10.0 | CUDA 지원 |
| numpy | 2.4.2 | |
| cuDNN | 9.2.0 (via conda) | `torch.backends.cudnn.version()` = 92000 |

### 3.3 GPU 검증 결과 (2026-03-16)

```
torch.cuda.is_available()  = True
torch.version.cuda         = 13.0
torch.cuda.get_device_name = NVIDIA Thor
Compute Capability         = 11.0
GPU Memory (Unified)       = 122.8 GB
cuDNN version              = 92000
```

**벤치마크 (matmul 2000x2000 x100회):**

| 디바이스 | 시간 | 배속 |
|---------|------|------|
| GPU (Thor) | 0.397s | **15.0x** |
| CPU (Cortex-X4) | 5.961s | 1.0x |

**MobileNetV2 GPU 추론 테스트:** 정상 (output shape: [1, 1000])

### 3.4 설치 이력

| 일자 | 변경 내용 | 방법 |
|------|----------|------|
| ~ 초기 | PyTorch 2.10.0 CPU 빌드 (pip) | `pip install torch` |
| 2026-03-16 | PyTorch 2.10.0 CUDA 13.0 빌드로 교체 | `conda install pytorch=2.10.0=cuda130_generic_py313* --channel conda-forge` |

> 교체 전 패키지 스냅샷: `PACKAGE_VERSIONS_20260316.txt`

---

## 4. 참고 사항

### 4.1 [참고] Swap 미설정

**현상:**
- Swap 공간 0B

**영향:**
- 128GB RAM이므로 일반적인 상황에서는 문제 없음
- 대규모 모델 학습 또는 다수 프로세스 동시 실행 시 OOM 가능성

**권장 조치:**
- 필요 시 zram 또는 NVMe swap 설정 (8~16GB 권장)

### 4.3 [참고] 기존 타겟 플랫폼과의 차이

**`SYSTEM_OVERVIEW.md` 기재 타겟:** Jetson Orin
**실제 디바이스:** Jetson AGX Thor

| 비교 항목 | Jetson Orin (기존 타겟) | Jetson AGX Thor (현재) |
|-----------|----------------------|----------------------|
| SoC | Tegra 234 | Tegra 264 |
| CPU | Cortex-A78AE (12코어) | Cortex-X4 (14코어) |
| GPU | Ampere (2048 CUDA cores) | Thor (Blackwell 계열) |
| 메모리 | 32/64GB | 128GB |
| CUDA | 12.x | 13.0 |
| L4T | R36.x | R38.4.0 |

Thor는 Orin 대비 상위 플랫폼이므로 호환성 이슈는 적으나, CUDA 13.0 / Driver 580 기반으로 의존 패키지 버전 확인이 필요합니다.

---

## 5. 시스템 리소스 현황 (분석 시점 스냅샷)

```
CPU 클럭:     972 MHz / 2601 MHz (idle 상태, MAXN 모드)
GPU 온도:     40C
GPU 사용률:   19% (Xorg + gnome-shell)
GPU 전력:     2W
RAM 사용:     5.7GB / 122GB (4.7%)
디스크 사용:  65GB / 937GB (8%)
```

**실행 중 GPU 프로세스:**

| PID | 타입 | 프로세스 | GPU 메모리 |
|-----|------|---------|-----------|
| 3189 | Graphics | Xorg | 113 MiB |
| 3403 | Graphics | gnome-shell | 75 MiB |
| 6438 | Graphics | nautilus | 18 MiB |

---

## 6. 원격 접속 (Remote Access)

### 6.1 VS Code Tunnel (브라우저에서 바로 접속)

Jetson Thor에 VS Code Tunnel이 systemd 서비스로 등록되어 있습니다.
**Jetson이 켜져 있으면 어디서든 접속 가능합니다.**

| 항목 | 값 |
|------|-----|
| 접속 URL | **https://vscode.dev/tunnel/jetson** |
| 터널 이름 | `jetson` |
| 인증 | GitHub 계정 (RyanAhn533) |
| 서비스 상태 확인 | `systemctl --user status code-tunnel.service` |
| 로그 확인 | `code tunnel service log` |
| 자동 시작 | 부팅 시 자동 실행 (systemd enabled) |

**접속 방법:**
1. 아무 브라우저에서 https://vscode.dev/tunnel/jetson 접속
2. GitHub 로그인 (RyanAhn533)
3. 바로 VS Code 환경 사용 가능 (터미널, 파일 편집, Git 전부 가능)

**또는 VS Code 데스크톱 앱에서:**
1. VS Code 열기
2. `Ctrl+Shift+P` → "Remote-Tunnels: Connect to Tunnel"
3. `jetson` 선택

### 6.2 SSH (같은 네트워크/VPN 내에서)

| 항목 | 값 |
|------|-----|
| 내부 IP | `223.195.35.106` (세종대 내부 네트워크) |
| USB IP | `192.168.55.1` (USB 직결 시) |
| 포트 | 22 |
| X11 포워딩 | 활성화 |

```bash
# 같은 네트워크에서 SSH 접속
ssh jetson@223.195.35.106

# GUI 앱 포워딩
ssh -X jetson@223.195.35.106

# 파일 전송
scp local_file.py jetson@223.195.35.106:~/work/
```

### 6.3 서비스 관리 명령어

```bash
# 터널 상태 확인
systemctl --user status code-tunnel.service

# 터널 재시작
systemctl --user restart code-tunnel.service

# 터널 로그 보기
code tunnel service log

# 터널 서비스 제거 (필요 시)
code tunnel service uninstall
```

---

*이 문서는 K-MER 멀티모달 운전자 감정인식 시스템의 실행 환경을 기록한 것입니다.*
*프로젝트 전체 아키텍처는 `SYSTEM_OVERVIEW.md`를 참조하십시오.*
