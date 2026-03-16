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
| PyTorch | 2.10.0 | **CPU 전용 빌드** |
| torchvision | 0.25.0 | CPU 전용 |
| torchaudio | 2.10.0 | CPU 전용 |
| numpy | 2.4.2 | |

---

## 4. 주요 이슈 및 권장 조치

### 4.1 [심각] PyTorch CUDA 미지원

**현상:**
- `torch.cuda.is_available()` → `False`
- `torch.version.cuda` → `None`
- miniforge3/pip으로 설치된 일반 CPU 빌드

**영향:**
- K-FER, K-MER Fusion 등 모든 딥러닝 추론이 CPU에서만 실행됨
- Thor GPU(CUDA 13.0)를 전혀 활용하지 못함
- 실시간 DMS 파이프라인 성능 목표 달성 불가

**권장 조치:**
- NVIDIA에서 제공하는 Jetson Thor용 PyTorch wheel (CUDA 13.0 빌드) 설치
- 또는 NVIDIA NGC 컨테이너(`nvcr.io/nvidia/l4t-pytorch`)를 Docker로 활용
- TensorRT 변환을 통한 최적 추론 경로 확보

### 4.2 [참고] Swap 미설정

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

*이 문서는 K-MER 멀티모달 운전자 감정인식 시스템의 실행 환경을 기록한 것입니다.*
*프로젝트 전체 아키텍처는 `SYSTEM_OVERVIEW.md`를 참조하십시오.*
