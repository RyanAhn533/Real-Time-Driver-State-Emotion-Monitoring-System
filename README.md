# K-MER On-Device Gateway

**차량 온디바이스 복합감정인지 → USB 게이트웨이 통신 시스템**

산업부 과제 RS-2024-00487049 | 세종대학교 HEART Lab  
Jetson AGX Thor 기반 실시간 운전자 감정 모니터링 + 모트렉스 USB 프로토콜 전송

> 최종 업데이트: 2026-04-06

---

## 시스템 개요

```
┌─────────────────────────────────────────────────────────────┐
│  Jetson AGX Thor (USB Device)                               │
│                                                             │
│  Camera ──→ FaceMesh + K-FER ──┐                           │
│  Microphone ──→ emotion2vec ───┤→ KMERFusion → Smoother ──→ PacketEncoder ──→ USB-C
│  Watch BLE ──→ Bio Expert ─────┘       │                    │     (12-byte)    (ttyGS0)
│                                        ↓                    │
│                                 Dashboard (OpenCV)          │
└─────────────────────────────────────────────────────────────┘
                                    │
                                    ↓ USB-C
                            ┌──────────────┐
                            │  게이트웨이    │ (슈퍼게이트/모트렉스)
                            └──────┬───────┘
                                   ↓
                            ┌──────────────┐
                            │  차량 네비     │ (H/U + STI Display)
                            └──────────────┘
```

---

## 디렉토리 구조

```
kmer-ondevice-gateway/
│
├── kmer_demo_final.py              # 실시간 데모 (카메라+모니터+USB 전송)
├── kmer_demo_final_backup.py       # 원본 백업 (UI/smoother 적용 전)
├── gateway_v2_packet_encoder.py    # 데모용 패킷 인코더 복사본
│
├── gateway_v2/                     # USB 게이트웨이 통신 모듈
│   ├── packet_encoder.py           # ★ 12-byte 모트렉스 패킷 인코더/디코더
│   ├── demo_usb_send.py            # USB 전송 데모 (시뮬/라이브/드라이런)
│   ├── gateway_sender.py           # USB Serial / TCP 전송 클래스
│   ├── receiver.py                 # TCP 수신 (노트북 테스트용)
│   ├── bulk_gadget.py              # USB Bulk Transfer 가젯 (FunctionFS)
│   ├── e2e_pipeline.py             # End-to-End 파이프라인 (카메라→패킷)
│   ├── demo_pipeline.py            # 파이프라인 데모
│   └── PROTOCOL_CHANGE_REPORT.md   # 프로토콜 변경 이력
│
├── sensing/                        # 센서 입력 + 추론
│   ├── kmer_inferencer_v2.py       # ★ 멀티모달 추론기 (FaceMesh→Expert→Fusion)
│   ├── kmer_inferencer.py          # v1 추론기 (이전 버전)
│   ├── fer_inferencer.py           # K-FER 단독 추론
│   ├── sensing_main.py             # 센서 스레드 진입점 (카메라/마이크/워치)
│   ├── inference.py                # 단독 추론 스크립트
│   ├── realsense_uvc.py            # RealSense/UVC 카메라
│   ├── rode.py                     # RODE 마이크
│   ├── watch.py                    # ADI Study Watch BLE
│   ├── pipeline/
│   │   ├── temporal_smoother.py    # ★ MajorityVote + EMA 스무딩
│   │   ├── e2e_pipeline.py         # E2E 파이프라인
│   │   ├── fatigue_tracker.py      # 피로도 추적
│   │   └── gateway_sender.py       # 파이프라인용 게이트웨이 전송
│   ├── core/
│   │   ├── buffers.py              # 프레임/오디오/바이오 버퍼
│   │   └── logger.py               # 로깅
│   └── config/
│       └── sensing_config.yaml     # 센서 설정
│
├── multimodal_dms/                 # Expert 모듈 + Fusion 모델
│   ├── experts/
│   │   ├── kfer_expert.py          # K-FER (한국인 표정 인식)
│   │   ├── kfer_trt_wrapper.py     # TensorRT 래퍼
│   │   ├── face_expert.py          # 얼굴 Expert
│   │   ├── audio_expert.py         # 오디오 Expert (emotion2vec + audeering)
│   │   ├── bio_expert.py           # 생체신호 Expert
│   │   ├── bio_expert_v2.py        # 생체신호 Expert v2
│   │   └── facs_aux.py             # FACS AU 보조
│   ├── fusion/
│   │   ├── kmer_fusion.py          # ★ KMERFusion 모델 (Cross-Attention)
│   │   ├── compound_emotion.py     # 복합감정 매핑
│   │   ├── dynamic_alpha.py        # 동적 알파 (Expert 가중치)
│   │   └── losses.py               # 학습 손실 함수
│   ├── gateway/
│   │   ├── packet_encoder.py       # 이전 8-byte 프로토콜 (deprecated)
│   │   └── demo_pipeline.py        # 이전 데모 파이프라인
│   └── kd/
│       ├── cross_label_kd.py       # Cross-label Knowledge Distillation
│       └── teacher_cache.py        # Teacher 캐시
│
├── docs/                           # 문서
│   ├── USB_통신_전체보고서_260403.md  # ★ USB 게이트웨이 연결 전체 보고서
│   ├── USB_송수신_설정가이드.txt      # USB 송수신 설정 방법
│   ├── USB_명령어.txt                # USB 관련 명령어
│   ├── KMER_실행가이드.txt           # K-MER 실행 가이드
│   ├── SYSTEM_ARCHITECTURE.md       # 시스템 아키텍처
│   ├── FULL_PIPELINE_DETAIL.md      # 파이프라인 상세
│   ├── PIPELINE_CODE_MAP.md         # 코드 맵
│   ├── OUTPUT_STATE_SPEC.md         # 출력 상태 스펙
│   ├── COMPLETE_GUIDE.md            # 전체 가이드
│   ├── PACKET_COMPATIBILITY.md      # 패킷 호환성
│   └── PROTOCOL_CHANGE_REPORT.md    # 프로토콜 변경 보고서
│
├── CLAUDE.md                       # Claude Agent 작업 지침 (환경/실행 방법 전부 포함)
├── docker_pip_freeze.txt           # Docker 컨테이너 pip 패키지 목록
├── docker_run_command.txt          # Docker 실행 명령
└── .gitignore
```

---

## USB 프로토콜 (12-byte, 모트렉스 규격)

### 패킷 구조
```
Byte  0: STX         = 0x02
Byte  1: VERSION     = 0x01
Byte  2: RESERVED_H  = 0x00
Byte  3: RESERVED_L  = 0x00
Byte  4: DEST        = 0x10 (모트렉스) / 0x00 (슈퍼게이트)
Byte  5: DATA_LEN[0] = 0x03 (LSB)
Byte  6: DATA_LEN[1] = 0x00
Byte  7: COMMAND     = 0xF0
Byte  8: BODY[0]     = 감정코드 (0x00~0x05, 0xFF=중립)
Byte  9: BODY[1]     = [감정강도 D7~D5][상태강도 D4~D2][Reserved D1~D0]
Byte 10: CHECKSUM    = XOR(Byte0~Byte9)
Byte 11: ETX         = 0x03
```

### 감정 코드
| Code | 감정 | K-FER |
|------|------|-------|
| 0x00 | 공포 | anxious |
| 0x01 | 놀람 | surprised |
| 0x02 | 분노 | angry |
| 0x03 | 슬픔 | sad |
| 0x04 | 행복 | happy |
| 0x05 | 혐오 | hurt |
| 0xFF | 중립 | neutral |

### 예시
```
공포(감정7,상태7): 02 01 00 00 10 03 00 F0 00 FC 1C 03
행복(감정7,상태7): 02 01 00 00 10 03 00 F0 04 FC 18 03
```

---

## 빠른 실행

### 1. 데모 실행 (Docker)
```bash
sudo chmod 666 /dev/ttyGS0
xhost +local:docker && docker start kmer_run
docker exec -e DISPLAY=$DISPLAY kmer_run \
  python3 /workspace/Real-Time-Driver-State-Emotion-Monitoring-System/kmer_demo_final.py --no_audio
```

### 2. USB 시뮬레이션 전송
```bash
sudo chmod 666 /dev/ttyGS0
cd /home/jetson/work/kmer-ondevice-gateway
python3 gateway_v2/demo_usb_send.py --simulate --port /dev/ttyGS0 --interval 10
```

### 3. USB 드라이런 (전송 없이 패킷 확인)
```bash
python3 gateway_v2/demo_usb_send.py --dry-run
```

---

## 주요 변경사항 (2026-04-03)

### 프로토콜
- 13-byte → **12-byte** (DATA_LEN 3바이트→2바이트)
- BODY[0] 비트패킹 → **감정코드 full byte**
- 상태 플래그 (Stress/LowAttn/Drowsy) 제거

### 데모
- **Temporal Smoother 적용** — 감정 안정화 (majority vote + EMA)
- **FaceMesh 시각화** — 얼굴 윤곽, 눈, 입 라인 표시
- **산업부 공식 UI** — 네이비/골드 톤 대시보드

### USB 게이트웨이 연결 테스트
| 방식 | 결과 | 비고 |
|------|------|------|
| CDC ACM | 노트북 OK, 게이트웨이 미수신 | setup request -95 |
| Mass Storage | 미디어 인식, 통신 불가 | 실시간 불가 |
| Bulk Transfer | 가젯 생성 OK, write 블로킹 | 호스트 미수신 |

**미해결**: 게이트웨이가 기대하는 USB VID/PID/클래스 확인 필요 (모트렉스 문의 중)

---

## 환경

- **장비**: Jetson AGX Thor (L4T R38.4.0, 커널 6.8.12-tegra)
- **Docker**: `kmer_run` (kmer_ready 이미지)
- **Python**: 3.12 (Docker 내)
- **GPU**: PyTorch 2.10.0+cu130, CUDA 13.0, TensorRT 10.15
- **센서**: RealSense D435, RODE Wireless GO II, ADI Study Watch

---

**MOTIE R&D RS-2024-00487049 | Sejong Univ. HEART Lab**
