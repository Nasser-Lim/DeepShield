# DeepShield — 뉴스룸 딥페이크 탐지 시스템

제보 이미지의 AI 생성 여부를 삼중 모델 앙상블로 판정하는 뉴스룸 전용 도구.

---

## 아키텍처

```
브라우저 (Next.js :3000)
    │
    │  POST /analyze  multipart image
    ▼
FastAPI 게이트웨이 (apps/api :8080)
    │  1. POST /upload  → file_id
    │  2. POST /infer   <- file_id
    ▼
RunPod 추론 서버 (services/runpod-inference :8000)
    ├─ MTCNN 얼굴 탐지  (얼굴 없으면 422 반환)
    ├─ Xception         가중치 0.50
    ├─ F3Net (FAD)      가중치 0.20  (score^1.8 recalibration 적용)
    └─ SPSL             가중치 0.30
         |
         v  가중 앙상블
판정: safe (< 0.35) · caution (0.35~0.75) · risk (>= 0.75)
```

**이미지는 추론 시간 동안 Pod의 `UPLOAD_DIR`에만 임시 저장됩니다. 외부 스토리지·인증 불필요.**

---

## 모델 구성

| 슬롯 | 모델 | 출처 | 입력 | 앙상블 가중치 |
|---|---|---|---|---|
| `effort` | Xception | DeepfakeBench v1.0.1 | 256x256 RGB 3ch | **0.50** |
| `xray` | F3Net (FAD) | DeepfakeBench v1.0.1 | 256x256 DCT 12ch | **0.20** |
| `spsl` | SPSL | DeepfakeBench v1.0.1 | 256x256 RGB+Phase 4ch | **0.30** |

- DeepfakeBench 코드 의존성 없음 — 순수 PyTorch + timm으로 아키텍처 자체 구현, `.pth` 가중치만 로드
- F3Net은 실사 사진에 대한 상향 bias가 있어 `score^1.8` power-law recalibration 적용
- 얼굴 탐지는 **MTCNN** (confidence >= 0.90) 사용. Haar cascade는 fallback 전용

### 현재 한계

세 모델 모두 FaceForensics++ (2018~2019년) 기반으로 학습됐습니다.

- Stable Diffusion, Midjourney, DALL-E 등 최신 Diffusion 모델 생성 이미지에 대한 일반화 능력이 제한적
- 고압축 JPEG 언론 사진에서 F3Net이 오탐할 수 있음
- 얼굴이 가려지거나 없는 이미지는 탐지 범위 밖 (422 에러 반환)

---

## 프로젝트 구조

```
DeepShield/
├── apps/
│   ├── api/                        FastAPI 게이트웨이 (Python)
│   │   ├── app/
│   │   │   ├── config.py           앙상블 가중치·threshold 설정
│   │   │   ├── routes/analyze.py   /analyze 엔드포인트
│   │   │   ├── services/
│   │   │   │   ├── ensemble.py     가중 앙상블 + F3Net recalibration
│   │   │   │   └── runpod_client.py RunPod HTTP 클라이언트
│   │   │   └── schemas/analysis.py Pydantic 스키마
│   │   └── dev.ps1
│   └── web/                        Next.js 14 대시보드 (TypeScript)
│       ├── app/page.tsx
│       ├── components/
│       │   ├── DropZone.tsx        이미지 업로드
│       │   ├── TrustMeter.tsx      조작 확률 게이지
│       │   ├── EvidenceViewer.tsx  히트맵 오버레이 (face_bbox 기반 위치)
│       │   ├── ModelScoreTabs.tsx  모델별 점수·설명 탭
│       │   └── ReportExport.tsx    PDF 내보내기
│       └── lib/
│           ├── api.ts              분석 API 호출
│           └── verdict.ts          판정 색상·레이블
├── services/
│   └── runpod-inference/           RunPod 추론 서버 (Python)
│       ├── server.py               FastAPI 서버 (upload + infer)
│       ├── models/
│       │   ├── _xception.py        순수 PyTorch Xception 구현
│       │   ├── effort.py           Xception 탐지기
│       │   ├── face_detect.py      MTCNN 얼굴 탐지 (Haar fallback)
│       │   ├── face_xray.py        F3Net + FAD head 탐지기
│       │   └── spsl.py             SPSL (RGB+Phase) 탐지기
│       ├── utils/
│       │   ├── heatmap.py          히트맵 생성·오버레이
│       │   └── io.py               이미지 로드 유틸
│       └── requirements.txt
├── docs/
│   └── runpod-setup.md             RunPod 배포 상세 가이드
├── .env.example
└── start.ps1                       로컬 전체 스택 기동
```

---

## 로컬 실행

### 사전 요구사항

- Python 3.11+
- Node.js 18+
- PowerShell

### 환경 변수 설정

```powershell
cp .env.example .env
```

`.env` 주요 항목:

```env
# 실제 RunPod 추론 서버 URL (로컬 추론 서버 사용 시 http://localhost:8000)
RUNPOD_INFERENCE_URL=https://[POD_ID]-8000.proxy.runpod.net
```

### 전체 스택 한 번에 실행

```powershell
.\start.ps1
```

세 서비스가 병렬로 기동됩니다. `Ctrl+C`로 전체 종료.

### 서비스별 개별 실행

```powershell
# 터미널 1 - 추론 서버 (port 8000)
cd services\runpod-inference && .\dev.ps1

# 터미널 2 - FastAPI 게이트웨이 (port 8080)
cd apps\api && .\dev.ps1

# 터미널 3 - Next.js 대시보드 (port 3000)
cd apps\web && npm run dev
```

### 접속 URL

| 서비스 | URL |
|---|---|
| 대시보드 | http://localhost:3000 |
| API 문서 | http://localhost:8080/docs |
| 추론 서버 문서 | http://localhost:8000/docs |

> 로컬 추론 서버는 `torch` 미설치 시 placeholder 모드(SHA-1 해시 기반 더미 점수)로 동작합니다.
> 실제 모델 추론은 RunPod GPU 환경에서만 동작합니다.

---

## RunPod 배포

자세한 절차는 [`docs/runpod-setup.md`](docs/runpod-setup.md) 참조.

### 요약

1. RunPod에서 GPU Pod 생성 (RTX A5000 권장, Volume 50GB, Mount Path: `/workspace`)
2. 웹 터미널에서 최초 셋업:

```bash
# 시스템 패키지
apt-get update && apt-get install -y git wget libgl1 libglib2.0-0

# 코드 클론
cd /workspace && git clone https://github.com/Nasser-Lim/DeepShield.git deepshield

# Python venv (Volume에 저장 - 재시작 후 재설치 불필요)
python3 -m venv /workspace/venv --system-site-packages
source /workspace/venv/bin/activate
pip install -r /workspace/deepshield/services/runpod-inference/requirements.txt
pip install mtcnn tensorflow-cpu

# 모델 가중치 다운로드 (~255MB)
mkdir -p /workspace/weights
wget -P /workspace/weights "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth"
wget -P /workspace/weights "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/f3net_best.pth"
wget -P /workspace/weights "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/spsl_best.pth"
```

3. 서버 기동:

```bash
cd /workspace/deepshield/services/runpod-inference && \
DEVICE=cuda UPLOAD_DIR=/tmp/deepshield/uploads \
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

4. 로컬 `.env` 업데이트:

```env
RUNPOD_INFERENCE_URL=https://[POD_ID]-8000.proxy.runpod.net
```

### Pod 재시작 후 복구 (한 줄)

```bash
cd /workspace/deepshield && git pull && source /workspace/venv/bin/activate && \
mkdir -p /tmp/deepshield/uploads && cd services/runpod-inference && \
DEVICE=cuda UPLOAD_DIR=/tmp/deepshield/uploads \
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## 앙상블 파라미터 조정

`apps/api/app/config.py`에서 수정합니다.

```python
# 앙상블 가중치 (합계가 1.0이 아니어도 런타임에 정규화)
weight_effort: float = 0.50   # Xception - 가장 범용적
weight_xray:   float = 0.20   # F3Net - 실사 오탐 있어 낮게 설정
weight_spsl:   float = 0.30   # SPSL

# 판정 임계값
threshold_safe: float = 0.35  # 미만 -> SAFE
threshold_risk: float = 0.75  # 이상 -> RISK (사이 -> CAUTION)
```

F3Net recalibration (`apps/api/app/services/ensemble.py`):

```python
# 실사 사진의 F3Net 과대평가를 보정. 0.85 -> 0.74 / 0.99 -> 0.98
def _recalibrate_xray(score: float) -> float:
    return score ** 1.8
```

---

## 에러 코드

| HTTP | 메시지 | 원인 |
|---|---|---|
| 415 | unsupported media type | JPEG/PNG/WebP/GIF 이외 파일 업로드 |
| 400 | empty upload | 빈 파일 |
| 422 | 이미지에서 얼굴을 감지할 수 없습니다 | MTCNN이 얼굴을 찾지 못함 (얼굴 없는 이미지, 가려진 얼굴) |
| 502 | inference failed | RunPod 서버 연결 실패 또는 내부 오류 |
