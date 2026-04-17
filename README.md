# DeepShield — 뉴스룸 디퓨전 생성 이미지 탐지 시스템

제보 이미지의 AI 생성 여부를 **DIRE (Diffusion Reconstruction Error, ICCV 2023)** 단일 모델로 판정합니다. Midjourney, Stable Diffusion, Nano-Banana 등 디퓨전 계열 생성물을 전체 이미지 기준으로 탐지합니다.

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
    └─ DIRE
         ├─ ADM UNet: DDIM reverse + forward (timestep_respacing=ddim20)
         ├─ |x - x_recon| -> DIRE map
         └─ ResNet-50: DIRE map -> 합성 확률
판정: safe (< 0.30) · caution (0.30 ~ 0.70) · risk (>= 0.70)
```

**이미지는 추론 시간 동안 Pod의 `UPLOAD_DIR`에만 임시 저장됩니다. 외부 스토리지·인증 불필요.**

---

## 모델 구성

| 슬롯 | 모델 | 역할 | 입력 |
|---|---|---|---|
| `dire` | ADM (256x256 uncond) + ResNet-50 | DDIM 왕복 재구성 오차맵 + 이진 분류 | 256×256 [-1, 1] 정규화 |

- DIRE는 얼굴이 아닌 **전체 이미지** 기반이므로 얼굴 탐지(MTCNN) 경로는 없음
- 공식 저장소(`ZhendongWang6/DIRE`)의 `guided_diffusion`을 `PYTHONPATH`로 주입해 사용
- `timestep_respacing=ddim20`으로 속도 최적화 (A5000 기준 이미지당 ~2-3초)
- ADM 가중치는 `DIRE_ADM_WEIGHTS` env로 전환 가능 (uncond 범용 / LSUN 침실)

### 알려진 한계

- 공식 ResNet-50 분류기는 LSUN-ADM 쌍으로 학습되어 있어 도메인 시프트가 클수록 정확도가 저하될 수 있음
- DDIM 왕복 재구성이 비싼 연산이므로 CPU 추론은 실용적이지 않음 (로컬 스모크 테스트용)
- PNG/JPG 파일 포맷 편향(Issue #30) 대응으로 서버에서 업로드 바이트를 PIL로 디코딩 후 처리

---

## 프로젝트 구조

```
DeepShield/
├── apps/
│   ├── api/                        FastAPI 게이트웨이 (Python)
│   │   ├── app/
│   │   │   ├── config.py           threshold 설정
│   │   │   ├── routes/analyze.py   /analyze 엔드포인트
│   │   │   ├── services/
│   │   │   │   ├── verdict.py      DIRE score -> safe/caution/risk
│   │   │   │   └── runpod_client.py RunPod HTTP 클라이언트
│   │   │   └── schemas/analysis.py Pydantic 스키마
│   │   └── dev.ps1
│   └── web/                        Next.js 14 대시보드 (TypeScript)
│       ├── app/page.tsx
│       ├── components/
│       │   ├── DropZone.tsx        이미지 업로드
│       │   ├── TrustMeter.tsx      조작 확률 게이지
│       │   ├── EvidenceViewer.tsx  전체 이미지 히트맵 오버레이
│       │   ├── ModelScoreTabs.tsx  DIRE 점수·설명
│       │   └── ReportExport.tsx    PDF 내보내기
│       └── lib/
│           ├── api.ts              분석 API 호출
│           └── verdict.ts          판정 색상·레이블
├── services/
│   └── runpod-inference/           RunPod 추론 서버 (Python)
│       ├── server.py               FastAPI 서버 (upload + infer)
│       ├── models/
│       │   ├── base.py             DetectorBase / DetectorOutput
│       │   └── dire.py             DireDetector (ADM + ResNet-50)
│       ├── utils/
│       │   ├── heatmap.py          히트맵 생성·오버레이
│       │   └── io.py               이미지 로드 유틸
│       └── requirements.txt
├── docs/
│   └── runpod-setup.md             RunPod 배포 상세 가이드 (dire_v1 볼륨 레이아웃)
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
RUNPOD_INFERENCE_URL=https://[POD_ID]-8000.proxy.runpod.net
DIRE_REPO_PATH=/workspace/dire_v1/repo
DIRE_ADM_WEIGHTS=/workspace/dire_v1/weights/256x256_diffusion_uncond.pt
DIRE_CLASSIFIER_WEIGHTS=/workspace/dire_v1/weights/classifier/lsun_adm.pth
DIRE_TIMESTEP_RESPACING=ddim20
```

### 전체 스택 한 번에 실행

```powershell
.\start.ps1
```

### 접속 URL

| 서비스 | URL |
|---|---|
| 대시보드 | http://localhost:3000 |
| API 문서 | http://localhost:8080/docs |
| 추론 서버 문서 | http://localhost:8000/docs |

---

## RunPod 배포

자세한 절차는 [`docs/runpod-setup.md`](docs/runpod-setup.md) 참조.

### 요약

1. RunPod에서 GPU Pod 생성 (RTX A5000 권장, Volume 50GB, Mount Path: `/workspace`)
2. `/workspace/dire_v1/`에 공식 DIRE 저장소·venv·가중치를 배치
3. env 설정 후 서버 기동

```bash
source /workspace/dire_v1/venv/bin/activate && export DIRE_REPO_PATH=/workspace/dire_v1/repo && export DIRE_ADM_WEIGHTS=/workspace/dire_v1/weights/256x256_diffusion_uncond.pt && export DIRE_CLASSIFIER_WEIGHTS=/workspace/dire_v1/weights/classifier/lsun_adm.pth && export DIRE_TIMESTEP_RESPACING=ddim20 && export PYTHONPATH=/workspace/dire_v1/repo:/workspace/dire_v1/repo/guided-diffusion && export UPLOAD_DIR=/workspace/dire_v1/uploads && export DEVICE=cuda && cd /workspace/ds_repo/services/runpod-inference && python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## 파라미터 조정

`apps/api/app/config.py`:

```python
threshold_safe: float = 0.30  # 미만 -> SAFE
threshold_risk: float = 0.70  # 이상 -> RISK (사이 -> CAUTION)
runpod_inference_timeout: float = 120.0  # DDIM 왕복 시간 고려
```

속도/정확도 트레이드오프는 Pod의 `DIRE_TIMESTEP_RESPACING`으로 조정합니다 (`ddim20` 기본, 더 빠르게 `ddim10`, 더 정확하게 `ddim50`).

---

## 에러 코드

| HTTP | 메시지 | 원인 |
|---|---|---|
| 415 | unsupported media type | JPEG/PNG/WebP/GIF 이외 파일 업로드 |
| 400 | empty upload / image load failed | 빈 파일 또는 디코딩 실패 |
| 404 | file_id not found | /upload 실패 후 /infer 호출 |
| 502 | inference failed | RunPod 서버 연결 실패 또는 내부 오류 |
