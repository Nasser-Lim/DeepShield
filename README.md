# DeepShield — 뉴스룸 신뢰도 방어 시스템

딥페이크 탐지 삼중 투표제. 로그인 불필요.

## 아키텍처

```
브라우저  →  Next.js (web:3000)
                ↓ POST /analyze  (multipart image)
         FastAPI Gateway (api:8080)
                ↓ POST /upload   → file_id
                ↓ POST /infer    ← file_id
         RunPod Inference Pod (inference:8000)
           ├─ Effort 탐지기     (가중치 0.40)
           ├─ Face X-ray 탐지기 (가중치 0.35)
           └─ SPSL 탐지기       (가중치 0.25)
                ↓ 가중 투표
         판정: safe (<30%) · caution (<70%) · risk (≥70%)
```

이미지는 추론 시간 동안 Pod의 `UPLOAD_DIR`에 임시 저장됩니다. 외부 스토리지·인증 불필요.

---

## 로컬 실행 (Docker 없이)

### 사전 요구사항

- Python 3.11 이상
- Node.js 18 이상
- PowerShell

### 한 번에 실행

```powershell
cp .env.example .env
.\start.ps1
```

터미널 하나에서 세 서비스가 병렬로 기동됩니다. `Ctrl+C`로 전체 종료.

### 서비스별 개별 실행

각각 별도 터미널에서 실행합니다.

```powershell
# 터미널 1 — AI 추론 서버 (port 8000)
cd services\runpod-inference
.\dev.ps1
```

```powershell
# 터미널 2 — FastAPI 게이트웨이 (port 8080)
cd apps\api
.\dev.ps1
```

```powershell
# 터미널 3 — Next.js 대시보드 (port 3000)
cd apps\web
.\dev.ps1
```

### 접속 URL

| 서비스 | URL |
|---|---|
| 대시보드 | http://localhost:3000 |
| API 문서 | http://localhost:8080/docs |
| 추론 서버 문서 | http://localhost:8000/docs |

### 첫 실행 시 자동으로 처리되는 것

- `services/runpod-inference/.venv` — Python 가상환경 생성 및 패키지 설치
- `apps/api/.venv` — Python 가상환경 생성 및 패키지 설치
- `apps/web/node_modules` — npm 패키지 설치

---

## RunPod 배포

자세한 내용은 [`docs/runpod-setup.md`](docs/runpod-setup.md) 참조.

1. RunPod에서 A5000 Pod 생성 (template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, Volume 50GB)
2. 웹 터미널에서 코드 배포 및 패키지 설치
3. `/volume/weights/`에 모델 가중치 저장
4. `python3 -m uvicorn server:app --host 0.0.0.0 --port 8000` 실행
5. 로컬 `.env`의 `RUNPOD_INFERENCE_URL`을 Pod 공개 URL로 업데이트

```
RUNPOD_INFERENCE_URL=https://[POD_ID]-8000.proxy.runpod.net
```

---

## 플레이스홀더 모델 교체

현재 `services/runpod-inference/models/`의 세 탐지기는 SHA-1 해시 기반 더미 점수를 반환합니다 (딥페이크 판별 능력 없음). 실제 가중치를 붙이려면 각 파일의 `load` / `predict` 메서드만 교체하면 됩니다.

| 파일 | 모델 | 저장소 |
|---|---|---|
| `models/effort.py` | Effort | github.com/HighwayWu/EFFORT |
| `models/face_xray.py` | Face X-ray | github.com/neverUseThisName/Face-X-Ray |
| `models/spsl.py` | SPSL | github.com/SCLBD/DeepfakeBench |

---

## 앙상블 가중치 조정

`.env`에서 수정합니다. 합계가 정확히 1.0이 아니어도 런타임에 정규화됩니다.

```
WEIGHT_EFFORT=0.40
WEIGHT_XRAY=0.35
WEIGHT_SPSL=0.25
```
