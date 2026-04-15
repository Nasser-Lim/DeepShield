# RunPod A5000 셋업 가이드

> **설계 원칙**
> - DeepfakeBench 코드 의존성 없음 — 순수 PyTorch + timm으로 모델 자체 구현, 가중치(.pth)만 로드
> - `/workspace/`가 Volume 마운트 경로 — 코드·가중치·venv 모두 저장되어 Pod 재시작 시 pip install 불필요

---

## Pod 생성 설정

| 항목 | 값 |
|---|---|
| GPU | RTX A5000 (24GB VRAM) |
| Template | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Container Disk | 30GB |
| Volume Disk | 50GB |
| **Volume Mount Path** | **`/workspace`** |
| Expose HTTP Port | `8000` |

> RunPod의 Volume은 `/workspace`에 마운트됩니다 (`/volume` 아님).
> Pod을 Stop/재시작해도 `/workspace`는 유지됩니다.

---

## 최종 디렉토리 구조

```
/workspace/                          (RunPod Volume - 영구 보존)
├── deepshield/                      (DeepShield 저장소)
│   └── services/runpod-inference/
│       ├── server.py
│       ├── models/
│       │   ├── _xception.py         (순수 PyTorch Xception 구현)
│       │   ├── effort.py            (Xception wrapper)
│       │   ├── face_xray.py         (F3Net + FAD wrapper)
│       │   └── spsl.py              (SPSL 4-channel wrapper)
│       ├── utils/
│       └── requirements.txt
├── weights/                         (모델 가중치)
│   ├── xception_best.pth            (~84MB)
│   ├── f3net_best.pth               (~87MB)
│   └── spsl_best.pth                (~84MB)
└── venv/                            (Python 가상환경 - 재시작 후에도 보존)
```

---

## 최초 셋업 절차 (Pod 처음 생성 시)

### Step 1 — 시스템 패키지 설치

```bash
apt-get update && apt-get install -y git wget curl unzip libgl1 libglib2.0-0 nano
```

---

### Step 2 — GPU 확인

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0), '|', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"
```

예상 출력:
```
CUDA: True | NVIDIA RTX A5000 | 24.0 GB
```

---

### Step 3 — 프로젝트 코드 클론

```bash
cd /workspace && git clone https://github.com/Nasser-Lim/DeepShield.git deepshield
```

---

### Step 4 — Python 가상환경 생성 (Volume에 저장)

```bash
python3 -m venv /workspace/venv --system-site-packages && source /workspace/venv/bin/activate && pip install --upgrade pip && pip install -r /workspace/deepshield/services/runpod-inference/requirements.txt
```

> `--system-site-packages` 옵션으로 base image의 torch/torchvision을 그대로 사용합니다.
> `/workspace/venv`는 Volume에 저장되므로 재시작 후에도 재설치 불필요합니다.

설치 패키지: fastapi, uvicorn, pydantic, numpy==1.26.4, opencv-python-headless, pillow, httpx, python-multipart, **timm==1.0.9**

---

### Step 5 — 모델 가중치 다운로드

```bash
mkdir -p /workspace/weights && cd /workspace/weights && wget -O xception_best.pth "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth" && wget -O f3net_best.pth "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/f3net_best.pth" && wget -O spsl_best.pth "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/spsl_best.pth"
```

확인:
```bash
ls -lh /workspace/weights/
```

예상 출력:
```
-rw-rw-rw- 1 root root 87M f3net_best.pth
-rw-rw-rw- 1 root root 84M spsl_best.pth
-rw-rw-rw- 1 root root 84M xception_best.pth
```

---

### Step 6 — 서버 기동

```bash
source /workspace/venv/bin/activate && export DEVICE=cuda UPLOAD_DIR=/tmp/deepshield/uploads && mkdir -p $UPLOAD_DIR && cd /workspace/deepshield/services/runpod-inference && python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

정상 기동 시 출력:
```
INFO:inference:Loading detectors on cuda
INFO:inference:Xception: missing=0 unexpected=7
INFO:inference:F3Net: missing=0 unexpected=7
INFO:inference:SPSL: missing=0 unexpected=7
INFO:inference:Detectors ready: ['effort', 'xray', 'spsl']
INFO:     Uvicorn running on http://0.0.0.0:8000
```

> `missing=0`이면 정상 (모든 파라미터 로드 완료).
> `unexpected=7`은 DeepfakeBench의 `adjust_channel` 레이어로, forward에 사용되지 않아 무시해도 됩니다.
> WARNING이 `missing=0 unexpected=7` 외에 추가로 발생하면 가중치 로드 실패를 의심하세요.

---

### Step 7 — 동작 확인

```bash
# 테스트 이미지 다운로드 (AI 생성 얼굴)
wget -O /tmp/face.jpg "https://thispersondoesnotexist.com"

# 업로드
FILE_ID=$(curl -s -X POST http://localhost:8000/upload -F "image=@/tmp/face.jpg" | python3 -c "import sys,json; print(json.load(sys.stdin)['file_id'])") && echo "file_id: $FILE_ID"

# 추론 (점수만 출력)
curl -s -X POST http://localhost:8000/infer -H "Content-Type: application/json" -d "{\"file_id\": \"$FILE_ID\"}" | python3 -c "
import sys, json
r = json.load(sys.stdin)
for k in ['effort','xray','spsl']:
    print(f'{k}: {r[k][\"score\"]:.4f}')
print(f'face_bbox: {r.get(\"face_bbox\")}')
"
```

예상 출력 (AI 생성 얼굴 기준):
```
effort: 0.3914
xray:   0.8405
spsl:   0.3907
face_bbox: [15, 50, 998, 974]
```

---

### Step 8 — 로컬 .env 업데이트

RunPod 대시보드 → Pod → **Connect** → **HTTP Service [8000]** 에서 공개 URL 확인 후:

```
RUNPOD_INFERENCE_URL=https://[POD_ID]-8000.proxy.runpod.net
```

로컬에서 재시작:
```powershell
.\start.ps1
```

---

## Pod 재시작 후 복구 절차

`/workspace/`는 Volume이라 코드·가중치·venv 모두 보존됩니다. 아래 한 줄만 실행하면 됩니다.

```bash
cd /workspace/deepshield && git pull && source /workspace/venv/bin/activate && export DEVICE=cuda UPLOAD_DIR=/tmp/deepshield/uploads && mkdir -p $UPLOAD_DIR && cd services/runpod-inference && python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### Startup Command 자동화 (선택)

RunPod 대시보드 → Pod 설정 → **Startup Command** 에 등록하면 Pod 시작 시 자동 기동됩니다.

```bash
bash -c "cd /workspace/deepshield && git pull && source /workspace/venv/bin/activate && mkdir -p /tmp/deepshield/uploads && cd services/runpod-inference && DEVICE=cuda UPLOAD_DIR=/tmp/deepshield/uploads python3 -m uvicorn server:app --host 0.0.0.0 --port 8000"
```

---

## 모델 구성

| 슬롯 | 모델 | 가중치 파일 | 입력 | 앙상블 가중치 |
|---|---|---|---|---|
| `effort` | Xception | `xception_best.pth` | 256×256, 3ch RGB | 0.40 |
| `xray` | F3Net (FAD) | `f3net_best.pth` | 256×256, 12ch (DCT) | 0.35 |
| `spsl` | SPSL | `spsl_best.pth` | 256×256, 4ch (RGB+phase) | 0.25 |

판정 기준: `final = 0.40×effort + 0.35×xray + 0.25×spsl`
- `< 0.30` → **safe**
- `0.30 ~ 0.70` → **caution**
- `≥ 0.70` → **risk**

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `WARNING: ... — using placeholder` | 가중치 로드 실패 | `/workspace/weights/` 파일 존재 확인, `missing` 키 로그 확인 |
| `No module named 'timm'` | venv 활성화 안 됨 | `source /workspace/venv/bin/activate` 실행 |
| `500 Internal Server Error` | 추론 중 예외 | uvicorn 터미널의 스택트레이스 확인 |
| `CUDA out of memory` | GPU 메모리 부족 | `nvidia-smi`로 좀비 프로세스 확인 후 kill |
| `/workspace` 비어 있음 | Volume 마운트 실패 | RunPod 대시보드에서 Volume 설정 확인 |
| Pod 공개 URL 502/503 | 서버 미기동 | 위 복구 절차 실행 |
