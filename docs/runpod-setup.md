# RunPod A5000 셋업 가이드

> **설계 원칙**
> - DeepfakeBench 코드 의존성 완전 제거 → 순수 PyTorch + timm으로 모델 자체 구현 (가중치만 로드)
> - Volume(`/workspace`)에 코드/가중치/venv 모두 저장 → Pod 재시작 시 추가 설치 불필요

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

> RunPod 기본 마운트 경로는 `/workspace` 입니다. Pod을 Stop/재시작해도 `/workspace`는 유지됩니다.
> 모델 가중치, 코드, Python 가상환경 모두 `/workspace/`에 저장합니다.

---

## 디렉토리 구조

```
/workspace/                          (RunPod Volume - 영구 보존)
├── deepshield/                      (DeepShield 저장소 클론)
│   └── services/runpod-inference/
│       ├── server.py
│       ├── models/
│       │   ├── effort.py            (Xception 자체 구현 wrapper)
│       │   ├── face_xray.py         (F3Net 자체 구현 wrapper)
│       │   └── spsl.py              (SPSL 자체 구현 wrapper)
│       ├── utils/
│       └── requirements.txt
│
├── weights/                         (모델 가중치 .pth 파일들)
│   ├── xception_best.pth
│   ├── f3net_best.pth
│   └── spsl_best.pth
│
└── venv/                            (Python 가상환경 - 재시작 후에도 보존)
    └── ...
```

---

## Step 1 — 시스템 패키지 설치

```bash
apt-get update && apt-get install -y git wget curl unzip libgl1 libglib2.0-0 nano
```

---

## Step 2 — GPU 확인

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0), '|', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"
```

예상 출력:
```
CUDA: True | NVIDIA RTX A5000 | 24.0 GB
```

---

## Step 3 — 프로젝트 코드 배포 (Volume에 저장)

```bash
cd /workspace && git clone https://github.com/Nasser-Lim/DeepShield.git deepshield && cd deepshield/services/runpod-inference
```

---

## Step 4 — Python 가상환경 생성 (Volume에 저장)

> venv를 `/workspace/venv`에 생성하면 Pod 재시작 시 pip install이 불필요합니다.

```bash
python3 -m venv /workspace/venv && source /workspace/venv/bin/activate && pip install --upgrade pip && pip install -r /workspace/deepshield/services/runpod-inference/requirements.txt
```

설치 패키지 (총 9개, ~1분 소요):
- fastapi, uvicorn[standard], pydantic, python-multipart
- numpy==1.26.4, opencv-python-headless, pillow, httpx
- timm==1.0.9 (Xception backbone 제공)

> torch / torchvision은 base image에 이미 포함되어 있어 venv는 `--system-site-packages` 없이도 자동 사용 가능 ... 이 아닙니다. venv는 격리되므로 torch도 별도 설치가 필요하면 다음 명령:
>
> ```bash
> pip install torch==2.4.1 torchvision==0.19.1
> ```
>
> 또는 시스템 패키지 공유 venv:
> ```bash
> python3 -m venv /workspace/venv --system-site-packages
> ```

---

## Step 5 — 모델 가중치 다운로드 (Volume에 저장)

세 모델(Xception, F3Net, SPSL) 가중치는 [SCLBD/DeepfakeBench v1.0.1](https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1)에서 제공됩니다.

```bash
mkdir -p /workspace/weights && cd /workspace/weights && wget -O xception_best.pth "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth" && wget -O f3net_best.pth "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/f3net_best.pth" && wget -O spsl_best.pth "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/spsl_best.pth"
```

확인:
```bash
ls -lh /workspace/weights/
```

---

## Step 6 — 가중치 키 구조 탐색 (모델 설계 기초 데이터)

DeepShield는 DeepfakeBench 코드를 임포트하지 않고, 자체 구현한 PyTorch 모델에 가중치만 로드합니다. 각 `.pth` 파일의 state_dict 키 prefix를 알아야 매핑 함수를 짤 수 있습니다.

```bash
source /workspace/venv/bin/activate && python3 -c "
import torch
for name in ['xception_best.pth', 'f3net_best.pth', 'spsl_best.pth']:
    path = f'/workspace/weights/{name}'
    ckpt = torch.load(path, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    keys = list(sd.keys())
    print(f'\n=== {name} ===')
    print(f'  Total keys: {len(keys)}')
    print(f'  First 8: {keys[:8]}')
    conv1 = [k for k in keys if \"conv1\" in k.lower()]
    if conv1:
        print(f'  conv1: {conv1[0]} shape={tuple(sd[conv1[0]].shape)}')
"
```

이 출력 결과를 가지고 `models/effort.py`, `face_xray.py`, `spsl.py`의 `_remap_keys()` 함수를 조정합니다.

---

## Step 7 — 서버 기동

```bash
source /workspace/venv/bin/activate && cd /workspace/deepshield/services/runpod-inference && export DEVICE=cuda UPLOAD_DIR=/tmp/deepshield/uploads WEIGHTS_DIR=/workspace/weights && mkdir -p $UPLOAD_DIR && python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

정상 기동 시 출력:
```
INFO:inference:Loading detectors on cuda
INFO:inference:Xception: missing=0 unexpected=0
INFO:inference:F3Net: missing=0 unexpected=0
INFO:inference:SPSL: missing=0 unexpected=0
INFO:inference:Detectors ready: ['effort', 'xray', 'spsl']
INFO:     Uvicorn running on http://0.0.0.0:8000
```

`Loading detectors on cuda` 후 WARNING이 없어야 합니다. WARNING이 있으면 placeholder로 fallback된 상태이므로 Step 6의 키 매핑을 조정해야 합니다.

---

## Step 8 — 동작 확인

테스트 이미지 다운로드:
```bash
wget -O /tmp/face.jpg "https://thispersondoesnotexist.com"
```

업로드 → 추론:
```bash
curl http://localhost:8000/healthz | python3 -m json.tool

FILE_ID=$(curl -s -X POST http://localhost:8000/upload -F "image=@/tmp/face.jpg" | python3 -c "import sys,json; print(json.load(sys.stdin)['file_id'])") && echo "file_id: $FILE_ID"

curl -s -X POST http://localhost:8000/infer -H "Content-Type: application/json" -d "{\"file_id\": \"$FILE_ID\"}" | python3 -c "
import sys, json
r = json.load(sys.stdin)
for k in ['effort','xray','spsl']:
    print(f'{k}: {r[k][\"score\"]:.4f}')
print(f'face_bbox: {r.get(\"face_bbox\")}')
"
```

예상 출력 (예시):
```
effort: 0.7234
xray: 0.6891
spsl: 0.9512
face_bbox: [15, 50, 998, 974]
```

같은 이미지를 두 번 보내도 점수가 동일하면 결정론적 추론이 정상 작동하는 것입니다.

---

## Step 9 — 로컬 .env 업데이트

RunPod 대시보드 → Pod → **Connect** → **HTTP Service [8000]** 에서 공개 URL 확인 후 로컬 `.env` 수정:

```
RUNPOD_INFERENCE_URL=https://[POD_ID]-8000.proxy.runpod.net
```

로컬에서 재시작:
```powershell
.\start.ps1
```

---

## Pod 재시작 시 복구 절차

Container Disk는 초기화되지만 `/workspace/` 전체가 Volume이라 코드, 가중치, venv 모두 보존됩니다.

```bash
cd /workspace/deepshield && git pull && source /workspace/venv/bin/activate && export DEVICE=cuda UPLOAD_DIR=/tmp/deepshield/uploads WEIGHTS_DIR=/workspace/weights && mkdir -p $UPLOAD_DIR && cd services/runpod-inference && python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

git pull로 최신 코드만 받고 (수초), pip install 불필요, 가중치 다운로드 불필요. 즉시 기동.

### Startup Command 자동화

RunPod 대시보드 → Pod 설정 → **Startup Command** 에 등록:

```bash
bash -c "cd /workspace/deepshield && git pull && source /workspace/venv/bin/activate && mkdir -p /tmp/deepshield/uploads && cd services/runpod-inference && DEVICE=cuda UPLOAD_DIR=/tmp/deepshield/uploads WEIGHTS_DIR=/workspace/weights python3 -m uvicorn server:app --host 0.0.0.0 --port 8000"
```

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `Detectors ready` 후 WARNING `... — using placeholder` | 가중치 로드 실패 (키 불일치 등) | Step 6 probe 결과 확인 후 `models/*.py`의 `_remap_keys()` 수정 |
| `No module named 'timm'` | venv 활성화 안 됨 | `source /workspace/venv/bin/activate` 먼저 실행 |
| `CUDA out of memory` | 다른 프로세스가 GPU 사용 중 | `nvidia-smi` 확인 후 좀비 프로세스 kill |
| Pod 공개 URL 502/503 | 서버 기동 안 됨 | Pod 터미널에서 uvicorn 로그 확인 |
| `/workspace` 가 비어있음 | Volume 마운트 실패 | RunPod 대시보드에서 Volume 설정 확인 |

---

## 리소스 사용량 (참고)

| 항목 | 크기 |
|---|---|
| `xception_best.pth` | ~80 MB |
| `f3net_best.pth` | ~85 MB |
| `spsl_best.pth` | ~80 MB |
| `/workspace/venv/` | ~1.5 GB |
| `/workspace/deepshield/` | ~1 MB |
| **총 Volume 사용량** | ~1.7 GB / 50 GB |
