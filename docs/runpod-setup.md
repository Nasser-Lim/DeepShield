# RunPod A5000 셋업 가이드

## Pod 생성 설정

| 항목 | 값 |
|---|---|
| GPU | RTX A5000 (24GB VRAM) |
| Template | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Container Disk | 30GB |
| Volume Disk | 50GB |
| Volume Mount Path | `/volume` |
| Expose HTTP Port | `8000` |

> Volume(`/volume`)은 Pod을 Stop/재시작해도 유지됩니다.
> 모델 가중치는 모두 `/volume/weights/`에 저장합니다.

---

## 디렉토리 구조

```
/volume/                          (RunPod Volume - 영구 보존)
└── weights/
    └── deepfakebench/            (SCLBD/DeepfakeBench - Xception + F3Net + SPSL 포함)
        ├── training/
        │   ├── detectors/
        │   │   ├── xception_detector.py
        │   │   ├── f3net_detector.py
        │   │   └── spsl_detector.py
        │   └── weights/          (가중치 .pth 파일)
        │       ├── xception_best.pth
        │       ├── f3net_best.pth
        │       └── spsl_best.pth
        └── pretrained/           (backbone 사전학습 가중치)

/workspace/                       (Container Disk - 재시작 시 초기화)
└── deepshield/
    └── services/
        └── runpod-inference/
            ├── server.py
            ├── models/
            │   ├── effort.py     (Xception wrapper)
            │   ├── face_xray.py  (F3Net wrapper)
            │   └── spsl.py       (SPSL wrapper)
            ├── utils/
            └── requirements.txt
```

---

## Step 1 — 시스템 패키지 설치

```bash
apt-get update && apt-get install -y git wget curl unzip libgl1 libglib2.0-0 nano
```

---

## Step 2 — Volume 디렉토리 초기화

> Volume이 처음 마운트된 경우 한 번만 실행합니다.

```bash
mkdir -p /volume/weights/deepfakebench/training/weights /volume/weights/deepfakebench/training/pretrained && ls -la /volume/weights/
```

---

## Step 3 — 프로젝트 코드 배포

```bash
mkdir -p /workspace/deepshield && cd /workspace/deepshield && git clone https://github.com/Nasser-Lim/DeepShield.git . && cd /workspace/deepshield/services/runpod-inference
```

---

## Step 4 — Python 패키지 설치

```bash
cd /workspace/deepshield/services/runpod-inference && pip install -r requirements.txt && pip install torchvision==0.19.1 && python3 -c "import fastapi, uvicorn, cv2, numpy; print('OK')"
```

---

## Step 5 — GPU 확인

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0), '|', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"
```

예상 출력:
```
CUDA: True | NVIDIA RTX A5000 | 24.0 GB
```

---

## Step 6 — 모델 저장소 클론 및 가중치 다운로드 (Volume에 저장)

> 세 모델(Xception, F3Net, SPSL)은 모두 [SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)에 통합되어 있습니다.

### DeepfakeBench 저장소 클론

```bash
cd /volume/weights && git clone https://github.com/SCLBD/DeepfakeBench.git deepfakebench && ls /volume/weights/deepfakebench/training/detectors/ | grep -E "xception|f3net|spsl"
```

### 모델 가중치 다운로드 (v1.0.1)

```bash
cd /volume/weights/deepfakebench && wget -O training/weights/xception_best.pth "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth" && wget -O training/weights/f3net_best.pth "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/f3net_best.pth" && wget -O training/weights/spsl_best.pth "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/spsl_best.pth"
```

### Backbone 사전학습 가중치 다운로드 (v1.0.0)

```bash
cd /volume/weights/deepfakebench && wget -O pretrained.zip "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip" && unzip pretrained.zip -d training/pretrained/ && rm pretrained.zip
```

### 전체 확인

```bash
find /volume/weights -name "*.pth" -o -name "*.pt" -o -name "*.ckpt" | sort && du -sh /volume/weights/
```

---

## Step 7 — 모델 코드 확인

`services/runpod-inference/models/` 파일들은 이미 실제 가중치 로드 코드를 포함하고 있습니다.  
가중치 파일이 없으면 자동으로 플레이스홀더(SHA-1 해시 기반 더미 점수)로 폴백합니다.

| 파일 | 모델 | 가중치 경로 |
|---|---|---|
| `effort.py` | Xception | `/volume/weights/deepfakebench/training/weights/xception_best.pth` |
| `face_xray.py` | F3Net | `/volume/weights/deepfakebench/training/weights/f3net_best.pth` |
| `spsl.py` | SPSL | `/volume/weights/deepfakebench/training/weights/spsl_best.pth` |

---

## Step 8 — 서버 기동

```bash
cd /workspace/deepshield/services/runpod-inference && export DEVICE=cuda && export UPLOAD_DIR=/tmp/deepshield/uploads && mkdir -p $UPLOAD_DIR && python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

정상 기동 시 출력:
```
INFO:inference:Loading detectors on cuda
INFO:inference:Detectors ready: ['effort', 'xray', 'spsl']
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Step 9 — 동작 확인

```bash
curl http://localhost:8000/healthz | python3 -m json.tool

FILE_ID=$(curl -s -X POST http://localhost:8000/upload -F "image=@/path/to/test.jpg" | python3 -c "import sys,json; print(json.load(sys.stdin)['file_id'])") && echo "file_id: $FILE_ID"

curl -s -X POST http://localhost:8000/infer -H "Content-Type: application/json" -d "{\"file_id\": \"$FILE_ID\"}" | python3 -m json.tool
```

예상 출력:
```json
{
  "effort": { "score": 0.83, "heatmap_b64": "..." },
  "xray":   { "score": 0.91, "heatmap_b64": "..." },
  "spsl":   { "score": 0.76, "heatmap_b64": "..." },
  "face_bbox": [40, 20, 180, 200],
  "overlay_b64": "..."
}
```

---

## Step 10 — 로컬 .env 업데이트

RunPod 대시보드 → Pod → **Connect** → **HTTP Service [8000]** 에서 공개 URL 확인 후:

```
# .env
RUNPOD_INFERENCE_URL=https://[POD_ID]-8000.proxy.runpod.net
```

로컬에서 재시작:
```powershell
# start.ps1 Ctrl+C 후
.\start.ps1
```

---

## Pod 재시작 시 복구 절차

Container Disk(`/workspace`)는 재시작 시 초기화됩니다.
Volume(`/volume/weights`)은 보존되므로 저장소 클론·가중치 다운로드는 생략하고 아래만 재실행합니다.

```bash
cd /workspace && git clone https://github.com/Nasser-Lim/DeepShield.git deepshield && cd /workspace/deepshield/services/runpod-inference && pip install -r requirements.txt && pip install torchvision==0.19.1 && export DEVICE=cuda UPLOAD_DIR=/tmp/deepshield/uploads && mkdir -p $UPLOAD_DIR && python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

> 재시작 자동화가 필요하면 Pod 생성 시 **Startup Command**에 위 명령어를 등록하세요.
