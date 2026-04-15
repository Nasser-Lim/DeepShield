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
/volume/                          ← RunPod Volume (영구 보존)
└── weights/
    ├── effort/
    │   ├── effort.pth            ← Effort 모델 가중치
    │   └── repo/                 ← git clone 소스
    ├── xray/
    │   ├── xray.pth              ← Face X-ray 가중치
    │   └── repo/
    └── spsl/
        ├── spsl.pth              ← SPSL 가중치
        └── repo/

/workspace/                       ← Container Disk (재시작 시 초기화)
└── deepshield/
    └── services/
        └── runpod-inference/
            ├── server.py
            ├── models/
            │   ├── effort.py
            │   ├── face_xray.py
            │   └── spsl.py
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
mkdir -p /volume/weights/effort/repo /volume/weights/xray/repo /volume/weights/spsl/repo && ls -la /volume/weights/
```

---

## Step 3 — 프로젝트 코드 배포

```bash
mkdir -p /workspace/deepshield
cd /workspace/deepshield

# Git으로 클론
git clone https://github.com/YOUR_REPO/DeepShield.git .

# 또는 로컬에서 scp로 업로드 (로컬 터미널에서 실행)
# scp -r ./services/runpod-inference root@[POD_IP]:/workspace/deepshield/services/

cd /workspace/deepshield/services/runpod-inference
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
CUDA available : True
Device         : NVIDIA RTX A5000
VRAM           : 24.0 GB
```

---

## Step 6 — 모델 가중치 다운로드 (Volume에 저장)

### Effort

```bash
cd /volume/weights/effort/repo && git clone https://github.com/HighwayWu/EFFORT.git .

pip install huggingface_hub && python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='HighwayWu/EFFORT', filename='effort.pth', local_dir='/volume/weights/effort/')"

ls -lh /volume/weights/effort/
```

### Face X-ray

```bash
cd /volume/weights/xray/repo && git clone https://github.com/neverUseThisName/Face-X-Ray.git .

wget -O /volume/weights/xray/xray.pth "https://저자_배포_링크/xray.pth"

ls -lh /volume/weights/xray/
```

### SPSL

```bash
cd /volume/weights/spsl/repo && git clone https://github.com/SCLBD/DeepfakeBench.git .

wget -O /volume/weights/spsl/spsl.pth "https://저자_배포_링크/spsl.pth"

ls -lh /volume/weights/spsl/
```

### 전체 가중치 확인

```bash
find /volume/weights -name "*.pth" -o -name "*.pt" -o -name "*.ckpt" | sort
du -sh /volume/weights/
```

---

## Step 7 — 모델 코드 교체

각 모델 파일의 `load` / `predict`를 실제 가중치 로드 코드로 교체합니다.

### effort.py

```bash
nano /workspace/deepshield/services/runpod-inference/models/effort.py
```

```python
import sys
import torch
import numpy as np
from .base import DetectorBase, DetectorOutput

class EffortDetector(DetectorBase):
    name = "effort"

    def load(self, device: str) -> None:
        sys.path.insert(0, "/volume/weights/effort/repo")
        from model import EffortNet          # 저자 실제 클래스명으로 교체
        self.model = EffortNet()
        ckpt = torch.load("/volume/weights/effort/effort.pth", map_location=device)
        self.model.load_state_dict(ckpt)
        self.model.to(device).eval()
        self.device = device

    def predict(self, face_bgr: np.ndarray) -> DetectorOutput:
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        x = tf(face_bgr[:, :, ::-1].copy()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = float(torch.sigmoid(self.model(x)).item())
        return DetectorOutput(score=score, heatmap=None)
```

### face_xray.py

```bash
nano /workspace/deepshield/services/runpod-inference/models/face_xray.py
```

```python
import sys
import torch
import numpy as np
from .base import DetectorBase, DetectorOutput

class FaceXrayDetector(DetectorBase):
    name = "xray"

    def load(self, device: str) -> None:
        sys.path.insert(0, "/volume/weights/xray/repo")
        from model import XrayNet            # 저자 실제 클래스명으로 교체
        self.model = XrayNet()
        ckpt = torch.load("/volume/weights/xray/xray.pth", map_location=device)
        self.model.load_state_dict(ckpt)
        self.model.to(device).eval()
        self.device = device

    def predict(self, face_bgr: np.ndarray) -> DetectorOutput:
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        x = tf(face_bgr[:, :, ::-1].copy()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
            score = float(torch.sigmoid(out).item())
        return DetectorOutput(score=score, heatmap=None)
```

### spsl.py

```bash
nano /workspace/deepshield/services/runpod-inference/models/spsl.py
```

```python
import sys
import torch
import numpy as np
from .base import DetectorBase, DetectorOutput

class SPSLDetector(DetectorBase):
    name = "spsl"

    def load(self, device: str) -> None:
        sys.path.insert(0, "/volume/weights/spsl/repo")
        from training.detectors.spsl_detector import SPSLDetector as Net  # 실제 경로 확인
        self.model = Net()
        ckpt = torch.load("/volume/weights/spsl/spsl.pth", map_location=device)
        self.model.load_state_dict(ckpt)
        self.model.to(device).eval()
        self.device = device

    def predict(self, face_bgr: np.ndarray) -> DetectorOutput:
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        x = tf(face_bgr[:, :, ::-1].copy()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
            score = float(torch.sigmoid(out).item())
        return DetectorOutput(score=score, heatmap=None)
```

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
Volume(`/volume/weights`)은 보존되므로 아래만 재실행합니다.

```bash
cd /workspace && git clone https://github.com/YOUR_REPO/DeepShield.git deepshield

cd /workspace/deepshield/services/runpod-inference && pip install -r requirements.txt && pip install torchvision==0.19.1

export DEVICE=cuda UPLOAD_DIR=/tmp/deepshield/uploads && mkdir -p $UPLOAD_DIR && python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

> 재시작 자동화가 필요하면 Pod 생성 시 **Startup Command**에 위 명령어를 등록하세요.
