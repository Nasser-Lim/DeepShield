# RunPod A5000 셋업 가이드

> **설계 원칙**
> - SBI / FatFormer / C2P-CLIP 순수 PyTorch 구현, 가중치(.pth)만 로드
> - `/workspace/`가 Volume 마운트 경로 — 코드·가중치·venv 모두 저장되어 Pod 재시작 시 pip install 불필요
> - 가중치는 로컬 PC에서 수동 다운로드 후 RunPod 웹터미널 파일 업로드로 배치

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

---

## 최종 디렉토리 구조

```
/workspace/
├── deepshield/                      (DeepShield 저장소)
│   └── services/runpod-inference/
│       ├── server.py
│       ├── models/
│       │   ├── effort.py            (SBI — EfficientNet-B4)
│       │   ├── face_xray.py         (FatFormer — CLIP ViT-L/14)
│       │   └── spsl.py              (C2P-CLIP — CLIP ViT-L/14)
│       └── requirements.txt
├── weights/
│   ├── sbi_best.pth                 (~70MB)
│   ├── fatformer_best.pth           (~1.2GB)
│   └── c2pclip_best.pth             (~1.1GB)
└── venv/                            (Python 가상환경 — 재시작 후에도 보존)
```

---

## 가중치 파일 획득 (로컬 PC에서 먼저)

| 모델 | 저장소 | 파일명 | 로컬 저장명 |
|---|---|---|---|
| SBI | https://github.com/mapooon/SelfBlendedImages | `FFraw.pth` (README → Pretrained Models → raw) | `sbi_best.pth` |
| FatFormer | https://github.com/Michel-liu/FatFormer | `fatformer_4class_ckpt.pth` (README → Model Zoo) | `fatformer_best.pth` |
| C2P-CLIP | https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection | `C2P_CLIP-GenImage_release_20250224.pth` (Google Drive) | `c2pclip_best.pth` |

---

## 최초 셋업 절차 (Pod 처음 생성 시)

### Step 1 — 시스템 패키지 + 코드 클론

```
apt-get update && apt-get install -y git wget libgl1 libglib2.0-0 && cd /workspace && git clone https://github.com/Nasser-Lim/DeepShield.git deepshield
```

---

### Step 2 — Python venv 생성 및 패키지 설치

```
python3 -m venv /workspace/venv --system-site-packages && source /workspace/venv/bin/activate && pip install --upgrade pip && pip install -r /workspace/deepshield/services/runpod-inference/requirements.txt && pip install mtcnn tensorflow-cpu open_clip_torch pytorch_wavelets
```

> `--system-site-packages` 옵션으로 base image의 torch/torchvision/cuda를 그대로 사용합니다.

---

### Step 3 — 가중치 파일 업로드

RunPod 웹터미널 좌측 파일 브라우저에서 `/workspace/weights/` 폴더를 만들고 로컬 PC의 가중치 3개를 업로드합니다.

업로드 확인:

```
ls -lh /workspace/weights/
```

예상 출력:
```
-rw-r--r-- 1 root root  70M sbi_best.pth
-rw-r--r-- 1 root root 1.2G fatformer_best.pth
-rw-r--r-- 1 root root 1.1G c2pclip_best.pth
```

---

### Step 4 — 가중치 키 구조 확인 (Probe)

```
source /workspace/venv/bin/activate && python3 -c "
import torch
for name, path in [('SBI','/workspace/weights/sbi_best.pth'),('FatFormer','/workspace/weights/fatformer_best.pth'),('C2P-CLIP','/workspace/weights/c2pclip_best.pth')]:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) else ckpt
    keys = list(sd.keys())
    print(f'=== {name} ({len(keys)} keys) first8={keys[:8]}')
"
```

결과를 개발자에게 공유하여 키 매핑을 확정합니다.

---

### Step 5 — 서버 기동

```
source /workspace/venv/bin/activate && mkdir -p /tmp/deepshield/uploads && cd /workspace/deepshield/services/runpod-inference && DEVICE=cuda UPLOAD_DIR=/tmp/deepshield/uploads python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

정상 기동 시 출력:
```
INFO:inference:FaceDetector: using MTCNN
INFO:inference:SBI: loaded ok
INFO:inference:FatFormer: loaded ok
INFO:inference:C2P-CLIP: loaded ok
INFO:inference:Detectors ready: ['effort', 'xray', 'spsl']
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

### Step 6 — 로컬 .env 업데이트

RunPod 대시보드 → Pod → **Connect** → **HTTP Service [8000]** 에서 공개 URL 확인 후:

```
RUNPOD_INFERENCE_URL=https://[POD_ID]-8000.proxy.runpod.net
```

로컬에서 재시작:
```powershell
.\start.ps1
```

---

## Pod 재시작 후 복구 (한 줄)

`/workspace/`는 Volume이라 코드·가중치·venv 모두 보존됩니다.

```
cd /workspace/deepshield && git pull && source /workspace/venv/bin/activate && mkdir -p /tmp/deepshield/uploads && cd services/runpod-inference && DEVICE=cuda UPLOAD_DIR=/tmp/deepshield/uploads python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## 모델 구성

| 슬롯 | 모델 | 아키텍처 | 입력 | 앙상블 가중치 |
|---|---|---|---|---|
| `effort` | SBI (CVPR 2022) | EfficientNet-B4 | 380×380 ImageNet 정규화 | 0.40 |
| `xray` | FatFormer (CVPR 2024) | CLIP ViT-L/14 + adapter | 224×224 CLIP 정규화 | 0.40 |
| `spsl` | C2P-CLIP (AAAI 2025) | CLIP ViT-L/14 + C2P | 224×224 CLIP 정규화 | 0.20 |

판정 기준: `final = (0.40×SBI + 0.40×FatFormer + 0.20×C2P-CLIP)`
- `< 0.30` → **safe**
- `0.30 ~ 0.70` → **caution**
- `≥ 0.70` → **risk**

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `WARNING: ... — using placeholder` | 가중치 로드 실패 | `/workspace/weights/` 파일 확인, probe 결과로 키 매핑 재확인 |
| `No module named 'open_clip'` | venv 활성화 안 됨 또는 미설치 | `source /workspace/venv/bin/activate && pip install open_clip_torch` |
| `No module named 'pytorch_wavelets'` | 미설치 | `pip install pytorch_wavelets` |
| `422 Unprocessable Entity` | MTCNN 얼굴 미감지 | 얼굴이 포함된 이미지 사용 |
| `CUDA out of memory` | GPU 메모리 부족 (FatFormer+C2P-CLIP 동시 로드 ~4GB) | `nvidia-smi`로 좀비 프로세스 확인 후 kill |
| Pod 공개 URL 502/503 | 서버 미기동 | 위 복구 절차 실행 |
