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
│   ├── sbi_best.pth                 (~135MB)
│   ├── fatformer_best.pth           (~1.9GB)
│   └── c2pclip_best.pth             (~1.2GB)
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
python3 -m venv /workspace/venv --system-site-packages && source /workspace/venv/bin/activate && pip install --upgrade pip && pip install -r /workspace/deepshield/services/runpod-inference/requirements.txt && pip install mtcnn tensorflow-cpu open_clip_torch pytorch_wavelets transformers accelerate
```

> `--system-site-packages` 옵션으로 base image의 torch/torchvision/cuda를 그대로 사용합니다.

---

### Step 3 — 가중치 파일 업로드

로컬에 다운로드된 파일을 `/workspace/weights/` 에 업로드합니다.

| 로컬 파일명 | RunPod 저장 경로 | 크기 |
|---|---|---|
| `FFraw.tar` | `/workspace/weights/sbi_best.pth` | ~135MB |
| `fatformer_4class_ckpt.pth` | `/workspace/weights/fatformer_best.pth` | ~1.9GB |
| `C2P_CLIP-GenImage_release_20250224.pth` | `/workspace/weights/c2pclip_best.pth` | ~1.2GB |

> `FFraw.tar`은 확장자가 .tar이지만 실제로는 PyTorch zip 포맷입니다. 내용 변경 없이 파일명만 `sbi_best.pth`로 바꿔서 업로드하면 됩니다.

#### 방법 A — RunPod 웹터미널 파일 브라우저 (권장, 소용량)

1. Pod → **Connect** → **Start Web Terminal** 클릭
2. 터미널 왼쪽 파일 브라우저에서 `/workspace/weights/` 폴더 생성
3. 폴더 클릭 → **Upload** 버튼으로 파일 3개를 위 이름으로 업로드

#### 방법 B — `runpodctl` CLI (대용량, 빠름)

**1. runpodctl 설치 (최초 1회)**

PowerShell에서:

```powershell
Invoke-WebRequest -Uri "https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-windows-amd64.exe" -OutFile "$env:USERPROFILE\runpodctl.exe"
```

> 다운로드 후 `$env:USERPROFILE\runpodctl.exe` 경로로 설치됩니다. 이후 명령에서 `runpodctl` 대신 `~\runpodctl.exe`로 실행합니다.

**2. 파일 전송**

파일 3개를 각각 전송합니다. `send` 실행 시 출력되는 `[CODE]`를 RunPod 웹터미널의 `receive` 명령에 붙여넣습니다.

로컬 PowerShell (파일마다 순서대로 실행):

```powershell
~\runpodctl.exe send "C:\Users\user\Downloads\sbi_best.pth"
```
```powershell
~\runpodctl.exe send "C:\Users\user\Downloads\fatformer_4class_ckpt.pth"
```
```powershell
~\runpodctl.exe send "C:\Users\user\Downloads\C2P_CLIP-GenImage_release_20250224.pth"
```

RunPod 웹터미널에서 수신 (각 `send`마다 코드 확인 후 실행):

```
mkdir -p /workspace/weights && runpodctl receive [CODE] && mv FFraw.tar /workspace/weights/sbi_best.pth
```
```
runpodctl receive [CODE] && mv fatformer_4class_ckpt.pth /workspace/weights/fatformer_best.pth
```
```
runpodctl receive [CODE] && mv C2P_CLIP-GenImage_release_20250224.pth /workspace/weights/c2pclip_best.pth
```

#### 업로드 확인

```
ls -lh /workspace/weights/
```

예상 출력:
```
-rw-r--r-- 1 root root 135M sbi_best.pth
-rw-r--r-- 1 root root 1.9G fatformer_best.pth
-rw-r--r-- 1 root root 1.2G c2pclip_best.pth
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
INFO:inference:SBI: missing=0 unexpected=0
INFO:inference:SBI: loaded ok
WARNING:inference:FatFormer: full inference not yet implemented — placeholder active
INFO:inference:C2P-CLIP: missing=N unexpected=2   (N=text_model 관련, 정상)
INFO:inference:C2P-CLIP: loaded ok
INFO:inference:FaceDetector: using MTCNN
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

## Pod 재시작 후 복구

> **주의**: RunPod의 **Stop → Start**는 Volume(`/workspace/`)은 보존하지만
> 컨테이너 자체는 재생성되어 **Public IP/URL이 바뀌고**, `/root/`, `/tmp/`
> 등 Volume 외부는 초기화됩니다. Pod를 완전히 **Delete**하고 새로 만드는
> 경우에도 같은 Volume을 재연결하면 아래 한 줄 복구가 동작합니다.

`/workspace/`는 Volume이라 코드(deepshield), 가중치(weights), venv 모두 보존됩니다.

**복구 한 줄:**

```
cd /workspace/deepshield && git pull && source /workspace/venv/bin/activate && pip install --quiet transformers accelerate open_clip_torch pytorch_wavelets mtcnn tensorflow-cpu && mkdir -p /tmp/deepshield/uploads && cd services/runpod-inference && DEVICE=cuda UPLOAD_DIR=/tmp/deepshield/uploads python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

> `pip install --quiet`는 이미 설치된 패키지면 즉시 통과하므로 안전망으로
> 포함했습니다. venv가 깨졌다면 이 명령이 재설치까지 수행합니다.

**재시작 후 반드시 할 것:**

1. **새 Public URL 확인**: Pod → **Connect** → **HTTP Service [8000]**
2. **로컬 `.env` 업데이트**: 새 URL로 `RUNPOD_INFERENCE_URL` 교체
3. **로컬 서비스 재시작**: `.\start.ps1`

```
RUNPOD_INFERENCE_URL=https://[NEW_POD_ID]-8000.proxy.runpod.net
```

---

## 모델 구성

| 슬롯 | 모델 | 아키텍처 | 입력 | 앙상블 가중치 |
|---|---|---|---|---|
| `effort` | SBI (CVPR 2022) | EfficientNet-B4 | 380×380 ImageNet 정규화 | 0.50 |
| `xray` | FatFormer (CVPR 2024) | CLIP ViT-L/14 + adapter | 224×224 CLIP 정규화 | **0.00 (placeholder)** |
| `spsl` | C2P-CLIP (AAAI 2025) | HF CLIPModel ViT-L/14 | 224×224 CLIP 정규화 | 0.50 |

> FatFormer는 language-guided alignment + text-guided interactor 추론 경로가
> 가중치만으로 재구현이 어려워 현재 중립값(0.5) placeholder로 동작합니다.
> 가중치 0으로 앙상블 기여는 없습니다. 추론 경로 이식은 후속 작업.

판정 기준: `final = (0.50×SBI + 0.00×FatFormer + 0.50×C2P-CLIP)`
- `< 0.30` → **safe**
- `0.30 ~ 0.70` → **caution**
- `≥ 0.70` → **risk**

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `FatFormer: ... placeholder active` | **정상** — FatFormer 추론 경로 미구현 | 무시 (weight_xray=0으로 앙상블 기여 없음) |
| `SBI: ... using placeholder` | 가중치 키 매핑 실패 | `ls /workspace/weights/sbi_best.pth` 확인, probe 재실행 |
| `C2P-CLIP: ... using placeholder` | 체크포인트 손상 / 누락 | 가중치 파일 크기 1.2GB 확인, 재업로드 |
| `No module named 'transformers'` | venv 패키지 누락 | 위 복구 한 줄에 이미 `pip install transformers`가 포함되어 있음 |
| `No module named 'open_clip'` / `'pytorch_wavelets'` / `'mtcnn'` | 미설치 | 복구 한 줄 실행 또는 개별 `pip install` |
| `422 Unprocessable Entity` | MTCNN 얼굴 미감지 | 얼굴이 포함된 이미지 사용 |
| `CUDA out of memory` | GPU 메모리 부족 | `nvidia-smi`로 좀비 프로세스 확인 후 `kill -9 PID` |
| 로컬 앱에서 502/503 | Pod 재시작 후 URL 변경 → `.env` 미업데이트 | RunPod에서 새 URL 확인 → `.env` 교체 → `.\start.ps1` |
| 서버는 기동했는데 `Detectors ready: []` | 세 모델 모두 로드 실패 | 가중치 파일 3개 존재 확인, 로그에서 각 모델 missing/unexpected 수 확인 |
