# RunPod A5000 셋업 가이드

> **설계 원칙**
> - SBI / UnivFD / C2P-CLIP 순수 PyTorch 구현, 가중치(.pth)만 로드
> - `/workspace/` 하위 경로명을 **고유 이름**으로 지정해 Pod 재시작 시 RunPod이 초기화하지 않도록 방지
> - 가중치·venv 모두 Volume에 보존 → Pod 재시작 후 서버 기동만 하면 됨
> - 추론 파이프라인에 **JPEG TTA**(Q=75 재압축) 적용 → 언론사 고압축 사진에서 압축 아티팩트로 인한 false positive 완화

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

## 디렉토리 구조

> 폴더명에 `ds_` 접두사를 사용해 RunPod 기본 폴더와 충돌을 방지합니다.

```
/workspace/
├── ds_repo/                         (DeepShield 저장소)
│   └── services/runpod-inference/
│       ├── server.py
│       ├── models/
│       │   ├── effort.py            (SBI    — EfficientNet-B4)
│       │   ├── face_xray.py         (UnivFD — HF CLIPModel ViT-L/14 + linear probe)
│       │   └── spsl.py              (C2P-CLIP — HF CLIPModel ViT-L/14)
│       └── requirements.txt
├── ds_weights/
│   ├── sbi_best.pth                 (~135MB)
│   ├── univfd_fc_weights.pth        (~3KB  — head-only linear probe)
│   └── c2pclip_best.pth             (~1.2GB)
└── ds_venv/                         (Python 가상환경 — 재시작 후에도 보존)
```

---

## 가중치 파일 (로컬 PC에서 먼저 확보)

| 모델 | 로컬 파일명 | RunPod 저장 경로 | 크기 |
|---|---|---|---|
| SBI | `sbi_best.pth` (FFraw.tar를 이름 변경) | `/workspace/ds_weights/sbi_best.pth` | ~135MB |
| UnivFD | `fc_weights.pth` ([공식 저장소 pretrained_weights/](https://github.com/WisconsinAIVision/UniversalFakeDetect)) | `/workspace/ds_weights/univfd_fc_weights.pth` | ~3KB |
| C2P-CLIP | `C2P_CLIP-GenImage_release_20250224.pth` | `/workspace/ds_weights/c2pclip_best.pth` | ~1.2GB |

> **UnivFD는 head-only 가중치(~3KB)**입니다. CLIP ViT-L/14 백본은
> HuggingFace에서 자동 다운로드됩니다 (C2P-CLIP이 이미 다운받아놓았으므로
> 캐시 공유). 최초 기동 시 `openai/clip-vit-large-patch14` 다운로드에
> 수 분이 걸릴 수 있습니다 (~1.7GB).

---

## 최초 셋업 절차 (Pod 처음 생성 시)

### Step 1 — 시스템 패키지 + 코드 클론

```
apt-get update && apt-get install -y git wget libgl1 libglib2.0-0 && git clone https://github.com/Nasser-Lim/DeepShield.git /workspace/ds_repo
```

---

### Step 2 — Python venv 생성 및 패키지 설치

```
/usr/bin/python3 -m venv /workspace/ds_venv --system-site-packages && source /workspace/ds_venv/bin/activate && pip install --upgrade pip && pip install -r /workspace/ds_repo/services/runpod-inference/requirements.txt && pip install open_clip_torch pytorch_wavelets transformers accelerate mtcnn tensorflow-cpu
```

> - `/usr/bin/python3`을 명시해 shell PATH 오염 방지
> - `--system-site-packages`로 base image의 torch/torchvision/cuda 재사용
> - `mtcnn tensorflow-cpu`: MTCNN 얼굴 감지기 의존성. venv 생성과 함께 설치해야 안전
> - tensorflow-cpu를 **단독으로** 나중에 설치하면 venv가 손상될 수 있으므로 반드시 이 명령 전체를 한 번에 실행

---

### Step 3 — 가중치 파일 업로드

로컬 PowerShell에서 `runpodctl`로 전송합니다.

**runpodctl 설치 (최초 1회, 로컬 PowerShell):**

```powershell
Invoke-WebRequest -Uri "https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-windows-amd64.exe" -OutFile "$env:USERPROFILE\runpodctl.exe"
```

**파일 전송 (로컬 PowerShell → RunPod 순서로 1개씩):**

로컬:
```powershell
~\runpodctl.exe send "C:\Users\user\Downloads\sbi_best.pth"
```
RunPod:
```
mkdir -p /workspace/ds_weights && cd /workspace/ds_weights && runpodctl receive [CODE]
```

로컬:
```powershell
~\runpodctl.exe send "C:\Users\user\Downloads\fc_weights.pth"
```
RunPod:
```
cd /workspace/ds_weights && runpodctl receive [CODE] && mv fc_weights.pth univfd_fc_weights.pth
```

로컬:
```powershell
~\runpodctl.exe send "C:\Users\user\Downloads\C2P_CLIP-GenImage_release_20250224.pth"
```
RunPod:
```
cd /workspace/ds_weights && runpodctl receive [CODE] && mv C2P_CLIP-GenImage_release_20250224.pth c2pclip_best.pth
```

**업로드 확인:**

```
ls -lh /workspace/ds_weights/
```

예상 출력:
```
-rw-r--r-- 1 root root 135M sbi_best.pth
-rw-r--r-- 1 root root 3.0K univfd_fc_weights.pth
-rw-r--r-- 1 root root 1.2G c2pclip_best.pth
```

---

### Step 4 — 서버 기동

```
source /workspace/ds_venv/bin/activate && mkdir -p /tmp/ds_uploads && cd /workspace/ds_repo/services/runpod-inference && DEVICE=cuda UPLOAD_DIR=/tmp/ds_uploads python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

정상 기동 시 출력:
```
INFO:inference:SBI: missing=0 unexpected=0
INFO:inference:SBI: loaded ok
INFO:inference:UnivFD: head loaded, backbone missing=N unexpected=0   (N=CLIP 백본 관련, 정상)
INFO:inference:UnivFD: loaded ok
INFO:inference:C2P-CLIP: missing=N unexpected=2   (N=text_model 관련, 정상)
INFO:inference:C2P-CLIP: loaded ok
INFO:inference:FaceDetector: using MTCNN
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

### Step 5 — 로컬 .env 업데이트

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

> **Stop → Start** 후 컨테이너가 재생성되어 **Public URL이 바뀌고** `/tmp/` 등
> Volume 외부는 초기화됩니다. `/workspace/ds_venv/`, `/workspace/ds_weights/`,
> `/workspace/ds_repo/`는 Volume에 보존됩니다.

**복구 한 줄:**

```
source /workspace/ds_venv/bin/activate && mkdir -p /tmp/ds_uploads && cd /workspace/ds_repo/services/runpod-inference && DEVICE=cuda UPLOAD_DIR=/tmp/ds_uploads python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

> venv가 손상된 경우 아래 **venv 재생성** 명령을 먼저 실행합니다.

**venv 손상 시 재생성:**

```
/usr/bin/python3 -m venv /workspace/ds_venv --system-site-packages && source /workspace/ds_venv/bin/activate && pip install --upgrade pip && pip install -r /workspace/ds_repo/services/runpod-inference/requirements.txt && pip install open_clip_torch pytorch_wavelets transformers accelerate mtcnn tensorflow-cpu
```

**재시작 후 반드시 할 것:**

1. **새 Public URL 확인**: Pod → **Connect** → **HTTP Service [8000]**
2. **로컬 `.env` 업데이트**: 새 URL로 `RUNPOD_INFERENCE_URL` 교체
3. **로컬 서비스 재시작**: `.\start.ps1`

---

## 모델 구성

| 슬롯 | 모델 | 아키텍처 | 입력 | 앙상블 가중치 |
|---|---|---|---|---|
| `effort` | SBI (CVPR 2022) | EfficientNet-B4 | 380×380 ImageNet 정규화 | 0.35 |
| `xray` | UnivFD (CVPR 2023) | CLIP ViT-L/14 + linear probe | 224×224 CLIP 정규화 | 0.35 |
| `spsl` | C2P-CLIP (AAAI 2025) | HF CLIPModel ViT-L/14 | 224×224 CLIP 정규화 | 0.30 |

> **JPEG TTA 적용**: 각 추론 시 원본 얼굴 크롭과 Q=75 재압축 크롭을 모두
> 모델에 입력하고 평균 점수를 최종 slot score로 사용합니다. 응답에
> `score_raw`, `score_tta`, `jpeg_tta_delta` 필드가 포함됩니다. delta가 0.3
> 이상이면 판정 신뢰도가 낮다는 신호입니다.

판정 기준: `final = (0.35×SBI + 0.35×UnivFD + 0.30×C2P-CLIP)`
- `< 0.30` → **safe**
- `0.30 ~ 0.70` → **caution**
- `≥ 0.70` → **risk**

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `UnivFD load failed ... fc head not found` | 가중치 파일이 head-only 형식 아님 | 공식 `fc_weights.pth` 재다운로드, `/workspace/ds_weights/univfd_fc_weights.pth`로 저장 |
| `SBI: ... using placeholder` | 가중치 파일 없음 | `ls /workspace/ds_weights/sbi_best.pth` 확인, 재업로드 |
| `UnivFD: ... using placeholder` | 가중치 파일 없음 | `ls /workspace/ds_weights/univfd_fc_weights.pth` 확인, 재업로드 |
| `C2P-CLIP: ... using placeholder` | 가중치 파일 없음 | `ls /workspace/ds_weights/c2pclip_best.pth` 확인, 재업로드 |
| `MTCNN unavailable ... falling back to Haar` | tensorflow 미설치 | venv 재생성 명령 실행 (mtcnn tensorflow-cpu 포함) |
| `No module named 'X'` | venv 손상 | 위 **venv 재생성** 명령 실행 |
| `No such file or directory: '/workspace/ds_venv/bin/python3'` | shell PATH 오염 (이전 venv 잔재) | `exec bash` 후 복구 명령 재실행 |
| `422 Unprocessable Entity` | MTCNN 얼굴 미감지 | 얼굴이 포함된 이미지 사용 |
| `CUDA out of memory` | GPU 메모리 부족 | `nvidia-smi`로 좀비 프로세스 확인 후 `kill -9 PID` |
| 로컬 앱에서 502/503 | Pod 재시작 후 URL 변경 | RunPod에서 새 URL 확인 → `.env` 교체 → `.\start.ps1` |
