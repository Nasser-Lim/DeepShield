# RunPod A5000 셋업 가이드 (DIRE 단일 모델)

> **설계 원칙**
> - DIRE (Diffusion Reconstruction Error, ICCV 2023) 단일 모델로 전체 이미지 디퓨전 탐지
> - `/workspace/dire_v1/` 고유 폴더명을 사용해 Pod 재시작 시 RunPod이 초기화하지 않도록 방지
> - ADM 가중치·분류기·venv·공식 저장소 모두 Volume에 보존 → Pod 재시작 후 서버 기동만 하면 됨
> - 얼굴 탐지 경로 없음 (DIRE는 전체 이미지 기반)

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

> 고유 이름 `dire_v1`을 사용해 버전/모델별로 분리합니다. 추후 다른 버전 비교 시 `dire_v2/`로 병행 배치 가능.

```
/workspace/
├── dire_v1/
│   ├── repo/                                # git clone of ZhendongWang6/DIRE (guided_diffusion 포함)
│   ├── venv/                                # --system-site-packages venv
│   ├── weights/
│   │   ├── 256x256_diffusion_uncond.pt      # OpenAI ADM 범용 (~2.1GB)
│   │   ├── lsun_bedroom.pt                  # LSUN Bedroom ADM (선택)
│   │   └── classifier/
│   │       └── lsun_adm.pth                 # ResNet-50 분류기 (~100MB)
│   └── uploads/                             # /tmp 대체, 영속 저장
└── ds_repo/                                 # DeepShield 저장소 (여전히 사용)
    └── services/runpod-inference/
        ├── server.py
        ├── models/
        │   ├── base.py
        │   └── dire.py
        └── requirements.txt
```

---

## 가중치 파일

| 역할 | 파일명 | RunPod 저장 경로 | 크기 | 취득 방법 |
|---|---|---|---|---|
| ADM 재구성 (범용) | `256x256_diffusion_uncond.pt` | `/workspace/dire_v1/weights/256x256_diffusion_uncond.pt` | ~2.1GB | Pod에서 wget |
| 이진 분류기 | `lsun_adm.pth` | `/workspace/dire_v1/weights/classifier/lsun_adm.pth` | ~270MB | 로컬 → runpodctl |

- ADM: OpenAI 공식 호스팅에서 Pod가 직접 wget (로컬 경유 불필요)
- 분류기: 저자 RecDrive `https://rec.ustc.edu.cn/share/ec980150-4615-11ee-be0a-eb822f25e070` (비밀번호: `dire`) 에서 로컬 다운로드 후 runpodctl 전송
- 나머지 분류기(`lsun_pndm.pth`, `lsun_iddpm.pth` 등)는 추후 비교 실험 시 추가 가능

`DIRE_ADM_WEIGHTS` 환경변수로 ADM 가중치를 런타임에 전환합니다.

---

## 최초 셋업 절차

### Step 1 — 시스템 패키지 + 저장소 클론

> DIRE 저장소(`/workspace/dire_v1/repo`)가 이미 클론되어 있으면 해당 부분은 건너뜁니다.

```
apt-get update && apt-get install -y git wget libgl1 libglib2.0-0 && mkdir -p /workspace/dire_v1/weights/classifier /workspace/dire_v1/uploads && git clone https://github.com/ZhendongWang6/DIRE.git /workspace/dire_v1/repo && git clone https://github.com/Nasser-Lim/DeepShield.git /workspace/ds_repo
```

### Step 2 — venv + 패키지 설치

```
/usr/bin/python3 -m venv /workspace/dire_v1/venv --system-site-packages && source /workspace/dire_v1/venv/bin/activate && pip install --upgrade pip && pip install -r /workspace/ds_repo/services/runpod-inference/requirements.txt && pip install blobfile einops tqdm
```

> `guided_diffusion`은 pip 패키지가 아닙니다. 저자 공식 저장소(`/workspace/dire_v1/repo`)를 `PYTHONPATH`로 주입하여 사용합니다 (아래 Step 4 참고).

### Step 3 — 가중치 확보

**분류기 — 로컬 PowerShell에서 runpodctl 전송:**

```powershell
~\runpodctl.exe send "C:\Users\user\Downloads\lsun_adm.pth"
```

RunPod 웹터미널:
```
cd /workspace/dire_v1/weights/classifier && runpodctl receive [CODE]
```

**ADM 디퓨전 모델 — Pod에서 직접 wget (~2.1GB, 로컬 경유 불필요):**

```
wget -P /workspace/dire_v1/weights/ https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
```

확인:
```
ls -lh /workspace/dire_v1/weights/ /workspace/dire_v1/weights/classifier/
```

### Step 4 — 서버 기동

```
source /workspace/dire_v1/venv/bin/activate && export DIRE_REPO_PATH=/workspace/dire_v1/repo && export DIRE_ADM_WEIGHTS=/workspace/dire_v1/weights/256x256_diffusion_uncond.pt && export DIRE_CLASSIFIER_WEIGHTS=/workspace/dire_v1/weights/classifier/lsun_adm.pth && export DIRE_TIMESTEP_RESPACING=ddim20 && export PYTHONPATH=/workspace/dire_v1/repo:/workspace/dire_v1/repo/guided-diffusion && export UPLOAD_DIR=/workspace/dire_v1/uploads && export DEVICE=cuda && cd /workspace/ds_repo/services/runpod-inference && python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

정상 기동 시 로그:
```
INFO:dire:Loading ADM weights from /workspace/dire_v1/weights/256x256_diffusion_uncond.pt
INFO:dire:Loading classifier weights from /workspace/dire_v1/weights/classifier/lsun_adm.pth
INFO:dire:classifier: missing=0 unexpected=0
INFO:dire:DireDetector loaded (device=cuda, respacing=ddim20)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 5 — 로컬 `.env` 업데이트

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

```
source /workspace/dire_v1/venv/bin/activate && export DIRE_REPO_PATH=/workspace/dire_v1/repo && export DIRE_ADM_WEIGHTS=/workspace/dire_v1/weights/256x256_diffusion_uncond.pt && export DIRE_CLASSIFIER_WEIGHTS=/workspace/dire_v1/weights/classifier/lsun_adm.pth && export DIRE_TIMESTEP_RESPACING=ddim20 && export PYTHONPATH=/workspace/dire_v1/repo:/workspace/dire_v1/repo/guided-diffusion && export UPLOAD_DIR=/workspace/dire_v1/uploads && export DEVICE=cuda && cd /workspace/ds_repo/services/runpod-inference && python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

> **새 Public URL**을 반드시 `.env`의 `RUNPOD_INFERENCE_URL`에 반영하고 `.\start.ps1` 재기동.

---

## 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `DIRE_REPO_PATH` | `/workspace/dire_v1/repo` | `guided_diffusion` 모듈 검색 경로 |
| `DIRE_ADM_WEIGHTS` | `/workspace/dire_v1/weights/256x256_diffusion_uncond.pt` | ADM UNet 가중치 (LSUN으로 스위치 가능) |
| `DIRE_CLASSIFIER_WEIGHTS` | `/workspace/dire_v1/weights/classifier/lsun_adm.pth` | ResNet-50 이진 분류기 |
| `DIRE_TIMESTEP_RESPACING` | `ddim20` | DDIM 재샘플링 스텝 수 (속도/정확도 트레이드오프) |
| `DIRE_IMAGE_SIZE` | `256` | ADM 입력 해상도 (기본값 유지 권장) |
| `UPLOAD_DIR` | `/workspace/dire_v1/uploads` | 업로드 임시 저장소 |
| `DEVICE` | `cpu` | GPU 사용 시 `cuda` |

---

## 판정 기준

- **단일 DIRE synthetic probability**를 그대로 사용 (앙상블 가중합 없음)
- `< 0.30` → **safe** (실제 이미지로 판정)
- `0.30 ~ 0.70` → **caution** (모호)
- `≥ 0.70` → **risk** (디퓨전 생성 의심)

---

## 성능 타겟

- `ddim20` 기준 A5000에서 이미지당 **≤ 3초**
- `ddim10`까지 낮추면 1.5초 내외 (정확도 약간 저하)
- CPU에서는 `ddim5` 정도로 내려도 수십 초 걸릴 수 있음 (스모크 테스트용)

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: guided_diffusion` | `PYTHONPATH` 미설정 | Step 4의 export 구문 재확인, `/workspace/dire_v1/repo` 존재 확인 |
| `FileNotFoundError: .../256x256_diffusion_uncond.pt` | 가중치 미업로드 | `ls -lh /workspace/dire_v1/weights/` 확인 후 재전송 |
| `classifier: missing=XXX unexpected=YYY` (큰 수) | 체크포인트 포맷 불일치 | `lsun_adm.pth`가 `{"model": state_dict}` 또는 `{"state_dict": ...}`로 래핑되어 있으면 자동 언랩하지만, 완전히 다른 아키텍처면 맞는 분류기 체크포인트 재확인 |
| `CUDA out of memory` | VRAM 부족 | `DIRE_TIMESTEP_RESPACING=ddim10`으로 낮추거나 use_fp16(기본 cuda에서 on) 확인 |
| `No module named 'X'` | venv 손상 | Step 2 재실행 |
| 로컬 앱 502/503 | Pod 재시작 후 URL 변경 | `.env`의 `RUNPOD_INFERENCE_URL` 교체 후 `.\start.ps1` |
| 추론 느림 (>10초) | respacing 기본 1000 스텝 적용됨 | `DIRE_TIMESTEP_RESPACING=ddim20` export 확인 |

---

## 분류기 전환 (실험 시)

RecDrive에서 추가로 받은 분류기(`lsun_pndm.pth`, `lsun_iddpm.pth`, `imagenet_adm.pth` 등)를 `/workspace/dire_v1/weights/classifier/`에 올린 뒤:

```
export DIRE_CLASSIFIER_WEIGHTS=/workspace/dire_v1/weights/classifier/imagenet_adm.pth
```

서버 재시작 후 동일 이미지로 스코어를 비교해 가장 분리도가 좋은 분류기를 선택합니다. ADM 본체(`256x256_diffusion_uncond.pt`)는 모든 분류기와 공용으로 사용합니다.
