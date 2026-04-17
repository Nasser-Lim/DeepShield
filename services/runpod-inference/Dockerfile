FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# DIRE official repo provides guided_diffusion; add to PYTHONPATH.
# On RunPod, override DIRE_REPO_PATH to /workspace/dire_v1/repo for volume-persisted code.
RUN git clone --depth 1 https://github.com/ZhendongWang6/DIRE.git /opt/dire
ENV PYTHONPATH=/opt/dire:/opt/dire/guided-diffusion
ENV DIRE_REPO_PATH=/opt/dire

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV INFER_HOST=0.0.0.0 INFER_PORT=8000
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
