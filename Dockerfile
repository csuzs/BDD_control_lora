FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04
LABEL maintainer="Hugging Face"
LABEL repository="diffusers"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:fkrull/deadsnakes

RUN apt install -y bash \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv && \
    rm -rf /var/lib/apt/lists

# make sure to use venv
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
RUN python3.10 -m pip install --no-cache-dir --upgrade pip uv==0.1.11 && \
    python3.10 -m uv pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    invisible_watermark && \
    python3.10 -m pip install --no-cache-dir \
    accelerate \
    datasets \
    hf-doc-builder \
    huggingface-hub \
    hf_transfer \
    Jinja2 \
    librosa \
    numpy==1.26.4 \
    scipy \
    tensorboard \
    transformers \
    pytorch-lightning  \
    hf_transfer

CMD ["/bin/bash"]
