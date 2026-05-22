# base-conda
# FROM continuumio/miniconda3 AS conda_base
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu20.04 AS cuda_base

ARG PYTHON_VERSION=3.9
ARG CONDA_ENV=cleanroom_env

ENV TZ=Asia/Seoul \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# ---------- 1) 깨진 NVIDIA repo 비활성화 + 기본 패키지 ----------
RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        wget curl git vim tmux build-essential ca-certificates \
        libgl1 libglib2.0-0 ffmpeg tzdata \
    && rm -rf /var/lib/apt/lists/*

# ---------- 2) Conda 대신 Miniforge 설치 (conda-forge 기반, ToS 없음) ----------
RUN wget --no-verbose \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
      -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm -f /tmp/miniforge.sh && \
    $CONDA_DIR/bin/conda clean -afy

# conda 명령은 bash -lc로 실행
SHELL ["/bin/bash", "-lc"]

# ---------- 3) conda 환경 & Python 패키지 ----------
RUN conda create -n ${CONDA_ENV} python=${PYTHON_VERSION} -y && \
    conda clean -afy

ENV CONDA_DEFAULT_ENV=${CONDA_ENV}
ENV PATH=$CONDA_DIR/envs/${CONDA_ENV}/bin:$PATH

CMD ["bash"]
