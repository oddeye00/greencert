FROM python:3.12.10-slim-bookworm

ENV PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      poppler-utils \
      texlive-fonts-recommended \
      texlive-latex-base \
      texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/greencert
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "reproduce.py", "smoke"]

