FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d

ENV CUDA_VISIBLE_DEVICES="" \
    GLOO_SOCKET_IFNAME=lo \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --user-group --uid 1000 --shell /bin/bash developer \
    && install -d -o developer -g developer /workspace

COPY .github/requirements-cpu.txt /tmp/requirements-cpu.txt
RUN python -m pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r /tmp/requirements-cpu.txt \
    && python -m pip check \
    && rm /tmp/requirements-cpu.txt

WORKDIR /workspace
COPY --chown=developer:developer . .
USER developer

CMD ["python", "-m", "unittest", "-v"]
