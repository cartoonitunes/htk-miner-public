# HTK CUDA miner image for vast.ai
#
# Must be a -devel image, not -runtime: htk_cuda_miner.py compiles keccak_miner.cu
# with cupy's nvcc backend at startup, so the CUDA toolkit has to be present.
# (It falls back to NVRTC, but nvcc gives noticeably better codegen here.)
#
# CUDA 12.4 covers every RTX 30xx/40xx host on vast.ai. sm_86 = 3090, sm_89 = 4090;
# cupy detects the real arch at runtime and compiles for it.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# cupy-cuda12x ships prebuilt wheels for the whole 12.x line.
RUN python3 -m pip install --no-cache-dir \
        "cupy-cuda12x>=13.0" \
        "numpy>=1.24" \
        "requests>=2.31"

WORKDIR /opt/htk
COPY keccak_miner.cu htk_cuda_miner.py htk_common.py miner_watchdog.py ./

# NOTE: no wallet key is ever baked in or passed to this image. The miner only
# publishes nonces to ntfy; signing happens on the operator's own machine.
ENV NTFY_STATUS_TOPIC="" \
    RIG_NAME="" \
    VAST_HOURLY_RATE="0" \
    GPU_DESC="" \
    HEARTBEAT_SECONDS="900"

# Fail fast and loudly if the GPU/driver combination cannot run the kernel.
HEALTHCHECK --interval=5m --timeout=30s --start-period=5m --retries=3 \
    CMD python3 -c "import cupy; cupy.cuda.runtime.getDeviceCount()" || exit 1

CMD ["sh", "-c", "python3 miner_watchdog.py --heartbeat ${HEARTBEAT_SECONDS}"]
