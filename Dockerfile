# SETU — reproducible image for the CPU path.
#
# The GDAL/PROJ/torch install is the single most likely cause of a failed live demo, which
# is why it is pinned into an image rather than left to a laptop. Two stages: the heavy
# scientific wheels resolve once and are cached, and only the source changes on a rebuild.

FROM python:3.11-slim-bookworm AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libGL and libglib are needed by OpenCV even in its headless build; leaving them out
# produces an ImportError at first use rather than at install time.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libgl1 \
      libglib2.0-0 \
      libexpat1 \
      curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/setu

COPY pyproject.toml README.md ./
COPY setu/__init__.py setu/_env.py setu/

# CPU-only torch. The CUDA wheels are several gigabytes and buy nothing on a judge's
# laptop, which is the environment this image exists to guarantee.
RUN pip install --upgrade pip wheel setuptools \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install ".[api]" kornia certifi


FROM deps AS runtime

WORKDIR /opt/setu
COPY . .
RUN pip install --no-deps -e .

# Model checkpoints download on first use; giving them a stable home means a warm
# container starts instantly.
ENV TORCH_HOME=/opt/setu/.cache/torch \
    SETU_MATCHANYTHING_DIR=/opt/setu/weights/matchanything \
    MPLBACKEND=Agg

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
