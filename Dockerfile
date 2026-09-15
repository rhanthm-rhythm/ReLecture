# ReLecture — pipeline orchestration layer
#
# This image runs the segment / transform / assemble stages and the Gradio UI.
# The GPU-bound servers (Whisper, Qwen3-TTS, Chatterbox, CosyVoice) are NOT
# included — they run separately (see servers/) and are reached via env vars.
#
# Build:
#   docker build -t relecture .
#
# Run the Gradio demo:
#   docker run --rm -p 7860:7860 --env-file .env relecture
#
# Run the CLI:
#   docker run --rm --env-file .env -v "$(pwd)/data:/app/data" relecture \
#     relecture run --source-video data/videos/lecture.mp4 --mode full
#
# Networking: the container calls services by the hostnames set in your .env
# (default: localhost:500x).  On Linux use --network=host so those addresses
# reach the host.  On Mac/Windows replace "localhost" with "host.docker.internal"
# in your .env before running.

FROM python:3.10-slim

# ffmpeg is required by the segment stage (audio extraction + video assembly)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv from the official image (no pip involved)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency manifests first so Docker can cache the install layer
COPY pyproject.toml uv.lock ./

# Install the project and all dependencies (no dev extras, use the locked versions)
RUN uv sync --frozen --no-dev

# Copy source and the default narrator voice used by the demo
COPY relecture/ relecture/
COPY assets/ assets/

# Gradio default port
EXPOSE 7860

CMD ["uv", "run", "python", "-m", "relecture", "ui"]
