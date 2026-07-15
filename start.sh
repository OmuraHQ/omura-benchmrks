#!/usr/bin/env bash
# Starts the Omura API server plus its sidecar processes:
#   - main.py           (FastAPI app + cataloger/vector-indexer/BellySeal/Walrus-blob threads)
#   - caption_service.py (Moondream captioning, separate .venv-caption)
#   - iv2_video_service.py (InternVideo2 video embeddings, separate .venv-iv2; skipped if
#                            IV2_CKPT/OMURA_VIDEO_HEADS aren't set to existing files)
#
# All GPU-using processes are pinned to GPU 7 only (CUDA_VISIBLE_DEVICES=7), overriding
# .env's CUDA_VISIBLE_DEVICES=6,7. OMURA_CUVS_GPU_ID is forced to 0 to match (it's an index
# into the visible-device list, not an absolute device id).
#
# Usage: ./start.sh [start|stop|status]

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/data/logs"
PID_DIR="$ROOT/data/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

export CUDA_VISIBLE_DEVICES=7
export OMURA_CUVS_GPU_ID=0

API_PORT="${OMURA_PORT:-19543}"
CAPTION_PORT="${OMURA_CAPTION_PORT:-18085}"
VIDEO_PORT="${IV2_SERVICE_PORT:-19560}"

is_running() {
    local pid_file="$1"
    [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

start_proc() {
    local name="$1" pid_file="$2" log_file="$3"
    shift 3
    if is_running "$pid_file"; then
        echo "[start.sh] $name already running (pid $(cat "$pid_file"))"
        return 0
    fi
    echo "[start.sh] starting $name -> $log_file"
    nohup "$@" >>"$log_file" 2>&1 &
    echo $! >"$pid_file"
    disown
}

wait_for_http() {
    local url="$1" name="$2" tries="${3:-30}"
    for ((i = 0; i < tries; i++)); do
        if curl -fsS -m 2 "$url" >/dev/null 2>&1; then
            echo "[start.sh] $name is up ($url)"
            return 0
        fi
        sleep 1
    done
    echo "[start.sh] WARNING: $name did not respond at $url after ${tries}s (check its log)"
    return 1
}

stop_proc() {
    local name="$1" pid_file="$2"
    if is_running "$pid_file"; then
        local pid
        pid="$(cat "$pid_file")"
        echo "[start.sh] stopping $name (pid $pid)"
        kill "$pid" 2>/dev/null
        for _ in $(seq 1 10); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
    else
        echo "[start.sh] $name not running"
    fi
    rm -f "$pid_file"
}

ensure_envs() {
    # 1. Main API server environment
    if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
        echo "[start.sh] Creating main .venv..."
        uv venv "$ROOT/.venv" --python 3.11
        uv sync
    fi

    # 2. Caption sidecar environment
    if [[ ! -x "$ROOT/.venv-caption/bin/python" ]]; then
        echo "[start.sh] Creating .venv-caption..."
        uv venv "$ROOT/.venv-caption" --python 3.11
        VIRTUAL_ENV="$ROOT/.venv-caption" uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
        VIRTUAL_ENV="$ROOT/.venv-caption" uv pip install transformers pillow einops accelerate
    fi

    # 3. Video service environment
    local iv2_dir="$ROOT/benchmarks/eval/internvideo2"
    if [[ ! -x "$iv2_dir/.venv-iv2/bin/python" ]]; then
        echo "[start.sh] Creating .venv-iv2..."
        uv venv "$iv2_dir/.venv-iv2" --python 3.10
        VIRTUAL_ENV="$iv2_dir/.venv-iv2" uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
            --index-url https://download.pytorch.org/whl/cu121
        VIRTUAL_ENV="$iv2_dir/.venv-iv2" uv pip install transformers==4.28.1 "tokenizers<0.14" timm==0.5.4 einops \
            decord opencv-python-headless==4.8.0.76 librosa==0.10.1 soundfile==0.12.1 \
            "datasets>=2.18,<3" pandas pyarrow easydict pyyaml termcolor scipy ftfy regex tqdm \
            "huggingface_hub>=0.23"
        VIRTUAL_ENV="$iv2_dir/.venv-iv2" uv pip install peft==0.5.0 "accelerate<0.30" open_clip_torch
        VIRTUAL_ENV="$iv2_dir/.venv-iv2" uv pip install numpy==1.24.4
        VIRTUAL_ENV="$iv2_dir/.venv-iv2" uv pip install "setuptools<81"

        # Install flash_attn stubs
        local sp_dir
        sp_dir="$iv2_dir/.venv-iv2/lib/python3.10/site-packages"
        mkdir -p "$sp_dir/flash_attn/modules" "$sp_dir/flash_attn/ops"
        
        cat << 'EOF' > "$sp_dir/flash_attn/__init__.py"
# Stub flash_attn: symbols exist so imports succeed; never invoked because the
# model is configured with use_flash_attn/use_fused_mlp/use_fused_rmsnorm = False.
__version__ = "0.0.0-stub"
def _unavail(*a, **k):
    raise RuntimeError("flash_attn stub: flash kernels disabled in this eval")
EOF

        cat << 'EOF' > "$sp_dir/flash_attn/flash_attn_interface.py"
def flash_attn_varlen_qkvpacked_func(*a, **k):
    raise RuntimeError("flash_attn stub invoked but flash disabled")
EOF

        cat << 'EOF' > "$sp_dir/flash_attn/bert_padding.py"
def unpad_input(*a, **k):
    raise RuntimeError("flash_attn stub")
def pad_input(*a, **k):
    raise RuntimeError("flash_attn stub")
EOF

        touch "$sp_dir/flash_attn/modules/__init__.py"
        cat << 'EOF' > "$sp_dir/flash_attn/modules/mlp.py"
class FusedMLP:
    def __init__(self,*a,**k):
        raise RuntimeError("flash_attn stub FusedMLP invoked but disabled")
EOF

        touch "$sp_dir/flash_attn/ops/__init__.py"
        cat << 'EOF' > "$sp_dir/flash_attn/ops/rms_norm.py"
class DropoutAddRMSNorm:
    def __init__(self,*a,**k):
        raise RuntimeError("flash_attn stub DropoutAddRMSNorm invoked but disabled")
EOF
    fi
}

do_start() {
    ensure_envs

    # 1. Main API server + in-process cataloger/vector-indexer/BellySeal/Walrus-blob threads
    #    (OMURA_INSTANCE_ROLE defaults to "all" -> single process handles ingestion + serving).
    start_proc "api server" "$PID_DIR/api.pid" "$LOG_DIR/api.log" \
        "$ROOT/.venv/bin/python" "$ROOT/main.py"

    # 2. Caption sidecar (Moondream) — separate venv (transformers 4.52), must match
    #    OMURA_CAPTION_SERVER_URL's port from .env.
    if [[ -x "$ROOT/.venv-caption/bin/python" ]]; then
        start_proc "caption service" "$PID_DIR/caption.pid" "$LOG_DIR/caption.log" \
            "$ROOT/.venv-caption/bin/python" "$ROOT/scripts/caption_service.py" \
            --port "$CAPTION_PORT"
    else
        echo "[start.sh] SKIP caption service: $ROOT/.venv-caption not found"
    fi

    # 3. Video embedding sidecar (InternVideo2) — separate venv (py3.10), needs the base
    #    6B checkpoint (IV2_CKPT) + finetuned heads (OMURA_VIDEO_HEADS). Both are large
    #    files not committed to the repo; skip cleanly if not configured on this host.
    local iv2_dir="$ROOT/benchmarks/eval/internvideo2"
    export OMURA_VIDEO_HEADS="${OMURA_VIDEO_HEADS:-$iv2_dir/data/finetune_v1/best_heads.pt}"
    export IV2_CKPT="${IV2_CKPT:-$ROOT/data/checkpoints/internvideo2/internvideo2-s2_6b-224p-f4_with_audio_encoder.pt}"
    if [[ -z "${IV2_CKPT:-}" || ! -f "${IV2_CKPT:-/nonexistent}" ]]; then
        echo "[start.sh] SKIP video service: IV2_CKPT not set to an existing checkpoint file" \
             "(export IV2_CKPT=/path/to/internvideo2-s2_6b-224p-f4_with_audio_encoder.pt)"
    elif [[ ! -f "$OMURA_VIDEO_HEADS" ]]; then
        echo "[start.sh] SKIP video service: OMURA_VIDEO_HEADS not found at $OMURA_VIDEO_HEADS"
    elif [[ ! -x "$iv2_dir/.venv-iv2/bin/python" ]]; then
        echo "[start.sh] SKIP video service: $iv2_dir/.venv-iv2 not found"
    else
        (cd "$iv2_dir" && start_proc "video service" "$PID_DIR/video.pid" "$LOG_DIR/video.log" \
            "$iv2_dir/.venv-iv2/bin/python" "$iv2_dir/scripts/iv2_video_service.py" \
            --port "$VIDEO_PORT")
    fi

    echo "[start.sh] waiting for services to come up..."
    wait_for_http "http://127.0.0.1:$API_PORT/health" "api server" 60
    [[ -f "$PID_DIR/caption.pid" ]] && wait_for_http "http://127.0.0.1:$CAPTION_PORT/health" "caption service" 60
    [[ -f "$PID_DIR/video.pid" ]] && wait_for_http "http://127.0.0.1:$VIDEO_PORT/health" "video service" 60

    echo "[start.sh] done. Logs: $LOG_DIR  PIDs: $PID_DIR"
}

do_stop() {
    stop_proc "video service" "$PID_DIR/video.pid"
    stop_proc "caption service" "$PID_DIR/caption.pid"
    stop_proc "api server" "$PID_DIR/api.pid"
}

do_status() {
    for entry in "api server:$PID_DIR/api.pid" "caption service:$PID_DIR/caption.pid" "video service:$PID_DIR/video.pid"; do
        name="${entry%%:*}"; pid_file="${entry#*:}"
        if is_running "$pid_file"; then
            echo "[start.sh] $name: RUNNING (pid $(cat "$pid_file"))"
        else
            echo "[start.sh] $name: stopped"
        fi
    done
}

case "${1:-start}" in
    start) do_start ;;
    stop) do_stop ;;
    status) do_status ;;
    *)
        echo "Usage: $0 [start|stop|status]" >&2
        exit 1
        ;;
esac
