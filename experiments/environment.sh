TAPS_PROFILE="${1:-eagle}"
case "$TAPS_PROFILE" in
    eagle|hass) ;;
    *) printf 'Choose eagle or hass.\n' >&2; return 2 ;;
esac
export TAPS_PROJECT_ROOT="${TAPS_PROJECT_ROOT:-/mnt/nfs/home/mohamadzbib/projects/taps-rebuild}"
export TAPS_RUNTIME_ROOT="${TAPS_RUNTIME_ROOT:-/mnt/nfs/home/mohamadzbib/projects/rl-infra}"
TAPS_ENVIRONMENT="$TAPS_RUNTIME_ROOT/envs/taps-$TAPS_PROFILE"
if [ ! -x "$TAPS_ENVIRONMENT/bin/python" ]; then
    printf 'Prepared cluster runtime is unavailable: %s\n' "$TAPS_ENVIRONMENT" >&2
    return 1
fi
unset LD_PRELOAD PYTHONHOME PYTHONPATH
source "$TAPS_ENVIRONMENT/bin/activate"
export PATH="$TAPS_RUNTIME_ROOT/tools:$PATH"
export PYTHONPATH="$TAPS_PROJECT_ROOT/src"
export UV_PROJECT_ENVIRONMENT="$TAPS_ENVIRONMENT"
export UV_PYTHON_INSTALL_DIR="$TAPS_RUNTIME_ROOT/python"
export UV_CACHE_DIR="$TAPS_PROJECT_ROOT/cache/uv"
export UV_NO_CONFIG=1 UV_LINK_MODE=copy
export HF_HOME="$TAPS_PROJECT_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$TAPS_PROJECT_ROOT/cache"
export CUDA_CACHE_PATH="$TAPS_PROJECT_ROOT/cache/cuda"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
cd "$TAPS_PROJECT_ROOT"
taps() {
    "$TAPS_RUNTIME_ROOT/tools/uv" run --no-sync --no-project python -B -m taps_rebuild "$@"
}
