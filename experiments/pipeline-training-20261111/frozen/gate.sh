set -euo pipefail
export PYTHONPATH=/mnt/nfs/home/mohamadzbib/projects/taps-rebuild/outputs/preparation/pipeline-training-20261111-v1/src
export HF_HOME=/mnt/nfs/home/mohamadzbib/projects/taps-rebuild/cache/native-eagle3/huggingface
export HF_HUB_CACHE=/mnt/nfs/home/mohamadzbib/projects/taps-rebuild/cache/native-eagle3/huggingface/hub
export HF_DATASETS_CACHE=/mnt/nfs/home/mohamadzbib/projects/taps-rebuild/cache/native-eagle3/datasets
export XDG_CACHE_HOME=/mnt/nfs/home/mohamadzbib/projects/taps-rebuild/cache/native-eagle3/xdg
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
cd /mnt/nfs/home/mohamadzbib/projects/taps-rebuild
export CUDA_VISIBLE_DEVICES=
/mnt/nfs/home/mohamadzbib/projects/taps-rebuild/envs/native-eagle3/bin/python -B /mnt/nfs/home/mohamadzbib/projects/taps-rebuild/outputs/preparation/pipeline-training-20261111-v1/gate_check.py
