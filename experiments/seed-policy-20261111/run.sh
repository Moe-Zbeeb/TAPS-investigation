set -euo pipefail
TAPS_ROOT=/mnt/nfs/home/mohamadzbib/projects/taps-rebuild
TAPS_BUNDLE="$TAPS_ROOT/outputs/preparation/seed-policy-20261111-v1"
TAPS_PREPARED="$TAPS_ROOT/outputs/validation-inputs/seed-policy-20261111-v1"
TAPS_OUTPUT="$TAPS_ROOT/outputs/seed-policy/first-turn-20261111-v1"
export PYTHONPATH="$TAPS_ROOT/src"
export PATH=/usr/local/bin:/usr/bin:/bin
cd "$TAPS_ROOT"
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -B -S "$TAPS_BUNDLE/driver.py" --action prepare --root "$TAPS_ROOT" --prepared "$TAPS_PREPARED" --output "$TAPS_OUTPUT"
/usr/bin/python3 -B -S "$TAPS_BUNDLE/driver.py" --action controller --root "$TAPS_ROOT" --prepared "$TAPS_PREPARED" --output "$TAPS_OUTPUT"
