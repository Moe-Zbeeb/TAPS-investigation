set -euo pipefail
TAPS_ROOT=/mnt/nfs/home/mohamadzbib/projects/taps-rebuild
TAPS_BUNDLE="$TAPS_ROOT/outputs/preparation/remaining-greedy-20261111-v3"
TAPS_PREPARED="$TAPS_ROOT/outputs/validation-inputs/remaining-greedy-20261111-v3"
TAPS_OUTPUT="$TAPS_ROOT/outputs/greedy-audits/remaining-fp32-20261111-v3"
export PYTHONPATH="$TAPS_ROOT/src"
export PATH=/usr/local/bin:/usr/bin:/bin
cd "$TAPS_ROOT"
if [ ! -e "$TAPS_PREPARED/manifest.json" ]; then
  CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -B -S "$TAPS_BUNDLE/driver.py" --action prepare --root "$TAPS_ROOT" --prepared "$TAPS_PREPARED" --output "$TAPS_OUTPUT"
fi
/usr/bin/python3 -B -S "$TAPS_BUNDLE/driver.py" --action controller --root "$TAPS_ROOT" --prepared "$TAPS_PREPARED" --output "$TAPS_OUTPUT"
