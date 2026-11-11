set -euo pipefail
TAPS_ROOT=/mnt/nfs/home/mohamadzbib/projects/taps-rebuild
TAPS_BUNDLE="$TAPS_ROOT/outputs/preparation/stopping-control-20261111-v1"
TAPS_PREPARED="$TAPS_ROOT/outputs/validation-inputs/registered-20261110-v2"
TAPS_OUTPUT="$TAPS_ROOT/outputs/greedy-audits/stopping-control-20261111-v1"
export PYTHONPATH="$TAPS_ROOT/src"
export PATH=/usr/local/bin:/usr/bin:/bin
cd "$TAPS_ROOT"
/usr/bin/python3 -B -S "$TAPS_BUNDLE/driver.py" controller --root "$TAPS_ROOT" --prepared "$TAPS_PREPARED" --output "$TAPS_OUTPUT"
