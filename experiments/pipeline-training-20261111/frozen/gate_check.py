import json
from pathlib import Path
from taps_rebuild import native_training as native
b=Path(__file__).resolve().parent
r=Path('/mnt/nfs/home/mohamadzbib/projects/taps-rebuild')
config=json.loads((b/'configs/gate-check-config.json').read_text())
result=native.research_gate(r,config)
assert result['status']=='passed' and result['scientific_training_authorized'] is True
native.atomic_json(b/'gate-check.json',result)
print(json.dumps(result))
