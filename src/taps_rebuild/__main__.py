import argparse
import importlib
import json
from pathlib import Path
import sys


COMMANDS = {'prepare-assets': 'assets', 'prepare-sources': 'sources', 'prepare-serving': 'runtime', 'prepare-native': 'native_runtime', 'data': 'data', 'data-gpu': 'data_pool', 'train': 'learning', 'native-train': 'native_training', 'compose': 'composition_cli', 'specialization': 'specialization', 'reproduce': 'attach', 'pool': 'pool', 'reference-group': 'reference_group', 'gpu-smoke': 'engineering', 'serving-smoke': 'serving_pool', 'verify-sampler': 'verification', 'report': 'metrics', 'serve': 'serving'}


def doctor(root):
    result = {'root': str(root), 'preparation': {}, 'runtimes': {}, 'scope': 'read_only_readiness_not_execution_proof'}
    config = json.loads((root / 'configs/reproduction.json').read_text())
    for name in ['assets', 'sources']:
        path = root / 'outputs/preparation' / (name + '.json')
        if path.exists():
            value = json.loads(path.read_text())
            result['preparation'][name] = {'status': value.get('status'), 'path': str(path), 'active_asset': value.get('active_asset')}
        else:
            result['preparation'][name] = {'status': 'missing', 'path': str(path)}
    for profile in ['eagle', 'hass']:
        python = Path(config['legacy_runtime_root']) / 'envs' / ('taps-' + profile) / 'bin/python'
        result['runtimes'][profile] = {'python': str(python), 'present': python.is_file()}
    for profile, directory in [('serving-vllm', 'runtime-vllm'), ('native-eagle3', 'runtime-native')]:
        receipt = root / 'outputs/preparation' / directory / 'runtime.json'
        result['runtimes'][profile] = {'python': str(root / 'envs' / profile / 'bin/python'), 'status': json.loads(receipt.read_text()).get('status') if receipt.exists() else 'not_prepared'}
    result['latest_runs'] = {}
    for name, pattern in [('engineering', 'outputs/engineering/*/summary.json'), ('reproduction', 'outputs/runs/*/pool.json'), ('serving', 'outputs/serving-pools/*/state.json')]:
        candidates = list(root.glob(pattern))
        if candidates:
            path = max(candidates, key=lambda candidate: candidate.stat().st_mtime_ns)
            value = json.loads(path.read_text())
            result['latest_runs'][name] = {'path': str(path), **{key: value[key] for key in ['status', 'gpu_count', 'scope', 'smoke_validation_passed', 'reproduction_gate_passed'] if key in value}}
    result['collection_destination'] = json.loads((root / 'configs/assets.json').read_text())['collection_destination']
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(prog='python -m taps_rebuild', description='TAPS preparation, engineering tests and measured research workflows')
    parser.add_argument('command', choices=['doctor', *COMMANDS])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == 'doctor':
        check = argparse.ArgumentParser()
        check.add_argument('--root', type=Path, required=True)
        doctor(check.parse_args(args.arguments).root.resolve())
        return
    module = importlib.import_module('.' + COMMANDS[args.command], 'taps_rebuild')
    sys.argv = [sys.argv[0], *args.arguments]
    module.main()


if __name__ == '__main__':
    main()
