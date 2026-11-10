import argparse
from pathlib import Path

from taps_rebuild import admission, pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-sampling', action='store_true')
    args = parser.parse_args()
    config = admission.read(args.config)
    phases = ('greedy', 'conditional', 'sampling') if args.include_sampling else ('greedy', 'conditional')
    report = admission.audit(args.root, args.root / config['driver'], args.root / config['prepared'], args.root / config['primary'], args.root / config['screen'], args.root / config['proof'], args.root / config['review'], phases=phases)
    pool.atomic_json(args.output, report)
    print({key: report[key] for key in ['status', 'phases_evaluated', 'checks', 'identity_sha256']})


if __name__ == '__main__':
    main()
