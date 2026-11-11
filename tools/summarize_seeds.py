"""Report mean/sample standard deviation for a predeclared list of seeds."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for shot in args.shots:
        measurements, missing, protocols = [], [], []
        for seed in args.seeds:
            directory = args.root / f'seed{seed}' / f'{shot}shot'
            path = directory / 'eval_metrics.json'
            if not path.exists():
                path = directory / 'image_metrics.json'
            if not path.exists():
                missing.append(seed)
                continue
            measurements.append(json.loads(path.read_text()))
            config = json.loads((directory / 'eval_config.json').read_text())
            protocols.append(tuple(config.get(key) for key in (
                'fold', 'eval_protocol', 'eval_label_sampling', 'number_of_queries', 'n_shot'
            )))
        if len(set(protocols)) > 1:
            raise ValueError('Cannot combine evaluations with different protocols')
        statistics = {}
        for key in ('i_auroc', 'i_ap', 'i_f1', 'p_auroc', 'au_pro', 'p_f1'):
            if measurements and all(key in row for row in measurements):
                values = np.asarray([100.0 * row[key] for row in measurements])
                statistics[key] = {'mean_percent': float(values.mean()),
                                   'std_pp': float(values.std(ddof=1)) if len(values) > 1 else None}
        rows.append({'shot': shot, 'complete': not missing, 'missing_seeds': missing,
                     'count': len(measurements), 'statistics': statistics})
    result = {'root': str(args.root), 'declared_seeds': args.seeds, 'std_ddof': 1, 'results': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
