"""Summarize paired, exhaustive evaluations without accepting sampled runs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run import PAPER_TARGETS


METRICS = ('i_auroc', 'i_ap', 'i_f1', 'p_auroc', 'au_pro', 'p_f1')

PROFILE_KEYS = (
    'fold', 'backbone_name', 'num_a_queries', 'proxy_orientation', 'proxy_reduction',
    'residual_matching', 'nls_values', 'loss_mode', 'image_size', 'image_topk',
    'image_proxy_weight', 'proxy_weight', 'support_rotations', 'support_scales',
    'pixel_expert_reduction', 'pixel_expert_weights', 'pixel_center_weight',
    'normal_tta_steps', 'checkpoint', 'image_experts', 'pixel_experts',
    'matching_layers', 'matching_pool_size',
    'image_support_rotations', 'image_final_layer_matching',
    'eval_image_size',
)


def collect(roots, required_metrics=METRICS):
    records = {}
    for root in roots:
        for path in root.rglob('eval_metrics.json'):
            folder = path.parent
            config = json.loads((folder / 'eval_config.json').read_text())
            manifest = json.loads((folder / 'evaluation_manifest.json').read_text())
            metrics = json.loads(path.read_text())
            if config['eval_protocol'] != 'exhaustive' or manifest['protocol'] != 'exhaustive':
                raise ValueError(f'Not an exhaustive evaluation: {folder}')
            count = manifest['number_of_queries']
            if (count != manifest['number_of_unique_queries']
                    or count != config['number_of_queries']
                    or count != len(manifest['queries'])
                    or manifest['query_support_overlap'] != 0):
                raise ValueError(f'Coverage audit failed: {folder}')
            if set(manifest['per_category_counts']) != set(metrics['per_category']):
                raise ValueError(f'Category coverage differs: {folder}')
            if not all(name in metrics for name in required_metrics):
                raise ValueError(f'Incomplete metrics: {folder}')
            key = (config['fold'], config['n_shot'], config['seed'])
            if key in records:
                raise ValueError(f'Duplicate run {key} in declared roots')
            records[key] = {'folder': str(folder), 'config': config,
                            'manifest': manifest, 'metrics': metrics}
    return records


def summarize(baseline, enhanced, seeds, fold=0):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Declare at least one distinct support seed')
    rows = []
    all_hashes = {record['manifest']['query_paths_sha256']
                  for runs in (baseline, enhanced) for record in runs.values()}
    if len(all_hashes) > 1:
        raise ValueError('Evaluation query sets/order differ across runs')
    for shot in (1, 2, 4):
        expected = PAPER_TARGETS[(fold, shot)]
        pairs = []
        missing = []
        for seed in seeds:
            key = (fold, shot, seed)
            if key not in baseline or key not in enhanced:
                missing.append(seed)
                continue
            first, second = baseline[key], enhanced[key]
            if first['manifest']['support_paths'] != second['manifest']['support_paths']:
                raise ValueError(f'Unpaired normal supports: {key}')
            pairs.append({'seed': seed, 'baseline_folder': first['folder'],
                          'enhanced_folder': second['folder'],
                          'number_of_queries': first['manifest']['number_of_queries'],
                          'baseline_percent': {m: 100 * first['metrics'][m] for m in METRICS},
                          'enhanced_percent': {m: 100 * second['metrics'][m] for m in METRICS},
                          'enhanced_minus_baseline_pp': {
                              m: 100 * (second['metrics'][m] - first['metrics'][m]) for m in METRICS},
                          'enhanced_minus_paper_pp': {
                              m: 100 * (second['metrics'][m] - expected[m]) for m in METRICS},
                          'all_six_exceed_paper': all(second['metrics'][m] > expected[m] for m in METRICS)})
        statistics = {}
        if pairs:
            for variant in ('baseline', 'enhanced'):
                statistics[variant] = {}
                for metric in METRICS:
                    values = np.asarray([pair[f'{variant}_percent'][metric] for pair in pairs])
                    statistics[variant][metric] = {
                        'mean_percent': float(values.mean()),
                        'std_pp': float(values.std(ddof=1)) if len(values) > 1 else None,
                        'minimum_percent': float(values.min()), 'maximum_percent': float(values.max())}
        mean_gaps = {metric: statistics['enhanced'][metric]['mean_percent'] - 100 * expected[metric]
                     for metric in METRICS} if pairs else {}
        rows.append({'shot': shot, 'complete': not missing, 'missing_seeds': missing,
                     'pairs': pairs, 'statistics': statistics,
                     'mean_enhanced_minus_paper_pp': mean_gaps,
                     'all_mean_metrics_exceed_paper': not missing and all(value > 0 for value in mean_gaps.values()),
                     'every_seed_exceeds_all_paper_metrics': not missing and all(
                         pair['all_six_exceed_paper'] for pair in pairs)})
    return {'protocol': 'exhaustive', 'fold': fold, 'declared_seeds': seeds,
            'query_paths_sha256': next(iter(all_hashes), None), 'std_ddof': 1,
            'support_pairing_verified': True,
            'number_of_verified_pairs': sum(len(row['pairs']) for row in rows),
            'complete': all(row['complete'] for row in rows),
            'all_18_mean_metrics_exceed_paper': all(row['all_mean_metrics_exceed_paper'] for row in rows),
            'results': rows,
            'paper_comparison_note': 'Descriptive only: different prevalence/support protocol from sampled episodes.',
            'selection_note': 'Target-set development results, not an independent held-out benchmark.'}


def publication_report(report, baseline, enhanced, profile_path, audit_path, release_path):
    """Build a portable local result artifact; never upload anything."""
    if not report['complete']:
        raise ValueError('Cannot publish an incomplete declared-seed benchmark')
    profile = json.loads(profile_path.read_text())
    audit = json.loads(audit_path.read_text())
    release = json.loads(release_path.read_text())
    profile_hash = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    if release['config_sha256'] != profile_hash:
        raise ValueError('Release manifest does not identify this evaluation profile')
    if (audit['declared_seeds'] != report['declared_seeds']
            or audit['omitted_test_images'] != 0 or not audit['nested_support_sets_verified']):
        raise ValueError('Complete-file/reference audit does not match declared seeds')
    relevant = {(report['fold'], shot, seed)
                for shot in (1, 2, 4) for seed in report['declared_seeds']}
    for key in relevant:
        config = enhanced[key]['config']
        for name in PROFILE_KEYS:
            if name in profile and config.get(name) != profile[name]:
                raise ValueError(f'Published profile differs from evaluated {key}: {name}')
    first = enhanced[min(relevant)]
    data_root = Path(first['config']['data_root']).resolve()
    queries = [{**query, 'path': Path(query['path']).relative_to(data_root).as_posix()}
               for query in first['manifest']['queries']]
    query_hash = hashlib.sha256('\n'.join(query['path'] for query in queries).encode()).hexdigest()
    count = len(queries)
    if count != audit['disk_test_images']:
        raise ValueError('Published query count differs from audited test files')
    runs = []
    for key in sorted(relevant):
        _, shot, seed = key
        manifest = enhanced[key]['manifest']
        support_root = Path(enhanced[key]['config']['data_root']).resolve()
        runs.append({
            'seed': seed, 'shot': shot,
            'supports': {product: [Path(path).relative_to(support_root).as_posix() for path in paths]
                         for product, paths in manifest['support_paths'].items()},
            'baseline_percent': {name: 100 * baseline[key]['metrics'][name] for name in METRICS},
            'enhanced_percent': {name: 100 * enhanced[key]['metrics'][name] for name in METRICS},
            'baseline_per_category_percent': {
                product: {name: 100 * scores[name] for name in METRICS}
                for product, scores in baseline[key]['metrics']['per_category'].items()},
            'enhanced_per_category_percent': {
                product: {name: 100 * scores[name] for name in METRICS}
                for product, scores in enhanced[key]['metrics']['per_category'].items()},
        })
    portable_summary = json.loads(json.dumps(report))
    portable_summary.pop('query_paths_sha256', None)
    for row in portable_summary['results']:
        for pair in row['pairs']:
            pair.pop('baseline_folder', None)
            pair.pop('enhanced_folder', None)
    return {
        'protocol': 'exhaustive', 'source': 'VisA' if report['fold'] == 0 else 'MVTec AD',
        'target': audit['target'], 'profile': str(profile_path), 'profile_sha256': profile_hash,
        'profile_configuration': profile, 'declared_seeds': report['declared_seeds'],
        'number_of_queries_per_run': count, 'normal_queries': sum(not q['anomaly'] for q in queries),
        'anomalous_queries': sum(bool(q['anomaly']) for q in queries),
        'query_relative_paths_sha256': query_hash,
        'per_category_counts': first['manifest']['per_category_counts'],
        'metadata_sha256': {domain: record['sha256'] for domain, record in audit['metadata'].items()},
        'metric_protocol': {'aggregation': 'category_macro_average', 'evaluation_size': profile['image_size'],
                            'gaussian_sigma': 4, 'aupro_max_fpr': 0.3,
                            'f1_threshold': 'per_category_evaluation_maximum'},
        'software': release['software'], 'cuda_runtime': release['cuda_runtime'],
        'dinov2_source_fingerprint': {name: value for name, value in release.get(
            'recorded_dinov2_source', {}).items() if name != 'path'},
        'weights': [{name: record[name] for name in (
            'path', 'size_bytes', 'sha256', 'frozen_encoder_signature', 'training_configuration_known')}
            for record in release['weights']],
        'coverage_verified_against_disk': True, 'query_support_overlap': 0,
        'nested_support_sets_verified': True, 'queries': queries, 'runs': runs,
        'summary': portable_summary,
        'limitations': [
            'Target data were used for method development; this is not an independent held-out benchmark.',
            'The natural-prevalence train-reference protocol is not established as identical to the paper protocol.',
            'The enhanced model is larger than the paper baseline and uses rotated views of the same K images.',
            'The original legacy head training configuration was not saved.',
            'A positive point-estimate gap does not establish stable superiority; inspect support-seed standard deviations and failures.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--baseline-root', type=Path, action='append', required=True)
    parser.add_argument('--enhanced-root', type=Path, action='append', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[111, 222, 333])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publication-output', type=Path,
                        help='Optional portable local JSON artifact; requires every declared run to be complete.')
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--dataset-audit', type=Path)
    parser.add_argument('--release-manifest', type=Path)
    args = parser.parse_args()
    baseline, enhanced = collect(args.baseline_root), collect(args.enhanced_root)
    report = summarize(baseline, enhanced, args.seeds, args.fold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for row in report['results']:
        print(f"{row['shot']}-shot: paired seeds={len(row['pairs'])}, missing={row['missing_seeds']}")
        if row['statistics']:
            for label, scores in row['statistics'].items():
                print(f"  {label}: " + ', '.join(
                    f"{name}={value['mean_percent']:.2f} ± "
                    + (f"{value['std_pp']:.2f}" if value['std_pp'] is not None else 'N/A')
                    for name, value in scores.items()))
    print(f'Saved {args.output}')
    if args.publication_output:
        if not all((args.profile, args.dataset_audit, args.release_manifest)):
            parser.error('--publication-output requires --profile, --dataset-audit and --release-manifest')
        public = publication_report(report, baseline, enhanced, args.profile,
                                    args.dataset_audit, args.release_manifest)
        args.publication_output.parent.mkdir(parents=True, exist_ok=True)
        args.publication_output.write_text(json.dumps(public, indent=2) + '\n', encoding='utf-8')
        print(f'Saved portable local benchmark artifact {args.publication_output}')


if __name__ == '__main__':
    main()
