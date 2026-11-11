"""Publish a complete fixed-profile, full-test benchmark with provenance checks.

Unlike the paired-baseline reporter, this does not invent a baseline for a
new transfer direction. It rejects missing seeds, partial metrics, changed
profiles, incomplete coverage, and training on the target domain.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from tools.check_release import file_digest, verify_manifest
from tools.summarize_full_evaluation import METRICS, collect
from run import PAPER_TARGETS


def compare_paper_tolerance(report, tolerance_pp=0.5):
    """Report both a performance floor and strict absolute reproduction error."""
    if not math.isfinite(tolerance_pp) or tolerance_pp < 0 or not report.get('complete'):
        raise ValueError('Require a complete benchmark and a finite nonnegative tolerance')
    if sorted(row['shot'] for row in report['summary']) != [1, 2, 4]:
        raise ValueError('Require all three support budgets exactly once')
    rows = []
    for summary in report['summary']:
        shot = summary['shot']
        expected = PAPER_TARGETS[report['fold'], shot]
        comparisons = {}
        for metric, reference in expected.items():
            mean = summary['statistics'][metric]['mean_percent']
            if not math.isfinite(mean) or not -1e-10 <= mean <= 100 + 1e-10:
                raise ValueError('Invalid percentage in paper comparison')
            delta = mean - 100 * reference
            comparisons[metric] = {
                'mean_percent': mean, 'paper_percent': 100 * reference,
                'difference_pp': delta,
                'no_more_than_tolerance_below': delta >= -tolerance_pp - 1e-10,
                'within_absolute_tolerance': abs(delta) <= tolerance_pp + 1e-10,
            }
        rows.append({'shot': shot, 'metrics': comparisons})
    failures = []
    for run in report.get('runs', []):
        for metric, reference in PAPER_TARGETS[report['fold'], run['shot']].items():
            measured = run['metrics_percent'][metric]
            delta = measured - 100 * reference
            if delta < -tolerance_pp - 1e-10:
                failures.append({'seed': run['seed'], 'shot': run['shot'], 'metric': metric,
                                 'measured_percent': measured, 'paper_percent': 100 * reference,
                                 'difference_pp': delta})
    return {
        'tolerance_pp': tolerance_pp, 'aggregation': 'declared_support_seed_mean',
        'note': 'Descriptive comparison: natural full-test and historical paper protocols may differ.',
        'all_mean_metrics_meet_performance_floor': all(
            item['no_more_than_tolerance_below'] for row in rows for item in row['metrics'].values()),
        'all_mean_metrics_within_absolute_tolerance': all(
            item['within_absolute_tolerance'] for row in rows for item in row['metrics'].values()),
        'all_individual_runs_meet_performance_floor': not failures if report.get('runs') else None,
        'individual_run_metrics_below_floor': failures,
        'results': rows,
    }


def build_report(records, profile_path, audit_path, manifest_path, required_metrics=METRICS):
    profile = json.loads(profile_path.read_text())
    audit = json.loads(audit_path.read_text())
    release = json.loads(manifest_path.read_text())
    if release['config_sha256'] != file_digest(profile_path):
        raise ValueError('Release manifest identifies a different profile')
    verify_manifest(manifest_path)
    fold, seeds, shots = profile['fold'], profile['eval_seeds'], profile['eval_shots']
    if not seeds or len(set(seeds)) != len(seeds) or shots != [1, 2, 4]:
        raise ValueError('Require distinct declared seeds and all three support budgets')
    expected = {(fold, shot, seed) for seed in seeds for shot in shots}
    if set(records) != expected:
        raise ValueError(f'Incomplete or unexpected runs: missing={expected - set(records)}, '
                         f'extra={set(records) - expected}')
    if (audit['fold'] != fold or audit['declared_seeds'] != seeds
            or audit['omitted_test_images'] or not audit['nested_support_sets_verified']):
        raise ValueError('Coverage audit does not match the declared benchmark')
    for entry in audit['metadata'].values():
        if file_digest(Path(entry['path'])) != entry['sha256']:
            raise ValueError('Dataset metadata changed after the coverage audit')
    if 'official_split_sha256' in audit:
        csv = Path(audit['metadata']['visa']['path']).parent / 'split_csv/1cls.csv'
        if file_digest(csv) != audit['official_split_sha256']:
            raise ValueError('Official VisA split changed after the audit')

    paths = {profile['checkpoint'], *profile.get('image_experts', [])}
    for shot in shots:
        paths.update(pattern.format(shot=shot) for pattern in profile.get('pixel_experts', []))
    if paths != {entry['path'] for entry in release['weights']}:
        raise ValueError('Release does not cover exactly the evaluated heads')
    training = {}
    for path in sorted(paths):
        state = torch.load(path, map_location='cpu', weights_only=False)
        config = state.get('args', {})
        if (config.get('fold') != fold or config.get('data_mode') != 'mvtec_visa'
                or config.get('mode') != 'train' or config.get('eval_frequency') != 0
                or config.get('checkpoint') or not state.get('training_configuration_known')
                or state.get('epoch') != config['epochs'] - 1):
            raise ValueError(f'Unverified source-only final-epoch training: {path}')
        training[path] = config

    audited_runs = {(entry['seed'], entry['shot']): entry for entry in audit['runs']}
    common_queries = None
    portable_queries = None
    runs = []
    ignored_profile_keys = {'output_dir', 'seed', 'eval_seeds', 'device', 'data_root',
                            'dinov2_repo', 'pretrained_weights_dir', 'backbone_weights'}
    for _, shot, seed in sorted(expected):
        record = records[fold, shot, seed]
        config, coverage, scores = record['config'], record['manifest'], record['metrics']
        # n_shot is changed by the budget loop, rather than being a profile drift.
        for key, value in profile.items():
            if key not in ignored_profile_keys | {'n_shot'} and config.get(key) != value:
                raise ValueError(f'Evaluation profile differs: seed={seed}, shot={shot}, {key}')
        reference = audited_runs[seed, shot]
        if (coverage['query_paths_sha256'] != reference['query_paths_sha256']
                or coverage['support_paths'] != reference['support_paths']
                or coverage['number_of_queries'] != audit['disk_test_images']):
            raise ValueError(f'Run differs from the independently audited split: {seed}, {shot}')
        if common_queries is not None and common_queries != coverage['queries']:
            raise ValueError('Query identities or labels changed between seeds or budgets')
        common_queries = coverage['queries']
        for metric in required_metrics:
            value = scores[metric]
            categories = [item[metric] for item in scores['per_category'].values()]
            # Exact AP can exceed one by machine epsilon (e.g. 1.0000000000000002).
            # Preserve raw values; only the validation bounds allow roundoff.
            if not all(math.isfinite(x) and -1e-12 <= x <= 1 + 1e-12
                       for x in [value, *categories]):
                raise ValueError(f'Invalid metric: {seed}, {shot}, {metric}')
            if not math.isclose(value, statistics.mean(categories), abs_tol=1e-10):
                raise ValueError(f'Metric is not a category macro mean: {metric}')
        data_root = Path(config['data_root']).resolve()
        portable_queries = [{**query, 'path': Path(query['path']).relative_to(data_root).as_posix()}
                            for query in coverage['queries']]
        runs.append({
            'seed': seed, 'shot': shot,
            'supports': {product: [Path(path).relative_to(data_root).as_posix() for path in selected]
                         for product, selected in coverage['support_paths'].items()},
            'metrics_percent': {metric: 100 * scores[metric] for metric in required_metrics},
            'per_category_percent': {
                product: {metric: 100 * values[metric] for metric in required_metrics}
                for product, values in scores['per_category'].items()},
        })
    summary = []
    for shot in shots:
        selected = [record for record in runs if record['shot'] == shot]
        values = {}
        for metric in required_metrics:
            samples = [record['metrics_percent'][metric] for record in selected]
            values[metric] = {'mean_percent': statistics.mean(samples),
                              'std_pp': statistics.stdev(samples) if len(samples) > 1 else None}
        summary.append({'shot': shot, 'count': len(selected), 'statistics': values})
    return {
        'complete': True, 'protocol': 'exhaustive', 'fold': fold,
        'source': 'MVTec AD' if fold == 1 else 'VisA', 'target': audit['target'],
        'profile': str(profile_path), 'profile_sha256': file_digest(profile_path),
        'profile_configuration': profile, 'declared_seeds': seeds, 'shots': shots,
        'number_of_queries_per_run': audit['disk_test_images'],
        'normal_queries': audit['normal_queries'], 'anomalous_queries': audit['anomalous_queries'],
        'number_of_evaluations': len(runs) * audit['disk_test_images'],
        'coverage_verified_against_official_split': True, 'query_support_overlap': 0,
        'official_anomaly_masks_verified': audit.get('official_anomaly_masks_verified', False),
        'nested_support_sets_verified': True, 'queries': portable_queries,
        'query_relative_paths_sha256': hashlib.sha256(
            '\n'.join(query['path'] for query in portable_queries).encode()).hexdigest(),
        'metadata_sha256': {domain: item['sha256'] for domain, item in audit['metadata'].items()},
        'official_split_sha256': audit.get('official_split_sha256'),
        'metric_protocol': {'aggregation': 'category_macro_average',
                            'input_size': profile['image_size'],
                            'evaluation_size': profile.get('eval_image_size') or profile['image_size'],
                            'gaussian_sigma': 4, 'aupro_max_fpr': 0.3, 'std_ddof': 1,
                            'f1_threshold': 'per_category_evaluation_maximum'},
        'weights': release['weights'], 'training_configurations': training,
        'software': release['software'], 'cuda_runtime': release['cuda_runtime'],
        'runs': runs, 'summary': summary,
        'limitations': [
            'The five seeds are support draws, not independent model training seeds.',
            'Natural full-test prevalence is not identical to the historical sampled paper protocol.',
            'Target data have been used during method development; this is not an independent held-out development benchmark.',
            'Source-head training configuration and inference configuration are recorded separately.',
            'These results do not establish an identical architecture or compute budget to the conference model.',
            'Whole-configuration results do not establish individual component gains.',
        ],
    }


def attach_paired_baseline(enhanced, baseline):
    """Attach independently validated control runs with identical queries/supports."""
    for key in ('fold', 'declared_seeds', 'shots', 'queries', 'metric_protocol'):
        if enhanced[key] != baseline[key]:
            raise ValueError(f'Baseline evaluation is not paired: {key}')
    first = {(run['seed'], run['shot']): run for run in baseline['runs']}
    second = {(run['seed'], run['shot']): run for run in enhanced['runs']}
    if set(first) != set(second):
        raise ValueError('Baseline evaluation has a different seed/budget grid')
    # Verify all pairs before mutating the report.
    for key, run in second.items():
        if run['supports'] != first[key]['supports']:
            raise ValueError(f'Baseline normal supports are not paired: {key}')
    for key, run in second.items():
        run['baseline_percent'] = first[key]['metrics_percent']
        run['baseline_per_category_percent'] = first[key]['per_category_percent']
        run['enhanced_minus_baseline_pp'] = {
            metric: run['metrics_percent'][metric] - first[key]['metrics_percent'][metric]
            for metric in METRICS}
    enhanced['paired_baseline'] = {
        'support_pairing_verified': True, 'number_of_pairs': len(first),
        'profile': baseline['profile'], 'profile_sha256': baseline['profile_sha256'],
        'profile_configuration': baseline['profile_configuration'],
        'summary': baseline['summary'], 'weights': baseline['weights'],
        'description': 'New MVTec-trained Small compatibility head, without reference rotations or expert fusion; not the historical conference checkpoint.',
    }
    return enhanced


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, action='append', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, action='append')
    parser.add_argument('--baseline-config', type=Path)
    parser.add_argument('--baseline-manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-gap-pp', type=float, default=0.5)
    cli = parser.parse_args()
    report = build_report(collect(cli.root), cli.config, cli.audit, cli.manifest)
    report['paper_comparison'] = compare_paper_tolerance(report, cli.target_gap_pp)
    baseline_options = (cli.baseline_root, cli.baseline_config, cli.baseline_manifest)
    if any(baseline_options):
        if not all(baseline_options):
            parser.error('Provide all three baseline options to attach a paired control')
        baseline = build_report(collect(cli.baseline_root), cli.baseline_config,
                                cli.audit, cli.baseline_manifest)
        attach_paired_baseline(report, baseline)
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with cli.output.open('x') as handle:
        handle.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'complete': report['complete'], 'target': report['target'],
                      'queries_per_run': report['number_of_queries_per_run'],
                      'summary': report['summary']}, indent=2))


if __name__ == '__main__':
    main()
