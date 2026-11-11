"""Prepare, audit and summarize the fixed converted-BraTS benchmark."""

import argparse
import json
from pathlib import Path
import statistics

from tools.summarize_benchmark import build_report
from tools.summarize_full_evaluation import collect
from utils.medical import MedicalEvalDataset, digest, prepare_split, validate_split


METRICS = ('i_auroc', 'p_auroc')


def audit_split(split_path, data_root):
    plan = json.loads(split_path.read_text())
    validate_split(plan, data_root)
    runs = []
    common = None
    for seed in plan['declared_seeds']:
        by_shot = {}
        for shot in (1, 2, 4):
            dataset = MedicalEvalDataset(data_root=data_root, medical_split=split_path,
                                         shot=(shot, 0), seed=seed, transform=None)
            manifest = dataset.evaluation_manifest
            if common is not None and common != manifest['queries']:
                raise ValueError('Medical queries changed across support draws')
            common = manifest['queries']
            by_shot[shot] = manifest['support_paths']['brain']
            runs.append({'seed': seed, 'shot': shot,
                         'query_paths_sha256': manifest['query_paths_sha256'],
                         'support_paths': manifest['support_paths']})
        if by_shot[4][:1] != by_shot[1] or by_shot[4][:2] != by_shot[2]:
            raise ValueError('Medical supports are not nested')
    return {
        'fold': 1, 'target': 'BraTS (supplied 2D conversion)',
        'declared_seeds': plan['declared_seeds'],
        'disk_test_images': plan['number_of_queries'],
        'original_test_images': plan['original_test_images'],
        'normal_queries': plan['normal_queries'], 'anomalous_queries': plan['anomalous_queries'],
        'omitted_test_images': 0, 'eligible_query_images_omitted': 0,
        'excluded_reference_copies': len(plan['excluded_queries']),
        'original_train_test_content_overlap': plan['original_train_test_content_overlap'],
        'nested_support_sets_verified': True, 'content_query_support_overlap': 0,
        'medical_split_sha256': digest(split_path),
        'metadata': {'brats': {'path': str((data_root / 'brats/meta.json').resolve()),
                              'sha256': plan['metadata_sha256']}},
        'runs': runs,
    }


def report_benchmark(roots, profile_path, split_path, audit_path, manifest_path, data_root):
    plan = json.loads(split_path.read_text())
    validate_split(plan, data_root)
    audit = json.loads(audit_path.read_text())
    profile = json.loads(profile_path.read_text())
    if (profile['data_mode'] != 'mvtec_brats' or profile['metric_set'] != 'auroc'
            or profile['eval_seeds'] != plan['declared_seeds']
            or Path(profile['medical_split']).resolve() != split_path.resolve()
            or audit['medical_split_sha256'] != digest(split_path)):
        raise ValueError('Medical profile, declared split and audit differ')
    records = collect(roots, required_metrics=METRICS)
    for record in records.values():
        if (record['manifest'].get('medical_split_sha256') != digest(split_path)
                or record['manifest'].get('content_query_support_overlap') != 0):
            raise ValueError('Evaluation lacks the declared content-disjoint split')
    result = build_report(records, profile_path, audit_path, manifest_path, required_metrics=METRICS)
    result.pop('coverage_verified_against_official_split')
    result.pop('official_anomaly_masks_verified')
    result.pop('official_split_sha256')
    result['coverage_verified_against_converted_disk'] = True
    result['medical_split_sha256'] = digest(split_path)
    result['medical_split'] = plan
    result['content_query_support_overlap'] = 0
    result['original_test_images'] = plan['original_test_images']
    result['excluded_reference_copies'] = len(plan['excluded_queries'])
    result['metric_protocol'] = {
        'metrics': list(METRICS), 'aggregation': 'single_brain_category',
        'evaluation_size': profile['image_size'], 'gaussian_sigma': 4,
        'pixel_auroc': 'exact_sklearn_roc_auc_score_on_all_evaluated_pixels', 'std_ddof': 1,
    }
    for run in result['runs']:
        run['metrics_percent']['mean_auroc'] = statistics.mean(run['metrics_percent'][m] for m in METRICS)
    for row in result['summary']:
        values = [run['metrics_percent']['mean_auroc'] for run in result['runs'] if run['shot'] == row['shot']]
        row['statistics']['mean_auroc'] = {'mean_percent': statistics.mean(values),
                                          'std_pp': statistics.stdev(values)}
    result['limitations'] = [
        'This is the supplied 2D JPG conversion, not a raw-volume BraTS evaluation.',
        'Patient/volume identities and MRI modality provenance are unavailable; content separation does not prove patient separation.',
        'Train and test share 154 normal image contents; the fixed query set excludes copies of the union of all declared references.',
        'The historical paper query/preprocessing protocol is not established as identical.',
        'Five support draws with fixed industrial-source weights are not independent training repetitions.',
        'No clinical validation or deployment assessment is implied.',
    ]
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['prepare', 'audit', 'report'])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[111, 222, 333, 444, 555])
    parser.add_argument('--split', type=Path)
    parser.add_argument('--root', action='append', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare_split(args.data_root, args.seeds)
    elif args.action == 'audit':
        if not args.split:
            parser.error('audit requires --split')
        result = audit_split(args.split, args.data_root)
    else:
        if not all((args.root, args.config, args.split, args.audit, args.manifest)):
            parser.error('report requires --root, --config, --split, --audit and --manifest')
        result = report_benchmark(args.root, args.config, args.split, args.audit,
                                  args.manifest, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in (
        'complete', 'number_of_queries', 'disk_test_images', 'normal_queries', 'anomalous_queries',
        'original_train_test_content_overlap', 'summary') if k in result}, indent=2))
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
