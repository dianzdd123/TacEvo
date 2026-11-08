"""Check task-specific local release files without importing the ML stack."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_MODULES = {
    'force': 'datasets/force', 'pose': 'datasets/pose/pose_loader',
    'shape': 'datasets/shape', 'grating': 'datasets/grating',
    'CIFAR-10': 'datasets/CIFAR_10',
}
NETWORK_DIRS = {
    'image_regression': 'pregens/image_regression_pregens',
    'single_label': 'pregens/single_label_pregens',
    'resnet': 'pregens/resnet_pregens',
}


def missing_components(root, task='force', network='image_regression'):
    """Return missing local modules/seed files; not a dependency or data check."""
    if task not in TASK_MODULES or network not in NETWORK_DIRS:
        raise ValueError('Unknown task or network type.')
    modules = ['pregens/init_archive_pregen', TASK_MODULES[task]]
    # resnet_training_utils imports this adapter unconditionally.
    modules.append('datasets/CIFAR_10')
    missing = []
    for module in dict.fromkeys(modules):
        base = Path(root) / module
        if not (base.with_suffix('.py').is_file() or (base / '__init__.py').is_file()):
            missing.append(module + '.py (or package with __init__.py)')
    seeds = Path(root) / NETWORK_DIRS[network]
    if not seeds.is_dir() or not any(seeds.glob('*.py')):
        missing.append(NETWORK_DIRS[network] + '/ (Python seed architectures)')
    return missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-type', choices=TASK_MODULES, default='force')
    parser.add_argument('--network-type', choices=NETWORK_DIRS, default='image_regression')
    args = parser.parse_args(argv)
    missing = missing_components(ROOT, args.task_type, args.network_type)
    for item in missing:
        print('MISSING:', item)
    print('Release incomplete; see docs/REUSE.md.' if missing else 'Local files found; dependencies, API access and experiment behaviour remain unvalidated.')
    return int(bool(missing))


if __name__ == '__main__':
    raise SystemExit(main())
