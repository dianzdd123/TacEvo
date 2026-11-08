"""Validate the selected release before loading dependencies or calling an LLM."""
import argparse
import subprocess
import sys
from check_release import ROOT, TASK_MODULES, NETWORK_DIRS, missing_components


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-type', choices=TASK_MODULES, default='force')
    parser.add_argument('--network-type', choices=NETWORK_DIRS, default='image_regression')
    parser.add_argument('--generations', type=int, default=20)
    parser.add_argument('--mutation', default='claude')
    parser.add_argument('--net-training-epochs', type=int, default=10)
    parser.add_argument('--n-niches', type=int, default=25)
    args = parser.parse_args(argv)
    if min(args.generations, args.net_training_epochs, args.n_niches) <= 0:
        parser.error('Generation, epoch and niche counts must be positive.')
    missing = missing_components(ROOT, args.task_type, args.network_type)
    if missing:
        parser.exit(1, 'Release incomplete; no experiment started.\n' + '\n'.join('  Missing: ' + x for x in missing) + '\nSee docs/REUSE.md.\n')
    return subprocess.call([sys.executable, '-m', 'pipeline.full_pipeline', *argv], cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
