import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from check_release import missing_components

class ReleaseChecks(unittest.TestCase):
    def test_empty_release(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(len(missing_components(Path(d))), 4)

    def test_complete_minimal_fixture_and_selected_task(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ['pregens/init_archive_pregen.py', 'datasets/force/__init__.py', 'datasets/CIFAR_10.py', 'pregens/image_regression_pregens/seed.py']:
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('# fixture')
            self.assertEqual(missing_components(root), [])
            self.assertTrue(any('pose_loader' in x for x in missing_components(root, task='pose')))

    def test_empty_seed_directory(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'pregens/image_regression_pregens').mkdir(parents=True)
            self.assertTrue(any('seed architectures' in x for x in missing_components(Path(d))))

    def test_unknown_configuration(self):
        with self.assertRaises(ValueError):
            missing_components(Path('.'), task='typo')
