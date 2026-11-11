"""Tests of evaluate.py with the model kept in models/best.

    python -m unittest discover -s tests -v
"""

import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest

os.environ['FLAPPY_HEADLESS'] = '1'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import evaluate  # noqa: E402
from env import FlappyBird  # noqa: E402

BEST = os.path.join(ROOT, 'models', 'best', 'model.zip')


def run(argv):
    """Run evaluate.main quietly. Returns (results, skipped, what it printed)."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        results, skipped = evaluate.main(argv)
    return results, skipped, printed.getvalue()


class NaturalOrder(unittest.TestCase):
    def test_two_comes_before_ten(self):
        names = ['process_0_model_10.zip', 'process_0_model_2.zip', 'process_1_model_1.zip',
                 'process_0_model_1.zip']
        self.assertEqual(sorted(names, key=evaluate.natural),
                         ['process_0_model_1.zip', 'process_0_model_2.zip',
                          'process_0_model_10.zip', 'process_1_model_1.zip'])


class TheKeptModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = evaluate.load(BEST)
        cls.env = FlappyBird(render=False, sound=False)

    def test_it_is_there(self):
        self.assertTrue(os.path.isfile(BEST))

    def test_it_passes_the_first_five_pipes_of_three_games(self):
        scores, capped, steps = evaluate.evaluate(self.model, self.env, episodes=3, seed=0,
                                                  max_score=5, deterministic=True)
        self.assertEqual(scores, [5, 5, 5])
        self.assertEqual(capped, 3)
        self.assertGreater(steps, 0)

    def test_the_same_seed_gives_the_same_numbers(self):
        first = evaluate.evaluate(self.model, self.env, 2, 11, 8, True)
        second = evaluate.evaluate(self.model, self.env, 2, 11, 8, True)
        self.assertEqual(first, second)

    def test_a_game_stops_at_the_cap(self):
        score, steps, capped = evaluate.play_one(self.model, self.env, seed=0, max_score=2,
                                                 deterministic=True)
        self.assertEqual((score, capped), (2, True))

    def test_frames_are_saved_as_pictures(self):
        with tempfile.TemporaryDirectory() as folder:
            evaluate.play_one(self.model, self.env, seed=0, max_score=2, deterministic=True,
                              frames_dir=folder, max_frames=5)
            self.assertEqual(sorted(os.listdir(folder)),
                             ['00000.png', '00001.png', '00002.png', '00003.png', '00004.png'])


class CommandLine(unittest.TestCase):
    def test_a_folder_with_an_empty_file(self):
        with tempfile.TemporaryDirectory() as folder:
            shutil.copy(BEST, os.path.join(folder, 'process_0_model_2.zip'))
            open(os.path.join(folder, 'process_0_model_10.zip'), 'w').close()  # 0 bytes
            results, skipped, printed = run(['--dir', folder, '--episodes', '1',
                                             '--max-score', '2', '--top', '5'])
        self.assertEqual(len(results), 1)
        self.assertEqual(len(skipped), 1)
        self.assertTrue(skipped[0].endswith('process_0_model_10.zip'))
        self.assertIn('not a readable model, skipped', printed)
        self.assertIn('Best 1 by mean score', printed)

    def test_the_default_is_the_kept_model(self):
        results, skipped, printed = run(['--episodes', '1', '--max-score', '2'])
        self.assertEqual(skipped, [])
        self.assertEqual(results[0][0], BEST)
        self.assertEqual(results[0][1], [2])
        self.assertIn('1 games per model, seeds 0..0, cap 2 pipes', printed)

    def test_a_missing_model_is_an_error(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stop:
            evaluate.main(['no-such-model.zip'])
        self.assertEqual(stop.exception.code, 2)

    def test_an_empty_folder_is_an_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stop:
                evaluate.main(['--dir', folder])
        self.assertEqual(stop.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
