"""Tests of the game environment. No window, no sound, no model.

    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

# No window and no sound: this has to be set before env.py starts pygame.
os.environ['FLAPPY_HEADLESS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import env as game  # noqa: E402
from env import FlappyBird  # noqa: E402


def play(environment, policy, seed, most_steps=3000):
    """Play one game with policy(observation) -> 0 or 1. Returns (observations, rewards, last step)."""
    obs, _ = environment.reset(seed=seed)
    observations, rewards = [obs], []
    for _ in range(most_steps):
        obs, reward, terminated, truncated, info = environment.step(policy(obs))
        observations.append(obs)
        rewards.append(reward)
        if terminated or truncated:
            return observations, rewards, (terminated, truncated, info)
    return observations, rewards, (False, False, info)


def never(_):
    return 0


def always(_):
    return 1


def by_hand(obs):
    """Flap when the bird has sunk to 30 pixels above the lower pipe. Passes a few pipes, no more."""
    return 1 if obs[0] < 30 else 0


class Reset(unittest.TestCase):
    def setUp(self):
        self.env = FlappyBird(render=False, sound=False)

    def test_first_observation_is_five_numbers(self):
        obs, info = self.env.reset(seed=0)
        self.assertEqual(np.asarray(obs).shape, (5,))
        self.assertEqual(info, {'score': 0})

    def test_observation_matches_the_declared_shape(self):
        obs, _ = self.env.reset(seed=0)
        self.assertEqual(np.asarray(obs).shape, self.env.observation_space.shape)

    def test_two_actions(self):
        self.assertEqual(self.env.action_space.n, 2)

    def test_reset_puts_the_score_back(self):
        play(self.env, by_hand, seed=0, most_steps=200)
        _, info = self.env.reset(seed=0)
        self.assertEqual(info['score'], 0)
        self.assertEqual(self.env.pipes_passed, 0)

    def test_bird_starts_at_a_sixth_of_the_width(self):
        self.env.reset(seed=0)
        self.assertEqual(self.env.bird.rect[0], int(game.SCREEN_WIDTH / 6))


class Seeds(unittest.TestCase):
    def setUp(self):
        self.env = FlappyBird(render=False, sound=False)

    def test_same_seed_same_game(self):
        first, rewards_1, _ = play(self.env, by_hand, seed=7, most_steps=400)
        second, rewards_2, _ = play(self.env, by_hand, seed=7, most_steps=400)
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(rewards_1, rewards_2)

    def test_other_seed_other_pipes(self):
        heights = set()
        for seed in range(6):
            self.env.reset(seed=seed)
            heights.add(self.env.pipe_group.sprites()[0].rect[1])
        self.assertGreater(len(heights), 1)

    def test_a_seed_also_fixes_the_games_after_it(self):
        def pipes_of_three_games():
            found = []
            self.env.reset(seed=3)
            for _ in range(3):
                found.append(self.env.pipe_group.sprites()[0].rect[1])
                self.env.reset()
            return found
        self.assertEqual(pipes_of_three_games(), pipes_of_three_games())


class Dying(unittest.TestCase):
    def setUp(self):
        self.env = FlappyBird(render=False, sound=False)

    def test_never_flapping_ends_on_the_ground(self):
        _, rewards, (terminated, truncated, info) = play(self.env, never, seed=0, most_steps=200)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(rewards[-1], -100)
        self.assertEqual(info['score'], 0)
        self.assertGreaterEqual(self.env.bird.rect.bottom, game.SCREEN_HEIGHT - game.GROUND_HEIGHT)

    def test_always_flapping_ends_at_the_ceiling(self):
        _, rewards, (terminated, truncated, _) = play(self.env, always, seed=0, most_steps=200)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(rewards[-1], -100)
        self.assertLessEqual(self.env.bird.rect.top, 0)

    def test_a_step_alive_is_worth_one_point(self):
        self.env.reset(seed=0)
        _, reward, terminated, _, _ = self.env.step(0)
        self.assertFalse(terminated)
        self.assertEqual(reward, 1)

    def test_a_flap_costs_a_little_on_the_next_step(self):
        self.env.reset(seed=0)
        self.env.step(1)
        _, reward, _, _, _ = self.env.step(0)
        self.assertAlmostEqual(reward, 0.95)


class Scoring(unittest.TestCase):
    def setUp(self):
        self.env = FlappyBird(render=False, sound=False)

    def test_a_hand_written_rule_passes_pipes(self):
        _, rewards, (_, _, info) = play(self.env, by_hand, seed=0, most_steps=600)
        self.assertGreaterEqual(info['score'], 1)
        # The step a pipe is passed pays 1 (alive) + 10 + 10, with or without the flap penalty
        self.assertTrue(any(reward in (21, 20.95) for reward in rewards))

    def test_the_score_is_the_number_of_passed_pipe_pairs(self):
        obs, _ = self.env.reset(seed=0)
        seen = 0
        for _ in range(600):
            obs, reward, terminated, truncated, info = self.env.step(by_hand(obs))
            if reward >= 20:
                seen += 1
            self.assertEqual(info['score'], seen)
            if terminated or truncated:
                break


class FlapSpeed(unittest.TestCase):
    def test_default_is_the_constant(self):
        environment = FlappyBird(render=False, sound=False)
        environment.reset(seed=0)
        environment.step(1)
        self.assertEqual(environment.bird.speed, -game.SPEED)

    def test_it_can_be_set(self):
        environment = FlappyBird(render=False, sound=False, flap_speed=20)
        environment.reset(seed=0)
        environment.step(1)
        self.assertEqual(environment.bird.speed, -20)

    def test_it_survives_a_reset(self):
        environment = FlappyBird(render=False, sound=False, flap_speed=20)
        environment.reset(seed=0)
        environment.reset(seed=1)
        environment.step(1)
        self.assertEqual(environment.bird.speed, -20)


class Headless(unittest.TestCase):
    def test_no_drawing_per_step(self):
        self.assertTrue(game.HEADLESS)
        environment = FlappyBird(render=True, sound=True)
        self.assertFalse(environment.render_every_step)

    def test_a_frame_can_still_be_drawn(self):
        environment = FlappyBird(render=False, sound=False)
        environment.reset(seed=0)
        environment.draw()
        self.assertEqual(environment.screen.get_size(), (game.SCREEN_WIDTH, game.SCREEN_HEIGHT))


if __name__ == '__main__':
    unittest.main()
