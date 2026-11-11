"""Watch a trained model play.

    python loadModel.py                         # the model in models/best
    python loadModel.py models/run1/process_0_model_12.zip
    python loadModel.py --fps 30 --games 3      # slower, and stop after three games

Close the window to stop.
"""

import argparse
import os
import sys

from stable_baselines3 import PPO

from env import SPEED, FlappyBird

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, 'models', 'best', 'model.zip')


def play_game(model_path, games=0, fps=60, flap_speed=SPEED, stochastic=False, sound=True, seed=None):
    """Play `games` games in the window (0: until the window is closed). Returns the scores."""

    # Create the Flappy Bird environment
    env = FlappyBird(render=True, sound=sound, fps=fps, flap_speed=flap_speed)
    obs, _ = env.reset(seed=seed)

    # Load the trained model, for playing only: the schedules of a model saved
    # by an older Python do not unpickle and are only needed for training.
    model = PPO.load(model_path, device='cpu',
                     custom_objects={'lr_schedule': lambda _: 0.0, 'clip_range': lambda _: 0.2})

    # Run the game with the trained model
    scores = []
    while True:
        action, _ = model.predict(obs, deterministic=not stochastic)
        obs, reward, terminated, truncated, info = env.step(int(action))

        if terminated or truncated:
            scores.append(info['score'])
            print(f'Game {len(scores)}: {info["score"]} pipes')
            if games and len(scores) >= games:
                return scores
            obs, info = env.reset()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Watch a trained model play Flappy Bird.')
    parser.add_argument('model', nargs='?', default=DEFAULT_MODEL,
                        help='a saved model (.zip). Default: models/best/model.zip')
    parser.add_argument('--games', type=int, default=0,
                        help='stop after this many games (default 0: until the window is closed)')
    parser.add_argument('--fps', type=int, default=60, help='steps a second (default 60)')
    parser.add_argument('--flap-speed', type=float, default=SPEED,
                        help=f'upward speed of a flap (default {SPEED})')
    parser.add_argument('--stochastic', action='store_true',
                        help='sample the action instead of taking the most likely one')
    parser.add_argument('--no-sound', action='store_true', help='play without sound')
    parser.add_argument('--seed', type=int, default=None, help='the same pipes every time')
    args = parser.parse_args(argv)

    if not os.path.isfile(args.model):
        parser.error(f'no such model: {args.model}')

    play_game(args.model, games=args.games, fps=args.fps, flap_speed=args.flap_speed,
              stochastic=args.stochastic, sound=not args.no_sound, seed=args.seed)
    return 0


if __name__ == '__main__':
    sys.exit(main())
