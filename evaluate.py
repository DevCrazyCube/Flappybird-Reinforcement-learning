"""Measure how well saved models play, without a window.

    python evaluate.py                                   # the model in models/best
    python evaluate.py models/best/model.zip --episodes 50
    python evaluate.py a.zip b.zip --max-score 200       # compare models
    python evaluate.py --dir models/run1 --top 10        # every model in a folder, best first
    python evaluate.py --frames out/frames               # save one run as pictures

Every model plays the same games: game i uses seed (--seed + i), so the pipes
are the same for each model and for each time you run this. A game stops when
the bird dies or when it has passed --max-score pipes (the cap). A model that
reaches the cap did not die; how long it would have gone on is not measured.

The score is the number of pipes passed, the number the game shows on screen.
"""

import argparse
import os
import re
import statistics
import sys
import time

# No window and no sound: this has to be set before env.py starts pygame.
os.environ['FLAPPY_HEADLESS'] = '1'
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import pygame  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from env import SPEED, FlappyBird  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, 'models', 'best', 'model.zip')


def load(model_path):
    """Load a saved model for playing only.

    The models were saved by an older Python; their learning-rate and
    clip-range schedules do not unpickle on a newer one. They are only needed
    for training, so they are replaced by constants here.
    """
    return PPO.load(model_path, device='cpu',
                    custom_objects={'lr_schedule': lambda _: 0.0, 'clip_range': lambda _: 0.2})


def play_one(model, env, seed, max_score, deterministic, frames_dir=None, max_frames=0):
    """Play one game. Returns (score, steps, reached_the_cap)."""
    obs, _ = env.reset(seed=seed)
    steps = 0
    frame = 0
    while True:
        if frames_dir and frame < max_frames:
            env.draw()
            pygame.image.save(env.screen, os.path.join(frames_dir, f'{frame:05d}.png'))
            frame += 1
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(int(action))
        steps += 1
        if terminated or truncated:
            return info['score'], steps, False
        if info['score'] >= max_score:
            return info['score'], steps, True


def natural(path):
    """Sort key that puts model_2 before model_10."""
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', path)]


def evaluate(model, env, episodes, seed, max_score, deterministic):
    """Let a loaded model play `episodes` games. Returns (scores, games at the cap, steps)."""
    if deterministic is False:
        # The stochastic policy draws from torch's generator; fix it so two
        # runs of this script give the same numbers.
        model.set_random_seed(seed)
    scores, capped, steps = [], 0, 0
    for i in range(episodes):
        score, n, cap = play_one(model, env, seed + i, max_score, deterministic)
        scores.append(score)
        capped += cap
        steps += n
    return scores, capped, steps


def main(argv=None):
    parser = argparse.ArgumentParser(description='Measure how well saved models play.')
    parser.add_argument('models', nargs='*', default=[DEFAULT_MODEL],
                        help='saved models (.zip). Default: models/best/model.zip')
    parser.add_argument('--dir', metavar='DIR', help='evaluate every .zip in DIR instead')
    parser.add_argument('--top', type=int, default=0,
                        help='afterwards, list the N best by mean score')
    parser.add_argument('--episodes', type=int, default=20, help='games per model (default 20)')
    parser.add_argument('--max-score', type=int, default=100,
                        help='stop a game after this many pipes (default 100)')
    parser.add_argument('--seed', type=int, default=0, help='seed of the first game (default 0)')
    parser.add_argument('--flap-speed', type=float, default=SPEED,
                        help=f'upward speed of a flap (default {SPEED}, the game as it is now)')
    parser.add_argument('--stochastic', action='store_true',
                        help='sample the action instead of taking the most likely one')
    parser.add_argument('--frames', metavar='DIR',
                        help='save the first game of the first model as PNG frames in DIR')
    parser.add_argument('--max-frames', type=int, default=600,
                        help='at most this many frames with --frames (default 600)')
    args = parser.parse_args(argv)

    if args.dir:
        if not os.path.isdir(args.dir):
            parser.error(f'no such folder: {args.dir}')
        args.models = sorted((os.path.join(args.dir, name) for name in os.listdir(args.dir)
                              if name.endswith('.zip')), key=natural)
        if not args.models:
            parser.error(f'no .zip models in {args.dir}')
    for path in args.models:
        if not os.path.isfile(path):
            parser.error(f'no such model: {path}')

    env = FlappyBird(render=False, sound=False, flap_speed=args.flap_speed)
    deterministic = not args.stochastic

    print(f'{args.episodes} games per model, seeds {args.seed}..{args.seed + args.episodes - 1}, '
          f'cap {args.max_score} pipes, flap speed {args.flap_speed:g}, '
          f'{"most likely action" if deterministic else "sampled action"}')
    print(f'{"model":<34} {"mean":>7} {"median":>7} {"min":>5} {"max":>5} {"at cap":>8} {"steps/s":>8}')

    results = []
    skipped = []
    for path in args.models:
        name = os.path.relpath(path)
        if len(name) > 34:
            name = '...' + name[-31:]
        try:
            model = load(path)
        except Exception as error:  # an empty or damaged file: say so and go on
            print(f'{name:<34} not a readable model, skipped ({type(error).__name__})', flush=True)
            skipped.append(path)
            continue
        started = time.perf_counter()
        scores, capped, steps = evaluate(model, env, args.episodes, args.seed, args.max_score, deterministic)
        seconds = time.perf_counter() - started
        print(f'{name:<34} {statistics.mean(scores):>7.1f} {statistics.median(scores):>7.1f} '
              f'{min(scores):>5} {max(scores):>5} {capped:>4}/{args.episodes:<3} '
              f'{steps / seconds:>8.0f}', flush=True)
        results.append((path, scores, capped))

    if skipped:
        print(f'\n{len(skipped)} file(s) skipped: not a readable model.')

    if args.top and results:
        print(f'\nBest {min(args.top, len(results))} by mean score:')
        best = sorted(results, key=lambda r: (-statistics.mean(r[1]), -min(r[1])))[:args.top]
        for path, scores, capped in best:
            print(f'  {statistics.mean(scores):>7.1f}  min {min(scores):>3}  '
                  f'at cap {capped}/{len(scores)}  {os.path.relpath(path)}')

    if args.frames and results:
        os.makedirs(args.frames, exist_ok=True)
        model = load(results[0][0])
        if not deterministic:
            model.set_random_seed(args.seed)
        score, steps, _ = play_one(model, env, args.seed, args.max_score, deterministic,
                                   frames_dir=args.frames, max_frames=args.max_frames)
        print(f'Saved {min(steps, args.max_frames)} frames of game {args.seed} '
              f'(score {score}) in {args.frames}')

    return results, skipped


if __name__ == '__main__':
    results, skipped = main()
    # 0: every model was measured. 1: a file was not a readable model.
    sys.exit(1 if skipped else 0)
