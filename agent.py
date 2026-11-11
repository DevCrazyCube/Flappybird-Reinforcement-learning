"""Train the agent with PPO (stable-baselines3).

    python agent.py                          # a new model, 4 processes, until you stop it
    python agent.py --from models/best/model.zip    # go on from a saved model
    python agent.py --name run2 --processes 2 --rounds 10
    python agent.py --window --processes 1   # watch it train (much slower)

Every process trains its own copy. After each round of --steps steps it saves
    models/<name>/process_<p>_model_<n>.zip
Stop with Ctrl+C; the models saved so far stay. Then pick the one to keep with
    python evaluate.py --dir models/<name> --top 10
"""

import argparse
import multiprocessing
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def train_model(process_num, args):
    # A window per process only when asked for; otherwise no window, no sound.
    # This has to be set before env.py starts pygame, so env is imported here.
    if not args.window:
        os.environ['FLAPPY_HEADLESS'] = '1'

    import torch
    from stable_baselines3 import PPO
    from env import FlappyBird

    models_dir = os.path.join(HERE, 'models', args.name)
    log_dir = os.path.join(HERE, 'logs', args.name, f'process_{process_num}')
    os.makedirs(models_dir, exist_ok=True)

    # TensorBoard logs only when tensorboard is installed
    try:
        import tensorboard  # noqa: F401
        os.makedirs(log_dir, exist_ok=True)
    except ImportError:
        log_dir = None

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Every process plays different games
    seed = None if args.seed is None else args.seed + process_num

    # env is the environment class inside env.py
    env = FlappyBird(render=args.window, sound=False)

    if args.start_from:
        # Go on from a saved model. The schedules of a model saved by an older
        # Python do not unpickle; they are set again from --learning-rate.
        model = PPO.load(args.start_from, env, verbose=args.verbose, device=device,
                         tensorboard_log=log_dir, seed=seed,
                         custom_objects={'learning_rate': args.learning_rate,
                                         'lr_schedule': lambda _: args.learning_rate,
                                         'clip_range': lambda _: 0.2})
        print(f'[process {process_num}] going on from {args.start_from}', flush=True)
    else:
        model = PPO('MlpPolicy', env, verbose=args.verbose, device=device,
                    tensorboard_log=log_dir, seed=seed, learning_rate=args.learning_rate)
        print(f'[process {process_num}] new model', flush=True)

    # Training loop: learn for --steps steps, save, and go on
    count = 0
    while args.rounds == 0 or count < args.rounds:
        count += 1
        model.learn(total_timesteps=args.steps, reset_num_timesteps=False, tb_log_name='PPO')
        path = os.path.join(models_dir, f'process_{process_num}_model_{count}')
        model.save(path)
        print(f'[process {process_num}] saved {os.path.relpath(path)}.zip '
              f'({model.num_timesteps:,} steps in all)', flush=True)

    env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train the Flappy Bird agent with PPO.')
    parser.add_argument('--name', default='run1',
                        help='this run: models go to models/<name>, logs to logs/<name> (default run1)')
    parser.add_argument('--from', dest='start_from', metavar='MODEL',
                        help='go on from this saved model (.zip) instead of a new one')
    parser.add_argument('--processes', type=int, default=4,
                        help='how many copies train side by side (default 4)')
    parser.add_argument('--steps', type=int, default=50000,
                        help='steps between two saves (default 50000)')
    parser.add_argument('--rounds', type=int, default=0,
                        help='stop after this many saves per process (default 0: until Ctrl+C)')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='default 0.0003')
    parser.add_argument('--seed', type=int, default=None, help='make a run repeatable')
    parser.add_argument('--device', default='auto', help='auto, cpu or cuda (default auto)')
    parser.add_argument('--window', action='store_true',
                        help='show the game while it trains (much slower)')
    parser.add_argument('--verbose', type=int, default=0, help="stable-baselines3's own output (0, 1, 2)")
    args = parser.parse_args(argv)

    if args.start_from and not os.path.isfile(args.start_from):
        parser.error(f'no such model: {args.start_from}')
    if args.processes < 1:
        parser.error('--processes must be 1 or more')

    # "spawn": every process starts clean, with its own pygame
    context = multiprocessing.get_context('spawn')
    processes = [context.Process(target=train_model, args=(i, args)) for i in range(args.processes)]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print('\nStopping. The models saved so far are kept.')
        for process in processes:
            process.terminate()
    return max((process.exitcode or 0) for process in processes)


if __name__ == '__main__':
    sys.exit(main())
