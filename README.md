# Flappy Bird, played by a PPO agent

A small neural network that learned to play Flappy Bird from five numbers, with
[stable-baselines3](https://github.com/DLR-RM/stable-baselines3) PPO on a
[Gymnasium](https://gymnasium.farama.org/) environment wrapped around a pygame Flappy Bird.

<p align="center">
  <img src="docs/play.gif" width="300" alt="The trained agent passing six pipes in a row">
</p>

<p align="center"><sub>The kept model on game seed 100, first six pipes. Made with <code>evaluate.py --frames</code>.</sub></p>

## How well it plays

Measured with `evaluate.py`, no window, the model's most likely action each step.
The score is the number of pipes passed, the number the game shows.

| Games | Seeds | Cap | Mean | Median | Lowest | Reached the cap |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 1000–1029 (not used to pick the model) | 500 pipes | **294.9** | 356 | 10 | 8 of 30 |
| 10 | 100–109 (used to pick it) | 200 pipes | 189.3 | 200 | 93 | 9 of 10 |

Read it like this:

- **It is good, not perfect.** It usually passes hundreds of pipes, and now and then it dies early
  (10 pipes in the worst of the 30 games).
- A game that reaches the cap is stopped; how long the bird would have lived is not measured.
- The model was picked from 576 saved models: all were measured on 5 games, the 22 that never
  died there on 10 more, and the best of those is `models/best/model.zip`. The first row is the
  honest one, because those 30 games played no part in the choice.
- One machine, one run of each measurement, CPU only.

## Try it

Python 3.12 (the versions this was run on are in `requirements.txt`).

```bash
git clone https://github.com/DevCrazyCube/Flappybird-Reinforcement-learning
cd Flappybird-Reinforcement-learning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

| You want to | Run |
| --- | --- |
| Watch the trained model play | `python loadModel.py` |
| Watch it slower, three games | `python loadModel.py --fps 30 --games 3` |
| Measure it without a window | `python evaluate.py --episodes 30 --seed 1000 --max-score 500` |
| Compare every model of a training run | `python evaluate.py --dir models/run1 --top 10` |
| Save one game as pictures | `python evaluate.py --frames out/frames` |
| Train a new model | `python agent.py --name run1` |
| Go on from the kept model | `python agent.py --from models/best/model.zip` |
| Play it yourself | `python flappy.py` (space to flap) |
| Run the tests | `python -m unittest discover -s tests` |

Every script explains its options with `--help`.

## How it works

```
 pygame game ──▶ env.py (Gymnasium) ──▶ PPO (stable-baselines3) ──▶ models/<run>/*.zip
                   ▲ 5 numbers, 2 actions                               │
                   └──────────── evaluate.py picks the one to keep ◀────┘
```

- **What the agent sees:** five numbers about the bird and the next pipe (positions, distances
  and speed). No pixels.
- **What it can do:** flap, or not.
- **What it is paid for:** +1 for each step alive, +10 to +20 for each pipe (more after 50 and
  100 pipes), −0.05 for a flap, −100 for dying.
- **Training:** `agent.py` runs several processes, each with its own copy of the model, and saves
  after every round. A run leaves hundreds of models behind; `evaluate.py` plays them all on the
  same seeded games and says which one to keep.

Things worth knowing before you change it:

- The observation space is declared as whole numbers 0–255 although the real values go outside
  that. It stays that way because the saved model only loads into the space it was trained on.
- The kept model was saved under an older Python. Its learning-rate schedule does not load on a
  newer one, so `evaluate.py` and `loadModel.py` replace it with a constant. That is fine for
  playing; to train on, start a fresh run or expect the learning rate you pass in.
- Only one model is in the repository now. The 576 models and the logs of the original run are
  in the git history before this change.

## Tests

`python -m unittest discover -s tests` runs 29 tests in a few seconds: the environment follows
the Gymnasium rules, the same seed gives the same game, a bird that never flaps dies, and
`evaluate.py` gives the same numbers twice and skips files that are not models.

Not tested: training itself (it takes hours), sound, and the window on Windows or macOS.

## Credits and licence

- The pygame Flappy Bird this is built on is by Leonardo Marques
  ([LeonMarqs/Flappy-bird-python](https://github.com/LeonMarqs/Flappy-bird-python)), MIT, see
  `LICENSE`.
- The sprites and sounds come from
  [zhaolingzhi/FlapPyBird-master](https://github.com/zhaolingzhi/FlapPyBird-master).
- The environment, the training, the evaluation and the tests are this repository's own work.
  The measuring script, the tests and this README were made by directing an AI coding agent
  (Claude Code).
