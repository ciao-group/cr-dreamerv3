"""Share of time a trained agent looks at each object category.

Loads the agent from the latest checkpoint of a training run, plays episodes
with the run's CrAtari config and writes one CSV row per episode, with the
same columns as scripts/human_object_gaze.py (grid variant):

  looked_at_grid/<c> = looked_at/<c>/sum / frames/sum
  chance_grid/<c>    = chance/<c>/sum / frames/sum
  on_screen/<c>      = on_screen/<c>/sum / frames/sum

It also writes agent_play_stats.csv with the avatar and gaze statistics of
embodied/core/play_stats.py (same columns as scripts/human_play_stats.py)
and the per-step trajectories in agent_trajectories.npz.

Episodes are cut after --max-frames emulator frames (default 18000 = 15 min
at 20 Hz, the length of an Atari-HEAD trial); 0 plays full episodes.

The agent part of the checkpoint is copied first, so that the running
training job can save and delete checkpoints meanwhile.

  python scripts/vision_grid/eval_objects.py --run-dir RUN_DIR --out-dir OUT
"""

import argparse
import csv
import pathlib
import pickle
import shutil
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import elements  # noqa: E402
import embodied  # noqa: E402
from embodied.core import object_gaze, objects, play_stats  # noqa: E402

KINDS = ('looked_at', 'chance', 'on_screen')
COLUMNS = {'looked_at': 'looked_at_grid', 'chance': 'chance_grid', 'on_screen': 'on_screen'}


def snapshot_checkpoint(run_dir: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
  """Copies agent.pkl and step.pkl of the latest checkpoint (without the
  replay buffer)."""
  ckpt = run_dir / 'ckpt'
  folder = ckpt / (ckpt / 'latest').read_text().strip()
  target = out_dir / 'ckpt' / folder.name
  if not (target / 'done').exists():
    target.mkdir(parents=True, exist_ok=True)
    for name in ('agent.pkl', 'step.pkl'):
      shutil.copy2(folder / name, target / name)
    (target / 'done').write_bytes(b'')
  print(f'Checkpoint: {folder}', flush=True)
  return target


STEP_KEYS = ('frames', 'reward', 'avatar_x', 'avatar_y', 'gaze')


class Episodes:
  """Sums the object logs per episode, cut at max_frames, and keeps the
  per-step trajectory.

  A transition holds the observation of an env step together with the
  action chosen on it, so the gaze square used during the step is the one
  of the previous transition of the same worker."""

  def __init__(self, max_frames: int):
    self.max_frames = max_frames
    self.running = {}
    self.prev_gaze = {}
    self.done = []

  def __call__(self, tran, worker, **kwargs):
    gaze, self.prev_gaze[worker] = self.prev_gaze.get(worker), int(tran['gaze_position'])
    if tran['is_first']:
      self.running[worker] = {'score': 0.0, 'steps': 0, 'frames': 0.0,
                              **{f'{k}/{c}': 0.0 for k in KINDS for c in objects.CATEGORIES},
                              'trajectory': {k: [] for k in STEP_KEYS}}
      return  # The reset observation covers no emulator frames.
    ep = self.running.get(worker)
    if ep is None:  # Episode already cut, wait for the next one.
      return
    traj = ep['trajectory']
    traj['frames'].append(float(tran['log/frames']))
    traj['reward'].append(float(tran['log/game_score']))
    traj['avatar_x'].append(float(tran['log/player_position_x_raw']))
    traj['avatar_y'].append(float(tran['log/player_position_y_raw']))
    traj['gaze'].append(gaze)
    ep['score'] += float(tran['log/game_score'])
    ep['steps'] += 1
    ep['frames'] += float(tran['log/frames'])
    for k in KINDS:
      for c in objects.CATEGORIES:
        ep[f'{k}/{c}'] += float(tran[f'log/{k}/{c}'])
    cut = self.max_frames and ep['frames'] >= self.max_frames
    if tran['is_last'] or cut:
      ep['cut'] = bool(cut and not tran['is_last'])
      self.done.append(ep)
      del self.running[worker]
      print(f'Episode {len(self.done)}: score {ep["score"]:.0f}, {ep["frames"]:.0f} frames'
            f'{" (cut)" if ep["cut"] else ""}', flush=True)


def write_play_stats(done, grid, game, run_dir, step, out_dir):
  """Avatar and gaze statistics per episode, and the trajectories."""
  n_squares = grid.count[0] * grid.count[1]
  rows, arrays = [], {}
  for i, ep in enumerate(done):
    t = ep['trajectory']
    traj = play_stats.agent_trajectory(t['frames'], t['reward'], t['avatar_x'], t['avatar_y'],
                                       t['gaze'], n_squares)
    rows.append({'run': str(run_dir), 'checkpoint_step': step, 'episode': i, 'rows': len(traj.time),
                 **play_stats.stats(traj, grid, game)})
    for k in STEP_KEYS:
      arrays[f'ep{i}/{k}'] = np.asarray(t[k], np.float32)
  np.savez_compressed(out_dir / 'agent_trajectories.npz', **arrays)
  path = out_dir / 'agent_play_stats.csv'
  with open(path, 'w', newline='') as f:
    writer = csv.DictWriter(f, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  print(f'Wrote {path}')
  from scripts.human_play_stats import print_summary
  print_summary(rows, grid, f'{len(rows)} agent episodes, checkpoint step {step}')


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--run-dir', required=True, help='training run with config.yaml and ckpt/')
  parser.add_argument('--out-dir', required=True)
  parser.add_argument('--episodes', type=int, default=30)
  parser.add_argument('--envs', type=int, default=10)
  parser.add_argument('--max-frames', type=int, default=18000)
  parser.add_argument('--platform', default='cuda')
  args = parser.parse_args()

  from dreamerv3 import main as dv3
  run_dir = pathlib.Path(args.run_dir)
  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  ckpt = snapshot_checkpoint(run_dir, out_dir)

  config = elements.Config.load(run_dir / 'config.yaml')
  config = config.update({
      'logdir': str(out_dir),
      'jax.platform': args.platform,
      'jax.policy_devices': [0],
      'jax.train_devices': [0],
      'run.envs': args.envs,
  })
  assert config.task.startswith('cr-atari_'), config.task

  agent = dv3.make_agent(config)
  cp = elements.Checkpoint()
  cp.agent = agent
  cp.load(str(ckpt), keys=['agent'])

  episodes = Episodes(args.max_frames)
  fns = [lambda i=i: dv3.make_env(config, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=True)
  driver.on_step(episodes)
  driver.reset(agent.init_policy)
  policy = lambda *a: agent.policy(*a, mode='eval')
  while len(episodes.done) < args.episodes:
    driver(policy, steps=100)
  driver.close()

  step = int(pickle.loads((ckpt / 'step.pkl').read_bytes()))
  env = config.env['cr-atari']
  grid = object_gaze.Grid(tuple(env.size), tuple(env.vision_square_size), tuple(env.vision_square_count))
  game = config.task.split('_', 1)[1]
  write_play_stats(episodes.done[:args.episodes], grid, game, run_dir, step, out_dir)
  rows = []
  for i, ep in enumerate(episodes.done[:args.episodes]):
    row = {'run': str(run_dir), 'checkpoint_step': step, 'episode': i, 'score': ep['score'],
           'steps': ep['steps'], 'frames': ep['frames'], 'minutes': ep['frames'] * 0.05 / 60,
           'cut': ep['cut']}
    for k in KINDS:
      for c in objects.CATEGORIES:
        row[f'{COLUMNS[k]}/{c}'] = ep[f'{k}/{c}'] / max(ep['frames'], 1.0)
    rows.append(row)
  path = out_dir / 'agent_object_gaze.csv'
  with open(path, 'w', newline='') as f:
    writer = csv.DictWriter(f, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  print(f'Wrote {path}')

  print(f'\nShare of time in % (mean ± sd over {len(rows)} episodes), checkpoint step {step}')
  print(f'{"category":9}' + ''.join(f'{col:>20}' for col in COLUMNS.values()))
  for c in objects.CATEGORIES:
    cells = []
    for col in COLUMNS.values():
      v = 100 * np.array([r[f'{col}/{c}'] for r in rows])
      cells.append(f'{v.mean():12.1f} ± {v.std():4.1f}')
    print(f'{c:9}' + ''.join(f'{s:>20}' for s in cells))


if __name__ == '__main__':
  main()
