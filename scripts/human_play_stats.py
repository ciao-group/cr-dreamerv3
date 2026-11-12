"""Avatar positions and gaze behaviour of Atari-HEAD players.

One CSV row per trial with the statistics of embodied/core/play_stats.py;
scripts/vision_grid/eval_objects.py writes the same columns for the agent
(agent_play_stats.csv). Use --max-frames / --max-score to compare with the
part of the game the agent reaches.

  python scripts/human_play_stats.py \
      --game-dir data/Atari-HEAD/asterix_raw/asterix --out human_play_stats.csv --max-frames 5000
"""

import argparse
import csv
import functools
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from embodied.core import object_gaze, play_stats  # noqa: E402


def trial(name, game_dir, grid, game, max_frames, max_score):
  traj = play_stats.human_trajectory(game_dir, name, grid, game, max_frames, max_score)
  return {'trial': name, 'subject': name.split('_')[1], 'rows': len(traj.time),
          **play_stats.stats(traj, grid, game)}


def print_summary(rows, grid, label):
  """Mean ± sd over rows of the scalar statistics, lane and gaze grids."""
  def col(key):
    return np.array([r[key] for r in rows], float)
  print(f'\n{label}: mean ± sd over {len(rows)}')
  for key in ('score', 'minutes', 'avatar_found', 'avatar_x_mean', 'avatar_x_sd',
              'lane_changes_per_min', 'avatar_x_speed', 'gaze_entropy', 'gaze_switches_per_s',
              'gaze_dwell_ms', 'gaze_avatar_same_square', 'gaze_avatar_near', 'gaze_avatar_dist_px'):
    v = col(key)
    print(f'  {key:26}{np.nanmean(v):10.3f} ± {np.nanstd(v):.3f}')
  lanes = sorted(k for k in rows[0] if k.startswith('lane/'))
  print('  lane time %  ' + ' '.join(f'{100 * col(k).mean():5.1f}' for k in lanes))
  drow = sorted((k for k in rows[0] if k.startswith('gaze_avatar_drow/')), key=lambda k: int(k.split('/')[1]))
  print('  gaze row - avatar row %  ' + ' '.join(f'{k.split("/")[1]}:{100 * col(k).mean():.1f}' for k in drow))
  dcol = sorted((k for k in rows[0] if k.startswith('gaze_avatar_dcol/')), key=lambda k: int(k.split('/')[1]))
  print('  gaze col - avatar col %  ' + ' '.join(f'{k.split("/")[1]}:{100 * col(k).mean():.1f}' for k in dcol))
  cols, nrows = grid.count
  print('  gaze time per square %')
  for r in range(nrows):
    print('    ' + ' '.join(f'{100 * col(f"gaze_cell/{r * cols + c}").mean():5.1f}' for c in range(cols)))


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--game-dir', default='data/Atari-HEAD/asterix_raw/asterix')
  parser.add_argument('--game', default='asterix')
  parser.add_argument('--out', default='human_play_stats.csv')
  parser.add_argument('--size', type=int, nargs=2, default=(96, 96))
  parser.add_argument('--square', type=int, nargs=2, default=(24, 24))
  parser.add_argument('--count', type=int, nargs=2, default=(5, 5))
  parser.add_argument('--workers', type=int, default=8)
  parser.add_argument('--max-frames', type=int, default=0)
  parser.add_argument('--max-score', type=float, default=0)
  args = parser.parse_args()

  game_dir = pathlib.Path(args.game_dir)
  names = sorted(p.stem for p in game_dir.glob('*.txt') if (game_dir / p.stem).is_dir())
  grid = object_gaze.Grid(tuple(args.size), tuple(args.square), tuple(args.count))
  run = functools.partial(trial, game_dir=str(game_dir), grid=grid, game=args.game,
                          max_frames=args.max_frames, max_score=args.max_score)
  with ProcessPoolExecutor(args.workers) as ex:
    rows = list(ex.map(run, names))
  with open(args.out, 'w', newline='') as f:
    writer = csv.DictWriter(f, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  print(f'Wrote {args.out}')
  print_summary(rows, grid, f'{len(rows)} trials')


if __name__ == '__main__':
  main()
