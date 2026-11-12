"""Share of time Atari-HEAD players looked at each object category.

Writes one CSV row per trial (= one 15 min episode in Asterix) and prints the
mean over trials. Compare with the agent's episode logs:
looked_at/<c>/sum / frames/sum (grid) and chance/<c>/sum / frames/sum.

  python scripts/human_object_gaze.py \
      --game-dir data/Atari-HEAD/asterix_raw/asterix --out human_object_gaze.csv

The grid options must match the agent's CrAtari config (size, vision square
size, vision square count). See embodied/core/object_gaze.py for the method.
"""

import argparse
import csv
import functools
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from embodied.core import object_gaze, objects  # noqa: E402

COLUMNS = ['looked_at_grid', 'chance_grid', 'looked_at_direct', 'chance_direct', 'on_screen']


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--game-dir', default='data/Atari-HEAD/asterix_raw/asterix')
  parser.add_argument('--game', default='asterix')
  parser.add_argument('--out', default='human_object_gaze.csv')
  parser.add_argument('--size', type=int, nargs=2, default=(96, 96))
  parser.add_argument('--square', type=int, nargs=2, default=(24, 24))
  parser.add_argument('--count', type=int, nargs=2, default=(5, 5))
  parser.add_argument('--workers', type=int, default=8)
  parser.add_argument('--max-frames', type=int, default=0,
                      help='cut each trial after this many game frames (0 = whole trial)')
  parser.add_argument('--max-score', type=float, default=0,
                      help='cut each trial before its score exceeds this (0 = no cut)')
  args = parser.parse_args()

  game_dir = pathlib.Path(args.game_dir)
  names = sorted(p.stem for p in game_dir.glob('*.txt') if (game_dir / p.stem).is_dir())
  missing = sorted(p.stem for p in game_dir.glob('*.txt') if not (game_dir / p.stem).is_dir())
  if missing:
    print(f'Skipping {len(missing)} trials without extracted frames: {missing}')
  grid = object_gaze.Grid(tuple(args.size), tuple(args.square), tuple(args.count))
  print(f'{len(names)} trials, grid {grid.count} with squares of {grid.square_px} screen px')

  run = functools.partial(object_gaze.trial_shares, str(game_dir), grid=grid, game=args.game,
                          max_frames=args.max_frames, max_score=args.max_score)
  with ProcessPoolExecutor(args.workers) as ex:
    rows = list(ex.map(run, names))

  keys = ['trial', 'subject', 'score', 'game_frames', 'minutes', 'frames', 'skipped_frames'] + [
      f'{col}/{c}' for c in objects.CATEGORIES for col in COLUMNS]
  with open(args.out, 'w', newline='') as f:
    writer = csv.DictWriter(f, keys)
    writer.writeheader()
    writer.writerows(rows)
  print(f'Wrote {args.out}')

  scores = np.array([r['score'] for r in rows])
  game_frames = np.array([r['game_frames'] for r in rows])
  print(f'\nScore {scores.mean():.0f} ± {scores.std():.0f}, {game_frames.mean():.0f} game frames per trial')
  print(f'Share of time in % (mean ± sd over {len(rows)} trials)')
  print(f'{"category":9}' + ''.join(f'{col:>20}' for col in COLUMNS))
  for c in objects.CATEGORIES:
    cells = []
    for col in COLUMNS:
      v = 100 * np.array([r[f'{col}/{c}'] for r in rows])
      cells.append(f'{v.mean():12.1f} ± {v.std():4.1f}')
    print(f'{c:9}' + ''.join(f'{s:>20}' for s in cells))


if __name__ == '__main__':
  main()
