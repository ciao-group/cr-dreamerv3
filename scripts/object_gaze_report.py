"""Comparison table and gaze heatmaps of humans and agent.

Reads the CSVs in results/object_gaze (see its README) and writes there:

  comparison.csv     one row per group (human cut, agent seed, agent mean)
                     and category: share of time looked at (grid), chance,
                     ratio, on screen; mean and sd over trials/episodes.
  gaze_heatmaps.png  where the gaze was: humans on screen pixels and on the
                     agent's grid, agent on the grid (when agent_play_stats
                     CSVs exist). Human maps are the mean over trials.
  gaze_grid.csv      time share per grid square for every map on the grid.

  python scripts/object_gaze_report.py --human-cut max_frames_5000 --max-frames 5000
"""

import argparse
import csv
import glob
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from embodied.core import object_gaze, objects  # noqa: E402


def read_csv(path):
  with open(path) as f:
    return list(csv.DictReader(f))


def column(rows, key):
  return np.array([float(r[key]) for r in rows])


def group_rows(name, rows):
  """Rows of comparison.csv for one group of trials or episodes."""
  out = []
  score = column(rows, 'score') if 'score' in rows[0] else np.full(len(rows), np.nan)
  frames = column(rows, 'game_frames' if 'game_frames' in rows[0] else 'frames')
  for c in objects.CATEGORIES:
    look, chance = column(rows, f'looked_at_grid/{c}'), column(rows, f'chance_grid/{c}')
    screen = column(rows, f'on_screen/{c}')
    out.append({
        'group': name, 'n': len(rows), 'score_mean': score.mean(), 'frames_mean': frames.mean(),
        'category': c,
        'looked_at_mean': look.mean(), 'looked_at_sd': look.std(),
        'looked_at_min': look.min(), 'looked_at_max': look.max(),
        'chance_mean': chance.mean(), 'ratio_mean': (look / chance).mean(),
        'on_screen_mean': screen.mean()})
  return out


def human_pixel_heatmap(game_dir, max_frames, max_score, bin_px=2, blur_px=4.0):
  """Share of time per screen pixel bin, mean over trials (every trial
  weighs the same). A sample weighs duration / samples of its frame."""
  import cv2
  maps = []
  shape = (object_gaze.SCREEN[1] // bin_px, object_gaze.SCREEN[0] // bin_px)
  for txt in sorted(pathlib.Path(game_dir).glob('*.txt')):
    ids, durations, rewards, gaze = object_gaze.read_trial(str(txt))
    end = len(ids)
    if max_frames:
      end = min(end, max_frames)
    if max_score:
      over = np.nonzero(np.cumsum(rewards) > max_score)[0]
      end = min(end, int(over[0])) if len(over) else end
    heat = np.zeros(shape)
    for dur, g in zip(durations[:end], gaze[:end]):
      if len(g) and np.isfinite(dur):
        np.add.at(heat, ((g[:, 1] // bin_px).astype(int), (g[:, 0] // bin_px).astype(int)), dur / len(g))
    heat = cv2.GaussianBlur(heat, (0, 0), blur_px / bin_px)
    maps.append(heat / heat.sum())
  return np.mean(maps, 0)


def grid_mean(paths):
  """Mean time share per grid square over all rows of play_stats CSVs."""
  rows = [r for p in paths for r in read_csv(p)]
  n = sum(1 for k in rows[0] if k.startswith('gaze_cell/'))
  return np.array([column(rows, f'gaze_cell/{k}').mean() for k in range(n)]), len(rows)


def grid_image(shares, grid):
  """Screen image where every pixel shows the share of the square whose
  centre is nearest, as the gaze is mapped to squares."""
  ys, xs = np.mgrid[0:object_gaze.SCREEN[1], 0:object_gaze.SCREEN[0]]
  cells = grid.cells(np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], -1))
  return shares[cells].reshape(xs.shape)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--results', default='results/object_gaze')
  parser.add_argument('--game-dir', default='data/Atari-HEAD/asterix_raw/asterix')
  parser.add_argument('--human-cut', default='max_frames_5000',
                      help='suffix of the human CSVs to compare with (full_trials, max_frames_5000, ...)')
  parser.add_argument('--max-frames', type=int, default=5000, help='same cut for the pixel heatmap')
  parser.add_argument('--max-score', type=float, default=0)
  parser.add_argument('--size', type=int, nargs=2, default=(96, 96))
  parser.add_argument('--square', type=int, nargs=2, default=(24, 24))
  parser.add_argument('--count', type=int, nargs=2, default=(5, 5))
  args = parser.parse_args()
  res = pathlib.Path(args.results)
  grid = object_gaze.Grid(tuple(args.size), tuple(args.square), tuple(args.count))

  # Comparison table.
  table = []
  for path in sorted((res / 'human').glob('*.csv')):
    if not path.name.startswith('play_stats'):
      table += group_rows(f'human_{path.stem}', read_csv(path))
  agent_paths = sorted((res / 'agent').glob('seed*.csv'))
  for path in agent_paths:
    table += group_rows(f'agent_{path.stem}', read_csv(path))
  if agent_paths:
    table += group_rows('agent_all_seeds', [r for p in agent_paths for r in read_csv(p)])
  with open(res / 'comparison.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, list(table[0]))
    writer.writeheader()
    writer.writerows({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()} for r in table)
  print(f'Wrote {res / "comparison.csv"}')

  # Heatmaps.
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  maps = []  # (title, image, grid shares or None)
  pixel = human_pixel_heatmap(args.game_dir, args.max_frames, args.max_score)
  maps.append((f'Mensch, Pixel\n(15 Trials, {args.human_cut})', pixel, None))
  human_grid, n = grid_mean([res / 'human' / f'play_stats_{args.human_cut}.csv'])
  maps.append((f'Mensch, Grid\n({n} Trials, {args.human_cut})', grid_image(human_grid, grid), human_grid))
  seeds = sorted(glob.glob(str(res / 'agent' / '*play_stats*.csv')))
  for path in seeds:
    shares, n = grid_mean([path])
    maps.append((f'Agent {pathlib.Path(path).stem}\n({n} Episoden)', grid_image(shares, grid), shares))
  if len(seeds) > 1:
    shares, n = grid_mean(seeds)
    maps.append((f'Agent alle Seeds\n({n} Episoden)', grid_image(shares, grid), shares))

  with open(res / 'gaze_grid.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['map', 'square', 'row', 'col', 'share'])
    for title, _, shares in maps:
      if shares is not None:
        for k, v in enumerate(shares):
          writer.writerow([title.replace('\n', ' '), k, k // grid.count[0], k % grid.count[0], f'{v:.5f}'])
  print(f'Wrote {res / "gaze_grid.csv"}')

  fig, axes = plt.subplots(1, len(maps), figsize=(3.2 * len(maps), 5.0), squeeze=False)
  extent = (0, object_gaze.SCREEN[0], object_gaze.SCREEN[1], 0)
  for ax, (title, image, shares) in zip(axes[0], maps):
    ax.imshow(image, cmap='Blues', extent=extent, interpolation='nearest', vmin=0)
    ax.axhspan(*objects.GAMES['asterix'].playfield, fill=False, ec='0.5', ls='--', lw=0.8)
    if shares is not None:
      cols, rows = grid.count
      sq = np.array(grid.squares(), float)
      centres = sq[:, :2] + sq[:, 2:] / 2
      for k, v in enumerate(shares):
        ink = 'white' if v > 0.6 * shares.max() else '0.15'
        ax.text(*centres[k], f'{100 * v:.0f}', ha='center', va='center', fontsize=7, color=ink)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
  fig.suptitle('Blickzeit pro Ort\nZahlen: % der Zeit pro Grid-Quadrat; gestrichelt: Spielfeld',
               fontsize=9)
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  fig.savefig(res / 'gaze_heatmaps.png', dpi=150)
  print(f'Wrote {res / "gaze_heatmaps.png"}')


if __name__ == '__main__':
  main()
