"""Print a compact text summary of eval_objects results.

Meant for copying the output out of the cluster: reads agent_play_stats.csv
and agent_object_gaze.csv of each eval directory (written by
eval_objects.sh) and prints one table of scalar statistics over all runs,
then the gaze heatmap on the grid per run. Only uses the standard library.

  python3 scripts/vision_grid/summarize_evals.py \
      /sc/projects/sci-ebel/chair/Martin/CR-Atari_Logs/asterix/vision_grid/grid5x5_square24x24_EMMA_foveated_effort?*/eval_objects/seed0

As reference the humans (Atari-HEAD, cut after 5000 frames) and the first
agent eval of seed0 (checkpoint 32.9M) from results/object_gaze are printed
first, unless --no-reference is given.

Columns (means over episodes / trials; % of time):
  n        episodes or trials
  step     checkpoint step in millions
  score    game score
  min      minutes per episode
  centre   gaze on the centre square
  ent      entropy of the gaze over the squares (bits, max 4.64 for 5x5)
  sw/s     gaze square changes per second
  dwell    mean dwell time per square (ms)
  same     gaze square = avatar square
  near     gaze at most one square from the avatar
  dist     distance square centre - avatar (px)
  pl en it rw  looked at player, enemy, item, reward on the grid, as ratio
           to chance (looked_at_grid / chance_grid)
"""

import argparse
import csv
import math
import pathlib

CATEGORIES = ('player', 'enemy', 'item', 'reward')
REPO = pathlib.Path(__file__).resolve().parent.parent.parent
REFERENCE = [
    ('human 5000 frames',
     REPO / 'results/object_gaze/human/play_stats_max_frames_5000.csv',
     REPO / 'results/object_gaze/human/max_frames_5000.csv'),
    ('old seed0 32.9M',
     REPO / 'results/object_gaze/agent/seed0_step32859408_play_stats.csv',
     REPO / 'results/object_gaze/agent/seed0_step32859408.csv'),
]


def read_csv(path):
  if not path.exists():
    return []
  with open(path, newline='') as f:
    return list(csv.DictReader(f))


def mean(rows, key):
  values = []
  for row in rows:
    try:
      value = float(row[key])
    except (KeyError, TypeError, ValueError):
      continue
    if math.isfinite(value):
      values.append(value)
  return sum(values) / len(values) if values else None


def fmt(value, width, digits=1, scale=1.0):
  if value is None:
    return '-'.rjust(width)
  return f'{value * scale:{width}.{digits}f}'


def run_name(path):
  # .../<model>/eval_objects/seed0 -> <model suffix>/seed0
  parts = path.parts
  model = parts[-3] if len(parts) >= 3 else path.name
  model = model.replace('grid5x5_square24x24_EMMA_foveated_', '')
  return f'{model}/{path.name}'


def table_row(name, play, gaze):
  step = mean(play, 'checkpoint_step')
  ratios = []
  for c in CATEGORIES:
    looked = mean(gaze, f'looked_at_grid/{c}')
    chance = mean(gaze, f'chance_grid/{c}')
    ratios.append(looked / chance if looked is not None and chance else None)
  return (f'{name[:34]:34} {len(play):3d} {fmt(step, 5, 1, 1e-6)} '
          f"{fmt(mean(play, 'score'), 6, 0)} {fmt(mean(play, 'minutes'), 4, 1)} "
          f"{fmt(mean(play, 'gaze_cell/12'), 6, 1, 100)} "
          f"{fmt(mean(play, 'gaze_entropy'), 4, 2)} "
          f"{fmt(mean(play, 'gaze_switches_per_s'), 4, 1)} "
          f"{fmt(mean(play, 'gaze_dwell_ms'), 5, 0)} "
          f"{fmt(mean(play, 'gaze_avatar_same_square'), 4, 1, 100)} "
          f"{fmt(mean(play, 'gaze_avatar_near'), 4, 1, 100)} "
          f"{fmt(mean(play, 'gaze_avatar_dist_px'), 4, 0)}  "
          + ' '.join(fmt(r, 4, 1) for r in ratios))


def heatmap(name, play, cols=5, rows=5):
  lines = [f'-- {name}: gaze time per square %']
  for r in range(rows):
    lines.append('   ' + ' '.join(
        fmt(mean(play, f'gaze_cell/{r * cols + c}'), 5, 1, 100) for c in range(cols)))
  return lines


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('evals', nargs='+', type=pathlib.Path,
                      help='eval directories with agent_play_stats.csv')
  parser.add_argument('--no-reference', action='store_true',
                      help='do not print humans and the old seed0 eval')
  args = parser.parse_args()

  groups = []
  if not args.no_reference:
    for name, play_path, gaze_path in REFERENCE:
      play = read_csv(play_path)
      if play:
        groups.append((name, play, read_csv(gaze_path)))
      else:
        print(f'(reference missing: {play_path})')
  for path in args.evals:
    play = read_csv(path / 'agent_play_stats.csv')
    if not play:
      print(f'(no agent_play_stats.csv in {path})')
      continue
    groups.append((run_name(path), play, read_csv(path / 'agent_object_gaze.csv')))
  if not groups:
    return

  print(f"{'run':34} {'n':>3} {'step':>5} {'score':>6} {'min':>4} {'centre':>6} "
        f"{'ent':>4} {'sw/s':>4} {'dwell':>5} {'same':>4} {'near':>4} {'dist':>4}  "
        + ' '.join(c[:2].rjust(4) for c in CATEGORIES))
  for name, play, gaze in groups:
    print(table_row(name, play, gaze))
  print()
  for name, play, _ in groups:
    for line in heatmap(name, play):
      print(line)


if __name__ == '__main__':
  main()
