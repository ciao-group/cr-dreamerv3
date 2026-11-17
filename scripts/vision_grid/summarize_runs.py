"""Print a compact text summary of vision grid training runs.

Meant for copying the output out of the cluster: reads metrics.jsonl of each
run directory, averages the episode metrics in bins of agent steps and prints
one small table per run. Only uses the standard library.

  python scripts/vision_grid/summarize_runs.py \
      /sc/projects/sci-ebel/chair/Martin/CR-Atari_Logs/asterix/vision_grid/grid5x5_square24x24_EMMA_foveated_effort*/seed0

Columns per bin (means over the episodes that ended in the bin):
  step     upper end of the bin in millions of steps
  eps      episodes
  game     game score (without penalties)
  rew/st   training reward per agent step (with penalties)
  len      episode length in agent steps
  eff/st   effort penalty per step (log/effort_penalty/avg)
  sac/st   saccade penalty per step (log/saccade_penalty/avg)
  pl en it rw  % of episode time the gaze square overlapped player, enemy,
           item, reward (looked_at/<c>/sum / frames/sum), chance in a
           separate line
"""

import argparse
import json
import math
import pathlib

CATEGORIES = ('player', 'enemy', 'item', 'reward')


def read_metrics(path, max_step):
  rows = []
  with open(path) as f:
    for line in f:
      try:
        row = json.loads(line)
      except json.JSONDecodeError:
        continue
      if 'episode/length' not in row:
        continue
      if max_step and row.get('step', 0) > max_step:
        break
      rows.append(row)
  return rows


def mean(values):
  values = [v for v in values if v is not None and math.isfinite(v)]
  return sum(values) / len(values) if values else None


def ratio(rows, num, den):
  # Ratio of sums over the logged rows, in percent.
  pairs = [(r.get(num), r.get(den)) for r in rows]
  pairs = [(a, b) for a, b in pairs if a is not None and b]
  if not pairs:
    return None
  return 100 * sum(a for a, _ in pairs) / sum(b for _, b in pairs)


def fmt(value, width, digits=1):
  if value is None:
    return '-'.rjust(width)
  return f'{value:{width}.{digits}f}'


def summarize(rows, bin_size):
  bins = {}
  for row in rows:
    index = int(math.ceil(row.get('step', 0) / bin_size))
    bins.setdefault(index, []).append(row)
  lines = []
  header = (f"{'step':>5} {'eps':>4} {'game':>7} {'rew/st':>6} {'len':>6} "
            f"{'eff/st':>6} {'sac/st':>6}  " + ' '.join(
                c[:2].rjust(4) for c in CATEGORIES))
  lines.append(header)
  for index in sorted(bins):
    group = bins[index]
    lengths = [r.get('episode/length') for r in group]
    scores = [r.get('episode/score') for r in group]
    per_step = [s / l for s, l in zip(scores, lengths)
                if s is not None and l]
    looked = [ratio(group, f'epstats/log/looked_at/{c}/sum',
                    'epstats/log/frames/sum') for c in CATEGORIES]
    chance = [ratio(group, f'epstats/log/chance/{c}/sum',
                    'epstats/log/frames/sum') for c in CATEGORIES]
    lines.append(
        f'{index * bin_size / 1e6:5.1f} {len(group):4d} '
        f"{fmt(mean([r.get('episode/game_score') for r in group]), 7, 0)} "
        f'{fmt(mean(per_step), 6, 2)} {fmt(mean(lengths), 6, 0)} '
        f"{fmt(mean([r.get('epstats/log/effort_penalty/avg') for r in group]), 6, 3)} "
        f"{fmt(mean([r.get('epstats/log/saccade_penalty/avg') for r in group]), 6, 3)}  "
        + ' '.join(fmt(v, 4, 0) for v in looked)
        + '  chance ' + ' '.join(fmt(v, 3, 0) for v in chance))
  return lines


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('runs', nargs='+', type=pathlib.Path,
                      help='run directories containing metrics.jsonl')
  parser.add_argument('--bin', type=float, default=1e6,
                      help='bin size in steps (1e6)')
  parser.add_argument('--max-step', type=float, default=0,
                      help='ignore rows after this step, 0 = all')
  args = parser.parse_args()
  for run in args.runs:
    path = run / 'metrics.jsonl'
    print(f'== {run.parent.name}/{run.name}')
    if not path.exists():
      print('   no metrics.jsonl')
      continue
    rows = read_metrics(path, args.max_step)
    if not rows:
      print('   no episodes logged')
      continue
    for line in summarize(rows, args.bin):
      print(line)


if __name__ == '__main__':
  main()
