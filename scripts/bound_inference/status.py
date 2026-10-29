"""Print the training health of all seeds of a bound inference model.

Usage: python scripts/bound_inference/status.py [MODEL_DIR]
Default MODEL_DIR: .../CR-Atari_Logs/asterix/bounds/amortized_mn0.0_md0.0
"""

import json
import pathlib
import sys

DEFAULT = ('/sc/projects/sci-ebel/chair/Martin/CR-Atari_Logs/'
           'asterix/bounds/amortized_mn0.0_md0.0')
KEYS = ('train/loss/bounds', 'epstats/log/paused/avg', 'fps/policy')


def last_values(path, keys):
  values = {}
  if not path.exists():
    return values
  with path.open() as f:
    for line in f:
      entry = json.loads(line)
      for key in keys:
        if key in entry:
          values[key] = entry[key]
      values['step'] = entry.get('step', values.get('step'))
  return values


def recent_scores(path, n=20):
  scores = []
  if path.exists():
    with path.open() as f:
      for line in f:
        entry = json.loads(line)
        if 'episode/game_score' in entry:
          scores.append(entry['episode/game_score'])
  return scores[-n:]


def main():
  model = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
  print(model)
  for run in sorted(model.glob('seed*')):
    metrics = last_values(run / 'metrics.jsonl', KEYS)
    scores = recent_scores(run / 'metrics.jsonl')
    mean = sum(scores) / len(scores) if scores else float('nan')
    parts = [f'{run.name}: step {metrics.get("step")}',
             f'game_score (last {len(scores)} ep) {mean:.0f}']
    parts += [f'{k} {metrics[k]:.4g}' for k in KEYS if k in metrics]
    for name in ('recovery/recovery.json', 'infer_train/inference.json'):
      if (run / name).exists():
        parts.append(f'{name} exists')
    print('  ' + ' | '.join(parts))


if __name__ == '__main__':
  main()
