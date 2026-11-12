"""Share of time human players looked at each object category (Atari-HEAD).

The counterpart of the looked_at logs of CrAtari: for every frame the objects
are detected in the PNG (embodied/core/objects.py), and every gaze sample of
the frame is tested against them in two ways:

grid: the sample is mapped to the agent's vision square with the nearest
  center, and the category counts as looked at when that square overlaps one
  of its objects, the same test as for the agent.
direct: the sample lies in an object box grown by 1 visual degree on every
  side, to allow for eye tracker error.

A frame was shown for duration(ms) and its gaze samples are spread evenly over
that time, so the time on a category is duration x the share of the frame's
samples on it. Summed over frames and divided by the summed duration this is
the share of time, like looked_at/<c>/sum / frames/sum of the agent. Pauses
and decision time count, as the frame was on screen.

chance is the share for a gaze drawn uniformly: over the grid squares (grid)
or over the screen pixels (direct).
"""

import dataclasses
import os

import numpy as np

from embodied.core import objects, vision

SCREEN = (160, 210)  # (W, H) of the ALE screen and the Atari-HEAD frames
VISUAL_DEGREES = (44.6, 28.5)  # (W, H) of the screen, as in CrAtari
ONE_DEGREE = (SCREEN[0] / VISUAL_DEGREES[0], SCREEN[1] / VISUAL_DEGREES[1])


@dataclasses.dataclass(frozen=True)
class Grid:
  """The agent's vision grid: observation size, vision square size in
  observation pixels, and number of squares, as in the CrAtari config."""
  size: tuple[int, int] = (96, 96)
  square: tuple[int, int] = (24, 24)
  count: tuple[int, int] = (5, 5)

  @property
  def square_px(self) -> tuple[int, int]:
    """Vision square on the screen, computed as in CrAtari."""
    return (self.square[0] * SCREEN[0] // self.size[0],
            self.square[1] * SCREEN[1] // self.size[1])

  def squares(self) -> list[tuple[int, int, int, int]]:
    w, h = self.square_px
    return [(*vision.convert_1d_vision_square_position_to_2d(k, self.count, SCREEN, (w, h)), w, h)
            for k in range(self.count[0] * self.count[1])]

  def cells(self, xy: np.ndarray) -> np.ndarray:
    """Vision square with the nearest center for every gaze sample (n, 2),
    vectorised version of vision.convert_2d_gaze_position_to_1d_..."""
    axes = []
    for a in (0, 1):
      centers = np.array([vision.vision_square_offset(i, self.count[a], self.square_px[a], SCREEN[a])
                          + self.square_px[a] / 2 for i in range(self.count[a])])
      axes.append(np.argmin(np.abs(xy[:, a, None] - centers[None]), axis=1))
    return axes[1] * self.count[0] + axes[0]


def frame_shares(found, gaze: np.ndarray, grid: Grid, squares=None,
                 margin=ONE_DEGREE) -> dict[str, float]:
  """Per category: share of the frame's gaze samples on it (grid and direct),
  whether it is on screen, and the chance levels.

  found: detected objects; gaze: (n, 2) samples in screen pixels, n > 0."""
  squares = grid.squares() if squares is None else squares
  hits = objects.square_hits(found, squares)
  cells = grid.cells(gaze)
  out = {}
  for category in objects.CATEGORIES:
    boxes = [o for o in found if o.category == category]
    inside = np.zeros(len(gaze), bool)
    covered = np.zeros((SCREEN[1], SCREEN[0]), bool)
    for o in boxes:
      x0, x1 = o.x - margin[0], o.x + o.w + margin[0]
      y0, y1 = o.y - margin[1], o.y + o.h + margin[1]
      inside |= (gaze[:, 0] >= x0) & (gaze[:, 0] < x1) & (gaze[:, 1] >= y0) & (gaze[:, 1] < y1)
      # Pixel (i, j) covers [j, j+1) x [i, i+1); count it when its center is inside.
      covered[max(0, int(np.ceil(y0 - 0.5))):max(0, int(np.ceil(y1 - 0.5))),
              max(0, int(np.ceil(x0 - 0.5))):max(0, int(np.ceil(x1 - 0.5)))] = True
    out[f'looked_at_grid/{category}'] = float(hits[category][cells].mean())
    out[f'looked_at_direct/{category}'] = float(inside.mean())
    out[f'on_screen/{category}'] = float(bool(boxes))
    out[f'chance_grid/{category}'] = float(hits[category].mean())
    out[f'chance_direct/{category}'] = float(covered.mean())
  return out


def read_trial(path: str):
  """Frame ids, durations (ms) and gaze samples (n, 2) per frame from an
  Atari-HEAD label file; samples outside the screen are dropped."""
  ids, durations, gaze = [], [], []
  with open(path) as f:
    next(f)
    for line in f:
      parts = line.strip().split(',')
      if len(parts) < 7:
        continue
      ids.append(parts[0])
      durations.append(float(parts[3]) if parts[3] != 'null' else np.nan)
      values = [v for v in parts[6:] if v != 'null']
      xy = np.array(values[:len(values) // 2 * 2], np.float64).reshape(-1, 2)
      ok = (xy[:, 0] >= 0) & (xy[:, 0] < SCREEN[0]) & (xy[:, 1] >= 0) & (xy[:, 1] < SCREEN[1])
      gaze.append(xy[ok])
  return ids, np.array(durations), gaze


def trial_shares(game_dir: str, name: str, grid: Grid, game: str = 'asterix') -> dict[str, float]:
  """Share of time per category over one trial (one episode in Atari-HEAD
  Asterix), weighted by frame duration. Frames without valid gaze or
  duration are left out of numerator and denominator."""
  from PIL import Image
  ids, durations, gaze = read_trial(os.path.join(game_dir, name + '.txt'))
  squares = grid.squares()
  totals, time, used = {}, 0.0, 0
  for fid, dur, g in zip(ids, durations, gaze):
    if not len(g) or not np.isfinite(dur):
      continue
    with Image.open(os.path.join(game_dir, name, fid + '.png')) as img:
      frame = np.array(img.convert('RGB'))
    shares = frame_shares(objects.detect(frame, game), g, grid, squares)
    for key, value in shares.items():
      totals[key] = totals.get(key, 0.0) + dur * value
    time += dur
    used += 1
  result = {key: value / time for key, value in totals.items()}
  result.update(trial=name, subject=name.split('_')[1], frames=used,
                skipped_frames=len(ids) - used, minutes=time / 60000)
  return result
