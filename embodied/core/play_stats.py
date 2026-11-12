"""Gameplay and gaze statistics, the same for humans (Atari-HEAD) and agent.

Both are turned into a Trajectory: one row per time slice with its duration,
the reward, the avatar position and how the gaze was spread over the vision
grid during it. A human row is one Atari-HEAD frame (duration(ms) of the
label file, gaze share per square = share of the frame's gaze samples mapped
to it). An agent row is one env step (frames x 50 ms, one-hot gaze on the
square chosen for that step; the old square during EMMA saccade frames is
ignored here).

The avatar position is the top-left corner of its 8x11 box in screen pixels.
For the agent it comes from the RAM (CrAtari logs it), for humans from the
colour detector; both agree on random play (x within 1 px, y in 99.5 % of
frames exactly).

Gaze is measured on the agent's grid for both, so the numbers are directly
comparable: the human sample counts for the square the agent would have
needed to see that spot.
"""

import dataclasses
import os

import numpy as np

from embodied.core import object_gaze, objects

AVATAR_SIZE = (8, 11)
# Asterix: the avatar walks on 8 lanes, its box top is at y = 26 + 16 * lane.
LANES = {'asterix': (26, 16, 8)}
X_BINS = 8  # avatar x histogram over the 160 px screen width


@dataclasses.dataclass
class Trajectory:
  time: np.ndarray  # (n,) ms
  reward: np.ndarray  # (n,)
  avatar: np.ndarray  # (n, 2) top-left x, y in screen px, nan if not found
  gaze: np.ndarray  # (n, squares) share of the row's time on each square


def cut(rewards: np.ndarray, max_frames: int = 0, max_score: float = 0) -> int:
  """Number of rows to keep: after `max_frames` rows, or before the row in
  which the cumulative reward first exceeds `max_score` (0 = no cut)."""
  end = len(rewards)
  if max_frames:
    end = min(end, max_frames)
  if max_score:
    over = np.nonzero(np.cumsum(rewards) > max_score)[0]
    if len(over):
      end = min(end, int(over[0]))
  return end


def human_trajectory(game_dir: str, name: str, grid: object_gaze.Grid, game: str = 'asterix',
                     max_frames: int = 0, max_score: float = 0) -> Trajectory:
  """Trajectory of one Atari-HEAD trial. Rows without a valid duration are
  dropped; rows without gaze samples keep their time for the avatar stats."""
  from PIL import Image
  ids, durations, rewards, gaze = object_gaze.read_trial(os.path.join(game_dir, name + '.txt'))
  end = cut(rewards, max_frames, max_score)
  n_squares = grid.count[0] * grid.count[1]
  rows = []
  for fid, dur, rew, g in zip(ids[:end], durations[:end], rewards[:end], gaze[:end]):
    if not np.isfinite(dur):
      continue
    with Image.open(os.path.join(game_dir, name, fid + '.png')) as img:
      frame = np.array(img.convert('RGB'))
    players = [o for o in objects.detect(frame, game) if o.category == 'player']
    if players:
      p = max(players, key=lambda o: o.w * o.h)
      avatar = (p.x, p.y)
    else:
      avatar = (np.nan, np.nan)
    share = np.zeros(n_squares)
    if len(g):
      share = np.bincount(grid.cells(g), minlength=n_squares) / len(g)
    rows.append((dur, rew, avatar, share))
  return Trajectory(
      time=np.array([r[0] for r in rows], float),
      reward=np.array([r[1] for r in rows], float),
      avatar=np.array([r[2] for r in rows], float).reshape(-1, 2),
      gaze=np.array([r[3] for r in rows], float).reshape(-1, n_squares))


def agent_trajectory(frames, rewards, avatar_x, avatar_y, cells, n_squares: int,
                     frame_ms: float = 50.0) -> Trajectory:
  """Trajectory of one agent episode from per-step logs: emulator frames of
  the step, game score, RAM avatar position and the gaze square used."""
  frames = np.asarray(frames, float)
  gaze = np.zeros((len(frames), n_squares))
  gaze[np.arange(len(frames)), np.asarray(cells, int)] = 1.0
  return Trajectory(
      time=frames * frame_ms, reward=np.asarray(rewards, float),
      avatar=np.stack([np.asarray(avatar_x, float), np.asarray(avatar_y, float)], -1),
      gaze=gaze)


def stats(traj: Trajectory, grid: object_gaze.Grid, game: str = 'asterix') -> dict[str, float]:
  """Time-weighted gameplay and gaze statistics of one episode or trial.

  Avatar: share of time per lane (lane/0 = top) and per x bin, mean and sd
  of x, lane changes per minute and horizontal speed.
  Gaze: share of time per square (gaze_cell/k, row-major as the gaze index),
  entropy of that distribution (bits, 25 squares: max 4.64), square changes
  per second and mean dwell time, and where the gaze is relative to the
  avatar: same square, at most one square away, distance between the square
  centre and the avatar centre, and the row/column offset distribution
  (positive = below / right of the avatar)."""
  out = {}
  time = traj.time
  total = time.sum()
  out['minutes'] = total / 60000
  out['score'] = float(traj.reward.sum())

  # Avatar.
  found = np.isfinite(traj.avatar[:, 0])
  t, xy = time[found], traj.avatar[found]
  out['avatar_found'] = t.sum() / total if total else np.nan
  y0, step, n_lanes = LANES[game]
  lane = np.clip(np.round((xy[:, 1] - y0) / step), 0, n_lanes - 1).astype(int)
  lane_time = np.bincount(lane, weights=t, minlength=n_lanes) / max(t.sum(), 1e-9)
  for k in range(n_lanes):
    out[f'lane/{k}'] = lane_time[k]
  xbin = np.clip((xy[:, 0] * X_BINS // object_gaze.SCREEN[0]).astype(int), 0, X_BINS - 1)
  x_time = np.bincount(xbin, weights=t, minlength=X_BINS) / max(t.sum(), 1e-9)
  for k in range(X_BINS):
    out[f'avatar_xbin/{k}'] = x_time[k]
  mean_x = np.average(xy[:, 0], weights=t) if t.sum() else np.nan
  out['avatar_x_mean'] = mean_x
  out['avatar_x_sd'] = np.sqrt(np.average((xy[:, 0] - mean_x) ** 2, weights=t)) if t.sum() else np.nan
  out['lane_changes_per_min'] = np.count_nonzero(np.diff(lane)) / (total / 60000)
  out['avatar_x_speed'] = np.abs(np.diff(xy[:, 0])).sum() / (total / 1000)  # px/s

  # Gaze on the grid.
  n_squares = traj.gaze.shape[1]
  has_gaze = traj.gaze.sum(1) > 0
  gt = time[has_gaze]
  g = traj.gaze[has_gaze]
  gaze_total = gt.sum()
  cell_time = (g * gt[:, None]).sum(0) / gaze_total
  for k in range(n_squares):
    out[f'gaze_cell/{k}'] = cell_time[k]
  nz = cell_time[cell_time > 0]
  out['gaze_entropy'] = float(-(nz * np.log2(nz)).sum())
  mode = g.argmax(1)
  switches = np.count_nonzero(np.diff(mode))
  out['gaze_switches_per_s'] = switches / (gaze_total / 1000)
  out['gaze_dwell_ms'] = gaze_total / (switches + 1)

  # Gaze relative to the avatar (rows with both).
  both = has_gaze & found
  w = traj.gaze[both] * time[both, None]  # time per (row, square)
  centre = traj.avatar[both] + np.array(AVATAR_SIZE) / 2
  a_cell = grid.cells(centre)
  cols, rows = grid.count
  a_row, a_col = a_cell // cols, a_cell % cols
  squares = np.array(grid.squares(), float)
  sq_centre = squares[:, :2] + squares[:, 2:] / 2
  s_row, s_col = np.arange(n_squares) // cols, np.arange(n_squares) % cols
  d_row = s_row[None] - a_row[:, None]  # (rows, squares)
  d_col = s_col[None] - a_col[:, None]
  dist = np.linalg.norm(sq_centre[None] - centre[:, None], axis=-1)
  wt = w.sum()
  out['gaze_avatar_same_square'] = w[(d_row == 0) & (d_col == 0)].sum() / wt
  out['gaze_avatar_near'] = w[(np.abs(d_row) <= 1) & (np.abs(d_col) <= 1)].sum() / wt
  out['gaze_avatar_dist_px'] = (w * dist).sum() / wt
  for d in range(-(rows - 1), rows):
    out[f'gaze_avatar_drow/{d}'] = w[d_row == d].sum() / wt
  for d in range(-(cols - 1), cols):
    out[f'gaze_avatar_dcol/{d}'] = w[d_col == d].sum() / wt
  return out


def summary_keys(grid: object_gaze.Grid, game: str = 'asterix') -> list[str]:
  """Column order of stats()."""
  empty = Trajectory(np.ones(1), np.zeros(1), np.array([[0.0, LANES[game][0]]]),
                     np.eye(grid.count[0] * grid.count[1])[:1])
  return list(stats(empty, grid, game))
