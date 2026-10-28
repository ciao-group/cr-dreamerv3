"""Loader for the Atari-HEAD dataset (Zhang et al. 2019).

A game directory contains, per trial, `<trial>.txt` with one row per frame
(frame_id, episode_id, score, duration(ms), unclipped_reward, action,
gaze_positions...) and `<trial>.tar.bz2` with the frames as PNG files.

Human trials are converted into the step format of the CR agent:

* Every agent step shows one (pooled) frame foveated at the gaze `fovea_xy`.
* Its targets are the motor action, whether the game was paused, and the gaze
  of the next step (the sensory action).
* Frames during which the human paused the game are split into one step per
  fixation: all but the last are pause steps on the same frame.
"""

import dataclasses
import os
import tarfile

import numpy as np

from embodied.core import scanpath_metrics as sm

FRAME_MS = 50.0  # Atari-HEAD runs at 20 Hz while a key is held.


@dataclasses.dataclass
class Trial:
  name: str
  subject: str
  frame_ids: list[str]
  durations: np.ndarray  # (T,) ms each frame was shown
  actions: np.ndarray  # (T,) ALE action ids
  rewards: np.ndarray  # (T,) unclipped rewards
  episode_ids: np.ndarray  # (T,) -1 when unknown
  gaze: list[np.ndarray]  # T arrays of shape (n, 2), NaN for missing
  frame_dir: str

  def __len__(self):
    return len(self.frame_ids)

  @property
  def onsets(self) -> np.ndarray:
    """Start time of every frame in ms."""
    return np.concatenate([[0.0], np.cumsum(self.durations)[:-1]])

  @property
  def score(self) -> float:
    return float(self.rewards.sum())

  def frame(self, index: int) -> np.ndarray:
    """RGB frame (210, 160, 3) as uint8."""
    from PIL import Image
    path = os.path.join(self.frame_dir, self.frame_ids[index] + '.png')
    with Image.open(path) as img:
      return np.array(img.convert('RGB'))

  def gaze_samples(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All gaze samples (x, y, t) with times spread evenly over each frame."""
    xs, ys, ts = [], [], []
    for onset, dur, g in zip(self.onsets, self.durations, self.gaze):
      if not len(g):
        continue
      xs.append(g[:, 0])
      ys.append(g[:, 1])
      ts.append(onset + np.arange(len(g)) * (dur / len(g)))
    if not xs:
      return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ts)

  def fixations(self, **kw) -> np.ndarray:
    """Fixations (x, y, t_start, duration) over the whole trial."""
    return sm.detect_fixations_idt(*self.gaze_samples(), **kw)


@dataclasses.dataclass
class Steps:
  """Human behaviour in agent steps (struct of arrays, length N)."""
  frames: np.ndarray  # (N, pooling) frame indices to max-pool
  fovea_xy: np.ndarray  # (N, 2) gaze applied to the observation
  next_gaze_xy: np.ndarray  # (N, 2) target sensory action
  action: np.ndarray  # (N,) target motor action, -1 on pause steps
  pause: np.ndarray  # (N,) target pause decision
  duration: np.ndarray  # (N,) ms spent on the step
  is_first: np.ndarray  # (N,) start of a new game

  def __len__(self):
    return len(self.action)

  @property
  def valid(self) -> np.ndarray:
    """Steps where both the applied and the target gaze are known."""
    return np.isfinite(self.fovea_xy).all(1) & np.isfinite(
        self.next_gaze_xy).all(1)


def list_trials(game_dir: str) -> list[str]:
  return sorted(
      os.path.splitext(f)[0] for f in os.listdir(game_dir)
      if f.endswith('.txt'))


def subject_of(trial_name: str) -> str:
  # Trial names look like '160_RZ_9166697_Feb-20-16-46-45'.
  return trial_name.split('_')[1]


def split_trials(trials, test_every: int = 3):
  """Deterministic trial-level split (split first, never shuffle frames
  across splits, as recommended by Atari-HEAD)."""
  trials = sorted(trials)
  test = trials[test_every - 1::test_every]
  train = [t for t in trials if t not in test]
  return train, test


def load_trial(game_dir: str, name: str, extract: bool = True) -> Trial:
  txt = os.path.join(game_dir, name + '.txt')
  frame_ids, durations, actions, rewards, episodes, gaze = [], [], [], [], [], []
  with open(txt) as f:
    next(f)
    for line in f:
      parts = line.strip().split(',')
      if len(parts) < 6:
        continue
      frame_ids.append(parts[0])
      episodes.append(_int(parts[1], -1))
      durations.append(_float(parts[3], FRAME_MS))
      rewards.append(_float(parts[4], 0.0))
      actions.append(_int(parts[5], 0))
      values = np.array([_float(v, np.nan) for v in parts[6:]], np.float64)
      values = values[:len(values) // 2 * 2].reshape(-1, 2)
      gaze.append(values)
  frame_dir = os.path.join(game_dir, name)
  if extract and not os.path.isdir(frame_dir):
    _extract(os.path.join(game_dir, name + '.tar.bz2'), game_dir)
  return Trial(
      name=name,
      subject=subject_of(name),
      frame_ids=frame_ids,
      durations=np.array(durations, np.float64),
      actions=np.array(actions, np.int32),
      rewards=np.array(rewards, np.float64),
      episode_ids=np.array(episodes, np.int32),
      gaze=gaze,
      frame_dir=frame_dir,
  )


def to_steps(
    trial: Trial,
    stride: int = 1,
    pooling: int = 1,
    pause_threshold_ms: float = 1.5 * FRAME_MS,
    fixation_kw: dict | None = None,
) -> Steps:
  """Convert a trial into agent steps.

  `stride` frames form one agent step (the env's action repeat). The step
  shows the last `pooling` frames of its group, max-pooled like the env.
  A group shown longer than `stride * pause_threshold_ms` counts as paused;
  if it contains several fixations, all but the last become pause steps.
  """
  fixations = trial.fixations(**(fixation_kw or {}))
  fix_starts = fixations[:, 2] if len(fixations) else np.zeros(0)
  onsets = trial.onsets
  n_groups = len(trial) // stride
  frames, fovea, action, pause, duration, first = [], [], [], [], [], []
  prev_episode = None
  for g in range(n_groups):
    lo, hi = g * stride, (g + 1) * stride
    group_frames = np.arange(max(hi - pooling, lo), hi)
    t0 = onsets[lo]
    t1 = onsets[hi - 1] + trial.durations[hi - 1]
    group_gaze = np.concatenate([trial.gaze[i] for i in range(lo, hi)])
    episode = trial.episode_ids[lo]
    new_game = g == 0 or (episode != prev_episode and episode >= 0)
    prev_episode = episode

    inside = np.nonzero((fix_starts >= t0) & (fix_starts < t1))[0]
    paused = (t1 - t0) > stride * pause_threshold_ms
    if paused and len(inside) >= 2:
      for k, idx in enumerate(inside):
        last = k == len(inside) - 1
        end = t1 if last else fix_starts[inside[k + 1]]
        start = t0 if k == 0 else fix_starts[idx]
        frames.append(group_frames)
        fovea.append(fixations[idx, :2])
        action.append(trial.actions[lo] if last else -1)
        pause.append(not last)
        duration.append(end - start)
        first.append(new_game and k == 0)
    else:
      frames.append(group_frames)
      fovea.append(_median_gaze(group_gaze))
      action.append(trial.actions[lo])
      pause.append(False)
      duration.append(t1 - t0)
      first.append(new_game)

  fovea = np.array(fovea, np.float64).reshape(-1, 2)
  next_gaze = np.concatenate([fovea[1:], np.full((1, 2), np.nan)])
  return Steps(
      frames=np.array(frames, np.int64).reshape(len(fovea), -1),
      fovea_xy=fovea,
      next_gaze_xy=next_gaze,
      action=np.array(action, np.int32),
      pause=np.array(pause, bool),
      duration=np.array(duration, np.float64),
      is_first=np.array(first, bool),
  )


def pause_durations(trial: Trial, threshold_ms: float = 1.5 * FRAME_MS):
  """Durations (ms) of frames on which the human paused the game."""
  return trial.durations[trial.durations > threshold_ms]


def _median_gaze(samples: np.ndarray) -> np.ndarray:
  samples = samples[np.isfinite(samples).all(1)] if len(samples) else samples
  if not len(samples):
    return np.full(2, np.nan)
  return np.median(samples, 0)


def _extract(archive: str, target: str):
  with tarfile.open(archive, 'r:bz2') as tar:
    members = [m for m in tar.getmembers()
               if m.isfile() and m.name.endswith('.png') and '..' not in m.name]
    tar.extractall(target, members=members)


def _float(value: str, default: float) -> float:
  try:
    return float(value)
  except ValueError:
    return default


def _int(value: str, default: int) -> int:
  try:
    return int(value)
  except ValueError:
    return default
