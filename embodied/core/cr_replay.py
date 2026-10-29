"""Replay human (or recorded agent) trajectories through a CR agent.

A replay source yields one `ReplayStep` per agent step: the raw RGB frames to
pool, the gaze cell the observation is foveated at, and the target actions.
Observations are built with the same foveation and preprocessing as CrAtari,
so the agent sees what the human saw through the agent's own perception.
"""

import dataclasses

import numpy as np

from embodied.core import atari_head
from embodied.core import bounds as bounds_lib
from embodied.envs.atari import preprocess_image


@dataclasses.dataclass
class ReplayStep:
  frames: list  # raw RGB screens (H, W, 3) to max-pool
  fovea: int  # gaze cell the observation is foveated at
  action: int  # target motor action, -1 if unknown or paused
  gaze: int  # target gaze cell, -1 if unknown
  pause: int  # target pause decision
  reward: float
  is_first: bool
  duration: float  # ms


def xy_to_cell(xy, env) -> int:
  """Map native screen coordinates to the env's gaze cell, -1 if unknown."""
  if xy is None or not np.isfinite(xy).all():
    return -1
  cw, ch = env.W // env.vision_square_count[0], env.H // env.vision_square_count[1]
  cx = int(np.clip(xy[0] // cw, 0, env.vision_square_count[0] - 1))
  cy = int(np.clip(xy[1] // ch, 0, env.vision_square_count[1] - 1))
  return cy * env.vision_square_count[0] + cx


def action_index(ale_action: int, env) -> int:
  """Map an ALE action id (as recorded in Atari-HEAD) to the index in the
  env's action set. Actions missing from a minimal set (`actions: needed`)
  lose their FIRE component, e.g. UPFIRE -> UP and FIRE -> NOOP. Returns -1
  if there is no equivalent."""
  ids = [int(a) for a in env.actionset]
  if ale_action < 0:
    return -1
  if ale_action in ids:
    return ids.index(ale_action)
  meaning = env.ACTION_MEANING[ale_action].replace('FIRE', '') or 'NOOP'
  if meaning in env.ACTION_MEANING:
    fallback = env.ACTION_MEANING.index(meaning)
    if fallback in ids:
      return ids.index(fallback)
  return -1


def cell_center(cell: int, env) -> np.ndarray:
  cw, ch = env.W // env.vision_square_count[0], env.H // env.vision_square_count[1]
  cx, cy = cell % env.vision_square_count[0], cell // env.vision_square_count[0]
  return np.array([(cx + 0.5) * cw, (cy + 0.5) * ch], np.float64)


def human_steps(trial, env, max_steps=0, split_pauses=True):
  """Yield ReplaySteps for an Atari-HEAD trial in the env's step format."""
  steps = atari_head.to_steps(
      trial, stride=env.repeat, pooling=env.pooling,
      pause_threshold_ms=(
          1.5 * atari_head.FRAME_MS if split_pauses else np.inf))
  fovea = env.center_gaze_position()
  cache = {}
  n = len(steps) if not max_steps else min(max_steps, len(steps))
  for i in range(n):
    cell = xy_to_cell(steps.fovea_xy[i], env)
    fovea = cell if cell >= 0 else fovea
    frames = []
    for idx in steps.frames[i]:
      if idx not in cache:
        cache = {k: v for k, v in cache.items() if k >= idx - 8}
        cache[idx] = trial.frame(int(idx))
      frames.append(cache[idx])
    lo, hi = steps.frames[i][0], steps.frames[i][-1] + 1
    reward = 0.0 if steps.pause[i] else float(trial.rewards[lo:hi].sum())
    yield ReplayStep(
        frames=frames,
        fovea=fovea,
        action=action_index(int(steps.action[i]), env),
        gaze=xy_to_cell(steps.next_gaze_xy[i], env),
        pause=int(steps.pause[i]),
        reward=reward,
        is_first=bool(steps.is_first[i]),
        duration=float(steps.duration[i]),
    )


def make_image(step: ReplayStep, env, periphery_sigma=None) -> np.ndarray:
  sigma = env.bounds['periphery_sigma'] if periphery_sigma is None else periphery_sigma
  old = env.bounds.current['periphery_sigma']
  env.bounds.current['periphery_sigma'] = sigma
  try:
    foveated = [env.foveate(f, step.fovea) for f in step.frames]
  finally:
    env.bounds.current['periphery_sigma'] = old
  return preprocess_image(
      foveated, env.size, env.gray, env.aggregate, env.resize)


def batch_obs(step: ReplayStep, env, candidates: list[dict], names) -> dict:
  """Observation batch with one row per candidate theta."""
  n = len(candidates)
  per_candidate_blur = (
      'periphery_sigma' in names and env.vision_mode != 'foveated')
  if per_candidate_blur:
    images = np.stack([
        make_image(step, env, c['periphery_sigma']) for c in candidates])
  else:
    images = np.repeat(make_image(step, env)[None], n, 0)
  obs = {
      'image': images,
      'reward': np.full(n, step.reward, np.float32),
      'is_first': np.full(n, step.is_first, bool),
      'is_last': np.zeros(n, bool),
      'is_terminal': np.zeros(n, bool),
  }
  if names:
    obs['bounds'] = np.stack([
        bounds_lib.to_vector(names, c) for c in candidates]).astype(np.float32)
  return obs


def batch_act(step: ReplayStep, act_keys, n: int) -> dict:
  """Teacher-forcing actions; unknown targets are replaced by valid dummies
  and masked out of the likelihood via `step_masks`."""
  act = {
      'action': max(step.action, 0),
      'gaze_position': step.gaze if step.gaze >= 0 else step.fovea,
      'pause': step.pause,
  }
  return {k: np.full(n, act[k], np.int32) for k in act_keys}


def step_masks(step: ReplayStep, act_keys) -> dict:
  return {
      'action': step.action >= 0 and not step.pause,
      'gaze_position': step.gaze >= 0,
      'pause': 'pause' in act_keys,
  }


def fixations_from_cells(cells, durations_ms, env, rng=None) -> np.ndarray:
  """Merge consecutive steps on the same cell into fixations (x, y, t, d).

  Positions are cell centres, optionally jittered uniformly within the cell
  (for saliency maps of grid-based agents)."""
  cells = np.asarray(cells, int)
  durations_ms = np.asarray(durations_ms, np.float64)
  out = []
  t = 0.0
  i = 0
  while i < len(cells):
    j = i
    while j + 1 < len(cells) and cells[j + 1] == cells[i]:
      j += 1
    dur = durations_ms[i:j + 1].sum()
    xy = cell_center(cells[i], env)
    if rng is not None:
      cw = env.W // env.vision_square_count[0]
      ch = env.H // env.vision_square_count[1]
      xy = xy + rng.uniform(-0.5, 0.5, 2) * (cw, ch)
    out.append((*xy, t, dur))
    t += dur
    i = j + 1
  return np.array(out, np.float64).reshape(-1, 4)


def quantize_fixations(fix, env) -> np.ndarray:
  """Snap fixations to cell centres and merge consecutive ones on the same
  cell, so human and grid-based agent scanpaths are compared fairly."""
  fix = np.asarray(fix, np.float64).reshape(-1, 4)
  cells = [xy_to_cell(xy, env) for xy in fix[:, :2]]
  return fixations_from_cells(cells, fix[:, 3], env)
