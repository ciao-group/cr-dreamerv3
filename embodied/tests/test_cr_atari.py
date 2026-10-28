import numpy as np
import pytest

from embodied.core import bounds as bounds_lib

pytest.importorskip('ale_py')
from embodied.envs.cr_atari import CrAtari  # noqa: E402


def _env(**kw):
  defaults = dict(
      size=(96, 96), repeat=1, vision_square_size=(24, 24),
      vision_square_count=(4, 4), vision_model='EMMA', sticky=False)
  defaults.update(kw)
  return CrAtari('asterix', **defaults)


def _act(env, gaze=5, action=0, reset=False, pause=0):
  act = {'action': np.int32(action), 'reset': reset,
         'gaze_position': np.int32(gaze)}
  if 'pause' in env.act_space:
    act['pause'] = np.int32(pause)
  return env.step(act)


def test_no_bounds_observation_by_default():
  env = _env()
  obs = _act(env, reset=True)
  assert 'bounds' not in env.obs_space
  assert 'bounds' not in obs
  assert 'pause' not in env.act_space
  assert obs['image'].shape == (96, 96, 1)


def test_sampled_bounds_are_observed_and_resampled():
  env = _env(bounds_sample='emma_K,time_cost')
  assert env.obs_space['bounds'].shape == (2,)
  first = _act(env, reset=True)['bounds']
  assert ((first >= 0) & (first <= 1)).all()
  second = _act(env, reset=True)['bounds']
  assert not np.allclose(first, second)
  obs = _act(env, gaze=0)
  np.testing.assert_allclose(obs['bounds'], second)
  assert obs['log/bound_emma_K'] == pytest.approx(
      bounds_lib.denormalize_value('emma_K', second[0]), rel=1e-5)


def test_pause_freezes_game_and_costs_time():
  env = _env(pause=True, bounds={'time_cost': 1.0})
  _act(env, reset=True)
  for _ in range(5):
    _act(env, action=3)
  ram = env.ale.getRAM().copy()
  frame = env.ale.getFrameNumber()
  obs = _act(env, gaze=15, action=3, pause=1)
  assert obs['log/paused']
  assert env.ale.getFrameNumber() == frame
  np.testing.assert_array_equal(env.ale.getRAM(), ram)
  assert obs['log/gaze_position'] == 15
  assert obs['log/step_time'] >= env.TIME_PER_FRAME
  assert obs['reward'] == pytest.approx(-obs['log/step_time'])
  obs = _act(env, gaze=15, action=3, pause=0)
  assert not obs['log/paused']
  assert env.ale.getFrameNumber() > frame


def test_pause_is_capped():
  env = _env(pause=True, max_pauses=2)
  _act(env, reset=True)
  paused = [bool(_act(env, gaze=i % 16, pause=1)['log/paused'])
            for i in range(4)]
  assert paused == [True, True, False, True]


def test_bound_values_change_emma_timing():
  times = []
  for t_prep in (0.10, 0.20):
    env = _env(bounds={'emma_t_prep': t_prep}, pause=True)
    _act(env, reset=True)
    times.append(_act(env, gaze=0, pause=1)['log/step_time'])
  assert times[1] > times[0]


def test_bounds_roundtrip():
  for name in bounds_lib.SPECS:
    for u in (0.0, 0.3, 1.0):
      value = bounds_lib.denormalize_value(name, u)
      assert bounds_lib.normalize_value(name, value) == pytest.approx(u, abs=1e-6)
  with pytest.raises(AssertionError):
    bounds_lib.Bounds(sample=['unknown'])
