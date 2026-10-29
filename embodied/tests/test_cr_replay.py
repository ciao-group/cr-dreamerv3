import numpy as np
import pytest

from embodied.core import atari_head
from embodied.core import cr_replay

pytest.importorskip('ale_py')
from embodied.envs.cr_atari import CrAtari  # noqa: E402
from test_atari_head import NAME, _make_trial  # noqa: E402


@pytest.fixture(scope='module')
def env():
  return CrAtari(
      'asterix', size=(96, 96), repeat=1, pooling=1,
      vision_square_size=(24, 24), vision_square_count=(4, 4),
      bounds_sample='emma_K,time_cost')


def test_cells_match_env_grid(env):
  assert cr_replay.xy_to_cell((0, 0), env) == 0
  assert cr_replay.xy_to_cell((159, 0), env) == 3
  assert cr_replay.xy_to_cell((0, 209), env) == 12
  assert cr_replay.xy_to_cell((np.nan, 3), env) == -1
  for cell in range(16):
    assert cr_replay.xy_to_cell(cr_replay.cell_center(cell, env), env) == cell


def test_human_steps_and_batches(env, tmp_path):
  _make_trial(tmp_path)
  trial = atari_head.load_trial(str(tmp_path), NAME)
  steps = list(cr_replay.human_steps(trial, env))
  assert [s.pause for s in steps] == [0, 0, 1, 0, 0]
  assert steps[0].fovea == cr_replay.xy_to_cell((20, 30), env)
  assert steps[2].gaze == cr_replay.xy_to_cell((40, 60), env)
  assert steps[1].reward == 10
  # Unknown fovea keeps the previous one.
  assert steps[4].fovea == steps[3].fovea

  candidates = [{'emma_K': 0.004, 'time_cost': 0.0},
                {'emma_K': 0.012, 'time_cost': 2.0}]
  obs = cr_replay.batch_obs(steps[0], env, candidates, env.bounds.names)
  assert obs['image'].shape == (2, 96, 96, 1)
  assert obs['bounds'].shape == (2, 2)
  np.testing.assert_allclose(obs['bounds'][1], [1.0, 1.0])
  assert sorted(obs) == sorted(
      k for k in env.obs_space if not k.startswith('log/'))

  act = cr_replay.batch_act(steps[2], ['action', 'gaze_position', 'pause'], 2)
  np.testing.assert_array_equal(act['pause'], [1, 1])
  np.testing.assert_array_equal(act['action'], [0, 0])
  masks = cr_replay.step_masks(steps[2], ['action', 'gaze_position'])
  assert masks == {'action': False, 'gaze_position': True, 'pause': False}


def test_fixations_from_cells(env):
  fix = cr_replay.fixations_from_cells([0, 0, 5, 5, 5, 0], [50] * 6, env)
  np.testing.assert_allclose(fix[:, 3], [100, 150, 50])
  np.testing.assert_allclose(fix[:, 2], [0, 100, 250])
  np.testing.assert_allclose(fix[1, :2], cr_replay.cell_center(5, env))
  human = np.array([[10, 10, 0, 100], [12, 12, 100, 80], [150, 200, 180, 60]])
  quant = cr_replay.quantize_fixations(human, env)
  np.testing.assert_allclose(quant[:, 3], [180, 60])


def test_action_index_for_minimal_action_set():
  env = CrAtari(
      'asterix', size=(96, 96), vision_square_size=(24, 24),
      vision_square_count=(4, 4), actions='needed')
  meanings = [env.ACTION_MEANING[int(a)] for a in env.actionset]
  index = lambda name: cr_replay.action_index(
      env.ACTION_MEANING.index(name), env)
  assert meanings[index('RIGHT')] == 'RIGHT'
  assert meanings[index('UPFIRE')] == 'UP'
  assert meanings[index('DOWNLEFTFIRE')] == 'DOWNLEFT'
  assert meanings[index('FIRE')] == 'NOOP'
  assert cr_replay.action_index(-1, env) == -1


def test_action_index_for_full_action_set(env):
  for ale_id in range(18):
    assert cr_replay.action_index(ale_id, env) == ale_id
