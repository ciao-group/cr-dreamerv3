"""Eye movement effort and saccade cost of CrAtari."""

import numpy as np
import pytest

from embodied.core import vision


def make(**kw):
  pytest.importorskip('ale_py')
  from embodied.envs.cr_atari import CrAtari
  return CrAtari('asterix', size=(96, 96), gray=False, vision_square_size=(24, 24),
                 vision_square_count=(5, 5), seed=0, **kw)


def run(env, gazes):
  obs = [env.step({'action': 0, 'reset': True, 'gaze_position': 12})]
  for g in gazes:
    obs.append(env.step({'action': 0, 'reset': False, 'gaze_position': g}))
  return obs


def amplitude(env, a, b):
  return vision.calc_gaze_shift_eccentricity_from_1d_vision_square_positions(
      a, b, env.vision_square_count, env.scaled_vision_square_size, (env.W, env.H),
      env.VISUAL_DEGREES_PER_PIXEL)


def effort(env, k, kstatic):
  return vision.calc_effort(k, env.vision_square_count, env.scaled_vision_square_size,
                            (env.W, env.H), env.VISUAL_DEGREES_PER_PIXEL, kstatic=kstatic)


def test_defaults_add_no_penalty():
  env = make(vision_model='EMMA')
  for o in run(env, [0, 24, 24]):
    assert o['log/effort_penalty'] == 0 and o['log/saccade_penalty'] == 0
    assert o['reward'] == o['log/game_score']


def test_effort_kstatic_scales_the_static_effort():
  env = make(vision_model='EMMA', eye_movement_effort=True, effort_kstatic=0.02)
  obs = run(env, [0, 12])
  assert obs[1]['log/effort_penalty'] == pytest.approx(effort(env, 0, 0.02))
  assert obs[1]['log/effort_penalty'] == pytest.approx(20 * effort(env, 0, 0.001))
  assert obs[2]['log/effort_penalty'] == 0  # screen centre
  assert obs[1]['reward'] == pytest.approx(obs[1]['log/game_score'] - obs[1]['log/effort_penalty'])


def test_saccade_cost_grows_with_amplitude():
  env = make(vision_model='EMMA', saccade_cost=0.1)
  obs = run(env, [13, 13, 10, 24])
  near, stay, far = obs[1], obs[2], obs[3]
  assert near['log/saccade_amplitude'] == pytest.approx(amplitude(env, 12, 13))
  assert near['log/saccade_penalty'] == pytest.approx(0.1 * amplitude(env, 12, 13))
  assert stay['log/saccade_penalty'] == 0 and stay['log/saccade_amplitude'] == 0
  assert far['log/saccade_penalty'] > near['log/saccade_penalty']
  assert obs[0]['log/saccade_penalty'] == 0  # reset
  for o in obs[1:]:
    assert o['reward'] == pytest.approx(o['log/game_score'] - o['log/saccade_penalty'])


def test_saccade_cost_without_emma_and_effort():
  env = make(saccade_cost=0.5)
  obs = run(env, [0])
  assert obs[1]['log/saccade_penalty'] == pytest.approx(0.5 * amplitude(env, 12, 0))
  assert obs[1]['log/effort_penalty'] == 0


def test_config_passes_the_options():
  import elements
  import ruamel.yaml as yaml
  import pathlib
  configs = yaml.YAML(typ='safe').load(
      (pathlib.Path(__file__).resolve().parents[2] / 'dreamerv3/configs.yaml').read_text())
  env = elements.Config(configs['defaults']['env']['cr-atari'])
  env = env.update({'effort_kstatic': '0.02', 'saccade_cost': '0'})
  assert env.effort_kstatic == 0.02 and env.saccade_cost == 0.0
  assert isinstance(env.saccade_cost, float)
