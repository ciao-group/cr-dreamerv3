"""Discount per emulator frame (agent.frame_discount, cr-atari.frames_obs)."""

import numpy as np
import pytest

jnp = pytest.importorskip('jax.numpy')


def test_cr_atari_frames_obs_counts_frames_per_step():
  pytest.importorskip('ale_py')
  from embodied.envs.cr_atari import CrAtari
  env = CrAtari('asterix', size=(96, 96), gray=False, vision_square_size=(24, 24),
                vision_square_count=(5, 5), seed=0, repeat=1, vision_model='EMMA',
                frames_obs=True)
  assert 'frames' in env.obs_space
  obs = env.step({'action': 0, 'reset': True, 'gaze_position': 12})
  assert obs['frames'] == 0
  frames = []
  for gaze in (12, 12, 0, 0, 24, 13):
    before = env.duration
    obs = env.step({'action': 0, 'reset': False, 'gaze_position': gaze})
    assert obs['frames'] == env.duration - before
    frames.append(int(obs['frames']))
  assert frames[0] == frames[1] == 1  # gaze stays: one frame
  assert frames[3] == 1
  assert frames[2] > 1 and frames[4] > 1  # saccades last longer


def test_cr_atari_without_frames_obs_is_unchanged():
  pytest.importorskip('ale_py')
  from embodied.envs.cr_atari import CrAtari
  env = CrAtari('asterix', size=(96, 96), vision_square_size=(24, 24),
                vision_square_count=(5, 5), seed=0)
  assert 'frames' not in env.obs_space
  assert 'frames' not in env.step({'action': 0, 'reset': True, 'gaze_position': 12})


def test_lambda_return_with_per_step_discount():
  from dreamerv3.agent import lambda_return
  rng = np.random.default_rng(0)
  B, T = 3, 6
  last = jnp.zeros((B, T), bool)
  term = jnp.zeros((B, T), bool)
  rew = jnp.asarray(rng.normal(size=(B, T)), jnp.float32)
  val = jnp.asarray(rng.normal(size=(B, T)), jnp.float32)
  disc = 0.99
  scalar = lambda_return(last, term, rew, val, val, disc, 0.95)
  same = lambda_return(last, term, rew, val, val, jnp.full((B, T), disc), 0.95)
  np.testing.assert_allclose(scalar, same, rtol=1e-6)
  # With lam = 0 the return is rew[t+1] + disc[t+1] * val[t+1].
  frames = jnp.asarray(rng.integers(1, 6, size=(B, T)), jnp.float32)
  ret = lambda_return(last, term, rew, val, val, disc ** frames, 0.0)
  expected = rew[:, 1:] + disc ** frames[:, 1:] * val[:, 1:]
  np.testing.assert_allclose(ret, expected, rtol=1e-5)


def test_repl_loss_uses_frames():
  import embodied.jax  # noqa: F401
  from dreamerv3 import agent

  class Value:
    def __init__(self, x):
      self.x = x

    def pred(self):
      return self.x

    def loss(self, target):
      return (self.x - target) ** 2

  class Norm:
    def stats(self):
      return 0.0, 1.0

    def __call__(self, x, update):
      return 0.0, 1.0

  B, T = 2, 5
  rng = np.random.default_rng(1)
  last = jnp.zeros((B, T), bool)
  term = jnp.zeros((B, T), bool)
  rew = jnp.asarray(rng.normal(size=(B, T)), jnp.float32)
  val = jnp.asarray(rng.normal(size=(B, T)), jnp.float32)
  args = (last, term, rew, val, Value(val), Value(val), Norm())
  base = agent.repl_loss(*args, lam=0.0, horizon=100)[1]['ret']
  ones = agent.repl_loss(*args, lam=0.0, horizon=100, frames=jnp.ones((B, T)))[1]['ret']
  np.testing.assert_allclose(base, ones, rtol=1e-6)
  frames = jnp.full((B, T), 4.0)
  four = agent.repl_loss(*args, lam=0.0, horizon=100, frames=frames)[1]['ret']
  np.testing.assert_allclose(four, rew[:, 1:] + 0.99 ** 4 * val[:, 1:], rtol=1e-5)
