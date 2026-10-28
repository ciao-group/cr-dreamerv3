"""Human bounds of the CR agent and their priors.

Every bound has a default (literature value) and a prior box. Bounds listed in
`sample` are drawn from their prior at every episode start and exposed to the
agent as a normalised vector, so a single agent learns the optimal policy for
the whole prior (amortisation). All other bounds stay at their configured
values.
"""

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class BoundSpec:
  default: float
  low: float
  high: float
  log: bool = False  # sample and normalise on a log scale
  doc: str = ''


SPECS = {
    # Oculomotor control (EMMA, Salvucci 2001).
    'emma_K': BoundSpec(
        0.006, 0.003, 0.012, log=True, doc='encoding time scale (s)'),
    'emma_k': BoundSpec(
        0.4, 0.2, 0.6, doc='encoding eccentricity exponent (1/deg)'),
    'emma_t_prep': BoundSpec(
        0.135, 0.10, 0.20, doc='saccade preparation time (s)'),
    'effort_k': BoundSpec(
        0.001, 0.0001, 0.004, log=True, doc='static eye effort per deg^2'),
    # Perception.
    'periphery_sigma': BoundSpec(
        2.0, 1.0, 8.0, log=True, doc='peripheral blur (px, native screen)'),
    # Motor control.
    'motor_delay': BoundSpec(
        0.07, 0.04, 0.12, doc='mean motor execution delay (s)'),
    'reaction_rate': BoundSpec(
        5.0, 3.5, 6.5, doc='mean LATER rate (1/s), ~1/reaction time'),
    # Utility.
    'time_cost': BoundSpec(
        0.0, 0.0, 2.0, doc='reward cost per real second'),
}


class Bounds:

  def __init__(self, values=None, sample=(), seed=None):
    values = dict(values or {})
    unknown = (set(values) | set(sample)) - set(SPECS)
    assert not unknown, f'Unknown bounds: {sorted(unknown)}'
    self.fixed = {k: float(values.get(k, s.default)) for k, s in SPECS.items()}
    self.names = tuple(k for k in SPECS if k in sample)
    self.rng = np.random.default_rng(seed)
    self.current = dict(self.fixed)
    self.frozen = False

  @classmethod
  def from_config(cls, values, sample, seed=None):
    """`sample` is 'all', '' / 'none' or a comma separated list."""
    if isinstance(sample, str):
      if sample == 'all':
        sample = list(SPECS)
      elif sample in ('', 'none'):
        sample = []
      else:
        sample = [s.strip() for s in sample.split(',') if s.strip()]
    return cls(values, sample, seed)

  def __getitem__(self, key):
    return self.current[key]

  def resample(self):
    self.current = dict(self.fixed)
    if self.frozen:
      return self.current
    for name in self.names:
      u = self.rng.uniform()
      self.current[name] = denormalize_value(name, u)
    return self.current

  def set(self, **values):
    """Overwrite current values, e.g. to evaluate an inferred theta."""
    for k, v in values.items():
      assert k in SPECS, k
      self.current[k] = float(v)

  def freeze(self, **values):
    """Keep the given values for all future episodes (no resampling), while
    still exposing the sampled bounds to the agent."""
    for k, v in values.items():
      assert k in SPECS, k
      self.fixed[k] = float(v)
    self.current = dict(self.fixed)
    self.frozen = True

  def vector(self) -> np.ndarray:
    """Current values of the sampled bounds, normalised to [0, 1]."""
    return np.array(
        [normalize_value(k, self.current[k]) for k in self.names], np.float32)


def normalize_value(name: str, value: float) -> float:
  s = SPECS[name]
  if s.log:
    lo, hi, value = np.log(s.low), np.log(s.high), np.log(max(value, 1e-12))
  else:
    lo, hi = s.low, s.high
  return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def denormalize_value(name: str, u: float) -> float:
  s = SPECS[name]
  if s.log:
    return float(np.exp(np.log(s.low) + u * (np.log(s.high) - np.log(s.low))))
  return float(s.low + u * (s.high - s.low))


def to_vector(names, values: dict) -> np.ndarray:
  return np.array([normalize_value(k, values[k]) for k in names], np.float32)


def from_vector(names, vector) -> dict:
  return {k: denormalize_value(k, float(u)) for k, u in zip(names, vector)}
