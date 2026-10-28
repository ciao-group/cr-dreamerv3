import math

import numpy as np
import pytest

from embodied.core import scanpath_metrics as sm


def _fix(points, duration=100.0):
  pts = np.asarray(points, np.float64)
  t = np.arange(len(pts)) * duration
  return np.column_stack([pts, t, np.full(len(pts), duration)])


class TestFixationDetection:

  def test_two_fixations_with_saccade(self):
    ts = np.arange(300, dtype=float)
    xs = np.where(ts < 150, 40.0, 120.0)
    ys = np.where(ts < 150, 50.0, 150.0)
    fix = sm.detect_fixations_idt(xs, ys, ts)
    assert fix.shape == (2, 4)
    np.testing.assert_allclose(fix[:, :2], [[40, 50], [120, 150]])
    np.testing.assert_allclose(fix[:, 3], [150, 150])

  def test_short_window_is_not_a_fixation(self):
    ts = np.arange(30, dtype=float)
    fix = sm.detect_fixations_idt(np.full(30, 10.0), np.full(30, 10.0), ts)
    assert len(fix) == 0

  def test_nan_samples_split_fixations(self):
    ts = np.arange(200, dtype=float)
    xs = np.full(200, 80.0)
    xs[95:105] = np.nan
    fix = sm.detect_fixations_idt(xs, np.full(200, 100.0), ts)
    assert len(fix) == 2


class TestSaliency:

  def test_identical_points_are_perfect(self):
    pts = np.array([[20, 30], [100, 180], [80, 100]])
    rep = sm.saliency_report(pts, pts)
    assert rep['cc'] == pytest.approx(1.0)
    assert rep['sim'] == pytest.approx(1.0)
    assert rep['kld'] == pytest.approx(0.0, abs=1e-6)
    assert rep['nss'] > 2
    assert rep['auc_judd'] > 0.95

  def test_disjoint_points_are_bad(self):
    a = np.array([[10, 10], [15, 20]])
    b = np.array([[150, 200], [140, 190]])
    rep = sm.saliency_report(a, b)
    assert rep['cc'] < 0.1
    assert rep['sim'] < 0.1
    assert rep['auc_judd'] < 0.6

  def test_saliency_map_is_distribution(self):
    smap = sm.saliency_map(np.array([[80, 105]]))
    assert smap.shape == (210, 160)
    assert smap.sum() == pytest.approx(1.0)


class TestPaired:

  def test_levenshtein(self):
    assert sm.levenshtein('kitten', 'sitting') == 3
    assert sm.normalized_levenshtein([1, 2, 3], [1, 2, 3]) == 0.0
    assert sm.normalized_levenshtein([1, 2], [3, 4]) == 1.0

  def test_scanmatch_bounds(self):
    grid = (4, 4)
    assert sm.scanmatch([0, 5, 10], [0, 5, 10], grid) == pytest.approx(1.0)
    assert sm.scanmatch([0, 0], [15, 15], grid) == 0.0
    near = sm.scanmatch([0, 5], [1, 5], grid)
    far = sm.scanmatch([0, 5], [15, 5], grid)
    assert near > far

  def test_dtw_identical_and_shifted(self):
    p = np.array([[0, 0], [10, 0], [20, 0]])
    assert sm.dtw(p, p) == 0.0
    assert sm.dtw(p, p + [0, 5]) == pytest.approx(5.0)

  def test_multimatch_identical(self):
    fix = _fix([[10, 10], [50, 60], [120, 30], [80, 190]])
    mm = sm.multimatch(fix, fix)
    for value in mm.values():
      assert value == pytest.approx(1.0)

  def test_multimatch_opposite_direction(self):
    a = _fix([[10, 100], [150, 100]])
    b = _fix([[150, 100], [10, 100]])
    mm = sm.multimatch(a, b)
    assert mm['direction'] == pytest.approx(0.0)
    assert mm['length'] == pytest.approx(1.0)

  def test_collapse_repeats(self):
    np.testing.assert_array_equal(
        sm.collapse_repeats([1, 1, 2, 2, 2, 1]), [1, 2, 1])


class TestDistributional:

  def test_entropies(self):
    assert sm.stationary_entropy([0] * 10, 16) == 0.0
    assert sm.stationary_entropy(list(range(16)), 16) == pytest.approx(1.0)
    assert sm.transition_entropy([0, 1, 0, 1, 0, 1], 16) == 0.0
    rng = np.random.default_rng(0)
    assert sm.transition_entropy(rng.integers(0, 16, 5000), 16) > 0.9

  def test_refixation(self):
    assert sm.refixation_rate([0, 1, 0, 1]) == 1.0
    assert sm.refixation_rate([0, 1, 2, 3]) == 0.0

  def test_amplitudes_in_degrees(self):
    fix = _fix([[0, 0], [160, 0]])
    amp = sm.saccade_amplitudes(fix)
    assert amp[0] == pytest.approx(44.6)

  def test_directions(self):
    fix = _fix([[0, 0], [10, 0], [10, 10]])
    dirs = sm.saccade_directions(fix)
    np.testing.assert_allclose(dirs, [0, math.pi / 2])

  def test_js_divergence(self):
    bins = np.linspace(0, 1, 11)
    assert sm.js_divergence([0.05] * 5, [0.05] * 5, bins) == 0.0
    assert sm.js_divergence([0.05] * 5, [0.95] * 5, bins) == pytest.approx(1.0)

  def test_report_and_score(self):
    rng = np.random.default_rng(1)
    pts = rng.uniform([0, 0], [160, 210], (200, 2))
    fix = _fix(pts)
    stats = sm.gaze_statistics(fix, (4, 4))
    rep = sm.distributional_report(stats, stats, extra={
        'pause': ([100, 200], [100, 200], np.linspace(0, 1000, 11))})
    assert sm.human_likeness_score(rep) == pytest.approx(0.0)
    other = sm.gaze_statistics(_fix(pts[:, ::-1] * [160 / 210, 210 / 160] * 0.1), (4, 4))
    rep2 = sm.distributional_report(other, stats)
    assert sm.human_likeness_score(rep2) > 0.1
