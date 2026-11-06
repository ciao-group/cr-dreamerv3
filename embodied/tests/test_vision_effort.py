"""Tests for vision.calc_effort (static eye-movement effort) and the vision
square grid it is computed on.

Setup like CrAtari: observation 96x96, vision square 24x24, ALE screen
160x210. calc_effort gets the square scaled to the ALE screen (40x52).

Debug in PyCharm: right-click this file -> "Debug 'pytest in test_vision_...'",
or set a breakpoint and run the file directly (see __main__ at the bottom).
"""

import numpy as np
import pytest

from embodied.core import vision

SIZE = (96, 96)
SCREEN = (160, 210)  # (W, H) of the ALE screen
VISION_SQUARE_SIZE = (24, 24)
SCALED_SQUARE = (
    VISION_SQUARE_SIZE[0] * SCREEN[0] // SIZE[0],
    VISION_SQUARE_SIZE[1] * SCREEN[1] // SIZE[1],
)  # (40, 52)
DEG_PER_PX = np.array((44.6, 28.5)) / np.array(SCREEN)
KSTATIC = 0.001
POWER = 2


def effort_grid(count):
  """Effort of every cell as an array of shape (rows, cols)."""
  values = [
      vision.calc_effort(
          position=i,
          vision_square_count=count,
          vision_square_size=SCALED_SQUARE,
          screen_size=SCREEN,
          visual_degrees_per_pixel=DEG_PER_PX,
          kstatic=KSTATIC,
          power_static=POWER,
      )
      for i in range(count[0] * count[1])
  ]
  return np.array(values).reshape(count[1], count[0])


def expected_effort(position, count):
  """Reference: kstatic * eccentricity^power, measured in visual degrees
  between the cell center and the screen center."""
  x, y = vision.convert_1d_vision_square_position_to_2d_center(
      vision_square_position=position,
      vision_square_count=count,
      vision_square_size=SCALED_SQUARE,
      screen_size=SCREEN,
  )
  offset = np.array([x - SCREEN[0] / 2, y - SCREEN[1] / 2])
  eccentricity = np.linalg.norm(offset * DEG_PER_PX)
  return KSTATIC * eccentricity ** POWER


@pytest.mark.parametrize('n', [4, 5])
def test_effort_is_non_negative(n):
  grid = effort_grid((n, n))
  assert np.all(grid >= 0), grid


@pytest.mark.parametrize('n', [4, 5])
def test_effort_matches_distance_to_screen_center(n):
  count = (n, n)
  grid = effort_grid(count)
  expected = np.array(
      [expected_effort(i, count) for i in range(n * n)]).reshape(n, n)
  np.testing.assert_allclose(grid, expected, rtol=1e-6, atol=1e-12)


def test_effort_is_zero_at_center_cell_5x5():
  grid = effort_grid((5, 5))
  center = 2 * 5 + 2  # cell 12
  assert grid.flatten()[center] == pytest.approx(0.0, abs=1e-12), grid
  assert int(grid.argmin()) == center, grid

def test_effort_is_zero_at_center_cell_7x7():
  grid = effort_grid((7, 7))
  center = 7**2 // 2  # cell 24
  assert grid.flatten()[center] == pytest.approx(0.0, abs=1e-12), grid
  assert int(grid.argmin()) == center, grid


def test_central_cells_have_equal_effort_4x4():
  # 4x4 has no center cell: the four inner cells are equally far away.
  grid = effort_grid((4, 4))
  inner = grid[1:3, 1:3]
  np.testing.assert_allclose(inner, inner[0, 0], rtol=1e-6)


@pytest.mark.parametrize('n', [4, 5, 6, 7])
def test_effort_is_mirror_symmetric(n):
  grid = effort_grid((n, n))
  np.testing.assert_allclose(grid, grid[:, ::-1], rtol=1e-6)  # left/right
  np.testing.assert_allclose(grid, grid[::-1, :], rtol=1e-6)  # top/bottom


@pytest.mark.parametrize('n', [4, 5, 6, 7])
def test_effort_grows_towards_corners(n):
  grid = effort_grid((n, n))
  assert grid[0, 0] == grid.max(), grid


@pytest.mark.parametrize('n', [4, 5, 6, 7])
def test_grid_stays_on_screen_and_is_centered(n):
  count = (n, n)
  for axis in (0, 1):
    offsets = [
        vision.vision_square_offset(i, n, SCALED_SQUARE[axis], SCREEN[axis])
        for i in range(n)]
    assert offsets[0] == 0
    assert offsets[-1] + SCALED_SQUARE[axis] == SCREEN[axis], offsets
    centers = np.array(offsets) + SCALED_SQUARE[axis] / 2
    assert centers.mean() == pytest.approx(SCREEN[axis] / 2, abs=0.5), centers


@pytest.mark.parametrize('n', [4, 5])
def test_2d_to_1d_returns_cell_of_its_center(n):
  count = (n, n)
  for i in range(n * n):
    center = vision.convert_1d_vision_square_position_to_2d_center(
        vision_square_position=i, vision_square_count=count,
        vision_square_size=SCALED_SQUARE, screen_size=SCREEN)
    assert vision.convert_2d_gaze_position_to_1d_vision_square_position(
        gaze_position=center, vision_square_count=count,
        screen_size=SCREEN, vision_square_size=SCALED_SQUARE) == i


def test_center_position():
  assert vision.center_vision_square_position((5, 5)) == 12
  assert vision.center_vision_square_position((4, 4)) == 10
  screen_center = (SCREEN[0] // 2, SCREEN[1] // 2)
  assert vision.convert_2d_gaze_position_to_1d_vision_square_position(
      gaze_position=screen_center, vision_square_count=(5, 5),
      screen_size=SCREEN, vision_square_size=SCALED_SQUARE) == 12


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
