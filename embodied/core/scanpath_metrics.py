"""Metrics to compare human and model gaze behaviour.

Coordinates are screen pixels (x, y) with y pointing down. Fixation arrays
have the columns (x, y, t_start, duration) with times in milliseconds.

Three families of metrics are provided:

* Saliency metrics (NSS, AUC-Judd, CC, KLD, SIM) following Bylinskii et al.
  2018 and Atari-HEAD (Zhang et al. 2019).
* Paired scanpath metrics for scanpaths recorded on the same frames
  (Levenshtein, ScanMatch-style alignment, DTW, MultiMatch).
* Distributional statistics that do not require aligned game states and can
  therefore compare closed-loop rollouts with human trials.
"""

import math

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import wasserstein_distance

EPS = 1e-12
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz

# Atari-HEAD setup: 160x210 px screen spanning 44.6 x 28.5 visual degrees.
ATARI_HEAD_SCREEN = (160, 210)
ATARI_HEAD_DEG_PER_PX = (44.6 / 160, 28.5 / 210)


# ---------------------------------------------------------------------------
# Fixation detection
# ---------------------------------------------------------------------------

def detect_fixations_idt(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    max_dispersion_deg: float = 1.0,
    min_duration_ms: float = 50.0,
    deg_per_px: tuple[float, float] = ATARI_HEAD_DEG_PER_PX,
) -> np.ndarray:
    """Dispersion-threshold fixation detection (I-DT, Salvucci & Goldberg).

    Samples with NaN coordinates (blinks, tracking loss) end the current
    window. Returns an array of shape (N, 4): x, y, t_start, duration.
    """
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    ts = np.asarray(ts, np.float64)
    xd = xs * deg_per_px[0]
    yd = ys * deg_per_px[1]
    fixations = []
    n = len(xs)
    period = _sample_period(ts)
    missing = np.isnan(xs) | np.isnan(ys)
    missing_before = np.concatenate([[0], np.cumsum(missing)])
    # Last sample of the minimal window starting at each sample.
    window_end = np.searchsorted(ts, ts + min_duration_ms, 'left') - 1
    start = 0
    while start < n:
        end = window_end[start]
        if (
            missing[start] or
            ts[end] - ts[start] < min_duration_ms - period or
            missing_before[end + 1] - missing_before[start] > 0
        ):
            start += 1
            continue
        if _dispersion(xd[start:end + 1], yd[start:end + 1]) > max_dispersion_deg:
            start += 1
            continue
        x_lo, x_hi = xd[start:end + 1].min(), xd[start:end + 1].max()
        y_lo, y_hi = yd[start:end + 1].min(), yd[start:end + 1].max()
        while end + 1 < n and not np.isnan(xs[end + 1]):
            nx, ny = xd[end + 1], yd[end + 1]
            nx_lo, nx_hi = min(x_lo, nx), max(x_hi, nx)
            ny_lo, ny_hi = min(y_lo, ny), max(y_hi, ny)
            if (nx_hi - nx_lo) + (ny_hi - ny_lo) > max_dispersion_deg:
                break
            x_lo, x_hi, y_lo, y_hi = nx_lo, nx_hi, ny_lo, ny_hi
            end += 1
        duration = ts[end] - ts[start] + _sample_period(ts)
        fixations.append((
            xs[start:end + 1].mean(), ys[start:end + 1].mean(),
            ts[start], duration))
        start = end + 1
    return np.array(fixations, np.float64).reshape(-1, 4)


def _dispersion(x, y):
    return (x.max() - x.min()) + (y.max() - y.min())


def _sample_period(ts):
    if len(ts) < 2:
        return 1.0
    return float(np.median(np.diff(ts)))


# ---------------------------------------------------------------------------
# Saliency metrics
# ---------------------------------------------------------------------------

def fixation_map(points: np.ndarray, screen=ATARI_HEAD_SCREEN) -> np.ndarray:
    """Binary map (H, W) with ones at the given (x, y) points."""
    w, h = screen
    fmap = np.zeros((h, w), np.float64)
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(1)]
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    np.add.at(fmap, (yi, xi), 1.0)
    return fmap


def saliency_map(
    points: np.ndarray,
    screen=ATARI_HEAD_SCREEN,
    sigma_deg: float = 1.0,
    deg_per_px: tuple[float, float] = ATARI_HEAD_DEG_PER_PX,
) -> np.ndarray:
    """Fixation map blurred with a Gaussian of `sigma_deg` visual degrees,
    normalised to a probability distribution."""
    fmap = fixation_map(points, screen)
    sigma = (sigma_deg / deg_per_px[1], sigma_deg / deg_per_px[0])
    smap = gaussian_filter(fmap, sigma)
    return smap / max(smap.sum(), EPS)


def nss(saliency: np.ndarray, points: np.ndarray) -> float:
    """Normalized Scanpath Saliency: mean standardised saliency at fixations."""
    fmap = fixation_map(points, saliency.shape[::-1]) > 0
    if not fmap.any():
        return float('nan')
    std = saliency.std()
    norm = (saliency - saliency.mean()) / max(std, EPS)
    return float(norm[fmap].mean())


def auc_judd(saliency: np.ndarray, points: np.ndarray) -> float:
    """AUC-Judd: ROC area with fixated pixels as positives."""
    fmap = fixation_map(points, saliency.shape[::-1]) > 0
    if not fmap.any():
        return float('nan')
    sal = saliency.ravel()
    pos = np.sort(saliency[fmap])[::-1]
    n_pix = sal.size
    n_fix = pos.size
    tp = [0.0]
    fp = [0.0]
    for i, thresh in enumerate(pos):
        above = (sal >= thresh).sum()
        tp.append((i + 1) / n_fix)
        fp.append((above - i - 1) / max(n_pix - n_fix, 1))
    tp.append(1.0)
    fp.append(1.0)
    return float(_trapezoid(tp, fp))


def cc(p: np.ndarray, q: np.ndarray) -> float:
    """Pearson correlation coefficient between two saliency maps."""
    p = (p - p.mean()) / max(p.std(), EPS)
    q = (q - q.mean()) / max(q.std(), EPS)
    return float((p * q).mean())


def kld(p_pred: np.ndarray, q_true: np.ndarray) -> float:
    """KL(Q || P) with Q the ground truth, as used in Atari-HEAD."""
    p = p_pred / max(p_pred.sum(), EPS)
    q = q_true / max(q_true.sum(), EPS)
    return float((q * np.log(EPS + q / (p + EPS))).sum())


def sim(p: np.ndarray, q: np.ndarray) -> float:
    """Histogram intersection of two normalised saliency maps."""
    p = p / max(p.sum(), EPS)
    q = q / max(q.sum(), EPS)
    return float(np.minimum(p, q).sum())


def saliency_report(model_points, human_points, **kw) -> dict:
    """All saliency metrics with the human map as ground truth."""
    pred = saliency_map(model_points, **kw)
    true = saliency_map(human_points, **kw)
    return {
        'nss': nss(pred, human_points),
        'auc_judd': auc_judd(pred, human_points),
        'cc': cc(pred, true),
        'kld': kld(pred, true),
        'sim': sim(pred, true),
    }


# ---------------------------------------------------------------------------
# Paired scanpath metrics
# ---------------------------------------------------------------------------

def points_to_cells(
    points: np.ndarray,
    grid: tuple[int, int],
    screen=ATARI_HEAD_SCREEN,
) -> np.ndarray:
    """Map (x, y) points to row-major grid cell indices."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    cx = np.clip((pts[:, 0] / screen[0] * grid[0]).astype(int), 0, grid[0] - 1)
    cy = np.clip((pts[:, 1] / screen[1] * grid[1]).astype(int), 0, grid[1] - 1)
    return cy * grid[0] + cx


def collapse_repeats(seq) -> np.ndarray:
    """Merge consecutive identical entries (successive steps on one cell are
    one fixation)."""
    seq = np.asarray(seq)
    if len(seq) == 0:
        return seq
    keep = np.concatenate([[True], seq[1:] != seq[:-1]])
    return seq[keep]


def levenshtein(a, b) -> int:
    a, b = list(a), list(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_levenshtein(a, b) -> float:
    """Edit distance divided by the longer length (0 = identical)."""
    n = max(len(a), len(b))
    return levenshtein(a, b) / n if n else 0.0


def scanmatch(a, b, grid: tuple[int, int], gap: float = 0.0) -> float:
    """ScanMatch-style Needleman-Wunsch similarity on grid cells
    (Cristino et al. 2010). Substitution scores decrease linearly with the
    Euclidean distance between cells; the result is normalised to [0, 1]."""
    a, b = list(a), list(b)
    if not a or not b:
        return 0.0
    diag = math.hypot(grid[0] - 1, grid[1] - 1) or 1.0

    def sub(i, j):
        dx = i % grid[0] - j % grid[0]
        dy = i // grid[0] - j // grid[0]
        return 1.0 - 2.0 * math.hypot(dx, dy) / diag

    n, m = len(a), len(b)
    score = np.zeros((n + 1, m + 1))
    score[:, 0] = gap * np.arange(n + 1)
    score[0, :] = gap * np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score[i, j] = max(
                score[i - 1, j - 1] + sub(a[i - 1], b[j - 1]),
                score[i - 1, j] + gap,
                score[i, j - 1] + gap)
    return float(max(score[n, m], 0.0) / max(n, m))


def dtw(p: np.ndarray, q: np.ndarray) -> float:
    """Dynamic time warping distance between two point sequences, divided by
    the warping path length."""
    p = np.asarray(p, np.float64).reshape(-1, 2)
    q = np.asarray(q, np.float64).reshape(-1, 2)
    n, m = len(p), len(q)
    if n == 0 or m == 0:
        return float('nan')
    dist = np.linalg.norm(p[:, None] - q[None], axis=-1)
    cost = np.full((n + 1, m + 1), np.inf)
    steps = np.zeros((n + 1, m + 1))
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            prev = min(
                (cost[i - 1, j - 1], steps[i - 1, j - 1]),
                (cost[i - 1, j], steps[i - 1, j]),
                (cost[i, j - 1], steps[i, j - 1]))
            cost[i, j] = dist[i - 1, j - 1] + prev[0]
            steps[i, j] = prev[1] + 1
    return float(cost[n, m] / steps[n, m])


def multimatch(
    fix_a: np.ndarray,
    fix_b: np.ndarray,
    screen=ATARI_HEAD_SCREEN,
) -> dict:
    """MultiMatch (Jarodzka et al. 2010; Dewhurst et al. 2012) without the
    scanpath simplification step.

    Saccade vectors are aligned by the cheapest monotonic path through the
    matrix of vector differences. Returns similarities in [0, 1] for the
    dimensions vector (shape), direction, length, position and duration.
    """
    fix_a = np.asarray(fix_a, np.float64).reshape(-1, 4)
    fix_b = np.asarray(fix_b, np.float64).reshape(-1, 4)
    keys = ('vector', 'direction', 'length', 'position', 'duration')
    if len(fix_a) < 2 or len(fix_b) < 2:
        return {k: float('nan') for k in keys}
    diag = math.hypot(*screen)
    va = np.diff(fix_a[:, :2], axis=0)
    vb = np.diff(fix_b[:, :2], axis=0)
    diff = np.linalg.norm(va[:, None] - vb[None], axis=-1)
    pairs = _cheapest_path(diff)
    ia = np.array([p[0] for p in pairs])
    ib = np.array([p[1] for p in pairs])

    vec = np.linalg.norm(va[ia] - vb[ib], axis=-1)
    ang_a = np.arctan2(va[ia, 1], va[ia, 0])
    ang_b = np.arctan2(vb[ib, 1], vb[ib, 0])
    ang = np.abs(np.angle(np.exp(1j * (ang_a - ang_b))))
    len_diff = np.abs(np.linalg.norm(va[ia], axis=-1) - np.linalg.norm(vb[ib], axis=-1))
    pos = np.linalg.norm(fix_a[ia, :2] - fix_b[ib, :2], axis=-1)
    da, db = fix_a[ia, 3], fix_b[ib, 3]
    dur = np.abs(da - db) / np.maximum(np.maximum(da, db), EPS)
    return {
        'vector': float(1 - vec.mean() / (2 * diag)),
        'direction': float(1 - ang.mean() / math.pi),
        'length': float(1 - len_diff.mean() / diag),
        'position': float(1 - pos.mean() / diag),
        'duration': float(1 - dur.mean()),
    }


def _cheapest_path(cost: np.ndarray) -> list[tuple[int, int]]:
    n, m = cost.shape
    acc = np.full((n, m), np.inf)
    acc[0, 0] = cost[0, 0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            best = min(
                acc[i - 1, j] if i else np.inf,
                acc[i, j - 1] if j else np.inf,
                acc[i - 1, j - 1] if i and j else np.inf)
            acc[i, j] = cost[i, j] + best
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i or j:
        options = []
        if i and j:
            options.append((acc[i - 1, j - 1], i - 1, j - 1))
        if i:
            options.append((acc[i - 1, j], i - 1, j))
        if j:
            options.append((acc[i, j - 1], i, j - 1))
        _, i, j = min(options)
        path.append((i, j))
    return path[::-1]


def paired_report(model_fix, human_fix, grid, screen=ATARI_HEAD_SCREEN) -> dict:
    """Paired scanpath metrics for two fixation sequences on the same frames."""
    model_fix = np.asarray(model_fix, np.float64).reshape(-1, 4)
    human_fix = np.asarray(human_fix, np.float64).reshape(-1, 4)
    ca = collapse_repeats(points_to_cells(model_fix[:, :2], grid, screen))
    cb = collapse_repeats(points_to_cells(human_fix[:, :2], grid, screen))
    out = {
        'levenshtein': normalized_levenshtein(ca, cb),
        'scanmatch': scanmatch(ca, cb, grid),
        'dtw_px': dtw(model_fix[:, :2], human_fix[:, :2]),
    }
    out.update({f'multimatch_{k}': v for k, v in multimatch(
        model_fix, human_fix, screen).items()})
    return out


# ---------------------------------------------------------------------------
# Distributional statistics
# ---------------------------------------------------------------------------

def saccade_amplitudes(fix, deg_per_px=ATARI_HEAD_DEG_PER_PX) -> np.ndarray:
    """Saccade amplitudes in visual degrees between consecutive fixations."""
    fix = np.asarray(fix, np.float64).reshape(-1, 4)
    d = np.diff(fix[:, :2], axis=0) * np.asarray(deg_per_px)
    return np.linalg.norm(d, axis=-1)


def saccade_directions(fix, deg_per_px=ATARI_HEAD_DEG_PER_PX) -> np.ndarray:
    """Saccade directions in radians in [-pi, pi] (0 = rightwards)."""
    fix = np.asarray(fix, np.float64).reshape(-1, 4)
    d = np.diff(fix[:, :2], axis=0) * np.asarray(deg_per_px)
    d = d[np.linalg.norm(d, axis=-1) > 0]
    return np.arctan2(d[:, 1], d[:, 0])


def stationary_entropy(cells, n_cells: int) -> float:
    """Normalised Shannon entropy of the gaze distribution over cells."""
    p = np.bincount(np.asarray(cells, int), minlength=n_cells) / max(len(cells), 1)
    return float(_entropy(p) / math.log2(n_cells))


def transition_entropy(cells, n_cells: int) -> float:
    """Normalised gaze transition entropy (Krejtz et al. 2015) of the
    first-order transition matrix between distinct consecutive cells."""
    cells = collapse_repeats(np.asarray(cells, int))
    if len(cells) < 2:
        return float('nan')
    counts = np.zeros((n_cells, n_cells))
    np.add.at(counts, (cells[:-1], cells[1:]), 1)
    rows = counts.sum(1)
    pi = rows / rows.sum()
    h = 0.0
    for i in np.nonzero(rows)[0]:
        h += pi[i] * _entropy(counts[i] / rows[i])
    return float(h / math.log2(n_cells))


def refixation_rate(cells) -> float:
    """Share of gaze shifts that return to the cell visited two shifts ago."""
    cells = collapse_repeats(np.asarray(cells, int))
    if len(cells) < 3:
        return float('nan')
    return float((cells[2:] == cells[:-2]).mean())


def _entropy(p):
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


def js_divergence(a, b, bins) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1]) of two samples binned
    on common bins."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    pa, edges = np.histogram(a, bins=bins)
    pb, _ = np.histogram(b, bins=edges)
    pa = pa / max(pa.sum(), EPS)
    pb = pb / max(pb.sum(), EPS)
    m = 0.5 * (pa + pb)
    kl = lambda p, q: (p[p > 0] * np.log2(p[p > 0] / q[p > 0])).sum()
    return float(0.5 * kl(pa, m) + 0.5 * kl(pb, m))


def wasserstein(a, b) -> float:
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    return float(wasserstein_distance(a, b))


def gaze_statistics(fix, grid, screen=ATARI_HEAD_SCREEN,
                    deg_per_px=ATARI_HEAD_DEG_PER_PX) -> dict:
    """Summary statistics of one or several concatenated scanpaths."""
    fix = np.asarray(fix, np.float64).reshape(-1, 4)
    cells = points_to_cells(fix[:, :2], grid, screen)
    n = grid[0] * grid[1]
    return {
        'saccade_amplitude': saccade_amplitudes(fix, deg_per_px),
        'saccade_direction': saccade_directions(fix, deg_per_px),
        'fixation_duration': fix[:, 3],
        'stationary_entropy': stationary_entropy(cells, n),
        'transition_entropy': transition_entropy(cells, n),
        'refixation_rate': refixation_rate(cells),
    }


def distributional_report(
    model_stats: dict, human_stats: dict, extra: dict | None = None,
) -> dict:
    """Compare two `gaze_statistics` results.

    `extra` maps further names to (model_samples, human_samples, bins), e.g.
    pause durations or gaze-avatar distances.
    """
    out = {}
    specs = {
        'saccade_amplitude': np.linspace(0, 50, 26),
        'saccade_direction': np.linspace(-math.pi, math.pi, 17),
        'fixation_duration': np.concatenate([np.linspace(0, 1000, 21), [np.inf]]),
    }
    for key, bins in specs.items():
        a, b = model_stats[key], human_stats[key]
        out[f'{key}_js'] = js_divergence(a, b, bins)
        if key != 'saccade_direction':
            out[f'{key}_w1'] = wasserstein(a, b)
    for key in ('stationary_entropy', 'transition_entropy', 'refixation_rate'):
        out[f'{key}_absdiff'] = abs(model_stats[key] - human_stats[key])
    for key, (a, b, bins) in (extra or {}).items():
        out[f'{key}_js'] = js_divergence(a, b, bins)
        out[f'{key}_w1'] = wasserstein(a, b)
    return out


def human_likeness_score(report: dict) -> float:
    """Composite distance in [0, 1] (lower = more human-like): mean of all
    JS divergences and absolute differences of normalised entropies."""
    vals = [v for k, v in report.items()
            if (k.endswith('_js') or k.endswith('_absdiff')) and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float('nan')
