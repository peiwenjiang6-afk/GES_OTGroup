"""Expanding-origin functional forecasts; losses are mean squared hourly errors."""
import numpy as np
from scipy import stats

ALPHA_GRID = np.linspace(0.001, 0.999, 999)
K_GRID = (2, 3, 5, 8, 10, 15)
RIDGE_GRID = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
METHODS = {
    "ges": "GES", "far": "FPCA-ridge FAR(1)", "random_walk": "Random walk",
    "seasonal_naive": "Seasonal naive (lag 52)", "frechet_mean": "Running Fréchet mean",
}


def rolling_ges(y, train_size, alpha_grid=ALPHA_GRID):
    """Select alpha from past one-step losses, update, then forecast the next curve."""
    y = np.asarray(y, dtype=float)
    states = np.repeat(y[[0]], len(alpha_grid), axis=0)
    cumulative = np.zeros(len(alpha_grid))
    a = alpha_grid[:, None]
    forecasts, selected = [], []
    for t in range(1, len(y) - 1):
        cumulative += np.mean((states - y[t]) ** 2, axis=1)
        states = (1.0 - a) * states + a * y[t]
        if t >= train_size - 1:
            best = int(np.argmin(cumulative))
            forecasts.append(states[best].copy())
            selected.append(alpha_grid[best])
    return np.asarray(forecasts), np.asarray(selected), cumulative / (len(y) - 2)


def _fpca(history):
    mu = history.mean(axis=0)
    x = history - mu
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return mu, x, vt


def _score_system(x, vt, requested_k):
    k = min(int(requested_k), len(x) - 1, x.shape[1])
    basis = vt[:k].T
    scores = x @ basis
    gram = scores[:-1].T @ scores[:-1]
    cross = scores[:-1].T @ scores[1:]
    scale = float(np.trace(gram) / k)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return basis, scores, gram, cross, scale


def _transition(gram, cross, scale, ridge):
    a = gram + float(ridge) * scale * np.eye(len(gram))
    try:
        return np.linalg.solve(a, cross)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a) @ cross


def select_far(y, train_size, k_grid=K_GRID, ridge_grid=RIDGE_GRID, n_val=20):
    """Tune K and ridge on the last 20 initial-training targets only."""
    total = np.zeros((len(k_grid), len(ridge_grid)))
    counts = np.zeros_like(total)
    for target in range(max(max(k_grid) + 5, train_size - n_val), train_size):
        mu, x, vt = _fpca(y[:target])
        for i, k in enumerate(k_grid):
            basis, scores, gram, cross, scale = _score_system(x, vt, k)
            for j, ridge in enumerate(ridge_grid):
                b = _transition(gram, cross, scale, ridge)
                forecast = mu + basis @ (scores[-1] @ b)
                total[i, j] += np.mean((y[target] - forecast) ** 2)
                counts[i, j] += 1
    loss = np.divide(total, counts, out=np.full_like(total, np.inf), where=counts > 0)
    i, j = np.unravel_index(int(np.argmin(loss)), loss.shape)
    return {"K": int(k_grid[i]), "ridge": float(ridge_grid[j]),
            "validation_mse": float(loss[i, j])}


def rolling_far(y, train_size, k, ridge):
    """Refit FPCA and the score transition at every origin; keep tuning fixed."""
    forecasts, radii = [], []
    for target in range(train_size, len(y)):
        mu, x, vt = _fpca(y[:target])
        basis, scores, gram, cross, scale = _score_system(x, vt, k)
        b = _transition(gram, cross, scale, ridge)
        forecasts.append(mu + basis @ (scores[-1] @ b))
        radii.append(float(np.max(np.abs(np.linalg.eigvals(b)))))
    return np.asarray(forecasts), np.asarray(radii)


def forecast_all(y, train_size):
    """Return forecasts, weekly losses, alpha path, tuning and FAR diagnostics."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or not np.isfinite(y).all() or not 52 <= train_size < len(y):
        raise ValueError("Expected finite curves with 52 <= train_size < number of weeks.")
    ges, alpha, profile = rolling_ges(y, train_size)
    tuning = select_far(y, train_size)
    far, radii = rolling_far(y, train_size, tuning["K"], tuning["ridge"])
    forecasts = {
        "ges": ges, "far": far, "random_walk": y[train_size - 1:-1],
        "seasonal_naive": y[np.arange(train_size, len(y)) - 52],
        "frechet_mean": np.cumsum(y, axis=0)[train_size - 1:-1]
        / np.arange(train_size, len(y))[:, None],
    }
    losses = {key: np.mean((y[train_size:] - value) ** 2, axis=1)
              for key, value in forecasts.items()}
    scale = float(np.mean((y[:train_size] - y[:train_size].mean(axis=0)) ** 2))
    if scale <= 0:
        raise ValueError("Nonpositive training variation.")
    return forecasts, losses, alpha, profile, tuning, radii, scale


def dm_test(d, bandwidth=5):
    """Notebook DM convention: Bartlett HAC and two-sided Student t(n-1)."""
    d = np.asarray(d, dtype=float)
    if not 0 <= bandwidth < len(d):
        raise ValueError("HAC bandwidth must be smaller than the evaluation sample.")
    dc = d - d.mean()
    variance = np.mean(dc ** 2)
    for lag in range(1, bandwidth + 1):
        variance += 2 * (1 - lag / (bandwidth + 1.0)) * np.mean(dc[lag:] * dc[:-lag])
    if variance <= 0:
        return np.nan, np.nan
    statistic = d.mean() / np.sqrt(variance / len(d))
    return float(statistic), float(2 * stats.t.sf(abs(statistic), df=len(d) - 1))
