"""Univariate Wasserstein exponential smoothing and comparison forecasts.

Quantile rows use the same equally spaced probability grid. All forecasts are
one step ahead, use only observations available at their origin, and start at
row ``train_size``. GES follows the original mortality_ges_toolkit.py recursion.

ATM_d(1) follows Zhu and Mueller (2023), doi:10.1093/jrsssb/qkad051. This is a
monotone piecewise-linear numerical adaptation, not an exact port of the
authors' natural-cubic-spline R implementation. Maps preserve common support
endpoints; reverse quantile pairs define inverse maps. The least-squares
criterion uses trapezoidal Lebesgue integration on the common spatial interval.
"""

import numpy as np


def _validate_q(Q, train_size):
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[1] < 2 or not np.isfinite(Q).all():
        raise ValueError("Q must be a finite time-by-quantile matrix.")
    if np.any(np.diff(Q, axis=1) < 0):
        raise ValueError("Each row of Q must be nondecreasing.")
    if not isinstance(train_size, (int, np.integer)) or not 3 <= train_size < len(Q):
        raise ValueError("train_size must be an integer between 3 and len(Q)-1.")
    return Q


def squared_w2(q1, q2):
    """Midpoint-grid approximation to squared W2 distance between quantile rows."""
    q1, q2 = np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)
    if q1.shape != q2.shape or q1.ndim != 1 or not q1.size:
        raise ValueError("Quantile rows must have the same nonempty 1D shape.")
    if not np.isfinite(q1).all() or not np.isfinite(q2).all():
        raise ValueError("Quantile rows must be finite.")
    return float(np.mean((q1 - q2) ** 2))


def rolling_ges(Q, train_size):
    """Return (forecasts, selected alpha), fitting on all past one-step losses.

    Initialize every state at Q[0]. For each origin, select alpha from
    0.001,...,0.999 by cumulative squared W2 loss, then forecast with that state.
    """
    Q = _validate_q(Q, train_size)
    grid = np.linspace(0.001, 0.999, 999)
    alpha = grid[:, None]
    states = np.repeat(Q[[0]], len(grid), axis=0)
    cumulative_loss = np.zeros(len(grid), dtype=float)
    forecasts, selected = [], []
    for t in range(1, len(Q) - 1):
        errors = states - Q[t]
        cumulative_loss += np.mean(errors**2, axis=1)
        states = (1.0 - alpha) * states + alpha * Q[t]
        if t >= train_size - 1:
            best = int(np.argmin(cumulative_loss))
            forecasts.append(states[best].copy())
            selected.append(float(grid[best]))
    return np.asarray(forecasts), np.asarray(selected)


def rolling_references(Q, train_size):
    """Return (random-walk forecasts, expanding Wasserstein-Frechet means)."""
    Q = _validate_q(Q, train_size)
    means = [np.mean(Q[:t + 1], axis=0)
             for t in range(train_size - 1, len(Q) - 1)]
    return Q[train_size - 1:-1].copy(), np.asarray(means)


def _support_bounds(history, padding, eps):
    lo, hi = float(np.min(history)), float(np.max(history))
    margin = max(padding * (hi - lo), 10 * eps, 1e-8)
    return lo - margin, hi + margin + eps


def _knots(q, u, lo, hi, eps):
    # The tiny probability-dependent shift resolves empirical quantile ties.
    q = np.asarray(q) + eps * np.asarray(u)
    if not lo < q[0] or not q[-1] < hi or np.any(np.diff(q) <= 0):
        raise ValueError("Quantile knots must increase strictly inside support.")
    return np.r_[lo, q, hi]


def _transport(q_from, q_to, u, x, lo, hi, eps):
    return np.interp(x, _knots(q_from, u, lo, hi, eps),
                     _knots(q_to, u, lo, hi, eps))


def _pair_statistics(q0, q1, q2, u, lo, hi, n_x, eps):
    x = np.linspace(lo, hi, n_x)
    xp = _transport(q0, q1, u, x, lo, hi, eps) - x
    xm = x - _transport(q1, q0, u, x, lo, hi, eps)
    y = _transport(q1, q2, u, x, lo, hi, eps) - x
    products = np.array([xp * xp, xp * y, xm * xm, xm * y, y * y])
    return np.trapezoid(products, x=x, axis=1) / (x[-1] - x[0])


def _estimate_beta(stat):
    dp, np_, dm, nm, yy = stat
    bp = float(np.clip(np_ / dp, 0, .999)) if dp > 1e-14 else 0.
    bm = float(np.clip(nm / dm, -.999, 0)) if dm > 1e-14 else 0.
    lp = yy - 2 * bp * np_ + bp * bp * dp
    lm = yy - 2 * bm * nm + bm * bm * dm
    return bp if lp <= lm else bm


def _forecast_last(history, u, beta, lo, hi, eps):
    current = history[-1]
    if beta >= 0:
        moved = _transport(history[-2], current, u, current, lo, hi, eps)
        return current + beta * (moved - current)
    moved = _transport(current, history[-2], u, current, lo, hi, eps)
    return current + abs(beta) * (moved - current)


def rolling_atmd1(Q, u_grid, train_size, *, n_x=2201, eps=1e-4, padding=.05):
    """Return (forecasts, beta) for the corrected expanding-origin ATM_d(1).

    Support uses past extrema plus 5% padding. Recompute all pair contributions
    whenever support changes; otherwise add the newest contribution. This cache
    preserves the full refit's criterion without using future extrema. The
    regularization affects transport knots only; observations remain raw.
    """
    Q = _validate_q(Q, train_size)
    u = np.asarray(u_grid, dtype=float)
    if (u.ndim != 1 or len(u) != Q.shape[1] or not np.isfinite(u).all()
            or np.any(np.diff(u) <= 0) or u[0] <= 0 or u[-1] >= 1):
        raise ValueError("u_grid must increase strictly inside (0,1), matching Q.")
    if not isinstance(n_x, (int, np.integer)) or n_x < 3:
        raise ValueError("n_x must be an integer of at least 3.")
    if not np.isfinite([eps, padding]).all() or eps <= 0 or padding < 0:
        raise ValueError("eps must be positive and padding nonnegative.")
    forecasts, betas = [], []
    previous = None
    cumulative = None
    for t in range(train_size - 1, len(Q) - 1):
        history = Q[:t + 1]
        lo, hi = _support_bounds(history, padding, eps)
        if previous != (lo, hi):
            cumulative = np.sum([
                _pair_statistics(Q[k - 1], Q[k], Q[k + 1], u, lo, hi, n_x, eps)
                for k in range(1, t)
            ], axis=0)
        else:
            cumulative += _pair_statistics(
                Q[t - 2], Q[t - 1], Q[t], u, lo, hi, n_x, eps)
        beta = _estimate_beta(cumulative)
        predicted = _forecast_last(history, u, beta, lo, hi, eps)
        if not np.isfinite(predicted).all() or np.any(np.diff(predicted) < -1e-10):
            raise RuntimeError("Nonfinite or nonmonotone ATM forecast.")
        forecasts.append(predicted)
        betas.append(beta)
        previous = (lo, hi)
    return np.asarray(forecasts), np.asarray(betas)
