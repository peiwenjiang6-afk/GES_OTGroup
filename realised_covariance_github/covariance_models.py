"""SPD geometry and the seven covariance forecasting methods from Section 5.1.

The AIRM eigenvalue floor is 1e-10. vech-HAR uses overlapping 1/5/20-day
averages; HAR-DRD uses nonoverlapping daily, lags 2-5 and lags 6-20.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

EPS = 1e-10

def sym(A):
    return 0.5 * (A + A.T)

def spd_project(A, eps=EPS):
    w, V = eigh(sym(A))
    w = np.maximum(w, eps)
    return sym(V * w @ V.T)

def spd_power(A, p):
    w, V = eigh(sym(A))
    w = np.maximum(w, EPS)
    return sym(V * w ** p @ V.T)

def spd_log(A):
    w, V = eigh(sym(A))
    w = np.maximum(w, EPS)
    return sym(V * np.log(w) @ V.T)

def spd_exp(A):
    w, V = eigh(sym(A))
    return sym(V * np.exp(w) @ V.T)

def airm_dist2(A, B):
    A_mhalf = spd_power(A, -0.5)
    C = sym(A_mhalf @ B @ A_mhalf)
    w = np.linalg.eigvalsh(C)
    w = np.maximum(w, EPS)
    return float(np.sum(np.log(w) ** 2))

def airm_geodesic(A, B, a):
    A_half = spd_power(A, 0.5)
    A_mhalf = spd_power(A, -0.5)
    C = sym(A_mhalf @ B @ A_mhalf)
    return sym(A_half @ spd_power(C, a) @ A_half)

def frobenius_loss(actual, forecast):
    return float(np.linalg.norm(actual - forecast, ord='fro'))

def gaussian_nll(return_pct, forecast_cov_pct2):
    """Per-day Gaussian NLL; return and covariance are both in percentage units."""
    H = spd_project(forecast_cov_pct2)
    r = np.asarray(return_pct, dtype=float)
    sign, logdet = np.linalg.slogdet(H)
    if sign <= 0:
        raise FloatingPointError('Forecast covariance is not positive definite.')
    quad = float(r @ np.linalg.solve(H, r))
    n = len(r)
    return 0.5 * (n * np.log(2.0 * np.pi) + logdet + quad)

def covariance_scores(actual_cov, forecast_cov, demeaned_return_pct):
    H = spd_project(forecast_cov)
    return {'AIRM2': airm_dist2(actual_cov, H), 'Frobenius': frobenius_loss(actual_cov, H), 'NLL': gaussian_nll(demeaned_return_pct, H)}

def ges_run(Y, alpha):
    level = Y[0].copy()
    loss = 0.0
    for t in range(1, len(Y)):
        loss += airm_dist2(Y[t], level)
        level = airm_geodesic(level, Y[t], alpha)
    return (level, loss / (len(Y) - 1))

def fit_ges(Y, alpha_grid):
    losses = []
    for alpha in alpha_grid:
        _, loss = ges_run(Y, alpha)
        losses.append(loss)
    best = np.argmin(losses)
    alpha_hat = alpha_grid[best]
    level, _ = ges_run(Y, alpha_hat)
    return (alpha_hat, level)

def ewma_run(Y, alpha):
    level = Y[0].copy()
    loss = 0.0
    for t in range(1, len(Y)):
        loss += airm_dist2(Y[t], level)
        level = (1 - alpha) * level + alpha * Y[t]
        level = spd_project(level)
    return (level, loss / (len(Y) - 1))

def fit_ewma(Y, alpha_grid):
    losses = []
    for alpha in alpha_grid:
        _, loss = ewma_run(Y, alpha)
        losses.append(loss)
    best = np.argmin(losses)
    alpha_hat = alpha_grid[best]
    level, _ = ewma_run(Y, alpha_hat)
    return (alpha_hat, level)

def frechet_mean(Y, tol=1e-06, max_iter=30):
    M = spd_project(np.mean(Y, axis=0))
    for _ in range(max_iter):
        M_half = spd_power(M, 0.5)
        M_mhalf = spd_power(M, -0.5)
        Delta = np.zeros_like(M)
        for A in Y:
            C = sym(M_mhalf @ A @ M_mhalf)
            Delta += spd_log(C)
        Delta /= len(Y)
        if np.linalg.norm(Delta, 'fro') < tol:
            break
        M = sym(M_half @ spd_exp(Delta) @ M_half)
    return spd_project(M)

def fit_gar(Y, phi_grid):
    mu = frechet_mean(Y)
    losses = []
    for phi in phi_grid:
        loss = 0.0
        for t in range(len(Y) - 1):
            forecast = airm_geodesic(mu, Y[t], phi)
            loss += airm_dist2(Y[t + 1], forecast)
        losses.append(loss / (len(Y) - 1))
    phi_hat = phi_grid[np.argmin(losses)]
    return (mu, phi_hat)

def gar_forecast(last_Y, mu, phi):
    return airm_geodesic(mu, last_Y, phi)

def vech(A):
    """Half-vectorize the lower triangle of a square matrix."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix.')
    idx = np.tril_indices(A.shape[0])
    return np.asarray(A[idx], dtype=float)

def unvech(v, n):
    """Reconstruct a symmetric n x n matrix from lower-triangular vech."""
    v = np.asarray(v, dtype=float)
    idx = np.tril_indices(n)
    if len(v) != len(idx[0]):
        raise ValueError(f'vech length {len(v)} is incompatible with n={n}.')
    A = np.zeros((n, n), dtype=float)
    A[idx] = v
    A[idx[1], idx[0]] = v
    return sym(A)

def fit_vech_har(Y):
    """Fit the scalar-slope vech-HAR on a rolling training window."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 3 or Y.shape[1] != Y.shape[2]:
        raise ValueError('Y must have shape (T, n, n).')
    T, n, _ = Y.shape
    if T <= 20:
        raise ValueError('Scalar vech-HAR requires more than 20 observations.')
    S = np.asarray([vech(A) for A in Y], dtype=float)
    d = S.shape[1]
    rows = []
    target = []
    for t in range(20, T):
        daily = S[t - 1]
        weekly = S[t - 5:t].mean(axis=0)
        monthly = S[t - 20:t].mean(axis=0)
        for j in range(d):
            intercept_dummy = np.zeros(d, dtype=float)
            intercept_dummy[j] = 1.0
            rows.append(np.r_[intercept_dummy, daily[j], weekly[j], monthly[j]])
            target.append(S[t, j])
    X = np.asarray(rows, dtype=float)
    y = np.asarray(target, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {'intercept': np.asarray(beta[:d], dtype=float), 'slopes': np.asarray(beta[d:], dtype=float), 'n_assets': int(n)}

def vech_har_forecast(Y_history, model):
    """One-step-ahead covariance forecast from a fitted scalar vech-HAR."""
    Y_history = np.asarray(Y_history, dtype=float)
    n = int(model['n_assets'])
    if len(Y_history) < 20:
        raise ValueError('Scalar vech-HAR requires at least 20 historical observations.')
    if Y_history.shape[1:] != (n, n):
        raise ValueError('Y_history dimension does not match the fitted vech-HAR model.')
    S = np.asarray([vech(A) for A in Y_history], dtype=float)
    daily = S[-1]
    weekly = S[-5:].mean(axis=0)
    monthly = S[-20:].mean(axis=0)
    intercept = np.asarray(model['intercept'], dtype=float)
    beta_D, beta_W, beta_M = np.asarray(model['slopes'], dtype=float)
    vhat = intercept + beta_D * daily + beta_W * weekly + beta_M * monthly
    return spd_project(unvech(vhat, n))

def covariance_to_correlation(A):
    """Convert an SPD covariance matrix to its correlation matrix."""
    A = spd_project(A)
    sd = np.sqrt(np.diag(A))
    R = A / np.outer(sd, sd)
    R = sym(R)
    np.fill_diagonal(R, 1.0)
    return R

def offdiag_indices(n):
    """Strict lower-triangular indices."""
    return np.tril_indices(n, k=-1)

def corr_vech(R):
    """Vector of unique off-diagonal correlations."""
    idx = offdiag_indices(R.shape[0])
    return np.asarray(R[idx], dtype=float)

def corr_unvech(x, n):
    """Reconstruct a correlation matrix from strict-lower entries."""
    idx = offdiag_indices(n)
    R = np.eye(n, dtype=float)
    R[idx] = x
    R[idx[1], idx[0]] = x
    return sym(R)

def nearest_correlation_spd(R, eps=1e-10):
    """Numerical safeguard only."""
    R = sym(R)
    vals, vecs = np.linalg.eigh(R)
    vals = np.maximum(vals, eps)
    R = vecs @ np.diag(vals) @ vecs.T
    R = sym(R)
    d = np.sqrt(np.maximum(np.diag(R), eps))
    R = R / np.outer(d, d)
    R = sym(R)
    np.fill_diagonal(R, 1.0)
    return R

def _har_components(X, t):
    """Oh--Patton HAR components for target X[t]."""
    daily = X[t - 1]
    weekly = X[t - 5:t - 1].mean(axis=0)
    monthly = X[t - 20:t - 5].mean(axis=0)
    return (daily, weekly, monthly)

def fit_har_drd(Y, corr_sum_cap=1.0 - 1e-08):
    """Fit the one-step HAR-DRD model on a rolling training window."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 3 or Y.shape[1] != Y.shape[2]:
        raise ValueError('Y must have shape (T, n, n).')
    T, n, _ = Y.shape
    if T <= 20:
        raise ValueError('HAR-DRD requires more than 20 observations.')
    realized_var = np.diagonal(Y, axis1=1, axis2=2)
    if np.any(realized_var <= 0):
        raise ValueError('HAR-DRD requires strictly positive realised variances.')
    log_var = np.log(realized_var)
    var_beta = np.zeros((n, 4), dtype=float)
    for i in range(n):
        rows = []
        target = []
        for t in range(20, T):
            daily, weekly, monthly = _har_components(log_var, t)
            rows.append([1.0, daily[i], weekly[i], monthly[i]])
            target.append(log_var[t, i])
        X_i = np.asarray(rows, dtype=float)
        y_i = np.asarray(target, dtype=float)
        beta_i, *_ = np.linalg.lstsq(X_i, y_i, rcond=None)
        var_beta[i] = beta_i
    corr_mats = np.asarray([covariance_to_correlation(A) for A in Y])
    corr_vec = np.asarray([corr_vech(R) for R in corr_mats])
    corr_mean = corr_vec.mean(axis=0)
    rows = []
    target = []
    for t in range(20, T):
        daily, weekly, monthly = _har_components(corr_vec, t)
        y_t = corr_vec[t] - corr_mean
        X_t = np.column_stack([daily - corr_mean, weekly - corr_mean, monthly - corr_mean])
        rows.append(X_t)
        target.append(y_t)
    X_corr = np.vstack(rows)
    y_corr = np.concatenate(target)
    ols_beta, *_ = np.linalg.lstsq(X_corr, y_corr, rcond=None)
    x0 = np.maximum(np.asarray(ols_beta, dtype=float), 0.0)
    if x0.sum() >= corr_sum_cap:
        x0 *= corr_sum_cap / x0.sum()

    def objective(beta):
        resid = y_corr - X_corr @ beta
        return float(resid @ resid)

    def gradient(beta):
        resid = y_corr - X_corr @ beta
        return -2.0 * (X_corr.T @ resid)
    constraints = [{'type': 'ineq', 'fun': lambda beta: corr_sum_cap - np.sum(beta), 'jac': lambda beta: -np.ones(3)}]
    opt = minimize(objective, x0=x0, jac=gradient, bounds=[(0.0, None)] * 3, constraints=constraints, method='SLSQP', options={'ftol': 1e-12, 'maxiter': 500, 'disp': False})
    if opt.success:
        corr_beta = np.asarray(opt.x, dtype=float)
    else:
        corr_beta = x0.copy()
    corr_beta = np.maximum(corr_beta, 0.0)
    if corr_beta.sum() >= corr_sum_cap:
        corr_beta *= corr_sum_cap / corr_beta.sum()
    return {'var_beta': var_beta, 'corr_beta': corr_beta, 'corr_mean': corr_mean, 'n_assets': n}

def har_drd_forecast(Y_history, model):
    """One-step-ahead HAR-DRD covariance forecast."""
    Y_history = np.asarray(Y_history, dtype=float)
    n = int(model['n_assets'])
    if len(Y_history) < 20:
        raise ValueError('HAR-DRD requires at least 20 historical observations.')
    if Y_history.shape[1:] != (n, n):
        raise ValueError('Y_history dimension does not match fitted HAR-DRD model.')
    realized_var = np.diagonal(Y_history, axis1=1, axis2=2)
    log_var = np.log(realized_var)
    daily_v = log_var[-1]
    weekly_v = log_var[-5:-1].mean(axis=0)
    monthly_v = log_var[-20:-5].mean(axis=0)
    B = np.asarray(model['var_beta'], dtype=float)
    log_var_hat = B[:, 0] + B[:, 1] * daily_v + B[:, 2] * weekly_v + B[:, 3] * monthly_v
    var_hat = np.exp(log_var_hat)
    corr_mats = np.asarray([covariance_to_correlation(A) for A in Y_history])
    corr_vec = np.asarray([corr_vech(R) for R in corr_mats])
    daily_r = corr_vec[-1]
    weekly_r = corr_vec[-5:-1].mean(axis=0)
    monthly_r = corr_vec[-20:-5].mean(axis=0)
    a, b, c = np.asarray(model['corr_beta'], dtype=float)
    corr_mean = np.asarray(model['corr_mean'], dtype=float)
    corr_hat_vec = (1.0 - a - b - c) * corr_mean + a * daily_r + b * weekly_r + c * monthly_r
    R_hat = corr_unvech(corr_hat_vec, n)
    if np.linalg.eigvalsh(R_hat).min() <= 1e-10:
        R_hat = nearest_correlation_spd(R_hat)
    D_hat = np.diag(np.sqrt(var_hat))
    H_hat = D_hat @ R_hat @ D_hat
    return spd_project(H_hat)
