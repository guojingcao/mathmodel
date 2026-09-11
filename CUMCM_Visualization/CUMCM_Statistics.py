"""Numerically safe model-evaluation metrics with transparent NaN handling."""
from __future__ import annotations
import numpy as np

def _pair(y_true, y_pred):
    a, b = np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel()
    if not len(a) or not len(b): raise ValueError("inputs must be non-empty")
    if len(a) != len(b): raise ValueError("y_true and y_pred must have the same length")
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any(): raise ValueError("inputs contain no finite paired values")
    return a[mask], b[mask]

def mse(y_true, y_pred) -> float:
    """Mean squared error, ignoring paired NaN/inf observations."""
    a, b = _pair(y_true, y_pred); return float(np.mean((a - b) ** 2))
def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""; return float(np.sqrt(mse(y_true, y_pred)))
def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    a, b = _pair(y_true, y_pred); return float(np.mean(np.abs(a - b)))
def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination; returns NaN for a constant target."""
    a, b = _pair(y_true, y_pred); total = np.sum((a-a.mean())**2)
    return float(1 - np.sum((a-b)**2)/total) if total else float("nan")
def adjusted_r2(y_true, y_pred, n_features: int) -> float:
    """Adjusted R² for a model with ``n_features`` predictors."""
    a, _ = _pair(y_true, y_pred)
    if n_features < 0 or len(a) <= n_features + 1: raise ValueError("need n > n_features + 1")
    score = r2_score(a, y_pred); return float(1 - (1-score)*(len(a)-1)/(len(a)-n_features-1))
def mape(y_true, y_pred) -> float:
    """MAPE (%) excluding zero true values, avoiding division-by-zero inflation."""
    a, b = _pair(y_true, y_pred); mask = a != 0
    return float(np.mean(np.abs((a[mask]-b[mask])/a[mask]))*100) if mask.any() else float("nan")
def smape(y_true, y_pred) -> float:
    """Symmetric MAPE (%) with zero denominators excluded."""
    a,b = _pair(y_true,y_pred); den=np.abs(a)+np.abs(b); mask=den!=0
    return float(np.mean(2*np.abs(a[mask]-b[mask])/den[mask])*100) if mask.any() else float("nan")
def rmsle(y_true, y_pred) -> float:
    """Root mean squared logarithmic error for non-negative values."""
    a,b=_pair(y_true,y_pred)
    if np.any(a < 0) or np.any(b < 0): raise ValueError("RMSLE requires non-negative values")
    return float(np.sqrt(np.mean((np.log1p(a)-np.log1p(b))**2)))
def aic(residual, n_parameters: int) -> float:
    """Gaussian-residual AIC up to an additive constant."""
    r=np.asarray(residual,float); r=r[np.isfinite(r)]
    if not len(r) or n_parameters < 0: raise ValueError("finite residuals and non-negative n_parameters required")
    return float(len(r)*np.log(np.mean(r*r)) + 2*n_parameters)
def bic(residual, n_parameters: int) -> float:
    """Gaussian-residual BIC up to an additive constant."""
    r=np.asarray(residual,float); r=r[np.isfinite(r)]
    if not len(r) or n_parameters < 0: raise ValueError("finite residuals and non-negative n_parameters required")
    return float(len(r)*np.log(np.mean(r*r)) + n_parameters*np.log(len(r)))
