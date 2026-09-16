"""Small math utilities (derivative/integral/FFT/normalization) and a
restricted AST-based expression evaluator used by the Function Mixer.

Extracted from app.py (Phase 1 de-spaghetti pass).
"""

import ast
import operator as op

import numpy as np


def _safe_dt(t):
    t = np.asarray(t, float)
    if len(t) < 2:
        return 1.0
    dt = float(t[1] - t[0])
    return dt if dt != 0 else 1.0

def deriv(y, t=None):
    """Numerical derivative dy/dt using numpy gradient on uniform grid."""
    y = np.asarray(y, float)
    if t is None:
        return np.gradient(y)
    dt = _safe_dt(t)
    return np.gradient(y, dt)

def integ(y, t=None):
    """Cumulative integral ∫ y dt using cumulative trapezoid (no scipy)."""
    y = np.asarray(y, float)
    if len(y) < 2:
        return np.zeros_like(y)
    if t is None:
        # assume dt=1
        return np.cumsum((y[:-1] + y[1:]) * 0.5)  # length n-1
    t = np.asarray(t, float)
    dt = _safe_dt(t)
    out = np.zeros_like(y)
    out[1:] = np.cumsum((y[:-1] + y[1:]) * 0.5 * dt)
    return out

def fft_amp(y):
    """One-sided FFT amplitude spectrum (positive frequencies)."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 2:
        return y
    Y = np.fft.rfft(y - np.nanmean(y))
    return np.abs(Y)

def fft_power(y):
    """One-sided FFT power spectrum."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 2:
        return y
    Y = np.fft.rfft(y - np.nanmean(y))
    return (np.abs(Y) ** 2)

def fft_freqs(t):
    """FFT frequency axis (Hz) for rfft based on time grid t (seconds)."""
    t = np.asarray(t, float)
    n = len(t)
    dt = _safe_dt(t)
    return np.fft.rfftfreq(n, d=dt)

def norm_minmax(y):
    y = np.asarray(y, float)
    lo, hi = np.nanmin(y), np.nanmax(y)
    return (y - lo) / (hi - lo) if hi != lo else np.zeros_like(y)

def norm_max(y):
    y = np.asarray(y, float)
    m = np.nanmax(np.abs(y))
    return y / m if m != 0 else y

def norm_amplitude(y):
    y = np.asarray(y, float)
    amp = np.nanmax(y) - np.nanmin(y)
    return y / amp if amp != 0 else y

def norm_zscore(y):
    y = np.asarray(y, float)
    s = np.nanstd(y)
    return (y - np.nanmean(y)) / s if s != 0 else np.zeros_like(y)

def norm_mean(y):
    y = np.asarray(y, float)
    m = np.nanmean(y)
    return y / m if m != 0 else y

_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

_ALLOWED_FUNCS = {
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "exp": np.exp,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "tanh": np.tanh,
    "clip": np.clip,
    "where": np.where,
    "min": np.minimum,
    "max": np.maximum,
    "deriv": deriv,  # deriv(A) or deriv(A, t)
    "integ": integ,  # integ(A) or integ(A, t)
    "fft_amp": fft_amp,  # fft_amp(A)
    "fft_power": fft_power,
    "amax": np.max,
    "amin": np.min,
    "mean": np.mean,
    "median": np.median,
    "norm_minmax": norm_minmax,
    "norm_max": norm_max,
    "norm_amplitude": norm_amplitude,
    "norm_zscore": norm_zscore,
    "norm_mean": norm_mean,
}

_ALLOWED_NAMES = {"A", "B", "pi", "e", "t"}

def safe_eval_expr(expr: str, env: dict):
    """
    Safely evaluate a math expression using AST.
    env must provide A and B as numpy arrays, plus optional constants.
    """
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # numbers
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Only numeric constants are allowed.")

        # names
        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_NAMES:
                raise ValueError(f"Unknown variable '{node.id}'. Allowed: A, B.")
            return env[node.id]

        # binary ops
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _ALLOWED_BINOPS:
                raise ValueError("Operator not allowed.")
            return _ALLOWED_BINOPS[type(node.op)](_eval(node.left), _eval(node.right))

        # unary ops
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _ALLOWED_UNARYOPS:
                raise ValueError("Unary operator not allowed.")
            return _ALLOWED_UNARYOPS[type(node.op)](_eval(node.operand))

        # function calls
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed.")
            fname = node.func.id
            if fname not in _ALLOWED_FUNCS:
                raise ValueError(
                    f"Function '{fname}' not allowed. Allowed: {', '.join(sorted(_ALLOWED_FUNCS.keys()))}"
                )
            args = [_eval(a) for a in node.args]
            return _ALLOWED_FUNCS[fname](*args)

        raise ValueError("Expression contains unsupported syntax.")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree)
