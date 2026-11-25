"""Label-free latency-vs-churn diagnostics and rolling-vs-carried comparisons.

Bridges the proved binary SJM laws (`Delay.lean`, `Frontier.lean`, `Control.lean`)
to real-valued multi-feature Jump Models without requiring ground-truth state labels
or downstream trading backtests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .decoder import OnlineJumpDecoder, compute_loss_matrix


def count_switches(labels: Sequence[int] | np.ndarray) -> int:
    arr = np.asarray(labels)
    if arr.size <= 1:
        return 0
    return int(np.sum(arr[1:] != arr[:-1]))


def characteristic_loss_scale(
    X: Any,
    centers: np.ndarray,
    feat_weights: np.ndarray | None = None,
) -> float:
    """Natural evidence unit s_loss on training data: median positive pointwise loss gap.

    Setting `jump_penalty = eta * s_loss` makes dimensionless `eta` comparable across
    different feature dimensions, units, and sensor modalities.
    """
    loss_mx = compute_loss_matrix(X, centers, feat_weights=feat_weights)
    K = loss_mx.shape[1]
    if K < 2:
        return 1.0
    if K == 2:
        gaps = np.abs(loss_mx[:, 0] - loss_mx[:, 1])
    else:
        sorted_losses = np.sort(loss_mx, axis=1)
        gaps = sorted_losses[:, 1] - sorted_losses[:, 0]
    finite_pos = gaps[np.isfinite(gaps) & (gaps > 0.0)]
    if finite_pos.size == 0:
        return 1.0
    return float(np.median(finite_pos))


@dataclass(frozen=True)
class StepDelayReport:
    delay_0_to_1: int
    delay_1_to_0: int
    worst_delay: int
    center_gap: float


def step_response_delays(
    centers: np.ndarray,
    jump_penalty: float,
    mode: str = "carried",
    window: int | None = None,
    feat_weights: np.ndarray | None = None,
) -> StepDelayReport:
    """Exact deterministic detection delay after a clean regime shift between centers.

    Generalizes `Delay.lean` from unit binary inputs (Delta = 1) to fitted real centers
    with per-step loss separation Delta = 0.5 * || (mu_1 - mu_0) * w ||_2^2:
      - At lambda = 0: both delays equal 1.
      - For lambda > 0 in carried mode:
            d_{0->1} = floor(lambda / Delta) + 1   (strict inequality needed, tie -> 0)
            d_{1->0} = max(1, ceil(lambda / Delta)) (tie -> 0 helps return to 0)
      - In rolling-W mode, delays are capped by floor(W/2) + 1 and ceil(W/2) respectively
        whenever Delta > 0 (or verified directly on the decoder for general K >= 2).
    """
    c = np.asarray(centers, dtype=np.float64)
    if c.ndim == 1:
        c = c[:, np.newaxis]
    if c.shape[0] < 2:
        raise ValueError("Need at least 2 states to measure step-response delay.")

    w = np.ones(c.shape[1], dtype=np.float64) if feat_weights is None else np.asarray(feat_weights, dtype=np.float64).ravel()
    # note: if `centers` is already in weighted space (as in `jumpmodels`), `OnlineJumpDecoder` multiplies X by w.
    # A clean physical observation at regime k's unweighted prototype has weighted coordinates equal to `c[k]`.
    diff01 = c[1] - c[0]
    delta = 0.5 * float(np.sum(diff01 * diff01))
    if delta <= 0.0:
        raise ValueError("Centers 0 and 1 are identical; step-response delay is undefined.")

    lam = float(jump_penalty)
    if c.shape[0] == 2:
        if lam == 0.0:
            d01, d10 = 1, 1
        else:
            ratio = lam / delta
            # Guard tiny float rounding around exact integers (e.g., 3.0 / 1.0)
            eps = 1e-12 * max(1.0, abs(ratio))
            d01 = int(math.floor(ratio + eps)) + 1
            d10 = max(1, int(math.ceil(ratio - eps)))
            if mode == "rolling":
                if window is None or window < 1:
                    raise ValueError("window must be >= 1 when mode='rolling'.")
                d01 = min(d01, window // 2 + 1)
                d10 = min(d10, (window + 1) // 2)
            elif mode != "carried":
                raise ValueError(f"Unsupported mode {mode!r}; expected 'carried' or 'rolling'.")
        return StepDelayReport(
            delay_0_to_1=d01,
            delay_1_to_0=d10,
            worst_delay=max(d01, d10),
            center_gap=delta,
        )

    # General K >= 2 fallback: direct simulation across all ordered pairs (i -> j)
    dec = OnlineJumpDecoder(centers=c, jump_penalty=lam, feat_weights=None)
    warmup = max(int(math.ceil(2.0 * lam / delta)) + 5, (window or 1) + 2)
    horizon = warmup + 5
    worst = 1
    d01_val, d10_val = 1, 1
    for i in range(c.shape[0]):
        for j in range(c.shape[0]):
            if i == j:
                continue
            seq = np.vstack([np.repeat(c[i : i + 1], warmup, axis=0), np.repeat(c[j : j + 1], horizon, axis=0)])
            out = dec.decode_carried(seq) if mode == "carried" else dec.decode_rolling(seq, window=int(window))  # type: ignore[arg-type]
            post = out[warmup:]
            hits = np.flatnonzero(post == j)
            lag = int(hits[0]) + 1 if hits.size > 0 else horizon + 1
            worst = max(worst, lag)
            if i == 0 and j == 1:
                d01_val = lag
            elif i == 1 and j == 0:
                d10_val = lag
    return StepDelayReport(delay_0_to_1=d01_val, delay_1_to_0=d10_val, worst_delay=worst, center_gap=delta)


class ConfirmationFilter:
    """Consecutive-proposal confirmation controller (`Control.lean`).

    Requires `L >= 1` consecutive proposals different from the active state before switching.
    Guarantees at least `L` steps between any two executed switches on EVERY input sequence,
    capping switches over any `H` steps at `ceil(H / L)` at the cost of `L - 1` extra delay.
    """

    def __init__(self, confirm_steps: int = 1, initial_state: int | None = None) -> None:
        if confirm_steps < 1:
            raise ValueError("confirm_steps L must be >= 1.")
        self.confirm_steps = int(confirm_steps)
        self._initial_state = initial_state
        self.active_state: int | None = initial_state
        self.streak: int = 0
        self.candidate: int | None = None

    def reset(self) -> None:
        self.active_state = self._initial_state
        self.streak = 0
        self.candidate = None

    def step(self, proposal: int) -> int:
        p = int(proposal)
        if self.active_state is None:
            self.active_state = p
            self.streak = 0
            self.candidate = None
            return p

        if self.confirm_steps == 1:
            self.active_state = p
            return p

        if p == self.active_state:
            self.streak = 0
            self.candidate = None
            return self.active_state

        if self.candidate == p:
            self.streak += 1
        else:
            self.candidate = p
            self.streak = 1

        if self.streak >= self.confirm_steps:
            self.active_state = p
            self.streak = 0
            self.candidate = None
        return self.active_state

    def apply(self, proposals: Sequence[int] | np.ndarray) -> np.ndarray:
        self.reset()
        return np.array([self.step(int(p)) for p in proposals], dtype=np.int64)


@dataclass(frozen=True)
class PenaltyPoint:
    eta: float
    jump_penalty: float
    online_switches: int
    switches_per_1000: float
    offline_switches: int
    delay_0_to_1: int
    delay_1_to_0: int
    worst_delay: int
    disagree_with_offline_rate: float


def evaluate_penalty_curve(
    X: Any,
    centers: np.ndarray,
    etas: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
    mode: str = "carried",
    window: int | None = None,
    feat_weights: np.ndarray | None = None,
    loss_scale: float | None = None,
) -> list[PenaltyPoint]:
    """Sweep normalized penalties `lambda = eta * s_loss` on unlabeled data X.

    Produces the label-free Churn-vs-Latency tradeoff table.
    """
    scale = characteristic_loss_scale(X, centers, feat_weights) if loss_scale is None else float(loss_scale)
    n_obs = np.asarray(X).shape[0]
    if n_obs == 0:
        raise ValueError("X must be non-empty.")

    curve: list[PenaltyPoint] = []
    for eta in etas:
        lam = float(eta) * scale
        dec = OnlineJumpDecoder(centers=centers, jump_penalty=lam, feat_weights=feat_weights)
        online_labels = (
            dec.decode_carried(X) if mode == "carried" else dec.decode_rolling(X, window=int(window))  # type: ignore[arg-type]
        )
        offline_labels = dec.decode_offline(X)
        delays = step_response_delays(centers, lam, mode=mode, window=window, feat_weights=feat_weights)

        on_sw = count_switches(online_labels)
        off_sw = count_switches(offline_labels)
        disagree = float(np.mean(online_labels != offline_labels))

        curve.append(
            PenaltyPoint(
                eta=float(eta),
                jump_penalty=lam,
                online_switches=on_sw,
                switches_per_1000=1000.0 * on_sw / max(1, n_obs - 1),
                offline_switches=off_sw,
                delay_0_to_1=delays.delay_0_to_1,
                delay_1_to_0=delays.delay_1_to_0,
                worst_delay=delays.worst_delay,
                disagree_with_offline_rate=disagree,
            )
        )
    return curve


def select_penalty_by_delay_budget(
    curve: Sequence[PenaltyPoint],
    max_delay_steps: int,
) -> PenaltyPoint:
    """Select the design meeting `worst_delay <= max_delay_steps` with fewest online switches.

    Ties in observed switch count prefer the larger penalty (maximizing hysteresis margin within
    the delay budget, analogous to `lambda_d` in `Frontier.lean`).
    """
    if max_delay_steps < 1:
        raise ValueError("max_delay_steps must be >= 1.")
    feasible = [pt for pt in curve if pt.worst_delay <= max_delay_steps]
    if not feasible:
        return min(curve, key=lambda pt: (pt.worst_delay, pt.online_switches))
    return min(feasible, key=lambda pt: (pt.online_switches, -pt.jump_penalty))


@dataclass(frozen=True)
class ComparisonRow:
    window: int
    carried_switches: int
    rolling_switches: int
    extra_rolling_switches: int
    disagreement_rate: float
    saturation_horizon_est: int


def compare_carried_vs_rolling(
    X: Any,
    centers: np.ndarray,
    jump_penalty: float,
    windows: Sequence[int] = (7, 15, 31, 63, 126, 252),
    feat_weights: np.ndarray | None = None,
) -> list[ComparisonRow]:
    """Compare `predict_online` (`carried`) against `rolling-W` across window lengths W.

    Highlights both:
    - Coalescence at large W (disagreement -> 0 once W routinely contains a +-lambda saturation),
    - Rolling-reset instability at short/medium W (where `rolling-W` can switch MORE than `carried`).
    """
    dec = OnlineJumpDecoder(centers=centers, jump_penalty=jump_penalty, feat_weights=feat_weights)
    carried = dec.decode_carried(X)
    c_sw = count_switches(carried)
    s_loss = characteristic_loss_scale(X, centers, feat_weights=feat_weights)
    sat_est = max(1, int(math.ceil(2.0 * float(jump_penalty) / max(s_loss, 1e-12))))

    rows: list[ComparisonRow] = []
    for w in windows:
        rolling = dec.decode_rolling(X, window=int(w))
        r_sw = count_switches(rolling)
        rows.append(
            ComparisonRow(
                window=int(w),
                carried_switches=c_sw,
                rolling_switches=r_sw,
                extra_rolling_switches=r_sw - c_sw,
                disagreement_rate=float(np.mean(carried != rolling)),
                saturation_horizon_est=sat_est,
            )
        )
    return rows
