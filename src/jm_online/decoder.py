"""Core streaming, rolling-window, and offline decoders for Statistical Jump Models.

Designed to be:
1. Dependency-light (`numpy` only — no `matplotlib` or `scikit-learn` required).
2. Bit-for-bit compatible with `Yizhan-Oliver-Shu/jump-models` (`JumpModel` and `SparseJumpModel`).
3. Transparent enough to read top-to-bottom in five minutes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_2d_float(X: Any) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D input X, got shape {arr.shape}.")
    return arr


def default_penalty_matrix(n_states: int, jump_penalty: float) -> np.ndarray:
    """Standard uniform jump penalty matrix: lambda off-diagonal, zero on diagonal."""
    if n_states < 1:
        raise ValueError("n_states must be >= 1.")
    if jump_penalty < 0:
        raise ValueError("jump_penalty must be >= 0.")
    return float(jump_penalty) * (np.ones((n_states, n_states), dtype=np.float64) - np.eye(n_states))


def compute_loss_matrix(
    X: Any,
    centers: np.ndarray,
    feat_weights: np.ndarray | None = None,
    prob_vecs: np.ndarray | None = None,
) -> np.ndarray:
    """Compute pointwise state losses matching `jumpmodels.jump.do_E_step`.

    For discrete models (`prob_vecs=None`), entry `(t, k)` is:
        0.5 * || x_t * feat_weights - centers_k ||_2^2
    Note: `jumpmodels` stores `centers_` in weighted feature space when `feat_weights` is used.
    """
    X_arr = _as_2d_float(X)
    centers_arr = np.asarray(centers, dtype=np.float64)
    if centers_arr.ndim == 1:
        centers_arr = centers_arr[:, np.newaxis]
    if feat_weights is not None:
        w = np.asarray(feat_weights, dtype=np.float64).reshape(1, -1)
        X_arr = X_arr * w

    diff = X_arr[:, np.newaxis, :] - centers_arr[np.newaxis, :, :]
    loss_mx = 0.5 * np.sum(diff * diff, axis=2)

    if prob_vecs is not None:
        pv = np.asarray(prob_vecs, dtype=np.float64)
        loss_mx = np.nan_to_num(loss_mx, nan=1e100, posinf=1e100, neginf=1e100)
        loss_mx = loss_mx @ pv.T
    else:
        loss_mx = np.where(np.isnan(loss_mx), np.inf, loss_mx)
    return loss_mx


def dp_forward_values(loss_mx: np.ndarray, penalty_mx: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Forward dynamic-programming value recursion V_t(k) = L_t(k) + min_i [V_{t-1}(i) + Lambda_{i,k}]."""
    T, K = loss_mx.shape
    values = np.empty((T, K), dtype=np.float64)
    if T == 0:
        return values

    # Fast scalar path for standard 2-state models (avoids per-step NumPy array allocation overhead)
    if K == 2:
        p00, p01 = float(penalty_mx[0, 0]), float(penalty_mx[0, 1])
        p10, p11 = float(penalty_mx[1, 0]), float(penalty_mx[1, 1])
        l0 = loss_mx[:, 0].tolist()
        l1 = loss_mx[:, 1].tolist()
        v0, v1 = l0[0], l1[0]
        if normalize and np.isfinite(v0) and np.isfinite(v1):
            m = v0 if v0 <= v1 else v1
            v0 -= m
            v1 -= m
        values[0, 0], values[0, 1] = v0, v1
        for t in range(1, T):
            a0, b0 = v0 + p00, v1 + p10
            m0 = a0 if a0 <= b0 else b0
            a1, b1 = v0 + p01, v1 + p11
            m1 = a1 if a1 <= b1 else b1
            v0 = l0[t] + m0
            v1 = l1[t] + m1
            if normalize and np.isfinite(v0) and np.isfinite(v1):
                m = v0 if v0 <= v1 else v1
                v0 -= m
                v1 -= m
            values[t, 0], values[t, 1] = v0, v1
        return values

    cur = loss_mx[0].copy()
    if normalize and np.all(np.isfinite(cur)):
        cur = cur - np.min(cur)
    values[0] = cur
    for t in range(1, T):
        cur = loss_mx[t] + np.min(cur[:, np.newaxis] + penalty_mx, axis=0)
        if normalize and np.all(np.isfinite(cur)):
            cur = cur - np.min(cur)
        values[t] = cur
    return values


def dp_viterbi_offline(loss_mx: np.ndarray, penalty_mx: np.ndarray) -> np.ndarray:
    """Full-window offline Viterbi decoding (uses future observations via backward traceback)."""
    T, K = loss_mx.shape
    if T == 0:
        return np.empty(0, dtype=np.int64)
    values = dp_forward_values(loss_mx, penalty_mx, normalize=False)
    if K == 2:
        p00, p01 = float(penalty_mx[0, 0]), float(penalty_mx[0, 1])
        p10, p11 = float(penalty_mx[1, 0]), float(penalty_mx[1, 1])
        v0_list = values[:, 0].tolist()
        v1_list = values[:, 1].tolist()
        out = [0] * T
        cur_lbl = 0 if v0_list[-1] <= v1_list[-1] else 1
        out[-1] = cur_lbl
        for t in range(T - 1, 0, -1):
            if cur_lbl == 0:
                c0, c1 = v0_list[t - 1] + p00, v1_list[t - 1] + p10
            else:
                c0, c1 = v0_list[t - 1] + p01, v1_list[t - 1] + p11
            cur_lbl = 0 if c0 <= c1 else 1
            out[t - 1] = cur_lbl
        return np.asarray(out, dtype=np.int64)

    labels = np.empty(T, dtype=np.int64)
    labels[-1] = int(np.argmin(values[-1]))
    for t in range(T - 1, 0, -1):
        labels[t - 1] = int(np.argmin(values[t - 1] + penalty_mx[:, labels[t]]))
    return labels


class OnlineJumpDecoder:
    """Unified online/rolling/offline decoder compatible with `jumpmodels`.

    Modes:
    - `carried` (`predict_online` in `jumpmodels`): propagates arrival costs indefinitely from t=0.
    - `rolling`: solves a fresh length-W optimization on the most recent W points and emits the last state.
    - `offline` (`predict` in `jumpmodels`): full-sample Viterbi with backward traceback (non-causal reference).
    """

    def __init__(
        self,
        centers: Any,
        jump_penalty: float = 0.0,
        feat_weights: Any | None = None,
        penalty_mx: Any | None = None,
        prob_vecs: Any | None = None,
    ) -> None:
        self.centers = np.asarray(centers, dtype=np.float64)
        if self.centers.ndim == 1:
            self.centers = self.centers[:, np.newaxis]
        self.n_components = int(self.centers.shape[0])
        self.n_features = int(self.centers.shape[1])
        self.jump_penalty = float(jump_penalty)
        self.feat_weights = None if feat_weights is None else np.asarray(feat_weights, dtype=np.float64).ravel()
        self.prob_vecs = None if prob_vecs is None else np.asarray(prob_vecs, dtype=np.float64)

        n_grid = self.n_components if self.prob_vecs is None else int(self.prob_vecs.shape[0])
        if penalty_mx is not None:
            self.penalty_mx = np.asarray(penalty_mx, dtype=np.float64)
        else:
            self.penalty_mx = default_penalty_matrix(n_grid, self.jump_penalty)
        if self.penalty_mx.shape != (n_grid, n_grid):
            raise ValueError(f"penalty_mx shape {self.penalty_mx.shape} != expected {(n_grid, n_grid)}.")

        self._carried_values: np.ndarray | None = None
        self._rolling_losses: deque[np.ndarray] = deque()

    @classmethod
    def from_jumpmodel(cls, model: Any, jump_penalty: float | None = None) -> "OnlineJumpDecoder":
        """Wrap a fitted `jumpmodels.JumpModel` or `jumpmodels.SparseJumpModel`."""
        is_sparse = hasattr(model, "jm_ins")
        jm = model.jm_ins if is_sparse else model
        if not hasattr(jm, "centers_"):
            raise ValueError("Model must be fitted (missing `centers_`).")

        feat_weights = getattr(jm, "feat_weights", None)
        prob_vecs = getattr(jm, "prob_vecs", None)

        if jump_penalty is None:
            eff_penalty = float(getattr(jm, "jump_penalty", 0.0))
            penalty_mx = getattr(jm, "jump_penalty_mx", None)
        else:
            scale = np.sqrt(float(model.n_features_all)) if is_sparse else 1.0
            eff_penalty = float(jump_penalty) / scale
            if prob_vecs is not None:
                base_pen = float(getattr(jm, "jump_penalty", 0.0))
                if base_pen > 0 and not getattr(jm, "mode_loss", False):
                    penalty_mx = np.asarray(jm.jump_penalty_mx, dtype=np.float64) * (eff_penalty / base_pen)
                else:
                    raise ValueError("Overriding jump_penalty on continuous mode_loss JM requires refitting penalty_mx.")
            else:
                penalty_mx = default_penalty_matrix(int(jm.centers_.shape[0]), eff_penalty)

        return cls(
            centers=jm.centers_,
            jump_penalty=eff_penalty,
            feat_weights=feat_weights,
            penalty_mx=penalty_mx,
            prob_vecs=prob_vecs,
        )

    def with_penalty(self, jump_penalty: float) -> "OnlineJumpDecoder":
        """Return a fresh discrete decoder sharing the same centers and weights with a new penalty."""
        if self.prob_vecs is not None:
            raise ValueError("with_penalty is supported for discrete jump models (`prob_vecs=None`).")
        return OnlineJumpDecoder(
            centers=self.centers,
            jump_penalty=jump_penalty,
            feat_weights=self.feat_weights,
        )

    def reset(self) -> None:
        """Reset streaming state for both carried and rolling modes."""
        self._carried_values = None
        self._rolling_losses.clear()

    def _decode_grid_index(self, grid_idx: np.ndarray) -> np.ndarray:
        if self.prob_vecs is None:
            return grid_idx.astype(np.int64)
        proba = self.prob_vecs[grid_idx]
        return np.argmax(proba, axis=1).astype(np.int64)

    def step(self, x_t: Any, mode: str = "carried", window: int | None = None) -> int:
        """Process a single incoming observation x_t in streaming fashion and return its state label."""
        arr = np.asarray(x_t, dtype=np.float64).reshape(1, -1)
        loss_t = compute_loss_matrix(arr, self.centers, self.feat_weights, self.prob_vecs)[0]

        if mode == "carried":
            if self._carried_values is None:
                cur = loss_t.copy()
            else:
                cur = loss_t + np.min(self._carried_values[:, np.newaxis] + self.penalty_mx, axis=0)
            if np.all(np.isfinite(cur)):
                cur = cur - np.min(cur)
            self._carried_values = cur
            idx = int(np.argmin(cur))
        elif mode == "rolling":
            if window is None or window < 1:
                raise ValueError("window must be an integer >= 1 when mode='rolling'.")
            self._rolling_losses.append(loss_t)
            while len(self._rolling_losses) > window:
                self._rolling_losses.popleft()
            win_loss = np.vstack(self._rolling_losses)
            vals = dp_forward_values(win_loss, self.penalty_mx, normalize=True)
            idx = int(np.argmin(vals[-1]))
        else:
            raise ValueError(f"Unsupported streaming mode {mode!r}; expected 'carried' or 'rolling'.")

        if self.prob_vecs is None:
            return idx
        return int(np.argmax(self.prob_vecs[idx]))

    def decode_carried(self, X: Any) -> np.ndarray:
        """Causal carried-score online inference (exact batch equivalent of `predict_online`)."""
        loss_mx = compute_loss_matrix(X, self.centers, self.feat_weights, self.prob_vecs)
        if loss_mx.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        values = dp_forward_values(loss_mx, self.penalty_mx, normalize=True)
        grid_idx = np.argmin(values, axis=1)
        return self._decode_grid_index(grid_idx)

    predict_online = decode_carried

    def decode_rolling(self, X: Any, window: int) -> np.ndarray:
        """Causal rolling-window terminal inference: each step solves a fresh window of length <= W."""
        if window < 1:
            raise ValueError("window must be >= 1.")
        loss_mx = compute_loss_matrix(X, self.centers, self.feat_weights, self.prob_vecs)
        T = loss_mx.shape[0]
        if T == 0:
            return np.empty(0, dtype=np.int64)

        # Fast exact 1D clipped-score path for standard 2-state symmetric discrete models
        if (
            self.prob_vecs is None
            and self.n_components == 2
            and np.allclose(self.penalty_mx, default_penalty_matrix(2, self.jump_penalty))
        ):
            _, labels = self.binary_clipped_score_from_losses(loss_mx, window=window)
            return labels

        grid_idx = np.empty(T, dtype=np.int64)
        for t in range(T):
            start = max(0, t - window + 1)
            vals = dp_forward_values(loss_mx[start : t + 1], self.penalty_mx, normalize=True)
            grid_idx[t] = int(np.argmin(vals[-1]))
        return self._decode_grid_index(grid_idx)

    def decode_offline(self, X: Any) -> np.ndarray:
        """Full-sample offline Viterbi decoding (`predict` in `jumpmodels`)."""
        loss_mx = compute_loss_matrix(X, self.centers, self.feat_weights, self.prob_vecs)
        grid_idx = dp_viterbi_offline(loss_mx, self.penalty_mx)
        return self._decode_grid_index(grid_idx)

    predict = decode_offline

    def binary_clipped_score_from_losses(
        self, loss_mx: np.ndarray, window: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact 1D recursion from `Model.lean`: y_t = clip_lam(y_{t-1} + (L_t(0) - L_t(1))).

        Label is 1 iff cost(1) < cost(0), which for lambda > 0 is `y_t > 0` (ties go to 0).
        """
        if loss_mx.shape[1] != 2 or self.prob_vecs is not None:
            raise ValueError("binary_clipped_score requires a discrete 2-state model.")
        g = loss_mx[:, 0] - loss_mx[:, 1]
        T = g.shape[0]
        lam = self.jump_penalty
        scores = np.empty(T, dtype=np.float64)
        labels = np.empty(T, dtype=np.int64)

        if lam == 0.0:
            scores[:] = 0.0
            labels[:] = (g > 0.0).astype(np.int64)
            return scores, labels

        if window is None:
            y = 0.0
            g_list = g.tolist()
            out_scores = [0.0] * T
            out_labels = [0] * T
            neg_lam = -lam
            for t in range(T):
                yt = y + g_list[t]
                if yt > lam:
                    y = lam
                elif yt < neg_lam:
                    y = neg_lam
                else:
                    y = yt
                out_scores[t] = y
                out_labels[t] = 1 if y > 0.0 else 0
            return np.asarray(out_scores, dtype=np.float64), np.asarray(out_labels, dtype=np.int64)
        else:
            if window < 1:
                raise ValueError("window must be >= 1.")
            # Since clip_lam(0 + 0) = 0, left-padding by (W - 1) zeros vectorizes all T windows in W steps
            g_pad = np.pad(g, (window - 1, 0), constant_values=0.0)
            scores = np.zeros(T, dtype=np.float64)
            for s in range(window):
                scores = np.clip(scores + g_pad[s : s + T], -lam, lam)
            labels = (scores > 0.0).astype(np.int64)
        return scores, labels

    def binary_clipped_score(self, X: Any, window: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return `(scores, labels)` using the 1D clipped-score recursion."""
        loss_mx = compute_loss_matrix(X, self.centers, self.feat_weights, None)
        return self.binary_clipped_score_from_losses(loss_mx, window=window)


@dataclass(frozen=True)
class FittedJumpModel:
    centers_: np.ndarray
    labels_: np.ndarray
    val_: float
    jump_penalty: float
    n_components: int


def _kmeans_plusplus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n, d = X.shape
    centers = np.empty((k, d), dtype=np.float64)
    first = int(rng.integers(0, n))
    centers[0] = X[first]
    dist_sq = np.sum((X - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        total = float(np.sum(dist_sq))
        if total <= 0.0 or not np.isfinite(total):
            centers[c] = X[int(rng.integers(0, n))]
        else:
            probs = dist_sq / total
            idx = int(rng.choice(n, p=probs))
            centers[c] = X[idx]
        new_dist_sq = np.sum((X - centers[c]) ** 2, axis=1)
        dist_sq = np.minimum(dist_sq, new_dist_sq)
    return centers


def fit_jump_model(
    X: Any,
    n_components: int = 2,
    jump_penalty: float = 0.0,
    n_init: int = 20,
    max_iter: int = 300,
    tol: float = 1e-8,
    random_state: int = 0,
    sort_by_feature: int | None = 0,
) -> FittedJumpModel:
    """Dependency-free coordinate-descent fitter for discrete Statistical Jump Models.

    Alternates exact Viterbi state assignment with cluster-mean updates (`Bemporad et al., 2018`).
    When `sort_by_feature` is an integer, states are ordered by ascending center value on that coordinate
    (e.g., state 0 = low/calm, state 1 = high/elevated); if `None`, states are ordered by center L2 norm.
    """
    X_arr = _as_2d_float(X)
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("Training array X must contain only finite values.")
    n, d = X_arr.shape
    if n < n_components:
        raise ValueError(f"Need at least {n_components} rows to fit {n_components} states, got {n}.")

    rng = np.random.default_rng(random_state)
    penalty_mx = default_penalty_matrix(n_components, jump_penalty)

    best_val = np.inf
    best_centers: np.ndarray | None = None
    best_labels: np.ndarray | None = None

    for _ in range(n_init):
        centers = _kmeans_plusplus_init(X_arr, n_components, rng)
        prev_val = np.inf
        labels = np.zeros(n, dtype=np.int64)

        for _ in range(max_iter):
            loss_mx = compute_loss_matrix(X_arr, centers)
            labels = dp_viterbi_offline(loss_mx, penalty_mx)
            jumps = int(np.sum(labels[1:] != labels[:-1]))
            val = float(np.sum(loss_mx[np.arange(n), labels]) + jump_penalty * jumps)

            new_centers = centers.copy()
            for k in range(n_components):
                mask = labels == k
                if np.any(mask):
                    new_centers[k] = np.mean(X_arr[mask], axis=0)

            shift = float(np.max(np.abs(new_centers - centers)))
            centers = new_centers
            if abs(prev_val - val) <= tol and shift <= tol:
                prev_val = val
                break
            prev_val = val

        if prev_val < best_val:
            best_val = prev_val
            best_centers = centers.copy()
            best_labels = labels.copy()

    assert best_centers is not None and best_labels is not None
    if sort_by_feature is not None:
        order = np.argsort(best_centers[:, sort_by_feature], kind="mergesort")
    else:
        order = np.argsort(np.linalg.norm(best_centers, axis=1), kind="mergesort")
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n_components)
    best_centers = best_centers[order]
    best_labels = inv_order[best_labels]

    return FittedJumpModel(
        centers_=best_centers,
        labels_=best_labels,
        val_=float(best_val),
        jump_penalty=float(jump_penalty),
        n_components=int(n_components),
    )
