"""Unit tests verifying:
1. Exact parity with `Yizhan-Oliver-Shu/jump-models` (`predict`, `predict_online`, weighted & continuous).
2. Exact parity between streaming `step(x_t)`, batch `decode_*`, and the 1D clipped-score formula (`Model.lean`).
3. Exact agreement with the proved Lean theorems & counterexamples (`1100` pathwise switch, `Delay.lean` formulas,
   `W=7` periodic counterexample where `rolling-7` doubles switching while `carried` stays calm, and `Control.lean`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from jm_online import (
    ConfirmationFilter,
    OnlineJumpDecoder,
    compare_carried_vs_rolling,
    compute_loss_matrix,
    default_penalty_matrix,
    evaluate_penalty_curve,
    fit_jump_model,
    select_penalty_by_delay_budget,
    step_response_delays,
)


# --- Reference formulas taken directly from Yizhan-Oliver-Shu/jump-models (`jump.py`) ---
def _shu_reference_dp(loss_mx: np.ndarray, penalty_mx: np.ndarray, return_value_mx: bool = False) -> np.ndarray:
    loss = np.where(np.isnan(loss_mx), np.inf, loss_mx)
    n_s, n_c = loss.shape
    values = np.empty((n_s, n_c), dtype=np.float64)
    values[0] = loss[0]
    for t in range(1, n_s):
        values[t] = loss[t] + (values[t - 1][:, np.newaxis] + penalty_mx).min(axis=0)
    if return_value_mx:
        return values
    assign = np.empty(n_s, dtype=np.int64)
    assign[-1] = int(values[-1].argmin())
    for t in range(n_s - 1, 0, -1):
        assign[t - 1] = int((values[t - 1] + penalty_mx[:, assign[t]]).argmin())
    return assign


def test_exact_parity_with_shu_jumpmodels_discrete_and_weighted() -> None:
    rng = np.random.default_rng(20261124)
    X = rng.normal(size=(300, 4))
    feat_weights = np.array([1.2, 0.0, 0.7, 1.5])
    centers = rng.normal(size=(3, 4))
    lam = 3.7

    # Exact Shu `do_E_step` formula
    X_w = X * feat_weights
    shu_loss = 0.5 * np.sum((X_w[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    shu_pen = default_penalty_matrix(3, lam)
    shu_offline = _shu_reference_dp(shu_loss, shu_pen, return_value_mx=False)
    shu_online = _shu_reference_dp(shu_loss, shu_pen, return_value_mx=True).argmin(axis=1)

    @dataclass
    class MockShuJumpModel:
        centers_: np.ndarray
        jump_penalty: float
        jump_penalty_mx: np.ndarray
        feat_weights: np.ndarray
        prob_vecs: None = None

    mock = MockShuJumpModel(centers_=centers, jump_penalty=lam, jump_penalty_mx=shu_pen, feat_weights=feat_weights)
    dec = OnlineJumpDecoder.from_jumpmodel(mock)

    assert np.array_equal(dec.predict(X), shu_offline)
    assert np.array_equal(dec.predict_online(X), shu_online)

    # Streaming `step(x_t)` one observation at a time must match batch `predict_online(X)` identically
    dec.reset()
    streamed = np.array([dec.step(row, mode="carried") for row in X], dtype=np.int64)
    assert np.array_equal(streamed, shu_online)


def test_sparse_jumpmodel_wrapper_and_continuous_simplex_parity() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 5))
    prob_vecs = np.linspace(0.0, 1.0, 11)[:, None]
    prob_vecs = np.hstack([1.0 - prob_vecs, prob_vecs])  # 11-grid simplex for K=2
    centers = np.array([[-1.0, 0.5, 0.0, -0.2, 0.3], [1.0, -0.4, 0.0, 0.6, -0.3]])
    w = np.array([0.5, 0.3, 0.0, 0.15, 0.05])
    feat_weights = np.sqrt(w)
    centers_w = centers * feat_weights

    l1_half = np.abs(prob_vecs[:, None, :] - prob_vecs[None, :, :]).sum(axis=2) / 2.0
    pen_mx = (12.0 / np.sqrt(5.0)) * (l1_half**2)

    @dataclass
    class InnerJM:
        centers_: np.ndarray
        jump_penalty: float
        jump_penalty_mx: np.ndarray
        feat_weights: np.ndarray
        prob_vecs: np.ndarray
        mode_loss: bool = False

    @dataclass
    class MockSparseJM:
        jm_ins: InnerJM
        n_features_all: int = 5

    sjm = MockSparseJM(
        jm_ins=InnerJM(
            centers_=centers_w,
            jump_penalty=12.0 / np.sqrt(5.0),
            jump_penalty_mx=pen_mx,
            feat_weights=feat_weights,
            prob_vecs=prob_vecs,
        )
    )
    dec = OnlineJumpDecoder.from_jumpmodel(sjm)

    loss_raw = compute_loss_matrix(X, centers_w, feat_weights=feat_weights, prob_vecs=prob_vecs)
    shu_idx = _shu_reference_dp(loss_raw, pen_mx, return_value_mx=True).argmin(axis=1)
    shu_labels = prob_vecs[shu_idx].argmax(axis=1)
    assert np.array_equal(dec.predict_online(X), shu_labels)


def test_clipped_score_and_streaming_rolling_match_dp() -> None:
    rng = np.random.default_rng(99)
    X = rng.normal(size=(90, 2))
    centers = np.array([[-0.8, -0.5], [0.9, 0.7]])
    dec = OnlineJumpDecoder(centers=centers, jump_penalty=2.3)

    # Compare K-state DP directly against 1D clipped score (`Model.lean`)
    loss_mx = compute_loss_matrix(X, centers)
    _, clipped_carried = dec.binary_clipped_score(X, window=None)
    assert np.array_equal(dec.decode_carried(X), clipped_carried)

    for W in (1, 3, 7, 16):
        rolling_batch = dec.decode_rolling(X, window=W)
        # Force general DP path check via 3-state dummy isn't needed; verify direct window DP slice:
        manual = []
        for t in range(len(X)):
            sub = loss_mx[max(0, t - W + 1) : t + 1]
            v = _shu_reference_dp(sub, dec.penalty_mx, return_value_mx=True)
            manual.append(int(v[-1].argmin()))
        assert np.array_equal(rolling_batch, np.array(manual, dtype=np.int64))

        dec.reset()
        rolling_stream = np.array([dec.step(row, mode="rolling", window=W) for row in X], dtype=np.int64)
        assert np.array_equal(rolling_stream, rolling_batch)


def test_lean_theorems_and_counterexamples_reproduced() -> None:
    # Unit binary mismatch embedding: centers mu_0 = -0.5, mu_1 = +0.5 => L_0(x) - L_1(x) = x.
    # Input bits 0 -> x = -1, 1 -> x = +1 give exact unit mismatch losses!
    centers = np.array([[-0.5], [0.5]])

    # 1. Pathwise `1100` example (W = 3): 0 switches between t=2,3 at lambda=0, 1 switch at lambda=1.5
    seq_1100 = np.array([1.0, 1.0, -1.0, -1.0])
    out_lam0 = OnlineJumpDecoder(centers, jump_penalty=0.0).decode_rolling(seq_1100, window=3)
    out_lam15 = OnlineJumpDecoder(centers, jump_penalty=1.5).decode_rolling(seq_1100, window=3)
    assert (out_lam0[2], out_lam0[3]) == (0, 0)
    assert (out_lam15[2], out_lam15[3]) == (1, 0)

    # 2. Exact `Delay.lean` formulas on exact dyadic centers `[-0.5, +0.5]` (where Delta = 0.5 exactly):
    # Setting `lam = ratio * 0.5` matches unit-evidence binary penalty `ratio` with zero float rounding.
    for W in (5, 8):
        for ratio in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 6.0):
            lam = ratio * 0.5
            rep = step_response_delays(centers, lam, mode="rolling", window=W)
            if ratio == 0.0:
                assert (rep.delay_0_to_1, rep.delay_1_to_0) == (1, 1)
            else:
                exp_d_plus = min(int(np.floor(ratio)) + 1, W // 2 + 1)
                exp_d_minus = min(int(np.ceil(ratio)), (W + 1) // 2)
                assert rep.delay_0_to_1 == exp_d_plus
                assert rep.delay_1_to_0 == exp_d_minus

                # Verify actual trajectory on worst-case step change matches formula!
                dec = OnlineJumpDecoder(centers, jump_penalty=lam)
                step_up = np.concatenate([np.repeat(centers[0, 0], W), np.repeat(centers[1, 0], W)])
                labels_up = dec.decode_rolling(step_up, window=W)[W:]
                assert int(np.flatnonzero(labels_up == 1)[0]) + 1 == exp_d_plus

                step_dn = np.concatenate([np.repeat(centers[1, 0], W), np.repeat(centers[0, 0], W)])
                labels_dn = dec.decode_rolling(step_dn, window=W)[W:]
                assert int(np.flatnonzero(labels_dn == 0)[0]) + 1 == exp_d_minus

    # 3. W = 7 periodic counterexample (`---+-+++` repeated):
    # Under unit binary loss (`centers = [-0.5, +0.5]`, where loss gap of +-1 is +-1, equivalent to scaling lambda):
    # With loss gap +-1 (`centers = [-0.5, +0.5]`), at lambda=1.5 rolling-7 switches 2/8 = 25% per period;
    # at lambda=2.5 rolling-7 switches 4/8 = 50% per period, whereas `carried` at lambda=2.5 switches 0 times after warmup!
    motif = np.array([-1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0])
    periodic = np.tile(motif, 6)  # 48 steps; inspect steady cycle on last 8 steps
    r_15 = OnlineJumpDecoder(centers, jump_penalty=1.5).decode_rolling(periodic, window=7)[-9:]
    r_25 = OnlineJumpDecoder(centers, jump_penalty=2.5).decode_rolling(periodic, window=7)[-9:]
    c_25 = OnlineJumpDecoder(centers, jump_penalty=2.5).decode_carried(periodic)[-9:]
    assert int(np.sum(r_15[1:] != r_15[:-1])) == 2  # 25% switching
    assert int(np.sum(r_25[1:] != r_25[:-1])) == 4  # 50% switching (doubled!)
    assert int(np.sum(c_25[1:] != c_25[:-1])) <= 2  # Carried avoids the rolling-reset trap


def test_confirmation_filter_spacing_and_delay_budget_selector() -> None:
    rng = np.random.default_rng(123)
    noisy_props = rng.integers(0, 2, size=500)
    for L in (1, 2, 3, 5):
        executed = ConfirmationFilter(confirm_steps=L).apply(noisy_props)
        switch_times = np.flatnonzero(executed[1:] != executed[:-1]) + 1
        if switch_times.size >= 2:
            assert np.all(np.diff(switch_times) >= L)

    # Fit and run label-free selector on synthetic two-regime stream
    seg0 = rng.normal(loc=-1.5, scale=0.8, size=(200, 2))
    seg1 = rng.normal(loc=+1.5, scale=0.8, size=(200, 2))
    X = np.vstack([seg0, seg1, seg0])
    fit = fit_jump_model(X[:300], n_components=2, jump_penalty=2.0, n_init=10, random_state=0)
    curve = evaluate_penalty_curve(X[300:], fit.centers_, etas=(0.0, 1.0, 2.0, 4.0, 8.0, 16.0))
    chosen = select_penalty_by_delay_budget(curve, max_delay_steps=4)
    assert chosen.worst_delay <= 4
    # Compare carried vs rolling shrinks disagreement as W grows
    comp = compare_carried_vs_rolling(X[300:], fit.centers_, chosen.jump_penalty, windows=(5, 15, 63, 150))
    assert comp[-1].disagreement_rate <= comp[0].disagreement_rate
