"""End-to-end real-world benchmarks for `jm_online` (zero stock-picking alpha).

Evaluates label-free delay-budget penalty selection and `carried` (`predict_online`)
vs. `rolling-W` online inference across three public monitoring domains:
1. Smart-building HVAC sensor regime detection (UCI Occupancy Detection, 1-min telemetry).
2. Urban air-quality smog alert monitoring (UCI Beijing PM2.5, hourly telemetry).
3. Financial market volatility stress monitoring (FRED CBOE VIXCLS, daily risk telemetry).
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jm_online import (  # noqa: E402
    OnlineJumpDecoder,
    characteristic_loss_scale,
    compare_carried_vs_rolling,
    evaluate_penalty_curve,
    fit_jump_model,
    select_penalty_by_delay_budget,
)
from jm_online.diagnostics import count_switches  # noqa: E402
from jm_online.plotting import plot_carried_vs_rolling, plot_latency_churn_curve  # noqa: E402

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
ETAS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)


def _standardize_fit_transform(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu = train_df.mean(axis=0).to_numpy(dtype=np.float64)
    sd = train_df.std(axis=0, ddof=0).to_numpy(dtype=np.float64)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (train_df.to_numpy(dtype=np.float64) - mu) / sd, (test_df.to_numpy(dtype=np.float64) - mu) / sd


def _empirical_onset_lags(true_y: np.ndarray, pred_y: np.ndarray, max_search: int = 60) -> list[int]:
    """Measure minutes after each 0->1 true occupancy transition until pred_y is 1."""
    lags: list[int] = []
    n = len(true_y)
    for t in range(1, n):
        if true_y[t - 1] == 0 and true_y[t] == 1:
            end = min(n, t + max_search)
            # Only score genuine sustained occupancy episodes (>= 10 min)
            if end - t >= 10 and np.all(true_y[t : t + 10] == 1):
                hits = np.flatnonzero(pred_y[t:end] == 1)
                if hits.size > 0:
                    lags.append(int(hits[0]) + 1)
    return lags


def run_occupancy_benchmark() -> dict:
    occ_dir = DATA_DIR / "occupancy"
    train_raw = pd.read_csv(occ_dir / "datatraining.txt")
    test1_raw = pd.read_csv(occ_dir / "datatest.txt")
    test2_raw = pd.read_csv(occ_dir / "datatest2.txt")
    test_raw = pd.concat([test1_raw, test2_raw], ignore_index=True)

    feat_cols = ["Temperature", "Humidity", "Light", "CO2"]
    X_train, X_test = _standardize_fit_transform(train_raw[feat_cols], test_raw[feat_cols])
    y_test = test_raw["Occupancy"].to_numpy(dtype=np.int64)

    # Unsupervised fit on training sensors ONLY (sorted by Light: state 0 = dark/unoccupied, 1 = lit/occupied)
    base_fit = fit_jump_model(X_train, n_components=2, jump_penalty=2.0, n_init=20, random_state=42, sort_by_feature=2)
    s_loss = characteristic_loss_scale(X_train, base_fit.centers_)

    # Label-free penalty calibration on X_train with HVAC delay budget <= 5 minutes
    curve = evaluate_penalty_curve(X_train, base_fit.centers_, etas=ETAS, mode="carried", loss_scale=s_loss)
    chosen = select_penalty_by_delay_budget(curve, max_delay_steps=5)

    # Evaluate on 12,417 unseen test minutes (~8.62 days; note boundary between test1 and test2)
    dec_zero = OnlineJumpDecoder(base_fit.centers_, jump_penalty=0.0)
    dec_sel = OnlineJumpDecoder(base_fit.centers_, jump_penalty=chosen.jump_penalty)

    pred_zero = np.concatenate([dec_zero.decode_carried(X_test[: len(test1_raw)]), dec_zero.decode_carried(X_test[len(test1_raw) :])])
    pred_sel = np.concatenate([dec_sel.decode_carried(X_test[: len(test1_raw)]), dec_sel.decode_carried(X_test[len(test1_raw) :])])

    days = len(X_test) / 1440.0
    sw_zero = count_switches(pred_zero[: len(test1_raw)]) + count_switches(pred_zero[len(test1_raw) :])
    sw_sel = count_switches(pred_sel[: len(test1_raw)]) + count_switches(pred_sel[len(test1_raw) :])

    lags_zero = _empirical_onset_lags(y_test, pred_zero)
    lags_sel = _empirical_onset_lags(y_test, pred_sel)

    comp = compare_carried_vs_rolling(
        X_test[len(test1_raw) :],
        base_fit.centers_,
        chosen.jump_penalty,
        windows=(7, 15, 31, 63, 127, 255),
    )

    plot_latency_churn_curve(
        curve,
        chosen,
        title="UCI Building Sensors: Label-Free HVAC Penalty Selection",
        delay_unit="minutes",
        out_path=OUT_DIR / "occupancy_pareto.png",
    )
    plot_carried_vs_rolling(
        comp,
        title="UCI Building Sensors (Test Stream 2): Carried vs. Rolling-W",
        out_path=OUT_DIR / "occupancy_carried_vs_rolling.png",
    )

    return {
        "domain": "UCI Building Occupancy Sensors (1-min)",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "delay_budget": "5 minutes",
        "loss_scale": round(s_loss, 4),
        "selected_eta": chosen.eta,
        "selected_lambda": round(chosen.jump_penalty, 4),
        "guaranteed_worst_delay_min": chosen.worst_delay,
        "test_switches_unpenalized": sw_zero,
        "test_switches_selected": sw_sel,
        "test_switches_per_day_unpenalized": round(sw_zero / days, 2),
        "test_switches_per_day_selected": round(sw_sel / days, 2),
        "heldout_camera_accuracy_unpenalized_pct": round(100.0 * float(np.mean(pred_zero == y_test)), 2),
        "heldout_camera_accuracy_selected_pct": round(100.0 * float(np.mean(pred_sel == y_test)), 2),
        "median_empirical_entry_lag_min_unpenalized": float(np.median(lags_zero)) if lags_zero else None,
        "median_empirical_entry_lag_min_selected": float(np.median(lags_sel)) if lags_sel else None,
        "carried_vs_rolling_test2": [asdict(r) for r in comp],
    }


def run_pm25_benchmark() -> dict:
    csv_path = DATA_DIR / "beijing_pm25" / "PRSA_data_2010.1.1-2014.12.31.csv"
    df = pd.read_csv(csv_path, na_values=["NA"])
    # Causal forward-fill only for missing hourly PM2.5 readings; drop initial leading NAs
    df["pm2.5"] = df["pm2.5"].ffill()
    df = df.dropna(subset=["pm2.5", "DEWP", "TEMP", "PRES", "Iws"]).reset_index(drop=True)

    log_pm = np.log1p(df["pm2.5"].clip(lower=0.0))
    feats = pd.DataFrame(
        {
            "log_pm25": log_pm,
            "ewm6_log_pm25": log_pm.ewm(halflife=6, min_periods=6).mean(),
            "log_wind": np.log1p(df["Iws"].clip(lower=0.0)),
        }
    )
    valid = feats.notna().all(axis=1)
    df = df.loc[valid].reset_index(drop=True)
    feats = feats.loc[valid].reset_index(drop=True)
    train_mask = df["year"] <= 2012
    test_mask = df["year"] >= 2013
    X_train, X_test = _standardize_fit_transform(feats.loc[train_mask], feats.loc[test_mask])
    pm_test = df.loc[test_mask, "pm2.5"].to_numpy(dtype=np.float64)

    # Sort by log_pm25 so State 0 = Clean Air, State 1 = Polluted/Smog Regime
    base_fit = fit_jump_model(X_train, n_components=2, jump_penalty=3.0, n_init=20, random_state=42, sort_by_feature=0)
    s_loss = characteristic_loss_scale(X_train, base_fit.centers_)

    # Agency smog-alert delay budget: worst-case step delay <= 4 hours
    curve = evaluate_penalty_curve(X_train, base_fit.centers_, etas=ETAS, mode="carried", loss_scale=s_loss)
    chosen = select_penalty_by_delay_budget(curve, max_delay_steps=4)

    pred_zero = OnlineJumpDecoder(base_fit.centers_, jump_penalty=0.0).decode_carried(X_test)
    pred_sel = OnlineJumpDecoder(base_fit.centers_, jump_penalty=chosen.jump_penalty).decode_carried(X_test)

    months = len(X_test) / (24.0 * 30.4375)
    sw_zero = count_switches(pred_zero)
    sw_sel = count_switches(pred_sel)

    comp = compare_carried_vs_rolling(
        X_test,
        base_fit.centers_,
        chosen.jump_penalty,
        windows=(7, 15, 24, 72, 168, 336),
    )

    plot_latency_churn_curve(
        curve,
        chosen,
        title="Beijing Hourly PM2.5: Label-Free Smog Alert Calibration",
        delay_unit="hours",
        out_path=OUT_DIR / "pm25_pareto.png",
    )
    plot_carried_vs_rolling(
        comp,
        title="Beijing Hourly PM2.5 (2013–2014 Test): Carried vs. Rolling-W",
        out_path=OUT_DIR / "pm25_carried_vs_rolling.png",
    )

    return {
        "domain": "Beijing Hourly Air-Quality Sensors (2010–2014)",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "delay_budget": "4 hours",
        "loss_scale": round(s_loss, 4),
        "selected_eta": chosen.eta,
        "selected_lambda": round(chosen.jump_penalty, 4),
        "guaranteed_worst_delay_hours": chosen.worst_delay,
        "test_switches_unpenalized": sw_zero,
        "test_switches_selected": sw_sel,
        "test_switches_per_month_unpenalized": round(sw_zero / months, 2),
        "test_switches_per_month_selected": round(sw_sel / months, 2),
        "test_median_pm25_clean_state": round(float(np.median(pm_test[pred_sel == 0])), 1),
        "test_median_pm25_smog_state": round(float(np.median(pm_test[pred_sel == 1])), 1),
        "carried_vs_rolling_test": [asdict(r) for r in comp],
    }


def run_vix_benchmark() -> dict:
    vix_path = DATA_DIR / "vix" / "VIXCLS.csv"
    df = pd.read_csv(vix_path, na_values=[".", ""])
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["VIXCLS"] = pd.to_numeric(df["VIXCLS"], errors="coerce")
    df = df.dropna(subset=["VIXCLS"]).sort_values("observation_date").reset_index(drop=True)

    log_vix = np.log(df["VIXCLS"])
    feats = pd.DataFrame(
        {
            "log_vix": log_vix,
            "d5_log_vix": log_vix.diff(5),
            "vol20_log_vix": log_vix.diff(1).ewm(halflife=20, min_periods=20).std(),
        }
    )
    valid = feats.notna().all(axis=1)
    df = df.loc[valid].reset_index(drop=True)
    feats = feats.loc[valid].reset_index(drop=True)

    train_mask = df["observation_date"] <= pd.Timestamp("2006-12-31")
    test_mask = df["observation_date"] >= pd.Timestamp("2007-01-01")
    X_train, X_test = _standardize_fit_transform(feats.loc[train_mask], feats.loc[test_mask])
    vix_test = df.loc[test_mask, "VIXCLS"].to_numpy(dtype=np.float64)

    # Sort by log_vix so State 0 = Calm Market, State 1 = Elevated Volatility Stress
    base_fit = fit_jump_model(X_train, n_components=2, jump_penalty=4.0, n_init=20, random_state=42, sort_by_feature=0)
    s_loss = characteristic_loss_scale(X_train, base_fit.centers_)

    # Risk-monitoring delay budget: worst-case step delay <= 3 trading days
    curve = evaluate_penalty_curve(X_train, base_fit.centers_, etas=ETAS, mode="carried", loss_scale=s_loss)
    chosen = select_penalty_by_delay_budget(curve, max_delay_steps=3)

    pred_zero = OnlineJumpDecoder(base_fit.centers_, jump_penalty=0.0).decode_carried(X_test)
    pred_sel = OnlineJumpDecoder(base_fit.centers_, jump_penalty=chosen.jump_penalty).decode_carried(X_test)

    years = len(X_test) / 252.0
    sw_zero = count_switches(pred_zero)
    sw_sel = count_switches(pred_sel)

    comp = compare_carried_vs_rolling(
        X_test,
        base_fit.centers_,
        chosen.jump_penalty,
        windows=(7, 15, 31, 63, 126, 252),
    )

    plot_latency_churn_curve(
        curve,
        chosen,
        title="CBOE VIX Risk Monitor: Label-Free Stress Alert Calibration",
        delay_unit="trading days",
        out_path=OUT_DIR / "vix_pareto.png",
    )
    plot_carried_vs_rolling(
        comp,
        title="CBOE VIX (2007–2026 Out-of-Sample): Carried vs. Rolling-W",
        out_path=OUT_DIR / "vix_carried_vs_rolling.png",
    )

    return {
        "domain": "FRED CBOE VIX Volatility Stress Monitor (1990–2026)",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "delay_budget": "3 trading days",
        "loss_scale": round(s_loss, 4),
        "selected_eta": chosen.eta,
        "selected_lambda": round(chosen.jump_penalty, 4),
        "guaranteed_worst_delay_days": chosen.worst_delay,
        "test_switches_unpenalized": sw_zero,
        "test_switches_selected": sw_sel,
        "test_switches_per_year_unpenalized": round(sw_zero / years, 2),
        "test_switches_per_year_selected": round(sw_sel / years, 2),
        "test_median_vix_calm_state": round(float(np.median(vix_test[pred_sel == 0])), 2),
        "test_median_vix_stress_state": round(float(np.median(vix_test[pred_sel == 1])), 2),
        "carried_vs_rolling_test": [asdict(r) for r in comp],
    }


def ensure_public_datasets() -> None:
    """Automatically download public UCI / FRED benchmark files if not cached locally."""
    occ_dir = DATA_DIR / "occupancy"
    if not (occ_dir / "datatraining.txt").exists():
        occ_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen("https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip") as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                zf.extractall(occ_dir)

    pm_dir = DATA_DIR / "beijing_pm25"
    if not (pm_dir / "PRSA_data_2010.1.1-2014.12.31.csv").exists():
        pm_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen("https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip") as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                zf.extractall(pm_dir)

    vix_file = DATA_DIR / "vix" / "VIXCLS.csv"
    if not vix_file.exists():
        vix_file.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS", vix_file)


def main() -> None:
    ensure_public_datasets()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [
        run_occupancy_benchmark(),
        run_pm25_benchmark(),
        run_vix_benchmark(),
    ]

    summary_rows = []
    for r in results:
        summary_rows.append(
            {
                "domain": r["domain"],
                "train_rows": r["train_rows"],
                "test_rows": r["test_rows"],
                "delay_budget": r["delay_budget"],
                "selected_eta": r["selected_eta"],
                "selected_lambda": r["selected_lambda"],
                "unpenalized_switches": r["test_switches_unpenalized"],
                "selected_switches": r["test_switches_selected"],
                "switch_reduction_pct": round(
                    100.0 * (1.0 - r["test_switches_selected"] / max(1, r["test_switches_unpenalized"])), 1
                ),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "benchmark_summary.csv", index=False)
    (OUT_DIR / "benchmark_details.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
