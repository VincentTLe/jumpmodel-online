"""Optional plotting helpers (`matplotlib` imported lazily so core stays dependency-light)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .diagnostics import ComparisonRow, PenaltyPoint


def plot_latency_churn_curve(
    curve: Sequence[PenaltyPoint],
    selected: PenaltyPoint,
    title: str,
    delay_unit: str,
    out_path: str | Path,
) -> Path:
    """Plot online switching rate (per 1,000 steps) against guaranteed step-response delay."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=160)
    xs = [pt.worst_delay for pt in curve]
    ys = [pt.switches_per_1000 for pt in curve]

    ax.plot(xs, ys, marker="o", linewidth=1.8, color="#1f5aa6", label="Candidate penalties (η · s_loss)")
    ax.scatter(
        [selected.worst_delay],
        [selected.switches_per_1000],
        s=110,
        color="#d9480f",
        zorder=5,
        label=f"Selected (η={selected.eta:g}, delay={selected.worst_delay} {delay_unit})",
    )

    last_label_pos: tuple[float, float] | None = None
    x_span = max(1e-9, float(max(xs) - min(xs)))
    y_span = max(1e-9, float(max(ys) - min(ys)))
    for idx, pt in enumerate(curve):
        is_selected = pt.eta == selected.eta and pt.worst_delay == selected.worst_delay
        if not is_selected and last_label_pos is not None:
            dx = abs(pt.worst_delay - last_label_pos[0]) / x_span
            dy = abs(pt.switches_per_1000 - last_label_pos[1]) / y_span
            if dx < 0.085 and dy < 0.06:
                continue
        y_off = 7 if (idx % 2 == 0 or is_selected) else -11
        ax.annotate(
            f"η={pt.eta:g}",
            (pt.worst_delay, pt.switches_per_1000),
            textcoords="offset points",
            xytext=(4, y_off),
            fontsize=8,
            fontweight="bold" if is_selected else "normal",
            color="#b03505" if is_selected else "#333333",
        )
        last_label_pos = (float(pt.worst_delay), float(pt.switches_per_1000))

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Worst-case step-response delay ({delay_unit})", fontsize=9.5)
    ax.set_ylabel("Online state switches per 1,000 steps", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_carried_vs_rolling(
    rows: Sequence[ComparisonRow],
    title: str,
    out_path: str | Path,
) -> Path:
    """Plot disagreement rate and switch counts of `rolling-W` relative to `carried` (`predict_online`)."""
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ws = [r.window for r in rows]
    disagree_pct = [100.0 * r.disagreement_rate for r in rows]
    roll_sw = [r.rolling_switches for r in rows]
    carried_sw = rows[0].carried_switches if rows else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=160)

    ax1.plot(ws, disagree_pct, marker="s", linewidth=1.8, color="#2b8a3e")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Rolling window length W (log scale)", fontsize=9.5)
    ax1.set_ylabel("Disagreement with carried (%) ", fontsize=9.5)
    ax1.set_title("When does Rolling-W match predict_online?", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2.plot(ws, roll_sw, marker="o", linewidth=1.8, color="#7048e8", label="Rolling-W switches")
    ax2.axhline(carried_sw, linestyle="--", linewidth=1.6, color="#d9480f", label=f"Carried switches ({carried_sw})")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Rolling window length W (log scale)", fontsize=9.5)
    ax2.set_ylabel("Total emitted switches", fontsize=9.5)
    ax2.set_title("Emitted switch count vs. window W", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(frameon=True, fontsize=8.5)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
