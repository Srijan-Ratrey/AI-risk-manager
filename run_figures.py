"""Figures for the README and the video."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = Path("reports/figures")
INK, ACCENT, WARN, MUTED = "#1a1a1a", "#0b6e4f", "#c1440e", "#8a8a8a"


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, weight="bold", color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=10, color=INK)
    ax.grid(alpha=0.18, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def cost_curve(results):
    curve = pd.read_csv("reports/cost_curve.csv")
    policies = pd.read_csv("reports/policies.csv")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(curve["threshold"], curve["cost_per_10k"] / 1e6, color=INK, lw=2, zorder=3)

    # Staggered offsets: the three optima sit close together on both axes,
    # so fixed offsets overlap and the labels become unreadable.
    marks = [
        ("cost-optimal", ACCENT, "o", (0.04, 0.62), "left"),
        ("F1-optimal", WARN, "s", (0.05, 0.22), "left"),
        ("accuracy-optimal", MUTED, "^", (0.03, 0.16), "left"),
    ]
    for label, colour, marker, (dx, dy), align in marks:
        row = policies[policies["policy"] == f"global @ {label}"].iloc[0]
        t, cost = row["threshold"], row["cost_per_10k_inr"] / 1e6
        ax.scatter([t], [cost], color=colour, s=95, zorder=5, marker=marker,
                   edgecolor="white", linewidth=1.5)
        ax.annotate(f"{label}\nt={t:.3f}   Rs {cost:.2f}M",
                    xy=(t, cost), xytext=(t + dx, cost + dy), ha=align,
                    fontsize=9, color=colour, weight="bold",
                    arrowprops=dict(arrowstyle="-", color=colour, lw=1, alpha=0.7))

    gap = results["headline_f1_minus_cost_per_10k"]
    ax.set_xlim(0, 0.8)
    # Clip to the decision-relevant band. Blocking everything (t -> 0) costs
    # Rs 16.4M/10k and would flatten everything worth looking at.
    ax.set_ylim(3.1, 5.0)
    ax.text(0.004, 4.94, "^ blocking everything: Rs 16.4M/10k (off scale)",
            fontsize=8.5, color=MUTED, va="top")
    style(ax, "Cost curve: rupees lost per 10,000 transactions",
          "decision threshold (calibrated probability)", "cost per 10k txns (Rs, millions)")
    ax.text(0.98, 0.95,
            f"Optimising F1 instead of cost:\nRs {gap:,.0f} per 10,000 transactions",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=WARN,
            weight="bold", bbox=dict(boxstyle="round,pad=0.5", fc="#fdf3ee", ec=WARN, lw=1))
    fig.tight_layout()
    fig.savefig(FIG / "cost_curve.png", dpi=170)


def reliability(results):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 0.35], [0, 0.35], ls="--", color=MUTED, lw=1.2, label="perfect calibration")
    for label, colour, name in (("before", WARN, "before isotonic"), ("after", ACCENT, "after isotonic")):
        curve = pd.read_csv(f"reports/reliability_{label}.csv")
        ece = results[f"test_ece_{label}"]
        ax.plot(curve["predicted"], curve["observed"], marker="o", ms=4.5, lw=1.8,
                color=colour, label=f"{name}  (ECE {ece:.4f})")
    ax.set_xlim(0, 0.35); ax.set_ylim(0, 0.35)
    style(ax, "Reliability diagram (test window)",
          "mean predicted probability", "observed fraud rate")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "reliability.png", dpi=170)


def threshold_curve(costs):
    amounts = np.logspace(1, 5.5, 400)
    thresholds = costs.optimal_threshold(amounts)
    lo, hi = costs.threshold_limits()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.semilogx(amounts, thresholds, color=ACCENT, lw=2.2)
    ax.axhline(lo, ls=":", color=MUTED, lw=1)
    ax.axhline(hi, ls=":", color=MUTED, lw=1)
    ax.text(11, lo + 0.002, f"small-amount limit {lo:.3f}", fontsize=8.5, color=MUTED)
    ax.text(11, hi - 0.006, f"large-amount limit {hi:.3f}", fontsize=8.5, color=MUTED)
    style(ax, "Cost-optimal threshold as a function of order value  t*(a)",
          "order value (Rs, log scale)", "block above this probability")
    ax.text(0.98, 0.92, "measured effect on test: none\n(CI spans zero -- see README)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=WARN,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fdf3ee", ec=WARN, lw=1))
    fig.tight_layout()
    fig.savefig(FIG / "threshold_curve.png", dpi=170)


def segment_cost():
    seg = pd.read_csv("reports/segments.csv")
    band = seg[seg["dimension"] == "amount_band"].copy()
    order = ["<$25", "$25-100", "$100-250", "$250-1k", ">$1k"]
    band["segment"] = pd.Categorical(band["segment"], order, ordered=True)
    band = band.sort_values("segment")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.bar(band["segment"].astype(str), band["cost_per_10k_inr"] / 1e6, color=WARN, width=0.62)
    style(ax1, "Where the money is lost", "order value band", "cost per 10k txns (Rs, millions)")
    ax2.bar(band["segment"].astype(str), band["pr_auc"], color=INK, width=0.62)
    style(ax2, "Where the model is weakest", "order value band", "PR-AUC")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle("The model is worst exactly where the money is",
                 fontsize=12.5, weight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "segments.png", dpi=170)


def ladder():
    table = pd.read_csv("reports/baseline_ladder.csv")
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    table = table.sort_values("cost_per_10k_inr", ascending=True)
    colours = [ACCENT if "Count rule" in n else INK for n in table["baseline"]]
    ax.barh(table["baseline"], table["cost_per_10k_inr"] / 1e6, color=colours, height=0.62)
    style(ax, "Baseline ladder: cost per 10,000 transactions (test window)",
          "cost per 10k txns (Rs, millions)", "")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "baseline_ladder.png", dpi=170)


def main() -> None:
    from src.costs import CostModel
    FIG.mkdir(parents=True, exist_ok=True)
    results = json.loads(Path("reports/results.json").read_text())
    costs = CostModel.load()

    cost_curve(results)
    reliability(results)
    threshold_curve(costs)
    segment_cost()
    ladder()
    print("figures written:")
    for path in sorted(FIG.glob("*.png")):
        print(f"  {path}  ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
