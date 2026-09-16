"""Figures for the ERFI study report (docs/report/erfi_studija_sr.tex).

    python docs/report/make_erfi_figures.py                       # flat Go1 study -> figures/erfi/
    python docs/report/make_erfi_figures.py erfi_study_rough_l2.5 rough   # -> figures/erfi/rough/

Reads experiments/erfi_study_l2.5 (main study, 2.5 Nm), experiments/erfi_study
(first sweep, 7.0 Nm) and experiments/erfi_limits_l* (50 M-step limit sweep) and
writes PNGs into docs/report/figures/erfi/. Axis labels are Serbian Cyrillic;
DejaVu Sans (matplotlib's default) covers the glyphs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments"
OUT = REPO / "docs/report/figures/erfi"  # overridden in main() for other studies

# Fixed hue per condition (reference categorical palette, slots 1-6), never cycled.
COLORS = {
    "none": "#2a78d6", "dr": "#eb6834", "rfi": "#1baf7a",
    "rao": "#eda100", "erfi_c": "#e87ba4", "erfi_50": "#008300",
}
NAMES = {
    "none": "без насумичавања", "dr": "насумичавање динамике",
    "rfi": "RFI", "rao": "RAO", "erfi_c": "ERFI-C", "erfi_50": "ERFI-50",
}
MARKERS = {"none": "o", "dr": "s", "rfi": "^", "rao": "v", "erfi_c": "D", "erfi_50": "P"}
ORDER = list(COLORS)

PARAMS = ["payload_kg", "push_N", "friction", "gravity", "kp_scale"]
XLABEL = {
    "payload_kg": "додатна маса на трупу (kg)",
    "push_N": "хоризонтална сила на труп, 3 s (N)",
    "friction": "коефицијент трења пода",
    "gravity": "гравитационо убрзање (m/s²)",
    "kp_scale": "множилац појачања Kp",
}
NOMINAL = {"payload_kg": 0.0, "push_N": 0.0, "friction": 0.6, "gravity": -9.81, "kp_scale": 1.0}
YLABEL = {
    "success_rate": "стопа успеха",
    "fall_rate": "стопа падова",
    "progress_m": "пређени пут (m)",
    "tracking_rmse": "RMSE брзине (m/s)",
}

GRID = dict(alpha=0.25, lw=0.6)
TEXT = "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": "#8a8986", "axes.labelcolor": "#0b0b0b",
    "xtick.color": TEXT, "ytick.color": TEXT, "figure.dpi": 100,
})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _legend(fig, conditions, ncol=6, y=-0.01):
    handles = [
        plt.Line2D([], [], color=COLORS[c], marker=MARKERS[c], lw=2, markersize=6,
                   markeredgecolor="white", markeredgewidth=1, label=NAMES[c])
        for c in conditions
    ]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, y), handlelength=2.2, columnspacing=1.4)


# ------------------------------------------------------------- learning curves

def load_curves(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.glob("*/seed*/curve.json")):
        spec = json.loads((p.parent / "spec.json").read_text())
        for r in json.loads(p.read_text()):
            rows.append({"condition": spec["condition"], "seed": spec["seed"],
                         "lim": spec["rao_lim"], **r})
    return pd.DataFrame(rows)


def fig_learning(root: Path, fname: str, title_note: str):
    df = load_curves(root)
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2), sharex=True, sharey=True)
    for ax, cond in zip(axes.flat, ORDER):
        sub = df[df.condition == cond]
        for seed, g in sub.groupby("seed"):
            ax.plot(g.step / 1e6, g.reward, color=COLORS[cond], lw=1.6,
                    alpha=[1.0, 0.7, 0.45][seed], label=f"семе {seed}")
        ax.set_title(NAMES[cond], color=COLORS[cond] if cond != "rao" else "#a86f00", loc="left")
        ax.axhline(19, color="#8a8986", lw=0.8, ls=":")
        ax.set_ylim(0, 31)
        ax.set_xlim(0, float(df.step.max()) / 1e6 * 1.02)
        ax.grid(**GRID)
        _despine(ax)
    axes[0, 0].text(float(df.step.max()) / 1e6, 19.6, "стајање у месту ≈ 19", ha="right", va="bottom",
                    fontsize=7.5, color=TEXT)
    for ax in axes[1]:
        ax.set_xlabel("кораци окружења (милиони)")
    for ax in axes[:, 0]:
        ax.set_ylabel("награда (евалуација)")
    handles = [plt.Line2D([], [], color="#0b0b0b", lw=1.6, alpha=a, label=f"семе {s}")
               for s, a in zip(range(3), [1.0, 0.7, 0.45])]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title_note, x=0.01, ha="left", fontsize=9.5, color=TEXT)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(OUT / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------- protocol curves

def fig_protocol(results: pd.DataFrame, metric: str, fname: str, ylim, hline=None):
    conditions = [c for c in ORDER if c in set(results.condition)]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.6))
    for ax, param in zip(axes.flat, PARAMS):
        sub = results[results.param == param]
        for cond in conditions:
            g = sub[sub.condition == cond].groupby("level")[metric]
            mean, lo, hi = g.mean(), g.min(), g.max()
            ax.plot(mean.index, mean.values, color=COLORS[cond], lw=2, marker=MARKERS[cond],
                    markersize=5.5, markeredgecolor="white", markeredgewidth=0.9)
            ax.fill_between(mean.index, lo.values, hi.values, color=COLORS[cond], alpha=0.10, lw=0)
        nominal = float(sub["nominal"].iloc[0]) if "nominal" in sub.columns and len(sub) else NOMINAL[param]
        ax.axvline(nominal, color="#8a8986", lw=1, ls="--")
        ax.text(nominal, ylim[1], "тренинг", color=TEXT, fontsize=7.5, ha="center", va="bottom")
        if hline is not None:
            ax.axhline(hline[0], color="#8a8986", lw=0.8, ls=":")
            ax.text(0.98, hline[0] + 0.08, hline[1], color=TEXT, fontsize=7.5,
                    ha="right", va="bottom", transform=ax.get_yaxis_transform())
        ax.set_ylim(*ylim)
        ax.set_xlabel(XLABEL[param])
        ax.set_ylabel(YLABEL[metric])
        ax.grid(**GRID)
        _despine(ax)
    axes.flat[-1].axis("off")
    _legend(fig, conditions, ncol=3, y=0.02)
    # legend lives in the empty sixth panel
    fig.legends[-1].set_bbox_to_anchor((0.83, 0.12))
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def best_seed_subset(results: pd.DataFrame) -> pd.DataFrame:
    """One run per condition: the walking seed with the highest mean success over all levels."""
    nominal = results[(results.param == "payload_kg") & (results.level == 0.0)]
    walking = set(nominal[nominal.progress_m >= 1.0].run)
    score = results[results.run.isin(walking)].groupby(["condition", "run"]).success_rate.mean()
    best = score.groupby("condition").idxmax().map(lambda t: t[1])
    print("best seeds:", dict(best))
    return results[results.run.isin(set(best))]


def fig_summary_bars(results: pd.DataFrame, fname: str):
    """Mean success over all levels, per condition and perturbation. Walking seeds only."""
    tab = results.groupby(["param", "condition"]).success_rate.mean().unstack("condition")[ORDER]
    tab = tab.loc[PARAMS]
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    n = len(ORDER)
    width = 0.8 / n
    x = np.arange(len(PARAMS))
    for i, cond in enumerate(ORDER):
        vals = tab[cond].values
        bars = ax.bar(x + (i - n / 2 + 0.5) * width, vals, width * 0.9, color=COLORS[cond],
                      label=NAMES[cond], linewidth=0)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=6.3, color=TEXT, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(["маса", "гурање", "трење", "гравитација", "Kp"])
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("средња стопа успеха\nпреко свих нивоа")
    ax.grid(axis="y", **GRID)
    _despine(ax)
    ax.legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- limit sweep

def fig_limits(fname: str):
    frames = []
    for d in ("erfi_limits_l1.5", "erfi_limits_l2.5", "erfi_limits_l3.5"):
        frames.append(load_curves(EXP / d).assign(budget="50M"))
    for d in ("erfi_study_l2.5", "erfi_study"):
        c = load_curves(EXP / d)
        frames.append(c[(c.condition.isin(["rao", "erfi_50"])) & (c.seed == 0)].assign(budget="200M"))
    df = pd.concat(frames)
    # single-hue sequential ramp for the torque limit
    ramp = {1.5: "#9ecae1", 2.5: "#4292c6", 3.5: "#08519c", 7.0: "#08306b"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    for ax, cond in zip(axes, ["rao", "erfi_50"]):
        sub = df[df.condition == cond]
        for (lim, budget), g in sub.groupby(["lim", "budget"]):
            ls = "-" if budget == "200M" else "--"
            ax.plot(g.step / 1e6, g.reward, color=ramp[lim], lw=1.8, ls=ls,
                    label=f"{lim:g} Nm, {budget} корака")
        ax.axhline(19, color="#8a8986", lw=0.8, ls=":")
        ax.set_title(NAMES[cond], loc="left")
        ax.set_xlabel("кораци окружења (милиони)")
        ax.set_xlim(0, 210)
        ax.grid(**GRID)
        _despine(ax)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    axes[0].set_ylabel("награда по епизоди (евалуација)")
    axes[0].set_ylim(0, 31)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def study_figures(study: str, tag: str | None = None, learning_title: str = "граница момента 2,5 Nm"):
    """Protocol figures + learning curves for one study directory. Sets OUT."""
    global OUT
    OUT = REPO / "docs/report/figures/erfi" / (tag or "")
    OUT.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(EXP / study / "results.csv")
    nominal = res[(res.param == "payload_kg") & (res.level == 0.0)]
    walking = set(nominal[nominal.progress_m >= 1.0].run)
    res_walk = res[res.run.isin(walking)]
    print(f"[{study}] collapsed runs:", sorted(set(res.run) - walking))
    fig_learning(EXP / study, "learning.png", learning_title)
    fig_protocol(res, "success_rate", "success_all.png", (-0.02, 1.02))
    if len(res_walk):
        fig_protocol(res_walk, "success_rate", "success_walking.png", (-0.02, 1.02))
        fig_protocol(res_walk, "fall_rate", "fall_walking.png", (-0.02, 1.02))
        fig_protocol(res_walk, "progress_m", "progress_walking.png", (0, 4.6), hline=(2.5, "праг успеха 2,5 m"))
        fig_protocol(res_walk, "tracking_rmse", "rmse_walking.png", (0, 0.62))
        fig_summary_bars(res_walk, "summary_bars.png")
        tab = res_walk.groupby(["condition", "param"]).success_rate.mean().unstack("param").reindex(ORDER)[PARAMS]
        tab.to_csv(OUT / "summary_walking.csv")
        print(tab.round(3))
        # best seed per condition (no band: one run per line)
        res_best = best_seed_subset(res)
        fig_protocol(res_best, "success_rate", "success_best.png", (-0.02, 1.02))
        fig_protocol(res_best, "fall_rate", "fall_best.png", (-0.02, 1.02))
        fig_protocol(res_best, "progress_m", "progress_best.png", (0, 4.6), hline=(2.5, "праг успеха 2,5 m"))
        fig_summary_bars(res_best, "summary_bars_best.png")
        res_best.groupby(["condition", "param"]).success_rate.mean().unstack("param").reindex(ORDER)[PARAMS].to_csv(OUT / "summary_best.csv")
    print("written to", OUT)


def main():
    import sys
    if len(sys.argv) > 1:  # another study, own sub-directory
        study_figures(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1],
                      learning_title=f"{sys.argv[1]}, граница момента 2,5 Nm")
        return
    global OUT
    OUT.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(EXP / "erfi_study_l2.5/results.csv")
    # a run "walks" if at the nominal setting it covers >= 1 m in 8 s
    nominal = res[(res.param == "payload_kg") & (res.level == 0.0)]
    walking = set(nominal[nominal.progress_m >= 1.0].run)
    res_walk = res[res.run.isin(walking)]
    print("collapsed runs:", sorted(set(res.run) - walking))

    fig_learning(EXP / "erfi_study_l2.5", "learning_2p5.png", "граница момента 2,5 Nm (главна студија)")
    fig_learning(EXP / "erfi_study", "learning_7p0.png", "граница момента 7,0 Nm (прва студија)")

    fig_protocol(res, "success_rate", "success_all.png", (-0.02, 1.02))
    fig_protocol(res_walk, "success_rate", "success_walking.png", (-0.02, 1.02))
    fig_protocol(res_walk, "fall_rate", "fall_walking.png", (-0.02, 1.02))
    fig_protocol(res_walk, "progress_m", "progress_walking.png", (0, 4.6), hline=(2.5, "праг успеха 2,5 m"))
    fig_protocol(res_walk, "tracking_rmse", "rmse_walking.png", (0, 0.62))
    fig_summary_bars(res_walk, "summary_bars.png")
    fig_limits("limits.png")

    tab = res_walk.groupby(["condition", "param"]).success_rate.mean().unstack("param").reindex(ORDER)[PARAMS]
    tab.to_csv(OUT / "summary_walking.csv")
    print(tab.round(3))
    print("written to", OUT)


if __name__ == "__main__":
    main()
