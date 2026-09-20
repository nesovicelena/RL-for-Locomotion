#!/usr/bin/env python3
"""Redraw the evaluation curve figures from the results CSVs alone.

No policy, no MuJoCo, no GPU: every PNG that eval.py and eval_terrain.py write
is a pure function of `results*.csv`, so the whole figure set can be rebuilt on
the laptop, in whatever language the thesis needs.

    python scripts/replot.py                          # every study, English
    python scripts/replot.py --lang sr                # Serbian Cyrillic
    python scripts/replot.py --studies erfi_study_curr_v3_l2.5 --lang sr
    python scripts/replot.py --lang sr --out-suffix _sr   # keep both versions

Figures land next to their CSV as `<metric>_curves<suffix><out-suffix>.png`,
i.e. the same names eval_terrain.py uses when --out-suffix is empty.

The factorial `combined_*` suites are skipped: a grid has no curve to draw,
which is why eval_terrain.py never wrote a PNG for them either.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from rl_locomotion.eval import perturb

REPO = Path(__file__).resolve().parent.parent
METRICS = ("success_rate", "fall_rate", "progress_m")


def experiments_root() -> Path:
    env = os.environ.get("RL_EXPERIMENTS_DIR")
    if env:
        return Path(env)
    if Path("/workspace").is_dir():
        return Path("/workspace/experiments")
    return REPO / "experiments"

# Serbian Cyrillic. The keys must stay in step with perturb.LABELS /
# perturb.CONDITION_NAMES; anything missing here falls back to the English.
SR = {
    "labels": {
        "payload_kg": "терет на трупу (kg)",
        "push_N": "сила гурања на труп, 3 s (N)",
        "push_N_sagittal": "сагитално гурање трупа, 3 s (N)",
        "push_N_lateral": "бочно гурање трупа, 3 s (N)",
        "friction": "коефицијент трења подлоге",
        "gravity": "гравитација (m/s²)",
        "kp_scale": "множилац Kp",
        "slope_deg": "нагиб узбрдо (степени)",
        "terrain_amplitude": "рељеф терена, од врха до дна (m)",
    },
    "conditions": {
        "none": "без рандомизације",
        "dr": "рандомизација динамике",
        "rfi": "RFI",
        "rao": "RAO",
        "erfi_c": "ERFI-C",
        "erfi_50": "ERFI-50",
    },
    "metrics": {
        "success_rate": "стопа успеха",
        "fall_rate": "стопа падова",
        "progress_m": "пређени пут (m)",
    },
    "training": "обука",
}

EN = {
    "labels": dict(perturb.LABELS),
    "conditions": dict(perturb.CONDITION_NAMES),
    "metrics": {m: m.replace("_", " ") for m in METRICS},
    "training": "training",
}

LANGS = {"en": EN, "sr": SR}


def figure(results: pd.DataFrame, metric: str, lang: dict):
    """perturb.plot_success_curves with the caller's vocabulary swapped in.

    The axis labels and legend read from module-level dicts, so translating is
    a matter of lending the function a different pair for the call; the ylabel
    and the `training` marker are written inline and get fixed afterwards.
    """
    saved = perturb.LABELS, perturb.CONDITION_NAMES
    perturb.LABELS = {**EN["labels"], **lang["labels"]}
    perturb.CONDITION_NAMES = {**EN["conditions"], **lang["conditions"]}
    try:
        fig = perturb.plot_success_curves(results, metric=metric)
    finally:
        perturb.LABELS, perturb.CONDITION_NAMES = saved

    ylabel = lang["metrics"].get(metric, metric.replace("_", " "))
    for ax in fig.axes:
        if not ax.get_ylabel():
            continue  # an axis turned off to pad the grid
        ax.set_ylabel(ylabel)
        for text in ax.texts:
            if text.get_text() == EN["training"]:
                text.set_text(lang["training"])
    return fig


def csv_files(study_root: Path) -> list[tuple[Path, str]]:
    """Every plottable results CSV in a study, paired with its figure suffix."""
    out = []
    for path in sorted(study_root.glob("results*.csv")):
        suite = path.stem[len("results"):]  # "" for the base protocol
        if suite.startswith("_combined"):
            continue
        out.append((path, suite))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--studies", nargs="*", default=None,
                   help="study directory names (default: every one holding a results CSV)")
    p.add_argument("--lang", choices=sorted(LANGS), default="en")
    p.add_argument("--metrics", nargs="*", default=list(METRICS), choices=list(METRICS))
    p.add_argument("--out-suffix", default="",
                   help="appended to the PNG name, e.g. _sr to sit beside the English set")
    p.add_argument("--root", type=Path, default=None, help="experiments root (default: the usual)")
    args = p.parse_args()

    root = args.root or experiments_root()
    lang = LANGS[args.lang]
    studies = args.studies or sorted(
        d.name for d in root.iterdir() if d.is_dir() and any(d.glob("results*.csv")))
    if not studies:
        raise SystemExit(f"no study under {root} holds a results CSV")

    n = 0
    for study in studies:
        study_root = root / study
        if not study_root.is_dir():
            print(f"== {study}: not found, skipping")
            continue
        print(f"\n== {study}")
        for path, suite in csv_files(study_root):
            results = pd.read_csv(path)
            if "param" not in results or results.empty:
                print(f"  {path.name}: no curves to draw, skipping")
                continue
            for metric in args.metrics:
                if metric not in results:
                    continue
                fig = figure(results, metric, lang)
                out = study_root / f"{metric}_curves{suite}{args.out_suffix}.png"
                fig.savefig(out, dpi=150, bbox_inches="tight")
                matplotlib.pyplot.close(fig)
                print(f"  {out.name}")
                n += 1
    print(f"\n{n} figures ({args.lang})")


if __name__ == "__main__":
    main()
