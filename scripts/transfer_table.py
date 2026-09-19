#!/usr/bin/env python3
"""Zero-shot cross-robot transfer table: each policy on its own robot versus on the other one.

    python scripts/transfer_table.py                 # every Go1/A1 study pair with results
    python scripts/transfer_table.py --walking-only  # drop policies that stand still at home

Reads, per study, results.csv (own robot, from scripts/eval.py) and
results_on_<other>.csv (the other robot, from scripts/eval.py --robot or
runpod/run_cross_robot.sh), and writes experiments/redo/transfer_table.csv with one
row per (recipe, trained robot, condition):

    success_own        mean success over all levels of the parameters present in BOTH files
                       (and over the policies present in both, so a partial or still-running
                        cross-robot sweep is compared like with like and flagged PARTIAL)
    success_foreign    the same, on the other robot's model
    drop               success_own - success_foreign  (what the robot swap costs)
    progress_own/foreign   progress_m at the nominal row (payload 0), metres in 8 s
    n_walking          policies with progress_own >= 1 m (the figures' walking criterion)

No GPU, no MuJoCo: pure pandas over the CSVs. Only parameters that appear in both
files enter the mean, so a QUICK run (payload and push) is compared like with like.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# (recipe label, Go1 study dir, A1 study dir); the two robots' counterparts.
PAIRS = [
    ("v1 flat",       "erfi_study_l2.5",          "erfi_study_a1_l2.5"),
    ("v1 rough",      "erfi_study_rough_l2.5",    "erfi_study_a1_rough_l2.5"),
    ("v3 flat",       "erfi_study_v3_l2.5",       "erfi_study_a1_v3_l2.5"),
    ("v3 rough",      "erfi_study_v3_rough_l2.5", "erfi_study_a1_v3_rough_l2.5"),
    ("v3 curriculum", "erfi_study_curr_v3_l2.5",  "erfi_study_curr_a1_v3_l2.5"),
]
ORDER = ["none", "dr", "rfi", "rao", "erfi_c", "erfi_50"]


def experiments_root() -> Path:
    env = os.environ.get("RL_EXPERIMENTS_DIR")
    return Path(env) if env else REPO / "experiments" / "redo"


def nominal_progress(df: pd.DataFrame) -> pd.Series:
    """progress_m at payload 0 (the unperturbed row), indexed by run."""
    row = df[(df.param == "payload_kg") & (df.level == 0.0)]
    return row.set_index("run").progress_m


def compare(own: pd.DataFrame, foreign: pd.DataFrame, walking_only: bool) -> pd.DataFrame:
    # Compare like with like: only the runs and parameters present in BOTH files.
    # A cross-robot sweep that is still running, or was interrupted, leaves a
    # partial results_on_<robot>.csv; without this intersection `success_own`
    # would average over all 18 policies and `success_foreign` over the handful
    # that finished, and `drop` would be a comparison of two different sets.
    params = sorted(set(own.param) & set(foreign.param))
    shared = sorted(set(own.run) & set(foreign.run))
    own = own[own.run.isin(shared) & own.param.isin(params)]
    foreign = foreign[foreign.run.isin(shared) & foreign.param.isin(params)]
    prog_own, prog_for = nominal_progress(own), nominal_progress(foreign)
    walking = set(prog_own[prog_own >= 1.0].index)
    if walking_only:
        own, foreign = own[own.run.isin(walking)], foreign[foreign.run.isin(walking)]
    rows = []
    for cond in ORDER:
        o, f = own[own.condition == cond], foreign[foreign.condition == cond]
        if o.empty:
            continue
        runs = sorted(set(o.run))
        rows.append({
            "condition": cond,
            "n_policies": len(runs),
            "n_walking": sum(r in walking for r in runs),
            "success_own": o.success_rate.mean(),
            "success_foreign": f.success_rate.mean() if len(f) else float("nan"),
            "progress_own": prog_own.reindex(runs).mean(),
            "progress_foreign": prog_for.reindex(runs).mean(),
        })
    out = pd.DataFrame(rows)
    out["drop"] = out.success_own - out.success_foreign
    out.attrs["params"], out.attrs["shared"] = params, shared
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--walking-only", action="store_true", help="keep only policies with >= 1 m at nominal on their own robot")
    p.add_argument("--out", type=Path, help="CSV path (default <root>/transfer_table[_walking].csv)")
    args = p.parse_args()
    root = experiments_root()

    frames = []
    for recipe, go1_dir, a1_dir in PAIRS:
        for trained, study, other in (("go1", go1_dir, "a1"), ("a1", a1_dir, "go1")):
            own_path, foreign_path = root / study / "results.csv", root / study / f"results_on_{other}.csv"
            if not own_path.exists() or not foreign_path.exists():
                print(f"skip {recipe:14s} {trained}: missing {own_path.name if not own_path.exists() else foreign_path.name}")
                continue
            own_df, foreign_df = pd.read_csv(own_path), pd.read_csv(foreign_path)
            table = compare(own_df, foreign_df, args.walking_only)
            table.insert(0, "evaluated_on", other)
            table.insert(0, "trained_robot", trained)
            table.insert(0, "recipe", recipe)
            n_shared, n_own = len(table.attrs["shared"]), own_df.run.nunique()
            partial = f"  [PARTIAL: {n_shared}/{n_own} policies]" if n_shared < n_own else ""
            print(f"\n{recipe}  trained on {trained}, evaluated on {other}{partial}  "
                  f"(params: {', '.join(table.attrs['params'])})")
            print(table.drop(columns=["recipe", "trained_robot", "evaluated_on"]).round(3).to_string(index=False))
            frames.append(table)

    if not frames:
        raise SystemExit(f"\nno cross-robot results under {root}; run runpod/run_cross_robot.sh first")
    result = pd.concat(frames, ignore_index=True)
    out = args.out or root / ("transfer_table_walking.csv" if args.walking_only else "transfer_table.csv")
    result.to_csv(out, index=False)

    # Pooled over recipes: the one number per condition and direction.
    pooled = (result.groupby(["trained_robot", "condition"])[["success_own", "success_foreign", "drop"]]
              .mean().reset_index())
    pooled["condition"] = pd.Categorical(pooled.condition, ORDER, ordered=True)
    print("\npooled over recipes (mean of the per-recipe means):")
    print(pooled.sort_values(["trained_robot", "condition"]).round(3).to_string(index=False))
    print(f"\ntable -> {out}")


if __name__ == "__main__":
    main()
