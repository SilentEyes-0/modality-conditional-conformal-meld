"""Aggregate per-run CSVs into summary tables (mean and std across runs) and the worst-pattern deviation."""
import glob, os
import numpy as np
import pandas as pd

KEYS = ["eval_split", "condition", "task", "variant", "design", "alpha", "method", "pattern"]


def read_all(out_dir, prefix, eval_split):
    files = sorted(glob.glob(f"{out_dir}/{prefix}_seed*_{eval_split}.csv"))
    return pd.concat([pd.read_csv(f, dtype={"pattern": str}) for f in files], ignore_index=True) if files else pd.DataFrame()


def summarise_conformal(df):
    g = df.groupby(KEYS, dropna=False)
    s = g.agg(runs=("coverage", "size"), n=("n", "mean"), coverage=("coverage", "mean"), coverage_sd=("coverage", "std"),
              cov_lo=("cov_lo", "mean"), cov_hi=("cov_hi", "mean"), size=("size", "mean"), size_sd=("size", "std"),
              empty=("empty", "mean"), singleton=("singleton", "mean"), full=("full", "mean"),
              level=("level", lambda x: ",".join(sorted(set(map(str, x)))))).reset_index()
    return s


def worst_pattern(summary):
    pats = summary[summary["pattern"].str.fullmatch(r"[01]{3}")].copy()
    pats["deviation"] = (pats["coverage"] - (1 - pats["alpha"])).abs()
    pats["undercoverage"] = ((1 - pats["alpha"]) - pats["coverage"]).clip(lower=0)
    k = ["eval_split", "condition", "task", "variant", "design", "alpha", "method"]
    return pats.groupby(k).agg(worst_deviation=("deviation", "max"), worst_undercoverage=("undercoverage", "max"),
                               mean_size=("size", "mean")).reset_index()


def summarise_classification(df):
    cols = [c for c in df.columns if c.endswith("_R") or c.endswith("_T")]
    return df.groupby(["eval_split", "system", "task", "pattern"]).agg({c: ["mean", "std"] for c in cols}).reset_index()


def aggregate(out_dir, eval_split):
    cdf = read_all(out_dir, "conformal", eval_split); kdf = read_all(out_dir, "classification", eval_split)
    if cdf.empty: raise FileNotFoundError("no conformal outputs found")
    s = summarise_conformal(cdf); s.to_csv(f"{out_dir}/summary_conformal_{eval_split}.csv", index=False)
    w = worst_pattern(s); w.to_csv(f"{out_dir}/worst_pattern_{eval_split}.csv", index=False)
    k = summarise_classification(kdf); k.columns = ["_".join([c for c in col if c]) for col in k.columns]
    k.to_csv(f"{out_dir}/summary_classification_{eval_split}.csv", index=False)
    return s, w, k
