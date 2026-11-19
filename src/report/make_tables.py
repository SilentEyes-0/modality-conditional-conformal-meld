"""Build result tables from the final test summaries.

Usage (from the project root):
    python -m src.report.make_tables --dir outputs_amendment --out report_out

Table numbering follows the P8 plan (R1 to R10) and may need mapping to the paper's shells.
"""
import argparse
import pandas as pd
from .common import (load, sort_patterns, pm, save, MAIN_METHODS, split_condition, split_design_b,
                     keep_pattern_rows, class_name, wilson)

MAIN_ROWS = [("S1-text", "100"), ("S1-audio", "010"), ("S1-video", "001"), ("S2", "111"), ("S3", "111")]


def classification_tables(c, out):
    c = c.copy()
    for tag, mets in (("acc", "acc"), ("macro_f1", "macro_f1"), ("weighted_f1", "weighted_f1")):
        c[tag] = [pm(a, b) for a, b in zip(c[f"{mets}_R_mean"], c[f"{mets}_R_std"])]
    # R1: main comparison
    keep = c[[(s, p) in MAIN_ROWS for s, p in zip(c.system, c.pattern)]]
    r1 = keep[["task", "system", "pattern", "acc", "macro_f1", "weighted_f1"]]
    r1 = r1.assign(_o=r1.apply(lambda r: MAIN_ROWS.index((r.system, r.pattern)), axis=1)).sort_values(["task", "_o"]).drop(columns="_o")
    save(r1, out, "R1_classification_main")
    # R2: calibration of probabilities, raw (R) vs temperature-scaled (T)
    r2 = c[["task", "system", "pattern"]].copy()
    for m in ("ece", "brier", "nll"):
        r2[f"{m}_raw"] = [pm(a, b) for a, b in zip(c[f"{m}_R_mean"], c[f"{m}_R_std"])]
        r2[f"{m}_scaled"] = [pm(a, b) for a, b in zip(c[f"{m}_T_mean"], c[f"{m}_T_std"])]
    r2 = r2.sort_values(["task", "system", "pattern"])
    save(r2, out, "R2_probability_calibration")
    # R3: performance by pattern for the two multimodal classifiers
    r3 = c[c.system.isin(["S2", "S3"])][["task", "system", "pattern", "acc", "macro_f1", "weighted_f1"]]
    r3 = sort_patterns(r3).sort_values(["task", "system"], kind="stable")
    save(r3, out, "R3_performance_by_pattern")


def coverage_tables(s, out, variants=("T", "R")):
    base = s[(s.design == "A") & (s.condition == "clean")]
    spec = [("R4", "emotion", 0.10), ("R5", "emotion", 0.05), ("R6", "sentiment", 0.10), ("R7", "sentiment", 0.05)]
    for rid, task, alpha in spec:
        for v in variants:
            x = keep_pattern_rows(base[(base.task == task) & (base.variant == v) & (base.alpha.round(4) == alpha) & (base.method.isin(MAIN_METHODS))])
            if x.empty:
                print("skip", rid, task, alpha, v, "(no rows)")
                continue
            pats = sort_patterns(x[["pattern"]].drop_duplicates())["pattern"].tolist()
            rows = []
            for p in pats:
                row = {"pattern": p}
                n = x[(x.pattern == p)]["n"]
                row["n"] = int(n.iloc[0]) if len(n) else ""
                for m in MAIN_METHODS:
                    r = x[(x.pattern == p) & (x.method == m)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    row[f"{m}: coverage"] = r.coverage
                    row[f"{m}: interval"] = f"[{r.cov_lo:.3f}, {r.cov_hi:.3f}]"
                    row[f"{m}: size"] = r["size"]
                rows.append(row)
            save(pd.DataFrame(rows), out, f"{rid}_{task}_alpha{alpha:.2f}_{v}_coverage_size")


def class_table(s, out, variants=("T", "R")):
    """Class-conditional coverage (Design A, clean): coverage within each true class, by method."""
    base = s[(s.design == "A") & (s.condition == "clean") & (s.method.isin(MAIN_METHODS)) & (s.pattern.str.startswith("class"))]
    for task in ("emotion", "sentiment"):
        for alpha in (0.10, 0.05):
            for v in variants:
                x = base[(base.task == task) & (base.variant == v) & (base.alpha.round(4) == alpha)]
                if x.empty:
                    continue
                rows = []
                for lab in sorted(x.pattern.unique(), key=lambda z: int(z.replace("class", ""))):
                    row = {"class": class_name(task, lab), "n": int(x[x.pattern == lab].n.iloc[0])}
                    for m in MAIN_METHODS:
                        r = x[(x.pattern == lab) & (x.method == m)]
                        if len(r):
                            row[f"{m}: coverage"] = r.iloc[0].coverage
                            lo, hi = wilson(r.iloc[0].coverage, row["n"])
                            row[f"{m}: approx. 95% interval"] = f"[{lo:.3f}, {hi:.3f}]"
                    rows.append(row)
                save(pd.DataFrame(rows), out, f"R11_class_coverage_{task}_alpha{alpha:.2f}_{v}")


def fallback_table(s, out, alpha=0.10, variant="T"):
    x = s[s.design.str.startswith("B-") & (s.method == "conditional") & (s.condition == "clean")
          & (s.variant == variant) & (s.alpha.round(4) == alpha)].copy()
    x = keep_pattern_rows(x)
    if x.empty:
        print("skip R8 (no Design B rows)")
        return
    x["profile"], x["n_min"] = zip(*x.design.map(split_design_b))
    r8 = x[["task", "profile", "n_min", "pattern", "level", "n", "coverage", "coverage_sd", "size"]]
    r8 = sort_patterns(r8).sort_values(["task", "profile", "n_min"], kind="stable")
    save(r8, out, f"R8_small_groups_fallback_alpha{alpha:.2f}_{variant}")


def corruption_table(s, out, alpha=0.10, variant="T"):
    x = s[(s.design == "A") & (s.variant == variant) & (s.alpha.round(4) == alpha) & (s.method.isin(MAIN_METHODS))].copy()
    x = keep_pattern_rows(x)
    if not (x.condition != "clean").any():
        print("skip R9 (no corruption rows)")
        return
    x["corruption"], x["calibration"] = zip(*x.condition.map(split_condition))
    for task in x.task.unique():
        t = x[x.task == task]
        rows = []
        for (cor, cal, p), g in t.groupby(["corruption", "calibration", "pattern"]):
            row = {"corruption": cor, "calibration": cal or "clean", "pattern": p}
            for m in MAIN_METHODS:
                r = g[g.method == m]
                if len(r):
                    row[f"{m}: coverage"] = r.iloc[0].coverage
                    row[f"{m}: size"] = r.iloc[0]["size"]
            rows.append(row)
        df = pd.DataFrame(rows)
        df = df.assign(_c=(df.corruption != "clean").astype(int)).sort_values(["_c", "corruption", "calibration"], kind="stable").drop(columns="_c")
        df = sort_patterns(df).sort_values(["corruption", "calibration"], kind="stable")
        save(df, out, f"R9_corruption_{task}_alpha{alpha:.2f}_{variant}")


def reliability_table(w, out, alpha=0.10, variant="T"):
    x = w[(w.condition == "clean") & (w.variant == variant) & (w.alpha.round(4) == alpha)]
    if x.empty:
        print("skip R10 (no rows)")
        return
    r10 = x[["task", "design", "method", "worst_deviation", "worst_undercoverage", "mean_size"]]
    r10 = r10.sort_values(["task", "design", "method"])
    save(r10, out, f"R10_group_reliability_alpha{alpha:.2f}_{variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs_amendment")
    ap.add_argument("--out", default="report_out")
    a = ap.parse_args()
    s, c, w = load(a.dir)
    classification_tables(c, a.out)
    coverage_tables(s, a.out)
    class_table(s, a.out)
    fallback_table(s, a.out)
    corruption_table(s, a.out)
    reliability_table(w, a.out)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
