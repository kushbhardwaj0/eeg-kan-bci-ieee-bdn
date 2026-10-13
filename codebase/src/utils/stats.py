"""
Statistical Significance Testing & Effect Size Analysis Module.

Paired Wilcoxon signed-rank tests, Cohen's d, rank-biserial correlation, Holm & Benjamini-Hochberg corrections for
the family of baseline comparisons, and the subject-level aggregation they must run on
(one score per subject: sessions and seeds averaged first).
"""

from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- paired tests
def compute_wilcoxon_test(
    scores_a: Union[np.ndarray, list],
    scores_b: Union[np.ndarray, list],
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Paired Wilcoxon signed-rank test between two arrays of per-subject scores.

    Returns ``statistic`` (W), ``p_value``, ``mean_diff``, ``std_diff``.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(f"Length mismatch for paired samples: {len(a)} vs {len(b)}")

    diffs = a - b
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    if np.allclose(diffs, 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "mean_diff": mean_diff, "std_diff": std_diff}

    try:
        res = stats.wilcoxon(a, b, alternative=alternative)
        stat, p_val = float(res.statistic), float(res.pvalue)
    except Exception:
        # Identical non-zero differences: rank variance is zero, use the exact sign-test bound.
        non_zero = diffs[diffs != 0]
        n_nz = len(non_zero)
        if n_nz > 0 and (np.all(non_zero > 0) or np.all(non_zero < 0)):
            stat = 0.0 if np.all(non_zero > 0) else float(n_nz * (n_nz + 1) / 2)
            p_val = float(2.0 ** (1 - n_nz)) if alternative == "two-sided" else float(2.0 ** (-n_nz))
        else:
            stat, p_val = 0.0, 1.0

    return {"statistic": stat, "p_value": p_val, "mean_diff": mean_diff, "std_diff": std_diff}


def compute_cohens_d(scores_a, scores_b) -> float:
    """
    Paired Cohen's d: mean(diff) / std(diff).

    Zero variance gives ``inf`` (or ``-inf``) when the mean difference is non-zero and ``nan``
    when every difference is zero. Never a fake finite number.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    diffs = a - b
    if len(diffs) < 2:
        return float("nan")
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    if std_diff < 1e-12:
        return float(np.sign(mean_diff) * np.inf) if mean_diff != 0 else float("nan")
    return mean_diff / std_diff


def compute_rank_biserial(scores_a, scores_b) -> float:
    """Rank-biserial correlation r = (W+ - W-) / (W+ + W-) for paired samples."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    diffs = a - b
    non_zero = diffs[diffs != 0]
    if len(non_zero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(non_zero))
    w_plus = np.sum(ranks[non_zero > 0])
    w_minus = np.sum(ranks[non_zero < 0])
    total = w_plus + w_minus
    return float((w_plus - w_minus) / total) if total > 1e-12 else 0.0


# --------------------------------------------------------------------------- corrections
def holm_correction(p_values: Iterable[float]) -> List[float]:
    """
    Holm-Bonferroni step-down adjusted p-values (controls the family-wise error rate).

    adj_(i) = max_{j<=i} min(1, (m - j + 1) * p_(j)) over the ascending-sorted p-values.
    NaN inputs stay NaN and are not counted in ``m``.
    """
    p = np.asarray(list(p_values), dtype=np.float64)
    adjusted = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    m = int(valid.sum())
    if m == 0:
        return adjusted.tolist()
    order = np.argsort(p[valid])
    sorted_p = p[valid][order]
    stepped = np.minimum(1.0, (m - np.arange(m)) * sorted_p)
    stepped = np.maximum.accumulate(stepped)
    out = np.empty(m)
    out[order] = stepped
    adjusted[valid] = out
    return adjusted.tolist()


def benjamini_hochberg_correction(p_values: Iterable[float]) -> List[float]:
    """
    Benjamini-Hochberg (FDR) adjusted p-values.
    """
    p = np.asarray(list(p_values), dtype=np.float64)
    adjusted = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    m = int(valid.sum())
    if m == 0:
        return adjusted.tolist()
    
    valid_p = p[valid]
    sort_idx = np.argsort(valid_p)
    p_adj_sorted = np.zeros(m)
    for k in range(m):
        i = sort_idx[k]
        p_adj = min(valid_p[i] * m / (k + 1), 1.0)
        if k > 0:
            p_adj = max(p_adj, p_adj_sorted[sort_idx[k - 1]])
        p_adj_sorted[i] = p_adj

    out = np.empty(m)
    out[sort_idx] = p_adj_sorted
    adjusted[valid] = out
    return adjusted.tolist()


# --------------------------------------------------------------------------- aggregation
def aggregate_per_subject(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    One score per (model, subject): the mean over every session and seed row.
    This is the unit the paired tests must run on (audit item 22).
    """
    if "model" not in results_df.columns or "score" not in results_df.columns:
        raise ValueError("DataFrame must contain 'model' and 'score' columns")
    if "subject" not in results_df.columns:
        raise ValueError("DataFrame must contain a 'subject' column")
    return results_df.groupby(["model", "subject"], as_index=False)["score"].mean()


def seed_spread(results_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Per model: mean and std of the subject-averaged score across seeds (nan with one seed)."""
    if "seed" not in results_df.columns:
        return {m: {"mean": float(g["score"].mean()), "std": float("nan"), "n_seeds": 1}
                for m, g in results_df.groupby("model")}
    per_seed = results_df.groupby(["model", "seed"])["score"].mean().reset_index()
    out = {}
    for m, g in per_seed.groupby("model"):
        out[m] = {"mean": float(g["score"].mean()),
                  "std": float(g["score"].std(ddof=1)) if len(g) > 1 else float("nan"),
                  "n_seeds": int(len(g))}
    return out


# --------------------------------------------------------------------------- suite
def run_statistical_suite(
    results_df: pd.DataFrame,
    target_model: str = "cnn_kan",
    correction: Optional[str] = "holm",
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare ``target_model`` against every other model on per-subject scores.

    Returns a dict baseline -> metrics including the raw ``p_value``, the adjusted
    ``p_adj`` across the family of baselines, effect sizes, and the seed spread.
    Raises ``ValueError`` if the target model is absent (never a silent ``{}``).
    """
    per_subject = aggregate_per_subject(results_df)
    models = list(per_subject["model"].unique())
    target_name = target_model.lower()
    matching = [m for m in models if str(m).lower() == target_name]
    if not matching:
        raise ValueError(f"target model '{target_model}' not in results; available: {sorted(map(str, models))}")
    target_key = matching[0]

    target_scores = per_subject[per_subject["model"] == target_key].set_index("subject")["score"]
    spread = seed_spread(results_df)

    comparisons: Dict[str, Dict[str, Any]] = {}
    for baseline in models:
        if baseline == target_key:
            continue
        base_scores = per_subject[per_subject["model"] == baseline].set_index("subject")["score"]
        common = target_scores.index.intersection(base_scores.index)
        if len(common) == 0:
            continue
        t = target_scores.loc[common].values
        b = base_scores.loc[common].values
        w = compute_wilcoxon_test(t, b)
        comparisons[baseline] = {
            "target_model": target_key,
            "baseline_model": baseline,
            "num_subjects": int(len(common)),
            "target_mean_acc": float(np.mean(t)),
            "baseline_mean_acc": float(np.mean(b)),
            "mean_diff": w["mean_diff"],
            "w_statistic": w["statistic"],
            "p_value": w["p_value"],
            "cohens_d": compute_cohens_d(t, b),
            "rank_biserial": compute_rank_biserial(t, b),
            "target_seed_std": spread.get(target_key, {}).get("std", float("nan")),
            "baseline_seed_std": spread.get(baseline, {}).get("std", float("nan")),
        }

    names = list(comparisons)
    raw = [comparisons[n]["p_value"] for n in names]
    if correction == "fdr" or correction == "bh":
        adjusted = benjamini_hochberg_correction(raw)
    elif correction == "holm":
        adjusted = holm_correction(raw)
    else:
        adjusted = list(raw)

    for n, p_adj in zip(names, adjusted):
        comparisons[n]["p_adj"] = float(p_adj)
        comparisons[n]["p_value_corrected"] = float(p_adj)
        comparisons[n]["correction"] = correction or "none"
        comparisons[n]["n_comparisons"] = len(names)
        comparisons[n]["statistically_significant"] = bool(p_adj < alpha)
    return comparisons
