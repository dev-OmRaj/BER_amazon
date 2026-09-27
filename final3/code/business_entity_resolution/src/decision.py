"""Turning pair probabilities into final matches, and the challenge metric.

Decision rule
  1. exclusivity: every S2/S3 record belongs to at most one S1 entity, so only its
     highest-probability S1 candidate is kept (ties broken by embedding cosine);
  2. selection, one of
       "threshold" : keep the pair if p >= t
       "expected_f": per S1, choose the prefix of its candidates (sorted by p) that
                     maximises the expected per-entity F0.5, including the empty set
                     (which scores 1 when the entity is a singleton). Pairs below the
                     floor t are never considered.
     The rule and t are chosen on out-of-fold training predictions.
"""
import numpy as np
import polars as pl


def exclusive(pairs: pl.DataFrame) -> pl.DataFrame:
    """pairs: s1_id, r_id, p, cos -> keep only the best S1 per r_id."""
    return pairs.sort(["p", "cos"], descending=True).unique("r_id", keep="first", maintain_order=False)


def select_threshold(pairs: pl.DataFrame, t: float) -> pl.DataFrame:
    return pairs.filter(pl.col("p") >= t).select("s1_id", "r_id")


def select_expected_f(pairs: pl.DataFrame, t: float) -> pl.DataFrame:
    """Per S1, keep the top-k candidates maximising expected F0.5 (independence approx.)."""
    df = pairs.filter(pl.col("p") >= t).sort(["s1_id", "p"], descending=[False, True]).with_columns(
        pl.col("p").cum_sum().over("s1_id").alias("cp"),
        pl.int_range(1, pl.len() + 1).over("s1_id").alias("k"),
        pl.col("p").sum().over("s1_id").alias("sp"),
        (1 - pl.col("p")).log().sum().over("s1_id").exp().alias("p_none"),
    ).with_columns((1.25 * pl.col("cp") / (pl.col("k") + 0.25 * pl.col("sp"))).alias("ef"))
    best = df.group_by("s1_id").agg(pl.col("ef").max().alias("ef_max"), pl.col("p_none").first(),
                                    pl.col("k").get(pl.col("ef").arg_max()).alias("k_best"))
    best = best.filter(pl.col("ef_max") > pl.col("p_none"))
    return df.join(best.select("s1_id", "k_best"), on="s1_id").filter(pl.col("k") <= pl.col("k_best")) \
        .select("s1_id", "r_id")


def decide(pairs: pl.DataFrame, rule: str, t: float) -> pl.DataFrame:
    ex = exclusive(pairs)
    return select_threshold(ex, t) if rule == "threshold" else select_expected_f(ex, t)


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> float:
    """Challenge metric: F0.5 per S1 entity (singletons included), averaged.

    pred, truth: s1_id, r_id pairs.  s1_ids: the S1 entities being evaluated.
    Per entity F0.5 = 1.25*TP / (n_pred + 0.25*n_true), and 1.0 if both are empty.
    """
    base = pl.DataFrame({"s1_id": s1_ids})
    tp = pred.join(truth, on=["s1_id", "r_id"]).group_by("s1_id").len("tp")
    npred = pred.group_by("s1_id").len("npred")
    ntrue = truth.group_by("s1_id").len("ntrue")
    df = base.join(tp, on="s1_id", how="left").join(npred, on="s1_id", how="left") \
        .join(ntrue, on="s1_id", how="left").fill_null(0)
    f = np.where((df["npred"] + df["ntrue"]).to_numpy() == 0, 1.0,
                 1.25 * df["tp"].to_numpy() / np.maximum(df["npred"].to_numpy() + 0.25 * df["ntrue"].to_numpy(), 1e-9))
    return float(f.mean())
