"""Blocking-quality report for a candidate set (per country and overall).

  reduction ratio    1 - |candidates| / |S1 x (S2+S3)| within the country label
                     (and vs. the full cross product ignoring country)
  cand per S1        mean / median / p90 / p99 / max candidates per Source-1 entity
  pair completeness  (train only) share of true pairs present in the candidate set,
                     measured on clusters outside the encoder buckets
  entity completeness(train only) share of S1 entities whose every true match is present

    python src/blocking_stats.py --split test --file candidates.parquet
"""
import argparse
import json

import polars as pl

from config import ENCODER_BUCKETS, split_dir


def report(split: str, cand: pl.DataFrame, name: str) -> dict:
    recs = pl.read_parquet(split_dir(split) / "records.parquet",
                           columns=["entity_id", "src", "country", "true_s1", "bucket"]).with_row_index("idx")
    s1 = recs.filter(pl.col("src") == 1)
    per_s1 = s1.select("idx", "country").join(
        cand.group_by("s1_idx").len("n"), left_on="idx", right_on="s1_idx", how="left").fill_null(0)
    n_r = recs.filter(pl.col("src") != 1).group_by("country").len("n_r")
    stats = {"name": name, "split": split, "countries": {}}
    labelled = recs["true_s1"].null_count() < recs.height
    if labelled:
        ids = recs["entity_id"].to_numpy()
        true = recs["true_s1"].to_numpy()
        hit = pl.DataFrame({"s1_id": ids[cand["s1_idx"].to_numpy()], "r_id": ids[cand["r_idx"].to_numpy()]}) \
            .filter(pl.Series(true[cand["r_idx"].to_numpy()] == ids[cand["s1_idx"].to_numpy()]))
        gt = recs.filter(pl.col("true_s1").is_not_null() & ~pl.col("bucket").is_in(list(ENCODER_BUCKETS))) \
            .select(pl.col("true_s1").alias("s1_id"), pl.col("entity_id").alias("r_id"), "country")
        gt = gt.join(hit.with_columns(pl.lit(True).alias("found")), on=["s1_id", "r_id"], how="left") \
            .with_columns(pl.col("found").fill_null(False))

    def summarise(sub_s1, cand_pairs, full_pairs, gt_sub):
        n = sub_s1["n"]
        out = {
            "s1_entities": sub_s1.height,
            "candidate_pairs": int(cand_pairs),
            "all_pairs_within_country": int(full_pairs),
            "reduction_ratio": 1 - cand_pairs / full_pairs if full_pairs else None,
            "cand_per_s1_mean": float(n.mean()), "cand_per_s1_median": float(n.median()),
            "cand_per_s1_p90": float(n.quantile(0.9)), "cand_per_s1_p99": float(n.quantile(0.99)),
            "cand_per_s1_max": int(n.max()), "s1_without_candidates": int((n == 0).sum()),
        }
        if gt_sub is not None and gt_sub.height:
            out["pair_completeness"] = float(gt_sub["found"].mean())
            ent = gt_sub.group_by("s1_id").agg(pl.col("found").all())
            out["entity_completeness"] = float(ent["found"].mean())
        return out

    total_full = 0
    for c in sorted(s1["country"].unique().to_list()):
        sub = per_s1.filter(pl.col("country") == c)
        nr = n_r.filter(pl.col("country") == c)["n_r"]
        full = sub.height * (int(nr[0]) if len(nr) else 0)
        total_full += full
        stats["countries"][c] = summarise(sub, sub["n"].sum(), full,
                                          gt.filter(pl.col("country") == c) if labelled else None)
    stats["overall"] = summarise(per_s1, per_s1["n"].sum(), total_full, gt if labelled else None)
    n_all_r = recs.filter(pl.col("src") != 1).height
    stats["overall"]["reduction_ratio_vs_full_cross_product"] = 1 - cand.height / (s1.height * n_all_r)

    path = split_dir(split) / f"blocking_stats_{name}.json"
    path.write_text(json.dumps(stats, indent=2))
    print(f"[blocking-stats] {split} / {name}")
    print(f"  {'country':<8} {'S1':>9} {'pairs':>12} {'reduction':>11} {'mean/S1':>8} {'med':>5} {'p99':>6}"
          + (f" {'pair-rec':>9} {'entity-rec':>10}" if labelled else ""))
    for c, v in list(stats["countries"].items()) + [("ALL", stats["overall"])]:
        line = (f"  {c:<8} {v['s1_entities']:>9,} {v['candidate_pairs']:>12,} {v['reduction_ratio']:>11.7f} "
                f"{v['cand_per_s1_mean']:>8.2f} {v['cand_per_s1_median']:>5.0f} {v['cand_per_s1_p99']:>6.0f}")
        if labelled and "pair_completeness" in v:
            line += f" {v['pair_completeness']:>9.5f} {v['entity_completeness']:>10.5f}"
        print(line)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--file", default="candidates.parquet")
    args = ap.parse_args()
    report(args.split, pl.read_parquet(split_dir(args.split) / args.file), args.file.replace(".parquet", ""))


if __name__ == "__main__":
    main()
