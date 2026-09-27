"""Step 4 - pair features for every candidate pair.

Feature groups (all country-agnostic; the country label itself is NOT a feature, so
France is scored with exactly the same model inputs as US/India):

  embedding   cos, rank, gap to the record's best neighbour, margin over the runner-up,
              number of candidates of the S2/S3 record and of the S1 record, rank of the
              pair inside the S1 record's candidate list
  name        rapidfuzz ratio / token_sort / token_set / partial / Jaro-Winkler on the
              normalised name, ratio / token_set / exact on the "core" name (legal forms
              and stop words removed), space-insensitive ratio (domain names), best DBA
              part, word-TF-IDF cosine, char-3gram TF-IDF cosine, S1 name frequency
  address     ratio / token_set / token_sort / partial, word- and char-TF-IDF cosine,
              number-set overlap (house numbers, PIN / ZIP codes), missing-address flags
  context     for key similarities: difference to the best value among the other
              candidates of the same S2/S3 record and of the same S1 record
  flags       source (S2/S3), native-script name, domain name, DBA

    python src/features.py --split train
"""
import argparse
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from tqdm import tqdm

from config import N_JOBS, pmap, split_dir

_VEC = {
    "w": dict(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 1)),
    "c": dict(analyzer="char_wb", ngram_range=(3, 3)),
}
_HV = None


def _hash_chunk(texts):
    return _HV.transform(texts)


def tfidf_matrix(texts, kind):
    """Hashed TF-IDF matrix (L2-normalised rows) computed in parallel chunks."""
    global _HV
    _HV = HashingVectorizer(n_features=2 ** 21, alternate_sign=False, norm=None, lowercase=False, **_VEC[kind])
    step = 200_000
    chunks = [texts[i:i + step] for i in range(0, len(texts), step)]
    mats = pmap(_hash_chunk, chunks, desc=f"tf-idf hash ({kind})", chunksize=1)
    import scipy.sparse as sp
    x = sp.vstack(mats).tocsr()
    return TfidfTransformer(sublinear_tf=True).fit_transform(x).astype(np.float32).tocsr()


def rowwise_cos(x, a, b, step=1_000_000, desc="row-wise cosine"):
    out = np.empty(len(a), dtype=np.float32)
    for i in tqdm(range(0, len(a), step), desc=desc, unit="Mpair"):
        out[i:i + step] = np.asarray(x[a[i:i + step]].multiply(x[b[i:i + step]]).sum(axis=1)).ravel()
    return out


def str_sims(a, b, prefix, scorers, empty_nan=True, step=1_000_000):
    out = {}
    empty = (np.array([len(x) == 0 for x in a]) | np.array([len(x) == 0 for x in b])) if empty_nan else None
    bar = tqdm(total=len(a) * len(scorers), desc=f"{prefix} similarities", unit="pair", unit_scale=True)
    for nm, sc in scorers.items():
        v = np.empty(len(a), dtype=np.float32)
        for i in range(0, len(a), step):  # chunked so the bar moves
            v[i:i + step] = cpdist(a[i:i + step], b[i:i + step], scorer=sc, workers=N_JOBS, dtype=np.float32)
            bar.update(len(v[i:i + step]))
        if sc is not JaroWinkler.normalized_similarity:
            v = v / 100.0
        if empty is not None:
            v[empty] = np.nan
        out[f"{prefix}_{nm}"] = v
    bar.close()
    return out


def _alt_best(args):
    alt_a, core_a, alt_b, core_b = args
    best = 0.0
    for x in (alt_a.split("|") if alt_a else [core_a]):
        for y in (alt_b.split("|") if alt_b else [core_b]):
            best = max(best, fuzz.token_set_ratio(x, y))
    return best / 100.0


def _num_closeness(args):
    """Best digit-string similarity and smallest relative numeric difference between the
    number sets of two addresses (house numbers are perturbed by the noise generator:
    '51101' vs '50737', typos '5740' vs '5749')."""
    xa, xb = args
    best_sim, best_rel = 0.0, 1.0
    for x in xa:
        for y in xb:
            best_sim = max(best_sim, fuzz.ratio(x, y))
            if len(x) < 10 and len(y) < 10:
                ix, iy = int(x), int(y)
                best_rel = min(best_rel, abs(ix - iy) / max(ix, iy, 1))
    return best_sim / 100.0, best_rel


def embedding_features(cand: pl.DataFrame) -> dict:
    """Blocking-derived features: similarity, rank and competition among neighbours."""
    feats = {}
    feats["cos"] = cand["cos"].to_numpy()
    feats["rank"] = cand["rank"].to_numpy().astype(np.float32)
    feats["gap_r"] = cand["cos1"].to_numpy() - feats["cos"]
    feats["margin_r"] = np.where(cand["rank"].to_numpy() == 0,
                                 feats["cos"] - cand["cos2"].to_numpy(), feats["cos"] - cand["cos1"].to_numpy())
    grp = cand.select(
        pl.len().over("r_idx").alias("n_cand_r"),
        pl.len().over("s1_idx").alias("n_cand_s1"),
        pl.col("cos").rank("ordinal", descending=True).over("s1_idx").alias("rank_s1"),
        (pl.col("cos").max().over("s1_idx") - pl.col("cos")).alias("gap_s1"),
    )
    for c in grp.columns:
        feats[c] = grp[c].to_numpy().astype(np.float32)
    if "chan" in cand.columns:  # v6: pair found only by the name-only channel
        feats["chan"] = cand["chan"].to_numpy().astype(np.float32)
    return feats


def number_overlap(recs: pl.DataFrame, a, b):
    """Overlap of the address number sets (house numbers, PIN / ZIP codes)."""
    nums = recs.select(pl.when(pl.col("addr_nums") == "").then(None).otherwise(pl.col("addr_nums"))
                       .str.split(" ").alias("n"))["n"]
    pn = pl.DataFrame({"na": nums.gather(a), "nb": nums.gather(b)}).select(
        pl.col("na").list.set_intersection("nb").list.len().alias("common"),
        pl.col("na").list.len().alias("la"), pl.col("nb").list.len().alias("lb"))
    common = pn["common"].fill_null(0).to_numpy().astype(np.float32)
    la = pn["la"].fill_null(0).to_numpy().astype(np.float32)
    lb = pn["lb"].fill_null(0).to_numpy().astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        jacc = np.where((la > 0) & (lb > 0), common / (la + lb - common), np.nan).astype(np.float32)
    return nums, la, lb, {"num_common": common, "num_only_r": la - common, "num_only_s1": lb - common,
                          "num_jacc": jacc}


def add_labels(df: pl.DataFrame, recs: pl.DataFrame, a, b) -> pl.DataFrame:
    """Train split only: label y (is the S1 the true owner of the S2/S3 record) and fold bucket."""
    if "true_s1" in recs.columns and recs["true_s1"].null_count() < recs.height:
        ids = recs["entity_id"].to_numpy()
        true = recs["true_s1"].to_numpy()
        df = df.with_columns(pl.Series("y", (true[a] == ids[b]).astype(np.int8)),
                             pl.Series("bucket", recs["bucket"].to_numpy()[a]))
    return df


def build(split: str):
    t = time.time()
    d = split_dir(split)
    recs = pl.read_parquet(d / "records.parquet")
    cand = pl.read_parquet(d / "candidates.parquet")
    a = cand["r_idx"].to_numpy()   # S2/S3 side
    b = cand["s1_idx"].to_numpy()  # S1 side
    print(f"[features] {split}: {cand.height:,} pairs", flush=True)

    col = {c: recs[c].to_numpy() for c in ("name", "name_core", "name_alt", "addr")}
    feats = {}

    # ---------------------------------------------------------------- embedding / blocking
    feats.update(embedding_features(cand))
    if "p1" in cand.columns:  # out-of-fold probability of the stage-1 prefilter
        feats["p1"] = cand["p1"].to_numpy().astype(np.float32)
    print(f"[features] embedding done ({time.time()-t:.0f}s)", flush=True)

    # ---------------------------------------------------------------- names
    na, nb = col["name"][a].tolist(), col["name"][b].tolist()
    ca, cb = col["name_core"][a].tolist(), col["name_core"][b].tolist()
    feats.update(str_sims(na, nb, "name", {
        "ratio": fuzz.ratio, "tsort": fuzz.token_sort_ratio, "tset": fuzz.token_set_ratio,
        "partial": fuzz.partial_ratio, "jw": JaroWinkler.normalized_similarity}))
    feats.update(str_sims(ca, cb, "core", {"ratio": fuzz.ratio, "tset": fuzz.token_set_ratio,
                                           "jw": JaroWinkler.normalized_similarity}))
    feats["core_eq"] = np.array([x == y for x, y in zip(ca, cb)], dtype=np.float32)
    feats.update(str_sims([x.replace(" ", "") for x in ca], [x.replace(" ", "") for x in cb], "nospace",
                          {"ratio": fuzz.ratio, "partial": fuzz.partial_ratio}))
    alt = col["name_alt"]
    has_alt = np.nonzero((alt[a] != "") | (alt[b] != ""))[0]
    alt_best = np.array(feats["core_tset"], copy=True)
    if len(has_alt):
        alt_best[has_alt] = pmap(_alt_best, [(alt[a[i]], ca[i], alt[b[i]], cb[i]) for i in has_alt],
                                 desc="DBA / alt-name match", chunksize=5000)
    feats["alt_best"] = alt_best
    del na, nb
    print(f"[features] name string sims done ({time.time()-t:.0f}s)", flush=True)

    for field, kinds in (("name_core", ("w",)), ("name", ("c",)), ("addr", ("w", "c"))):
        texts = col[field].tolist()
        for k in kinds:
            x = tfidf_matrix(texts, k)
            feats[f"tfidf_{k}_{field}"] = rowwise_cos(x, a, b, desc=f"tf-idf cosine ({k}, {field})")
            del x
    print(f"[features] tf-idf cosines done ({time.time()-t:.0f}s)", flush=True)

    # name ambiguity: how many S1 records of the same country share this core name
    s1 = recs.with_row_index("idx").filter(pl.col("src") == 1)
    freq = s1.group_by(["country", "name_core"]).len("f")
    s1f = s1.join(freq, on=["country", "name_core"], how="left").select("idx", "f")
    fmap = np.zeros(recs.height, dtype=np.float32)
    fmap[s1f["idx"].to_numpy()] = s1f["f"].to_numpy()
    feats["s1_name_freq"] = np.log1p(fmap[b])

    # ---------------------------------------------------------------- addresses
    aa, ab = col["addr"][a].tolist(), col["addr"][b].tolist()
    feats.update(str_sims(aa, ab, "addr", {
        "ratio": fuzz.ratio, "tset": fuzz.token_set_ratio, "tsort": fuzz.token_sort_ratio,
        "partial": fuzz.partial_ratio}))
    del aa, ab
    nums, la, lb, num_feats = number_overlap(recs, a, b)
    feats.update(num_feats)
    nl = nums.to_list()
    both = np.nonzero((la > 0) & (lb > 0))[0]
    num_sim = np.full(len(a), np.nan, dtype=np.float32)
    num_rel = np.full(len(a), np.nan, dtype=np.float32)
    if len(both):
        res = pmap(_num_closeness, [(nl[a[i]], nl[b[i]]) for i in both], desc="house-number closeness",
                   chunksize=20000)
        res = np.array(res, dtype=np.float32)
        num_sim[both], num_rel[both] = res[:, 0], res[:, 1]
    feats["num_best_sim"] = num_sim
    feats["num_min_reldiff"] = num_rel
    missing = recs["addr_missing"].to_numpy()
    feats["addr_missing_r"] = missing[a].astype(np.float32)
    feats["addr_missing_s1"] = missing[b].astype(np.float32)
    print(f"[features] address done ({time.time()-t:.0f}s)", flush=True)

    # ---------------------------------------------------------------- flags
    feats["src"] = recs["src"].to_numpy()[a].astype(np.float32)
    feats["r_native"] = recs["name_native"].to_numpy()[a].astype(np.float32)
    dom = recs["name_is_domain"].to_numpy()
    feats["r_domain"] = dom[a].astype(np.float32)
    feats["s1_domain"] = dom[b].astype(np.float32)
    feats["any_dba"] = (recs["name_has_dba"].to_numpy()[a] | recs["name_has_dba"].to_numpy()[b]).astype(np.float32)
    feats["len_name_r"] = np.array([len(x.split()) for x in ca], dtype=np.float32)
    feats["len_name_s1"] = np.array([len(x.split()) for x in cb], dtype=np.float32)

    # ---------------------------------------------------------------- context (competition)
    df = pl.DataFrame({"r_idx": a, "s1_idx": b, **feats})
    ctx_cols = ["core_tset", "name_jw", "addr_tset", "tfidf_c_name", "tfidf_c_addr", "num_jacc", "num_best_sim"]
    df = df.with_columns(
        [(pl.col(c) - pl.col(c).max().over("r_idx")).alias(f"{c}_dr") for c in ctx_cols]
        + [(pl.col(c) - pl.col(c).max().over(["s1_idx", "src"])).alias(f"{c}_ds1") for c in ctx_cols]
    )

    df = add_labels(df, recs, a, b)
    df.write_parquet(d / "features.parquet")
    print(f"[features] {split}: wrote {df.shape} ({time.time()-t:.0f}s)")


def feature_columns(df: pl.DataFrame):
    return [c for c in df.columns if c not in ("r_idx", "s1_idx", "y", "bucket")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    build(ap.parse_args().split)


if __name__ == "__main__":
    main()
