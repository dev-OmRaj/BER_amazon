"""Learn normalisation dictionaries from the training ground truth.

1. token_dict  : native-script token -> Latin token.
   For every true pair whose S2/S3 side contains Indic-script tokens we count
   co-occurrences (native token w, Latin token v of the S1 record, same field).
   Among the most frequent v for a given w we pick the one whose consonant skeleton
   is closest to the `anyascii` transliteration of w.  E.g.
       "प्राइवेट" -> "private",  "शक्ति" -> "shakti",  "ಕರ್ನಾಟಕ" -> "karnataka".

2. comp_dict   : (country, address component) -> canonical component.
   For true pairs we look at comma-separated address components present in the S2/S3
   record but not in the S1 record, and vice versa.  Systematic substitutions are
   kept, e.g. ("US","north carolina") -> "NC", ("India","mh") -> "Maharashtra",
   ("India","calcutta") -> "Kolkata", ("India","महाराष्ट्र") -> "Maharashtra".
   The target is always the Source-1 spelling (Source 1 is the reference source).

Only the ENCODER_BUCKETS clusters are used, so the matcher's validation folds never
contribute to the dictionaries (no leakage into out-of-fold scores).
"""
import pickle
import re
import time

import polars as pl
from anyascii import anyascii
from rapidfuzz import fuzz
from tqdm import tqdm

from config import ENCODER_BUCKETS, WORK_DIR, pmap
from text_norm import comp_key, is_native, latin_token, raw_tokens

_VOWELS = re.compile(r"[aeiouyh]+")
_PHON = str.maketrans({"f": "p", "q": "k", "c": "k", "z": "j", "w": "v", "x": "k", "b": "p", "d": "t", "g": "k"})


def skeleton(s: str) -> str:
    """Consonant skeleton with voicing/aspiration merged ("shakti" and "skti" -> "skt")."""
    s = _VOWELS.sub("", s.lower()).translate(_PHON)
    return re.sub(r"(.)\1+", r"\1", s)


def _pair_tokens(args):
    r_text, s1_text = args
    nat = [t for t in raw_tokens(r_text) if is_native(t)]
    if not nat:
        return None
    lat = [latin_token(t) for t in raw_tokens(s1_text)]
    lat = [t for t in lat if t and not t.isdigit()]
    return list(set(nat)), list(set(lat))


def learn_token_dict(pairs: pl.DataFrame) -> dict:
    rows = []
    for r_col, s_col in (("r_name", "s1_name"), ("r_addr", "s1_addr")):
        sub = pairs.select(r_col, s_col).drop_nulls()
        sub = sub.filter(pl.col(r_col).str.contains(r"[ऀ-෿]"))
        res = pmap(_pair_tokens, sub.rows(), desc=f"dict tokens ({r_col})", chunksize=2000)
        nat, lat = [], []
        for x in res:
            if x:
                nat.append(x[0])
                lat.append(x[1])
        rows.append(pl.DataFrame({"w": nat, "v": lat}))
    df = pl.concat(rows)
    n_w = df.explode("w").group_by("w").len("n_w")
    co = df.explode("w").explode("v").group_by(["w", "v"]).len("c")
    top = (
        co.join(n_w, on="w")
        .sort(["w", "c"], descending=[False, True])
        .group_by("w", maintain_order=True).head(8)
    )
    out = {}
    for w, grp in tqdm(top.group_by("w"), total=top["w"].n_unique(), desc="dict token choice", unit="tok"):
        w = w[0]
        tw = skeleton(anyascii(w))
        best, best_score = None, 0.0
        for v, c, n in grp.select("v", "c", "n_w").iter_rows():
            sim = fuzz.ratio(tw, skeleton(v)) / 100.0
            score = (c / n) * sim * sim
            if score > best_score and sim >= 0.4 and (c >= 2 or sim >= 0.8):
                best, best_score = v, score
        if best is not None and best_score >= 0.15:
            out[w] = best
    return out


def _comp_keys(addr):
    if not addr:
        return []
    keys = {comp_key(c) for c in addr.split(",")}
    # components of at most 4 words: states / districts / cities, not street lines
    return [k for k in keys if k and len(k.split()) <= 4 and len(k) <= 40]


def learn_comp_dict(pairs: pl.DataFrame, min_count=20, min_share=0.5) -> dict:
    sub = pairs.select("country", "r_addr", "s1_addr").drop_nulls()
    rk = pmap(_comp_keys, sub["r_addr"].to_list(), desc="dict components (S2/S3)", chunksize=5000)
    sk = pmap(_comp_keys, sub["s1_addr"].to_list(), desc="dict components (S1)", chunksize=5000)
    df = pl.DataFrame({"country": sub["country"], "rk": rk, "sk": sk})
    n_rc = df.select("country", "rk").explode("rk").group_by(["country", "rk"]).len("n")
    df = df.with_columns(
        pl.col("rk").list.set_difference(pl.col("sk")).alias("r_only"),
        pl.col("sk").list.set_difference(pl.col("rk")).alias("s_only"),
    ).select("country", "r_only", "s_only")
    co = df.explode("r_only").drop_nulls("r_only").explode("s_only").drop_nulls("s_only")
    co = co.group_by(["country", "r_only", "s_only"]).len("c")
    best = (
        co.sort("c", descending=True).group_by(["country", "r_only"], maintain_order=True).first()
        .join(n_rc, left_on=["country", "r_only"], right_on=["country", "rk"])
        .filter((pl.col("c") >= min_count) & (pl.col("c") / pl.col("n") >= min_share))
    )
    raw = {(c, r): s for c, r, s in best.select("country", "r_only", "s_only").iter_rows()}
    # resolve chains (ap -> andhra pradesh -> telangana) so every record, S1 included,
    # ends on the same canonical component
    out = {}
    for (c, k), v in raw.items():
        seen = {k}
        while (c, v) in raw and v not in seen:
            seen.add(v)
            v = raw[(c, v)]
        if v != k:
            out[(c, k)] = v
    return out


def build(records: pl.DataFrame, gt: pl.DataFrame):
    """records: train records with columns entity_id, country, name_raw, addr_raw, bucket."""
    t = time.time()
    s1 = records.filter(pl.col("src") == 1).select(
        pl.col("entity_id").alias("s1_id"), pl.col("name_raw").alias("s1_name"),
        pl.col("addr_raw").alias("s1_addr"), "bucket")
    r = records.filter(pl.col("src") != 1).select(
        pl.col("entity_id").alias("r_id"), pl.col("name_raw").alias("r_name"),
        pl.col("addr_raw").alias("r_addr"), "country")
    pairs = (gt.join(s1, on="s1_id").filter(pl.col("bucket").is_in(list(ENCODER_BUCKETS)))
             .join(r, on="r_id"))
    print(f"[dicts] {pairs.height:,} training pairs from buckets {ENCODER_BUCKETS}")
    token_dict = learn_token_dict(pairs)
    print(f"[dicts] token_dict: {len(token_dict):,} native tokens ({time.time()-t:.0f}s)")
    comp_dict = learn_comp_dict(pairs)
    print(f"[dicts] comp_dict: {len(comp_dict):,} component synonyms ({time.time()-t:.0f}s)")
    path = WORK_DIR / "dicts.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"token_dict": token_dict, "comp_dict": comp_dict}, f)
    return token_dict, comp_dict


def load():
    with open(WORK_DIR / "dicts.pkl", "rb") as f:
        d = pickle.load(f)
    return d["token_dict"], d["comp_dict"]
