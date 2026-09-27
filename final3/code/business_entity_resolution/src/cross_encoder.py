"""Cross-encoder pair scorer: a transformer reads both records together.

The matcher's other features are hand-made similarity numbers.  A cross-encoder attends
across the two records token by token and learns the noise patterns of the data directly
(perturbed house numbers, neighbouring cities, DBA / domain names, transliteration,
reordered legal forms).  Its probability becomes one more matcher feature (`ce`).

Model   : the fine-tuned bi-encoder (multilingual-e5-small, MIT, 118M) + linear head on the
          mean-pooled pair encoding; input "name | address" of both records, RAW text
          (accents, case and native script kept), max 128 tokens.
Training: candidate pairs (after the prefilter) of the ENCODER_BUCKETS clusters of the
          given train-like split only - clusters the matcher never trains or validates on,
          so `ce` is an out-of-fold feature for folds A/B.  Binary cross-entropy, 1 epoch.
Scoring : every candidate pair of a split; the score is added as column `ce` to that
          split's features.parquet (the candidate set itself is unchanged).

    python src/cross_encoder.py train --split train_dense [--max-pairs 1200000]
    python src/cross_encoder.py score --split train_dense
    python src/cross_encoder.py score --split test

Cross-fitted variant (v5): two more models, each trained on one matcher fold (starting
from the v4 cross-encoder), each scoring only the OTHER fold -> out-of-fold column ce_cf;
on test both are averaged:
    python src/cross_encoder.py train --buckets A --init ce --out cfA --max-pairs 1500000
    python src/cross_encoder.py train --buckets B --init ce --out cfB --max-pairs 1500000
    python src/cross_encoder.py score --split train_dense --models cfA,cfB --col ce_cf
"""
import argparse
import json
import math
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import ENCODER_BUCKETS, ENCODER_DIR, FOLD_A_BUCKETS, FOLD_B_BUCKETS, SEED, WORK_DIR, split_dir

CE_DIR = WORK_DIR / "cross_encoder"
MAX_LEN = 128


def record_text(recs: pl.DataFrame) -> np.ndarray:
    return recs.select(pl.concat_str([pl.col("name_raw").fill_null(""), pl.lit(" | "),
                                      pl.col("addr_raw").fill_null("")])).to_series().to_numpy()


class CrossEncoder(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.enc = AutoModel.from_pretrained(path)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, batch):
        h = self.enc(**batch).last_hidden_state
        m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1e-6)).squeeze(-1)


def load_pairs(split):
    d = split_dir(split)
    recs = pl.read_parquet(d / "records.parquet", columns=["entity_id", "name_raw", "addr_raw", "true_s1", "bucket"])
    cand = pl.read_parquet(d / "candidates.parquet", columns=["r_idx", "s1_idx"])
    return recs, cand


BUCKET_SETS = {"enc": ENCODER_BUCKETS, "A": FOLD_A_BUCKETS, "B": FOLD_B_BUCKETS}


def model_path(name: str):
    """"ce" -> work/cross_encoder (v4 model); any other name -> work/cross_encoder_<name>."""
    return CE_DIR if name == "ce" else WORK_DIR / f"cross_encoder_{name}"


def load_model(path):
    model = CrossEncoder(path)
    head = path / "head.pt"
    if head.exists():
        model.head.load_state_dict(torch.load(head))
    return model


def train(args):
    torch.manual_seed(SEED)
    recs, cand = load_pairs(args.split)
    buckets = BUCKET_SETS[args.buckets]
    bucket = recs["bucket"].to_numpy()
    cand = cand.filter(pl.Series(np.isin(bucket[cand["r_idx"].to_numpy()], buckets)))
    if cand.height > args.max_pairs:
        cand = cand.sample(args.max_pairs, seed=SEED)
    text, ids, true = record_text(recs), recs["entity_id"].to_numpy(), recs["true_s1"].to_numpy()
    a, b = cand["s1_idx"].to_numpy(), cand["r_idx"].to_numpy()
    y = (true[b] == ids[a]).astype(np.float32)
    init = ENCODER_DIR if args.init == "encoder" else model_path(args.init)
    out_dir = model_path(args.out)
    print(f"[ce] {len(y):,} training pairs from buckets {buckets}, positive rate {y.mean():.3f}; "
          f"init {init} -> {out_dir}")
    tok = AutoTokenizer.from_pretrained(init)
    model = load_model(init).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(y) / args.batch)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * s / steps))))
    order = np.random.default_rng(SEED).permutation(len(y))
    bar = tqdm(range(steps), desc=f"cross-encoder train ({args.out})", unit="step")
    t = time.time()
    for s in bar:
        idx = order[s * args.batch:(s + 1) * args.batch]
        batch = tok(text[a[idx]].tolist(), text[b[idx]].tolist(), padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(batch)
        loss = F.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(y[idx]).cuda())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % 20 == 0:
            bar.set_postfix(loss=f"{loss.item():.4f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.enc.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    torch.save(model.head.state_dict(), out_dir / "head.pt")
    # buckets whose pairs this model has seen (directly or through its initialisation)
    seen = set(buckets)
    if (init / "buckets.json").exists():
        seen |= set(json.loads((init / "buckets.json").read_text()))
    elif init == CE_DIR:
        seen |= set(ENCODER_BUCKETS)
    seen = sorted(int(x) for x in seen)
    (out_dir / "buckets.json").write_text(json.dumps(seen))
    print(f"[ce] saved to {out_dir}, trained on buckets {seen} ({time.time()-t:.0f}s)")


@torch.no_grad()
def predict_pairs(model, tok, text, a, b, batch_size, desc):
    lens = np.fromiter((len(x) for x in text), dtype=np.int32, count=len(text))
    order = np.argsort(lens[a] + lens[b], kind="stable")  # length-sorted batches -> little padding
    out = np.empty(len(a), dtype=np.float32)
    with tqdm(total=len(a), desc=desc, unit="pair", unit_scale=True) as bar:
        for i in range(0, len(a), batch_size):
            idx = order[i:i + batch_size]
            batch = tok(text[a[idx]].tolist(), text[b[idx]].tolist(), padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to("cuda")
            out[idx] = torch.sigmoid(model(batch).float()).cpu().numpy()
            bar.update(len(idx))
    return out


def score(args):
    """Score every candidate pair with one or more cross-encoders and add column --col.

    With several models each pair gets the mean of the models that never trained on the
    pair's fold bucket (cross-fitting); on the test split every model is eligible.
    """
    recs, cand = load_pairs(args.split)
    reused = None
    if args.reuse_from:  # keep scores already computed for the same pairs in another split
        prev = pl.read_parquet(split_dir(args.reuse_from) / "features.parquet", columns=["r_idx", "s1_idx", args.col])
        reused = cand.join(prev, on=["r_idx", "s1_idx"], how="inner")
        cand = cand.join(prev.select("r_idx", "s1_idx"), on=["r_idx", "s1_idx"], how="anti")
        print(f"[ce] reusing {reused.height:,} {args.col} scores from {args.reuse_from}; scoring {cand.height:,} new pairs")
    text = record_text(recs)
    a, b = cand["s1_idx"].to_numpy(), cand["r_idx"].to_numpy()
    row_bucket = recs["bucket"].to_numpy()[b]
    total = np.zeros(len(a), dtype=np.float64)
    count = np.zeros(len(a), dtype=np.int32)
    t = time.time()
    for name in args.models.split(","):
        path = model_path(name)
        seen = json.loads((path / "buckets.json").read_text()) if (path / "buckets.json").exists() \
            else list(ENCODER_BUCKETS)
        # out-of-fold for folds A/B; encoder-bucket rows (never used to train or evaluate the
        # matcher) get every model, like the test split
        eligible = np.nonzero(~np.isin(row_bucket, seen) | np.isin(row_bucket, ENCODER_BUCKETS)
                              | (row_bucket < 0))[0]
        tok = AutoTokenizer.from_pretrained(path)
        model = load_model(path).cuda().eval().half()
        total[eligible] += predict_pairs(model, tok, text, a[eligible], b[eligible], args.batch,
                                         f"cross-encoder {name} score {args.split}")
        count[eligible] += 1
        del model
        torch.cuda.empty_cache()
    missing = int((count == 0).sum())
    if missing:
        print(f"[ce] warning: {missing:,} pairs have no eligible model (left missing)")
    out = np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)
    d = split_dir(args.split)
    feats = pl.read_parquet(d / "features.parquet")
    if args.col in feats.columns:
        feats = feats.drop(args.col)
    ce = pl.DataFrame({"r_idx": cand["r_idx"], "s1_idx": cand["s1_idx"], args.col: out})
    if reused is not None:
        ce = pl.concat([ce, reused.select("r_idx", "s1_idx", pl.col(args.col).cast(pl.Float32))])
    n_before = feats.height
    feats = feats.join(ce, on=["r_idx", "s1_idx"], how="left")
    assert feats.height == n_before and feats[args.col].null_count() <= missing, \
        "candidate set and features are out of sync"
    feats.write_parquet(d / "features.parquet")
    if "y" in feats.columns:
        ev = feats.filter(pl.col("bucket").is_in(list(FOLD_A_BUCKETS + FOLD_B_BUCKETS)))
        yy = ev["y"].to_numpy()
        p = np.clip(ev[args.col].to_numpy(), 1e-6, 1 - 1e-6)
        print(f"[ce] {args.split} {args.col} (folds A+B, out-of-fold): log-loss "
              f"{-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)):.4f}, accuracy {((p >= 0.5) == yy).mean():.4f}")
    print(f"[ce] {args.split}: added column {args.col} to features.parquet ({time.time()-t:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--split", default="train_dense")
    tr.add_argument("--buckets", default="enc", choices=sorted(BUCKET_SETS), help="training clusters")
    tr.add_argument("--init", default="encoder", help="'encoder' (bi-encoder) or a cross-encoder name, e.g. ce")
    tr.add_argument("--out", default="ce", help="model name: ce -> work/cross_encoder, X -> cross_encoder_X")
    tr.add_argument("--max-pairs", type=int, default=1_200_000)
    tr.add_argument("--batch", type=int, default=128)
    tr.add_argument("--lr", type=float, default=3e-5)
    sc = sub.add_parser("score")
    sc.add_argument("--split", required=True)
    sc.add_argument("--models", default="ce", help="comma-separated model names (cross-fitted mean)")
    sc.add_argument("--col", default="ce", help="feature column to write")
    sc.add_argument("--reuse-from", default=None, help="split whose existing scores for the same pairs are reused")
    sc.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()
    train(args) if args.cmd == "train" else score(args)


if __name__ == "__main__":
    main()
