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
"""
import argparse
import math
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import ENCODER_BUCKETS, ENCODER_DIR, SEED, WORK_DIR, split_dir

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


def train(args):
    torch.manual_seed(SEED)
    recs, cand = load_pairs(args.split)
    bucket = recs["bucket"].to_numpy()
    cand = cand.filter(pl.Series(np.isin(bucket[cand["r_idx"].to_numpy()], ENCODER_BUCKETS)))
    if cand.height > args.max_pairs:
        cand = cand.sample(args.max_pairs, seed=SEED)
    text, ids, true = record_text(recs), recs["entity_id"].to_numpy(), recs["true_s1"].to_numpy()
    a, b = cand["s1_idx"].to_numpy(), cand["r_idx"].to_numpy()
    y = (true[b] == ids[a]).astype(np.float32)
    print(f"[ce] {len(y):,} training pairs from buckets {ENCODER_BUCKETS}, positive rate {y.mean():.3f}")
    tok = AutoTokenizer.from_pretrained(ENCODER_DIR)
    model = CrossEncoder(ENCODER_DIR).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(y) / args.batch)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * s / steps))))
    order = np.random.default_rng(SEED).permutation(len(y))
    bar = tqdm(range(steps), desc="cross-encoder train", unit="step")
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
    CE_DIR.mkdir(parents=True, exist_ok=True)
    model.enc.save_pretrained(CE_DIR)
    tok.save_pretrained(CE_DIR)
    torch.save(model.head.state_dict(), CE_DIR / "head.pt")
    print(f"[ce] saved to {CE_DIR} ({time.time()-t:.0f}s)")


@torch.no_grad()
def score(args):
    recs, cand = load_pairs(args.split)
    tok = AutoTokenizer.from_pretrained(CE_DIR)
    model = CrossEncoder(CE_DIR)
    model.head.load_state_dict(torch.load(CE_DIR / "head.pt"))
    model = model.cuda().eval().half()
    text = record_text(recs)
    a, b = cand["s1_idx"].to_numpy(), cand["r_idx"].to_numpy()
    lens = np.fromiter((len(x) for x in text), dtype=np.int32, count=len(text))
    order = np.argsort(lens[a] + lens[b], kind="stable")  # length-sorted batches -> little padding
    out = np.empty(len(a), dtype=np.float32)
    t = time.time()
    with tqdm(total=len(a), desc=f"cross-encoder score {args.split}", unit="pair", unit_scale=True) as bar:
        for i in range(0, len(a), args.batch):
            idx = order[i:i + args.batch]
            batch = tok(text[a[idx]].tolist(), text[b[idx]].tolist(), padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to("cuda")
            out[idx] = torch.sigmoid(model(batch).float()).cpu().numpy()
            bar.update(len(idx))
    d = split_dir(args.split)
    feats = pl.read_parquet(d / "features.parquet")
    if "ce" in feats.columns:
        feats = feats.drop("ce")
    ce = pl.DataFrame({"r_idx": cand["r_idx"], "s1_idx": cand["s1_idx"], "ce": out})
    feats = feats.join(ce, on=["r_idx", "s1_idx"], how="left")
    assert feats["ce"].null_count() == 0, "candidate set and features are out of sync"
    feats.write_parquet(d / "features.parquet")
    if "y" in feats.columns:
        yy = feats["y"].to_numpy()
        p = np.clip(feats["ce"].to_numpy(), 1e-6, 1 - 1e-6)
        print(f"[ce] {args.split}: log-loss {-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)):.4f}, "
              f"accuracy {((p >= 0.5) == yy).mean():.4f}")
    print(f"[ce] {args.split}: added column ce to features.parquet ({time.time()-t:.0f}s, "
          f"{len(a) / max(time.time()-t, 1):,.0f} pairs/s)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--split", default="train_dense")
    tr.add_argument("--max-pairs", type=int, default=1_200_000)
    tr.add_argument("--batch", type=int, default=128)
    tr.add_argument("--lr", type=float, default=3e-5)
    sc = sub.add_parser("score")
    sc.add_argument("--split", required=True)
    sc.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()
    train(args) if args.cmd == "train" else score(args)


if __name__ == "__main__":
    main()
