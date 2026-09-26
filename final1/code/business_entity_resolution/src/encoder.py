"""Step 2 - fine-tune a multilingual bi-encoder and embed every record.

Model  : intfloat/multilingual-e5-small (MIT licence, 118M parameters)
Input  : "query: <normalised name> | <normalised address>"
Loss   : symmetric in-batch contrastive loss (InfoNCE / MultipleNegativesRanking),
         temperature 0.05. Every batch holds one country only (harder in-batch
         negatives); pairs that share an S1 cluster are masked out as negatives.
Data   : true (S1, S2/S3) pairs of the ENCODER_BUCKETS clusters only.

    python src/encoder.py train [--max-pairs 1500000 --epochs 1 --batch 512]
    python src/encoder.py embed --split train
    python src/encoder.py embed --split test
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

from config import ENCODER_BASE_MODEL, ENCODER_BUCKETS, ENCODER_DIR, ENCODER_MAX_LEN, SEED, split_dir
from io_utils import read_ground_truth


def record_text(df: pl.DataFrame) -> list:
    return df.select(
        pl.concat_str([pl.lit("query: "), pl.col("name"), pl.lit(" | "), pl.col("addr")])
    ).to_series().to_list()


def mean_pool(out, mask):
    h = out.last_hidden_state
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)


class Encoder:
    def __init__(self, path, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(device)
        self.device = device

    def encode_batch(self, texts):
        b = self.tok(texts, padding=True, truncation=True, max_length=ENCODER_MAX_LEN, return_tensors="pt").to(self.device)
        return F.normalize(mean_pool(self.model(**b), b["attention_mask"]), dim=-1)


# ---------------------------------------------------------------- training
def build_training_pairs(max_pairs):
    recs = pl.read_parquet(split_dir("train") / "records.parquet")
    text = recs.select("entity_id", "country", "bucket", pl.Series("text", record_text(recs)))
    s1 = text.filter(pl.col("entity_id").str.starts_with("S1-")).rename(
        {"entity_id": "s1_id", "text": "a"})
    r = text.select(pl.col("entity_id").alias("r_id"), pl.col("text").alias("b"))
    pairs = (read_ground_truth().join(s1, on="s1_id").filter(pl.col("bucket").is_in(list(ENCODER_BUCKETS)))
             .join(r, on="r_id"))
    if pairs.height > max_pairs:
        pairs = pairs.sample(max_pairs, seed=SEED)
    return pairs.select("s1_id", "country", "a", "b")


def make_batches(pairs: pl.DataFrame, batch, rng):
    """Single-country batches in random order."""
    pairs = pairs.sample(fraction=1.0, shuffle=True, seed=int(rng.integers(1 << 30)))
    batches = []
    for _, grp in pairs.group_by("country"):
        n = grp.height // batch
        for i in range(n):
            batches.append(grp.slice(i * batch, batch))
    rng.shuffle(batches)
    return batches


def train(args):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    pairs = build_training_pairs(args.max_pairs)
    print(f"[encoder] {pairs.height:,} training pairs")
    enc = Encoder(ENCODER_BASE_MODEL)
    enc.model.train()
    opt = torch.optim.AdamW(enc.model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = pairs.height // args.batch
    total = steps_per_epoch * args.epochs
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * s / total))))
    t = time.time()
    bar = tqdm(total=total, desc="encoder train", unit="step")
    for ep in range(args.epochs):
        for b in make_batches(pairs, args.batch, rng):
            ids = b["s1_id"].to_list()
            codes = torch.from_numpy(np.unique(np.array(ids), return_inverse=True)[1]).cuda()
            same = (codes[:, None] == codes[None, :]) if len(set(ids)) < len(ids) else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qa = enc.encode_batch(b["a"].to_list())
                qb = enc.encode_batch(b["b"].to_list())
            logits = (qa.float() @ qb.float().T) / args.temp
            if same is not None:  # other positives of the same S1 are not negatives
                eye = torch.eye(len(ids), dtype=torch.bool, device="cuda")
                logits = logits.masked_fill(same & ~eye, -1e4)
            labels = torch.arange(len(ids), device="cuda")
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.model.parameters(), 1.0)
            opt.step()
            sched.step()
            bar.update(1)
            if bar.n % 20 == 0:
                bar.set_postfix(epoch=ep, loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.1e}")
    bar.close()
    print(f"[encoder] trained {total} steps ({time.time()-t:.0f}s)")
    ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    enc.model.save_pretrained(ENCODER_DIR)
    enc.tok.save_pretrained(ENCODER_DIR)
    print(f"[encoder] saved to {ENCODER_DIR}")


# ---------------------------------------------------------------- embedding
@torch.no_grad()
def embed(args):
    path = args.model or ENCODER_DIR
    enc = Encoder(str(path))
    enc.model.eval().half()
    recs = pl.read_parquet(split_dir(args.split) / "records.parquet", columns=["name", "addr"])
    texts = record_text(recs)
    order = np.argsort([len(x) for x in texts])  # length-sorted batches -> little padding
    out = np.lib.format.open_memmap(split_dir(args.split) / "emb.npy", mode="w+",
                                    dtype=np.float16, shape=(len(texts), enc.model.config.hidden_size))
    t = time.time()
    with tqdm(total=len(texts), desc=f"embed {args.split}", unit="rec", unit_scale=True) as bar:
        for i in range(0, len(texts), args.batch):
            idx = order[i:i + args.batch]
            out[idx] = enc.encode_batch([texts[j] for j in idx]).cpu().numpy().astype(np.float16)
            bar.update(len(idx))
    out.flush()
    print(f"[embed] {args.split}: wrote {out.shape} ({time.time()-t:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--max-pairs", type=int, default=1_500_000)
    tr.add_argument("--epochs", type=int, default=1)
    tr.add_argument("--batch", type=int, default=512)
    tr.add_argument("--lr", type=float, default=5e-5)
    tr.add_argument("--temp", type=float, default=0.05)
    em = sub.add_parser("embed")
    em.add_argument("--split", required=True)
    em.add_argument("--batch", type=int, default=2048)
    em.add_argument("--model", default=None, help="default: the fine-tuned encoder")
    args = ap.parse_args()
    train(args) if args.cmd == "train" else embed(args)


if __name__ == "__main__":
    main()
