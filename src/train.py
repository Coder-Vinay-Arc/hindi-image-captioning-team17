"""
train.py -- training loop for the captioning models (baseline now, transformer later).

What it does: trains with teacher forcing + cross-entropy (pad ignored) and, for the attention
model, the doubly-stochastic attention penalty. After every epoch it computes the dev loss, prints a
sample caption, and saves checkpoints:
    checkpoints/<name>_last.pth   (every epoch, used for --resume)
    checkpoints/<name>_best.pth   (lowest dev loss -> use this for evaluation)
plus a loss history at results/<name>_history.json.

Run (from the project root):
    # quick sanity check on the 500-row sample (CPU is fine)
    python src/train.py --model baseline --train_split sample --val_split sample --epochs 2

    # real training (use a GPU, e.g. Colab)
    python src/train.py --model baseline --epochs 10 --batch_size 64

Loading a trained model elsewhere (e.g. evaluate.py):
    from train import load_checkpoint
    model, vocab = load_checkpoint("checkpoints/baseline_best.pth", device)
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `models` and `dataset` import from anywhere
from dataset import Vocab, get_dataloader  # noqa: E402


def build_model(name, vocab_size, **kw):
    if name == "baseline":
        from models.baseline import CaptionModel
        return CaptionModel(vocab_size, **kw)
    if name == "transformer":
        from models.transformer import CaptionModel  # added later (src/models/transformer.py)
        return CaptionModel(vocab_size, **kw)
    raise ValueError(f"unknown model: {name}")


def save_checkpoint(path, model, vocab, model_name, optimizer=None, epoch=0, best_val=float("inf")):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_name": model_name,
        "config": model.config,
        "vocab": vocab.itos,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "best_val": best_val,
    }, path)


def load_checkpoint(path, device="cpu"):
    """Rebuild a trained model + its vocab from a checkpoint. Returns (model, vocab)."""
    ckpt = torch.load(path, map_location=device)
    cfg = dict(ckpt["config"])
    cfg.pop("vocab_size")
    model = build_model(ckpt["model_name"], len(ckpt["vocab"]), pretrained=False, **cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), Vocab(ckpt["vocab"])


def run_epoch(model, loader, criterion, device, optimizer=None, alpha_c=1.0, grad_clip=5.0, log_every=50):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    for i, (images, caps, lengths) in enumerate(loader):
        images, caps = images.to(device), caps.to(device)
        with torch.set_grad_enabled(train):
            logits, alphas = model(images, caps)
            targets = caps[:, 1:]  # predict everything after <start>
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if alphas is not None and alpha_c > 0:  # doubly-stochastic attention: each region should get ~1 total attention
                loss = loss + alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += loss.item() * images.size(0)
        n += images.size(0)
        if train and (i + 1) % log_every == 0:
            print(f"    batch {i + 1}/{len(loader)}  loss {loss.item():.3f}", flush=True)
    return total / max(n, 1)


@torch.no_grad()
def show_sample(model, loader, vocab, device):
    model.eval()
    images, caps, _ = next(iter(loader))
    ids, _ = model.caption(images[:2].to(device), vocab.start_id, vocab.end_id, beam_size=1)
    for k in range(len(ids)):
        print(f"    ref : {vocab.decode(caps[k])}\n    pred: {vocab.decode(ids[k])}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Train a captioning model")
    ap.add_argument("--model", default="baseline", choices=["baseline", "transformer"])
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="dev")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--fine_tune", action="store_true", help="unfreeze the last ResNet block")
    ap.add_argument("--dropout", type=float, default=None, help="override the model's default dropout (e.g. 0.4)")
    ap.add_argument("--alpha_c", type=float, default=1.0, help="attention regularisation weight")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--name", default=None, help="checkpoint prefix (default: model name)")
    ap.add_argument("--resume", default=None, help="path to a *_last.pth to continue from")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    sys.stdout.reconfigure(encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.name or args.model
    print(f"device: {device}")

    vocab_path = os.path.join(args.data_root, "vocab.json")
    if os.path.exists(vocab_path):
        vocab = Vocab.load(vocab_path)
    else:  # fall back to building it from the train split
        vocab = Vocab.build(os.path.join(args.data_root, "train", "captions.csv"))
        vocab.save(vocab_path)
    print(f"vocab size: {len(vocab)}")

    train_loader = get_dataloader(args.train_split, vocab, args.data_root, args.batch_size,
                                  num_workers=args.num_workers)
    val_loader = get_dataloader(args.val_split, vocab, args.data_root, args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

    extra = {} if args.dropout is None else {"dropout": args.dropout}
    model = build_model(args.model, len(vocab), fine_tune=args.fine_tune, **extra).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    print(f"trainable parameters: {sum(p.numel() for p in params):,}")

    start_epoch, best_val, history = 0, float("inf"), []
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_val = ckpt["epoch"], ckpt["best_val"]
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    os.makedirs("results", exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer, args.alpha_c)
        va = run_epoch(model, val_loader, criterion, device, None, args.alpha_c)
        print(f"epoch {epoch + 1}/{args.epochs}  train {tr:.3f}  dev {va:.3f}  ({time.time() - t0:.0f}s)")
        show_sample(model, val_loader, vocab, device)

        history.append({"epoch": epoch + 1, "train_loss": tr, "dev_loss": va})
        with open(f"results/{name}_history.json", "w") as f:
            json.dump(history, f, indent=1)

        save_checkpoint(os.path.join(args.ckpt_dir, f"{name}_last.pth"), model, vocab, args.model,
                        optimizer, epoch + 1, best_val)
        if va < best_val:
            best_val = va
            save_checkpoint(os.path.join(args.ckpt_dir, f"{name}_best.pth"), model, vocab, args.model,
                            None, epoch + 1, best_val)
            print(f"    saved best (dev {va:.3f})")


if __name__ == "__main__":
    main()
