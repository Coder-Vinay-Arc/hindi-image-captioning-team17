"""
train.py -- Training loop for the Hindi captioning models (baseline now, transformer later).

Reads the preprocessed HVG crops through src/dataset.py, trains with teacher forcing and
cross-entropy (+ the "doubly stochastic" attention regulariser from Show, Attend and Tell),
evaluates dev loss every epoch, prints a few sample captions, and saves checkpoints:

    checkpoints/<name>_last.pth   every epoch (use --resume to continue from it)
    checkpoints/<name>_best.pth   lowest dev loss so far (give this one to evaluate.py)
    results/<name>_history.json   train/dev loss per epoch (for the loss-curve plot)

Usage (run from the repo root):
    # quick check on the 500-row sample split (works on CPU)
    python src/train.py --model baseline --train_split sample --val_split sample --epochs 2 --batch_size 8

    # real run (GPU / Colab)
    python src/train.py --model baseline --epochs 10
    # then fine-tune ResNet layer4, starting from the frozen model's best weights
    python src/train.py --model baseline --epochs 5 --fine_tune --init_from checkpoints/baseline_best.pth --name baseline_ft
    python src/train.py --model baseline --epochs 15 --resume checkpoints/baseline_last.pth

Load a trained model elsewhere:
    from models import load_checkpoint
    model, vocab, ckpt = load_checkpoint("checkpoints/baseline_best.pth")
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `dataset` / `models` import from src/
from dataset import HVGDataset, Vocab, make_collate  # noqa: E402
from models import build_model, load_checkpoint, save_checkpoint  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Train a Hindi captioning model")
    ap.add_argument("--model", default="baseline", choices=["baseline", "transformer"])
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="dev")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--encoder_lr", type=float, default=1e-4, help="LR for ResNet layer4 when --fine_tune")
    ap.add_argument("--fine_tune", action="store_true", help="unfreeze the last ResNet block")
    ap.add_argument("--alpha_c", type=float, default=1.0, help="attention regularisation weight (0 = off)")
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--patience", type=int, default=3, help="stop after N epochs without dev improvement")
    ap.add_argument("--subset", type=int, default=None, help="only use the first N training examples")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--name", default=None, help="checkpoint/result prefix (default: --model)")
    ap.add_argument("--resume", default=None, help="path to a *_last.pth checkpoint (continue same run)")
    ap.add_argument("--init_from", default=None,
                    help="start a NEW run from a checkpoint's weights (e.g. fine-tune the frozen best model)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def make_loader(root, split, vocab, batch_size, shuffle, num_workers, subset=None):
    ds = HVGDataset(root, split, vocab)
    if subset:
        ds = Subset(ds, range(min(subset, len(ds))))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=make_collate(vocab.pad_id), pin_memory=torch.cuda.is_available())


def compute_loss(model, images, caps, criterion, pad_id, alpha_c):
    logits, alphas = model(images, caps)
    targets = caps[:, 1:]                                   # predict everything after <start>
    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    if alphas is not None and alpha_c > 0:
        # Doubly stochastic attention (Xu et al. eq. 14): over the whole caption, every image
        # region should receive roughly equal total attention. Padding steps are masked out.
        mask = (targets != pad_id).unsqueeze(2).float()
        loss = loss + alpha_c * ((1.0 - (alphas * mask).sum(dim=1)) ** 2).mean()
    return loss


def run_epoch(model, loader, criterion, device, args, pad_id, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    use_amp = device.type == "cuda"
    total, n = 0.0, 0
    log_every = max(1, len(loader) // 10)
    for i, (images, caps, _) in enumerate(loader):
        images, caps = images.to(device, non_blocking=True), caps.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=use_amp):
            loss = compute_loss(model, images, caps, criterion, pad_id, args.alpha_c)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if (i + 1) % log_every == 0:
                print(f"    batch {i + 1}/{len(loader)}  loss {loss.item():.3f}", flush=True)
        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / max(n, 1)


@torch.no_grad()
def show_samples(model, loader, vocab, device, k=3):
    images, caps, _ = next(iter(loader))
    ids, _ = model.caption(images[:k].to(device), vocab.start_id, vocab.end_id, beam_size=1)
    for j, pred in enumerate(ids):
        print(f"    ref : {vocab.decode(caps[j])}\n    pred: {vocab.decode(pred)}", flush=True)


def main():
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8")                # Windows consoles + Hindi text
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.name or args.model
    print(f"device: {device}")

    # ---- vocab: always the one built from TRAIN (shared by every model so scores are comparable)
    if args.resume:
        model, vocab, ckpt = load_checkpoint(args.resume, device)
    else:
        vocab_path = os.path.join(args.data_root, "vocab.json")
        if os.path.exists(vocab_path):
            vocab = Vocab.load(vocab_path)
        else:
            vocab = Vocab.build(os.path.join(args.data_root, "train", "captions.csv"))
            vocab.save(vocab_path)
            print(f"built vocab from train split -> {vocab_path}")
        model = build_model(args.model, len(vocab), fine_tune=args.fine_tune).to(device)
        ckpt = None
        if args.init_from:
            init = torch.load(args.init_from, map_location=device, weights_only=False)
            if init["vocab"] != vocab.itos:
                sys.exit(f"--init_from checkpoint was trained with a different vocab than {vocab_path}")
            model.load_state_dict(init["state_dict"])
            print(f"initialised weights from {args.init_from}")
    print(f"vocab size: {len(vocab)}")

    train_loader = make_loader(args.data_root, args.train_split, vocab, args.batch_size, True,
                               args.num_workers, args.subset)
    val_loader = make_loader(args.data_root, args.val_split, vocab, args.batch_size, False,
                             args.num_workers)
    print(f"train examples: {len(train_loader.dataset)}  dev examples: {len(val_loader.dataset)}")

    # ---- optimiser: decoder at --lr, unfrozen encoder block (if any) at the smaller --encoder_lr
    encoder = getattr(model, "encoder", None)
    enc_params = [p for p in encoder.parameters() if p.requires_grad] if encoder is not None else []
    dec_params = [p for n_, p in model.named_parameters() if p.requires_grad and not n_.startswith("encoder.")]
    groups = [{"params": dec_params, "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.encoder_lr})
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    print(f"trainable parameters: {sum(p.numel() for g in groups for p in g['params']):,}")

    start_epoch, best_val, history, bad_epochs = 0, float("inf"), [], 0
    if ckpt is not None:
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_val = ckpt.get("best_val", float("inf"))
        history = ckpt.get("history", [])
        print(f"resumed from {args.resume} (epoch {start_epoch}, best dev {best_val:.3f})")

    os.makedirs("results", exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, args, vocab.pad_id, optimizer, scaler)
        va = run_epoch(model, val_loader, criterion, device, args, vocab.pad_id)
        scheduler.step(va)
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch + 1}/{args.epochs}  train {tr:.3f}  dev {va:.3f}  lr {lr:.1e}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        show_samples(model, val_loader, vocab, device)

        history.append({"epoch": epoch + 1, "train_loss": tr, "dev_loss": va, "lr": lr})
        with open(os.path.join("results", f"{name}_history.json"), "w") as f:
            json.dump(history, f, indent=1)

        improved = va < best_val
        if improved:
            best_val, bad_epochs = va, 0
            save_checkpoint(os.path.join(args.ckpt_dir, f"{name}_best.pth"), model, vocab, args.model,
                            epoch + 1, best_val=best_val)
            print(f"    new best dev loss {va:.3f} -> saved {name}_best.pth")
        else:
            bad_epochs += 1
        save_checkpoint(os.path.join(args.ckpt_dir, f"{name}_last.pth"), model, vocab, args.model,
                        epoch + 1, optimizer, best_val=best_val, history=history,
                        scheduler=scheduler.state_dict())
        if bad_epochs >= args.patience:
            print(f"no dev improvement for {args.patience} epochs -- stopping early")
            break

    print(f"done. best dev loss {best_val:.3f}  ->  {os.path.join(args.ckpt_dir, name + '_best.pth')}")


if __name__ == "__main__":
    main()
