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

    # fine-tune ResNet layer4, starting from the trained frozen baseline, with a 10x smaller LR for the CNN
    python src/train.py --model baseline --epochs 5 --batch_size 64 --fine_tune --freeze_bn \
        --init_from checkpoints/baseline_best.pth --lr 1e-4 --encoder_lr 1e-5 --name baseline_ft

    # only measure the dev loss / dev cross-entropy of an existing checkpoint (no training)
    python src/train.py --model baseline --init_from checkpoints/baseline_best.pth --eval_only

Optional flags added for the baseline-tuning experiments (all OFF by default, so the original
command above still reproduces the reported baseline exactly):
    --encoder_lr   separate learning rate for the unfrozen ResNet block (default: same as --lr)
    --init_from    start a NEW run from a checkpoint's weights (fresh optimiser, epoch counter at 0)
    --freeze_bn    keep the ResNet BatchNorm layers in eval mode while training (see run_epoch)
    --eval_only    compute dev loss + dev cross-entropy for --init_from and exit
The history file now also stores "dev_ce" = dev cross-entropy WITHOUT the attention penalty. Use it to
compare runs that use different --alpha_c values (their "dev_loss" includes a differently-weighted
penalty, so it is not comparable between them).

Loading a trained model elsewhere (e.g. evaluate.py):
    from train import load_checkpoint
    model, vocab = load_checkpoint("checkpoints/baseline_best.pth", device)
"""

# Standard library: argparse (command-line options), json (save the loss history), os (paths), sys (imports
# and exit), time (time per epoch).
import argparse
import json
import os
import sys
import time

# PyTorch is the deep-learning library; torch.nn gives us the loss function and the layers.
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `models` and `dataset` import from anywhere
from dataset import Vocab, get_dataloader  # noqa: E402


# Creates the model chosen with --model. Both models have the same interface, so the rest of the code does not
# care which one it is.
def build_model(name, vocab_size, **kw):
    if name == "baseline":
        from models.baseline import CaptionModel
        return CaptionModel(vocab_size, **kw)
    if name == "transformer":
        from models.transformer import CaptionModel  # added later (src/models/transformer.py)
        return CaptionModel(vocab_size, **kw)
    raise ValueError(f"unknown model: {name}")


# Saves everything needed to load the model later: weights, config, vocab, optimizer state, epoch and best dev
# loss.
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


# Used by evaluate.py and attention_maps.py. The saved config rebuilds the same model; pretrained=False
# because the trained weights are loaded anyway.
def load_checkpoint(path, device="cpu"):
    """Rebuild a trained model + its vocab from a checkpoint. Returns (model, vocab)."""
    ckpt = torch.load(path, map_location=device)
    cfg = dict(ckpt["config"])
    cfg.pop("vocab_size")
    model = build_model(ckpt["model_name"], len(ckpt["vocab"]), pretrained=False, **cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), Vocab(ckpt["vocab"])


def freeze_batchnorm(model):
    """Put every BatchNorm layer of the CNN encoder back into eval mode.

    Why: model.train() switches ALL sub-modules to training mode, including the BatchNorm layers inside
    the "frozen" ResNet-50. Their weights get no gradient (requires_grad=False), but in training mode
    BatchNorm still overwrites its running mean/variance with statistics of our small region crops on
    every forward pass. So without this, the frozen encoder slowly drifts away from the ImageNet features
    it was pretrained with, and at eval time it uses those drifted statistics.
    Keeping BN in eval mode uses the ImageNet statistics everywhere (standard practice when fine-tuning
    a pretrained CNN with small batches). Only called when --freeze_bn is given.
    """
    encoder = getattr(model, "encoder", None)  # both baseline and transformer expose .encoder
    if encoder is None:
        return
    for m in encoder.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def run_epoch(model, loader, criterion, device, optimizer=None, alpha_c=1.0, grad_clip=5.0, log_every=50,
              freeze_bn=False):
    """One pass over `loader`. Trains if an optimizer is given, otherwise only evaluates.

    Returns (mean loss, mean cross-entropy), both averaged per example:
      - loss = cross-entropy + alpha_c * attention penalty (what is optimised / used to pick *_best.pth)
      - ce   = cross-entropy alone, comparable between runs with different --alpha_c values
    """
    # If an optimizer is given we train (and learn from the loss). Without it we only evaluate, for example on
    # the dev set.
    train = optimizer is not None
    # Train mode switches dropout on (random switching-off of values against overfitting). Eval mode switches
    # it off, so evaluation is not random.
    model.train(train)
    if train and freeze_bn:
        freeze_batchnorm(model)  # must come AFTER model.train(), which would switch BN back to training mode
    # Running sums, used to compute the average loss over all examples at the end.
    total, total_ce, n = 0.0, 0.0, 0
    # Loop over batches: images (B,3,224,224), caps (B,T) with <start> first and <end> last.
    for i, (images, caps, lengths) in enumerate(loader):
        images, caps = images.to(device), caps.to(device)  # (B, 3, 224, 224), (B, T)
        # Gradients are only needed for learning. Switching them off during evaluation saves memory and time.
        with torch.set_grad_enabled(train):
            logits, alphas = model(images, caps)  # (B, T-1, vocab), (B, T-1, 49) or None
            targets = caps[:, 1:]  # predict everything after <start>
            # Teacher forcing: the model reads the REAL caption words and must predict the NEXT word at every
            # position, so the targets are the caption shifted by one. Cross-entropy gives a large loss when
            # the model gives a low probability to the correct next word.
            ce = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss = ce
            # Doubly-stochastic attention penalty (baseline only): over the whole caption every image region
            # should be looked at about once, so the total attention of each region should be close to 1.
            if alphas is not None and alpha_c > 0:  # doubly-stochastic attention: each region should get ~1 total attention
                loss = loss + alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        # The learning step. The order matters: clear old gradients, compute new ones, clip them, then update
        # the weights.
        if train:
            # Remove the gradients of the previous batch (PyTorch would add them up otherwise).
            optimizer.zero_grad()
            # Backpropagation: computes how much every weight contributed to the loss (the gradients).
            loss.backward()
            # Gradient clipping: if the gradients are too big they are scaled down, so one bad batch cannot
            # make a huge, unstable update.
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # Adam moves every weight a small step in the direction that lowers the loss.
            optimizer.step()
        # Multiply by the batch size so the final average is per example, even if the last batch is smaller.
        total += loss.item() * images.size(0)
        total_ce += ce.item() * images.size(0)
        n += images.size(0)
        # Print the loss every log_every batches, so we can see the training progress.
        if train and (i + 1) % log_every == 0:
            print(f"    batch {i + 1}/{len(loader)}  loss {loss.item():.3f}", flush=True)
    return total / max(n, 1), total_ce / max(n, 1)


# Prints a couple of predictions after every epoch, so we can see by eye if the model is learning (no
# gradients needed here).
@torch.no_grad()
def show_sample(model, loader, vocab, device):
    model.eval()
    images, caps, _ = next(iter(loader))
    # beam_size=1 means greedy decoding (always take the best word). Good enough for a quick look.
    ids, _ = model.caption(images[:2].to(device), vocab.start_id, vocab.end_id, beam_size=1)
    for k in range(len(ids)):
        print(f"    ref : {vocab.decode(caps[k])}\n    pred: {vocab.decode(ids[k])}", flush=True)


# Reads the options, builds the data and the model, then runs the epoch loop and saves the checkpoints.
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
    # ---- options for the baseline-tuning experiments (issue #13); all off by default ----
    ap.add_argument("--encoder_lr", type=float, default=None,
                    help="learning rate for the unfrozen ResNet block with --fine_tune (default: same as --lr). "
                         "Pretrained CNN weights need much smaller steps than the freshly trained decoder.")
    ap.add_argument("--init_from", default=None,
                    help="start a NEW run from this checkpoint's weights (fresh optimiser, epochs from 0), "
                         "e.g. fine-tune the trained frozen baseline instead of training from scratch")
    ap.add_argument("--freeze_bn", action="store_true",
                    help="keep the encoder's BatchNorm layers in eval mode while training (see freeze_batchnorm)")
    ap.add_argument("--eval_only", action="store_true",
                    help="with --init_from: print dev loss and dev cross-entropy of that checkpoint, then exit")
    args = ap.parse_args()

    # Fixed random seed so a run can be repeated. Use the GPU if there is one.
    torch.manual_seed(args.seed)
    sys.stdout.reconfigure(encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.name or args.model
    print(f"device: {device}")

    # Load the shared vocabulary. Never rebuild it: all team members must use the same one.
    vocab_path = os.path.join(args.data_root, "vocab.json")
    if os.path.exists(vocab_path):
        vocab = Vocab.load(vocab_path)
    else:  # fall back to building it from the train split
        vocab = Vocab.build(os.path.join(args.data_root, "train", "captions.csv"))
        vocab.save(vocab_path)
    print(f"vocab size: {len(vocab)}")

    # Data loaders: the train split is shuffled, the dev split is not.
    train_loader = get_dataloader(args.train_split, vocab, args.data_root, args.batch_size,
                                  num_workers=args.num_workers)
    val_loader = get_dataloader(args.val_split, vocab, args.data_root, args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

    # Build the model. --dropout is passed only when the user gives it, otherwise the model's own default is
    # used.
    extra = {} if args.dropout is None else {"dropout": args.dropout}
    model = build_model(args.model, len(vocab), fine_tune=args.fine_tune, **extra).to(device)
    # The loss function. ignore_index skips <pad> positions, because padding is not a real word.
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    # --init_from: copy the weights of an already trained model into the freshly built one. The new model
    # may differ only in TRAINING options (--fine_tune, --dropout, ...), not in its layers, so the state
    # dicts match. The vocab must be identical, otherwise the embedding/output rows mean different words.
    if args.init_from:
        init = torch.load(args.init_from, map_location=device)
        if init["vocab"] != vocab.itos:
            sys.exit(f"{args.init_from} was trained with a different vocab than {vocab_path}")
        model.load_state_dict(init["state_dict"])
        print(f"initialised weights from {args.init_from} (trained for {init['epoch']} epochs)")

    # --eval_only: just measure the dev loss of the loaded checkpoint (e.g. to get "dev_ce" for the original
    # baseline, whose history file was written before dev_ce existed), then stop.
    if args.eval_only:
        if not args.init_from:
            sys.exit("--eval_only needs --init_from <checkpoint>")
        va, va_ce = run_epoch(model, val_loader, criterion, device, None, args.alpha_c)
        print(f"{args.init_from} on {args.val_split}: dev loss {va:.3f}  dev ce {va_ce:.3f}")
        return

    # Optimiser. Two parameter groups so the pretrained CNN block can get a smaller learning rate:
    #   decoder (+ anything else trainable outside the encoder) -> --lr
    #   unfrozen encoder block (only with --fine_tune)          -> --encoder_lr (defaults to --lr)
    # Without --fine_tune the encoder group is empty and this is identical to the original single-LR Adam.
    enc_params = [p for n_, p in model.named_parameters() if p.requires_grad and n_.startswith("encoder.")]
    dec_params = [p for n_, p in model.named_parameters() if p.requires_grad and not n_.startswith("encoder.")]
    groups = [{"params": dec_params, "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.encoder_lr or args.lr})
    optimizer = torch.optim.Adam(groups)
    params = dec_params + enc_params
    print(f"trainable parameters: {sum(p.numel() for p in params):,} "
          f"(encoder {sum(p.numel() for p in enc_params):,})")

    # Training state. --resume restores the weights, optimizer and epoch of an earlier run.
    start_epoch, best_val, history = 0, float("inf"), []
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_val = ckpt["epoch"], ckpt["best_val"]
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    # Folder for the loss history files.
    os.makedirs("results", exist_ok=True)
    # One epoch = one full pass over the training set. After each epoch we measure the loss on the dev set,
    # which the model has never learned from. If the train loss keeps falling but the dev loss rises, the
    # model is overfitting.
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        # Train for one epoch (an optimizer is given, so the weights are updated).
        tr, _ = run_epoch(model, train_loader, criterion, device, optimizer, args.alpha_c,
                          freeze_bn=args.freeze_bn)
        # Evaluate on the dev set (no optimizer, so no learning).
        va, va_ce = run_epoch(model, val_loader, criterion, device, None, args.alpha_c)
        print(f"epoch {epoch + 1}/{args.epochs}  train {tr:.3f}  dev {va:.3f}  dev ce {va_ce:.3f}  "
              f"({time.time() - t0:.0f}s)")
        show_sample(model, val_loader, vocab, device)

        # dev_ce = dev cross-entropy without the attention penalty (comparable across --alpha_c values)
        history.append({"epoch": epoch + 1, "train_loss": tr, "dev_loss": va, "dev_ce": va_ce})
        with open(f"results/{name}_history.json", "w") as f:
            json.dump(history, f, indent=1)

        # Always save the latest state, so an interrupted run can be resumed.
        save_checkpoint(os.path.join(args.ckpt_dir, f"{name}_last.pth"), model, vocab, args.model,
                        optimizer, epoch + 1, best_val)
        # Keep a separate *_best.pth for the lowest dev loss (this one is used for testing), because the last
        # epoch may already be overfitted.
        if va < best_val:
            best_val = va
            save_checkpoint(os.path.join(args.ckpt_dir, f"{name}_best.pth"), model, vocab, args.model,
                            None, epoch + 1, best_val)
            print(f"    saved best (dev {va:.3f})")


# Run main() only when the file is executed directly.
if __name__ == "__main__":
    main()
