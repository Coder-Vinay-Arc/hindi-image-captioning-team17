"""
models -- model registry + checkpoint save/load shared by train.py, evaluate.py and others.

    from models import build_model, save_checkpoint, load_checkpoint

    model = build_model("baseline", vocab_size=len(vocab), fine_tune=False)
    save_checkpoint("checkpoints/baseline_best.pth", model, vocab, "baseline", epoch=5)
    model, vocab, ckpt = load_checkpoint("checkpoints/baseline_best.pth", device="cpu")

To add a new model (e.g. the transformer): implement a class with the same interface as
models.baseline.CaptionModel -- a `config` dict attribute, forward(images, captions) ->
(logits, alphas or None), and caption(images, start_id, end_id, max_len, beam_size) ->
(ids, alphas) -- then register it in build_model() below.
"""

import os

import torch


def build_model(name, vocab_size, pretrained=True, **kwargs):
    """Create a captioning model by name. kwargs go to the model constructor."""
    if name == "baseline":
        from models.baseline import CaptionModel
        return CaptionModel(vocab_size, pretrained=pretrained, **kwargs)
    if name == "transformer":
        from models.transformer import CaptionModel as TransformerCaptionModel  # Vishal's model
        return TransformerCaptionModel(vocab_size, pretrained=pretrained, **kwargs)
    raise ValueError(f"unknown model '{name}'")


def save_checkpoint(path, model, vocab, model_name, epoch, optimizer=None, **extra):
    """Save everything needed to rebuild the model without the training code's arguments."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_name": model_name,
        "config": model.config,
        "state_dict": model.state_dict(),
        "vocab": vocab.itos,
        "epoch": epoch,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        **extra,
    }, path)


def load_checkpoint(path, device="cpu"):
    """Returns (model in eval mode, Vocab, raw checkpoint dict)."""
    from dataset import Vocab

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt["config"])
    vocab_size = cfg.pop("vocab_size")
    # pretrained=False: the weights come from the checkpoint, no need to download ImageNet ones
    model = build_model(ckpt["model_name"], vocab_size, pretrained=False, **cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), Vocab(ckpt["vocab"]), ckpt
