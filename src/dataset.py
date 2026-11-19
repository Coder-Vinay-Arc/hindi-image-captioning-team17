"""
dataset.py -- PyTorch Dataset / DataLoader for the preprocessed Hindi Visual Genome crops.

Expects the output of preprocess_hvg.py:
    data/processed/<split>/captions.csv   (file_name, hindi_caption, english_caption)
    data/processed/<split>/images/*.jpg

Provides:
    tokenize(text)            -> list of Hindi tokens (normalised, whitespace split)
    Vocab                     -> word <-> index, with <pad>=0 <start>=1 <end>=2 <unk>=3
    HVGDataset                -> returns (image_tensor, caption_ids_tensor)
    get_dataloader(...)       -> DataLoader that pads captions per batch

Usage in training code:
    from dataset import Vocab, get_dataloader
    vocab = Vocab.build("data/processed/train/captions.csv")   # once, on TRAIN only
    vocab.save("data/processed/vocab.json")
    vocab = Vocab.load("data/processed/vocab.json")            # everyone else
    loader = get_dataloader("train", vocab, batch_size=32)
    for images, captions, lengths in loader: ...
        # images: (B,3,224,224)  captions: (B,T) padded with 0  lengths: (B,) incl. <start>/<end>

Quick self-test (build vocab + print one batch):
    python src/dataset.py --root data/processed --split sample
"""

import argparse
import json
import os
import re
from collections import Counter

import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"

# Punctuation to strip: ASCII punctuation + Devanagari danda/double danda.
_PUNCT = re.compile(r"[\"'.,!?;:()\[\]{}\-_/\\|।॥“”‘’]")


def tokenize(text):
    """Normalise a Hindi caption and split on whitespace."""
    text = str(text).replace("‍", "").replace("‌", "")  # zero-width joiners
    text = _PUNCT.sub(" ", text)
    return text.split()


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, captions_csv, min_freq=2):
        """Build vocab from a captions.csv (use the TRAIN split only)."""
        df = pd.read_csv(captions_csv)
        counts = Counter(tok for c in df["hindi_caption"] for tok in tokenize(c))
        words = sorted(w for w, n in counts.items() if n >= min_freq)
        return cls([PAD, START, END, UNK] + words)

    def __len__(self):
        return len(self.itos)

    @property
    def pad_id(self):
        return self.stoi[PAD]

    @property
    def start_id(self):
        return self.stoi[START]

    @property
    def end_id(self):
        return self.stoi[END]

    @property
    def unk_id(self):
        return self.stoi[UNK]

    def encode(self, text):
        """Caption string -> [<start>, w1, ..., wn, <end>] as ids."""
        ids = [self.stoi.get(t, self.unk_id) for t in tokenize(text)]
        return [self.start_id] + ids + [self.end_id]

    def decode(self, ids, skip_special=True):
        """List/tensor of ids -> Hindi string (stops at <end>)."""
        words = []
        for i in ids:
            i = int(i)
            if i == self.end_id:
                break
            if skip_special and i in (self.pad_id, self.start_id):
                continue
            words.append(self.itos[i])
        return " ".join(words)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


def default_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet
    ])


class HVGDataset(Dataset):
    def __init__(self, root, split, vocab, transform=None, max_len=40):
        self.dir = os.path.join(root, split)
        self.df = pd.read_csv(os.path.join(self.dir, "captions.csv"))
        self.vocab = vocab
        self.transform = transform or default_transform()
        self.max_len = max_len  # max tokens incl. <start>/<end>

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.dir, "images", row["file_name"])).convert("RGB")
        ids = self.vocab.encode(row["hindi_caption"])
        if len(ids) > self.max_len:  # truncate but keep <end>
            ids = ids[: self.max_len - 1] + [self.vocab.end_id]
        return self.transform(img), torch.tensor(ids, dtype=torch.long)

    def reference(self, idx):
        """Raw Hindi reference caption (for BLEU/CIDEr in evaluate.py)."""
        return self.df.iloc[idx]["hindi_caption"]


def make_collate(pad_id):
    def collate(batch):
        images, caps = zip(*batch)
        lengths = torch.tensor([len(c) for c in caps], dtype=torch.long)
        return torch.stack(images), pad_sequence(caps, batch_first=True, padding_value=pad_id), lengths
    return collate


def get_dataloader(split, vocab, root="data/processed", batch_size=32, shuffle=None,
                   num_workers=0, transform=None, max_len=40):
    """DataLoader for 'train' / 'dev' / 'test' / 'sample'. Shuffles only train by default."""
    ds = HVGDataset(root, split, vocab, transform=transform, max_len=max_len)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=make_collate(vocab.pad_id))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles can't print Hindi by default
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed")
    ap.add_argument("--split", default="sample")
    ap.add_argument("--vocab_from", default="train", help="split used to build the vocab")
    ap.add_argument("--min_freq", type=int, default=2)
    args = ap.parse_args()

    vocab_path = os.path.join(args.root, "vocab.json")
    vocab = Vocab.build(os.path.join(args.root, args.vocab_from, "captions.csv"), args.min_freq)
    vocab.save(vocab_path)
    print(f"vocab size: {len(vocab)} (saved to {vocab_path})")

    loader = get_dataloader(args.split, vocab, root=args.root, batch_size=4)
    images, caps, lengths = next(iter(loader))
    print("images:", tuple(images.shape), "captions:", tuple(caps.shape), "lengths:", lengths.tolist())
    print("first caption decoded:", vocab.decode(caps[0]))
