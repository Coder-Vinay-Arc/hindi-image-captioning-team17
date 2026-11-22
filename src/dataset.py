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

# Standard library: argparse (self-test options), json (save / load the vocab), os (paths), re (regex to
# remove punctuation), Counter (count words).
import argparse
import json
import os
import re
from collections import Counter

# pandas reads captions.csv, torch is the deep-learning library, PIL opens the images, torchvision resizes and
# normalises them.
import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Special tokens. They always get the first four ids: <pad>=0, <start>=1, <end>=2, <unk>=3.
PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"

# Punctuation to strip: ASCII punctuation + Devanagari danda/double danda.
_PUNCT = re.compile(r"[\"'.,!?;:()\[\]{}\-_/\\|।॥“”‘’]")


# A computer cannot read sentences, so first we cut a Hindi caption into words: remove invisible characters,
# turn punctuation into spaces, split on spaces.
def tokenize(text):
    """Normalise a Hindi caption and split on whitespace."""
    # Zero-width joiner / non-joiner are invisible characters. They would make the same word look different to
    # the computer, so we remove them.
    text = str(text).replace("‍", "").replace("‌", "")  # zero-width joiners
    # Replace punctuation by spaces, so 'सड़क,' and 'सड़क' become the same word.
    text = _PUNCT.sub(" ", text)
    # Split on spaces -> a list of words.
    return text.split()


# The model can only work with numbers, not words. Vocab is the dictionary that converts word <-> number (id).
class Vocab:
    # itos = 'index to string' (list of words). stoi = 'string to index' (dictionary), the reverse lookup.
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    # Builds the vocabulary from the TRAIN captions only (so nothing from dev/test leaks in).
    @classmethod
    def build(cls, captions_csv, min_freq=2):
        """Build vocab from a captions.csv (use the TRAIN split only)."""
        df = pd.read_csv(captions_csv)
        # Count how often every word appears in the training captions.
        counts = Counter(tok for c in df["hindi_caption"] for tok in tokenize(c))
        # Keep only words that appear at least min_freq times. Very rare words are not worth learning, they
        # become <unk>. Sorting makes the order the same in every run.
        words = sorted(w for w, n in counts.items() if n >= min_freq)
        # Special tokens go first, so their ids are always 0, 1, 2, 3.
        return cls([PAD, START, END, UNK] + words)

    # len(vocab) = number of words the model knows.
    def __len__(self):
        return len(self.itos)

    # Small helpers that give the id of each special token.
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

    # Caption text -> list of ids. Example: 'एक कुत्ता' -> [<start>, id(एक), id(कुत्ता), <end>].
    def encode(self, text):
        """Caption string -> [<start>, w1, ..., wn, <end>] as ids."""
        # stoi.get(word, unk_id): a word that is not in the vocabulary becomes <unk>.
        ids = [self.stoi.get(t, self.unk_id) for t in tokenize(text)]
        # <start> and <end> teach the model where a caption begins and where it must stop.
        return [self.start_id] + ids + [self.end_id]

    # Reverse of encode: ids -> Hindi sentence. Stops at <end> and ignores <pad> and <start>.
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

    # Save the word list as JSON, so every team member uses exactly the same vocabulary.
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


# Image preprocessing: resize to 224x224, make a tensor, and normalise with the ImageNet mean / std (the same
# numbers ResNet-50 was trained with, so its features stay meaningful).
def default_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet
    ])


# A Dataset gives one training example at a time. Here: one (image, caption ids) pair for every row of
# captions.csv.
class HVGDataset(Dataset):
    # root + split choose the folder, e.g. data/processed/train. max_len cuts very long captions.
    def __init__(self, root, split, vocab, transform=None, max_len=40):
        self.dir = os.path.join(root, split)
        # Table with the file name and the Hindi caption of every example.
        self.df = pd.read_csv(os.path.join(self.dir, "captions.csv"))
        self.vocab = vocab
        self.transform = transform or default_transform()
        self.max_len = max_len  # max tokens incl. <start>/<end>

    # Number of examples = number of rows in captions.csv.
    def __len__(self):
        return len(self.df)

    # Called by the DataLoader for example number idx.
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Open the crop and force 3 colour channels (RGB).
        img = Image.open(os.path.join(self.dir, "images", row["file_name"])).convert("RGB")
        # Turn the Hindi caption into ids.
        ids = self.vocab.encode(row["hindi_caption"])
        if len(ids) > self.max_len:  # truncate but keep <end>
            ids = ids[: self.max_len - 1] + [self.vocab.end_id]
        # Return the image as a tensor (3,224,224) and the caption ids as a tensor (T,).
        return self.transform(img), torch.tensor(ids, dtype=torch.long)

    def reference(self, idx):
        """Raw Hindi reference caption (for BLEU/CIDEr in evaluate.py)."""
        return self.df.iloc[idx]["hindi_caption"]


# The DataLoader uses this to join several examples into one batch. Captions have different lengths, so
# shorter ones are padded with <pad> to make a rectangle.
class Collate:
    """Pads captions in a batch. A class (not a closure) so DataLoader workers can pickle it on Windows."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    # batch = list of (image, caption) pairs. Output: images (B,3,224,224), padded captions (B,T) and real
    # lengths (B,).
    def __call__(self, batch):
        # Split the list of pairs into a list of images and a list of captions.
        images, caps = zip(*batch)
        # Real length of every caption (before padding).
        lengths = torch.tensor([len(c) for c in caps], dtype=torch.long)
        # Stack the images into one tensor, and pad the captions with <pad> up to the longest caption of the
        # batch.
        return torch.stack(images), pad_sequence(caps, batch_first=True, padding_value=self.pad_id), lengths


# Small helper that creates the Collate object.
def make_collate(pad_id):
    return Collate(pad_id)


# Main function the other files use to get batches of data.
def get_dataloader(split, vocab, root="data/processed", batch_size=32, shuffle=None,
                   num_workers=0, transform=None, max_len=40):
    """DataLoader for 'train' / 'dev' / 'test' / 'sample'. Shuffles only train by default."""
    ds = HVGDataset(root, split, vocab, transform=transform, max_len=max_len)
    # Shuffle only the training data (so the model does not learn the order). Dev / test keep a fixed order so
    # predictions line up with references.
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=make_collate(vocab.pad_id))


# Self-test: build the vocab, save it, and print the shape of one batch.
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles can't print Hindi by default
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed")
    ap.add_argument("--split", default="sample")
    ap.add_argument("--vocab_from", default="train", help="split used to build the vocab")
    ap.add_argument("--min_freq", type=int, default=2)
    args = ap.parse_args()

    # Build the vocab from the chosen split (normally train) and save it.
    vocab_path = os.path.join(args.root, "vocab.json")
    vocab = Vocab.build(os.path.join(args.root, args.vocab_from, "captions.csv"), args.min_freq)
    vocab.save(vocab_path)
    print(f"vocab size: {len(vocab)} (saved to {vocab_path})")

    # Take one small batch and print its shapes, to check that data loading works.
    loader = get_dataloader(args.split, vocab, root=args.root, batch_size=4)
    images, caps, lengths = next(iter(loader))
    print("images:", tuple(images.shape), "captions:", tuple(caps.shape), "lengths:", lengths.tolist())
    print("first caption decoded:", vocab.decode(caps[0]))
