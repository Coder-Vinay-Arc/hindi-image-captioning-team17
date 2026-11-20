# Image Captioning in Hindi with Attention

**Deep Learning Project — Team 17 (Section B)**

Generating Hindi captions for images directly using Hindi Visual Genome and attention-based deep learning models — compared against an English captioning + translation pipeline.

## Team Members

| Name | Email |
|---|---|
| Vinay Kumar Gupta | su-24153@sitare.org |
| Rahul Kumar Verma | su-24106@sitare.org |
| Vishal Bhardwaj | su-24155@sitare.org |
| Vivek Kumar | su-24159@sitare.org |

## Problem Statement

Most image captioning research and models are built for English, while Hindi remains a low-resource language for this task. In this project, we generate Hindi captions for images directly, and compare whether direct Hindi captioning performs better than an English captioning + machine translation pipeline.

## Current Status (updated 20 Nov 2026)

- [x] Repo created, all members added as collaborators
- [x] Dataset strategy decided: Hindi Visual Genome (region-crop approach)
- [x] Dataset downloaded and preprocessed (train / dev / test) — see [Dataset](#dataset)
- [x] `src/dataset.py` — PyTorch Dataset, Hindi tokenizer, vocab, DataLoader (merged to `main`)
- [x] `src/models/baseline.py` + `src/train.py` — CNN + LSTM + Attention, training loop, checkpoints (merged)
- [x] `src/evaluate.py` — BLEU-1..4 and CIDEr, usable for all three approaches (merged)
- [x] Baseline trained on the full training set and scored on the test set — see [Results](#results)
- [x] Transformer model (`src/models/transformer.py`) trained and scored on the test set — see [Results](#results)
- [x] Transformer attention-map figures (`src/attention_maps.py`, `results/transformer_attention_*.png`)
- [ ] English captioning + translation pipeline (`src/translation_pipeline.py`) — not started
- [ ] Human evaluation, baseline attention-map figures — not started
- [ ] Report — not started

## Task Division

| Member | Branch | Task | Status |
|---|---|---|---|
| Vinay Kumar Gupta | `vinay-data-pipeline`, `vinay-baseline` | Data pipeline, Dataset class, baseline training, `evaluate.py` | Done (merged) |
| Rahul Kumar Verma | `rahul-baseline-model` | Baseline model: CNN + LSTM + Attention | Baseline already exists in `main` — Rahul can extend/tune it (e.g. `--fine_tune`, more epochs) |
| Vishal Bhardwaj | `vishal-transformer-model` | Transformer decoder (`src/models/transformer.py`) | Done — see [Transformer training setup](#transformer-training-setup) |
| Vivek Kumar | `vivek-translation-eval` | English captioning + translation pipeline | To do — see [Scoring another approach](#scoring-another-approach-eg-translation-pipeline) |

> Detailed project rules and task division are in [`CLAUDE.md`](./CLAUDE.md).

## Dataset

- **Source:** [Hindi Visual Genome 1.1](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3267) (LINDAT/CLARIN). The release ships the images, so no separate Visual Genome download is needed.
- **Processed data (Google Drive folder, everything is here):** https://drive.google.com/drive/folders/1jF_T4IGqQlnZO42QYHk8bQuIgdeUTLwi?usp=sharing
- **Files in that folder:** `hvg-processed.zip.zip` (contains `train`, `test`, `sample` and `vocab.json`) and `dev.zip` (the `dev` split). Download both and extract **both** into `data/processed/`, so you end up with `data/processed/{train,dev,test,sample}` and `data/processed/vocab.json`. Trained checkpoints and result files are saved in the same folder (`checkpoints/`, `results/`).
- The data is **not** in git (size). `data/` is in `.gitignore`.

### Region-crop decision
In Hindi Visual Genome a caption describes a **region (bounding box)** of an image, not the whole image. We treat every region crop as an independent (image, caption) pair, which is standard for Visual Genome-style data and gives more training pairs. `src/preprocess_hvg.py` crops each box, resizes it to 224x224 and writes `captions.csv`. Regions smaller than 8 px or without a Hindi caption are dropped.

| Split | Rows in raw file | Kept after preprocessing |
|---|---|---|
| train | 28,930 | 28,705 |
| dev | 998 | 993 |
| test | 1,595 | 1,586 |
| sample (first 500 train rows, for quick tests) | 500 | 500 |

### Processed data layout
```
data/processed/
├── vocab.json                  # shared vocabulary (3,615 tokens, built from train, min_freq=2)
├── train/ dev/ test/ sample/
│   ├── captions.csv            # file_name, hindi_caption, english_caption
│   └── images/*.jpg            # 224x224 region crops
```
Special tokens: `<pad>=0`, `<start>=1`, `<end>=2`, `<unk>=3`. **Everyone must use the same `vocab.json`** so that results are comparable. Do not rebuild it.

### Rebuilding the data from scratch (only if needed)
```bash
# raw files: download hindi-visual-genome-11.zip from LINDAT and extract it into data/raw/
python src/preprocess_hvg.py --annotations data/raw/hindi-visual-genome-train.txt --images_dir data/raw/hindi-visual-genome-train.images --out_dir data/processed/train
python src/preprocess_hvg.py --annotations data/raw/hindi-visual-genome-dev.txt   --images_dir data/raw/hindi-visual-genome-dev.images   --out_dir data/processed/dev
python src/preprocess_hvg.py --annotations data/raw/hindi-visual-genome-test.txt  --images_dir data/raw/hindi-visual-genome-test.images  --out_dir data/processed/test
python src/dataset.py --root data/processed --split dev      # builds data/processed/vocab.json
```

## Approach

1. **Baseline (done):** ResNet-50 encoder (ImageNet-pretrained, frozen) → 7x7 grid of 2048-d features → LSTM decoder with additive (Bahdanau) soft attention ("Show, Attend and Tell").
2. **Advanced (done):** the same frozen ResNet-50 grid features → 3-layer Transformer decoder with cross-attention over the 49 image cells.
3. **Comparison pipeline (to do):** English captioning model → machine translation to Hindi.

### Baseline training setup
| | |
|---|---|
| Encoder | ResNet-50, frozen (`--fine_tune` unfreezes the last block) |
| Decoder | LSTMCell, 512 hidden, 256 embedding, 256 attention dim, dropout 0.5 |
| Trainable parameters | 12.36 M |
| Loss | cross-entropy (pad ignored) + doubly-stochastic attention penalty (weight 1.0) |
| Optimiser | Adam, lr 4e-4, gradient clipping 5.0 |
| Run | 10 epochs, batch size 64, Colab T4 GPU (~2.5 min / epoch) |
| Best checkpoint | epoch 8 (dev loss 3.619); dev loss rises slightly after that (mild overfitting) |
| Decoding | beam search, beam size 3 |

### Transformer training setup
| | |
|---|---|
| Encoder | same frozen ResNet-50 as the baseline, features projected to 512-d + learned grid-position embeddings |
| Decoder | 3 pre-norm Transformer layers, 8 heads, d_model 512, feed-forward 1024, dropout 0.3 |
| Trainable parameters | 14.28 M |
| Loss | cross-entropy (pad ignored); no attention penalty |
| Optimiser | Adam, lr 2e-4, gradient clipping 5.0 |
| Run | 15 epochs, batch size 64, Colab T4 GPU (~2 min / epoch) |
| Best checkpoint | epoch 8 (dev loss 3.201); dev loss rises after that (train loss keeps falling to 1.93, dev reaches 3.56) |
| Decoding | beam search, beam size 3 |
| Ablation | dropout 0.4, 12 epochs: best epoch 8, dev loss 3.224 |

The final Transformer was chosen by dev loss, not by test score. The Transformer's dev loss is plain cross-entropy, while the baseline's also includes the attention penalty, so the two dev losses are not directly comparable.

## Results

Test set (1,586 region crops), scores x100:

| Approach | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr |
|---|---|---|---|---|---|
| CNN + LSTM + Attention (baseline) | 22.82 | 12.00 | 7.00 | 4.31 | 68.22 |
| ResNet-50 + Transformer decoder | 21.42 | 10.94 | 5.99 | 3.18 | 59.13 |
| ResNet-50 + Transformer decoder, dropout 0.4 (ablation) | 22.20 | 11.55 | 6.23 | 3.03 | 60.88 |
| English captioning + translation | — | — | — | — | — |

The Transformer is below the baseline on every metric (CIDEr 59.13 vs 68.22). The dropout 0.4 ablation scored slightly higher on test than the dropout 0.3 model, but its dev loss was slightly worse (3.224 vs 3.201) and each setting was run once, so the final Transformer is the dropout 0.3 model chosen by dev loss.

Notes: captions are short region descriptions (about 4-5 words), so BLEU-3/4 are naturally low. Many predictions are generic ("एक प्लेट पर एक सैंडविच"), and rare words become `<unk>` because the vocabulary only keeps words that appear at least twice in train. Scores and predictions are in `results/` (`<name>_test_scores.json`, `<name>_test_preds.csv`).

## Repository Structure

```
├── data/              # raw/ and processed/ (NOT committed)
├── notebooks/         # Colab / exploration notebooks
├── src/
│   ├── preprocess_hvg.py   # crops regions, writes captions.csv
│   ├── dataset.py          # tokenizer, Vocab, HVGDataset, get_dataloader
│   ├── models/
│   │   ├── baseline.py     # CNN + LSTM + Attention
│   │   └── transformer.py  # ResNet-50 + Transformer decoder
│   ├── train.py            # training loop, save/load checkpoints
│   ├── evaluate.py         # BLEU, CIDEr (reusable for every approach)
│   ├── attention_maps.py   # attention-map figures for random test images
│   └── compare_results.py  # Markdown score table from results/*_scores.json
├── checkpoints/       # *.pth (NOT committed)
├── results/           # score tables, predictions, attention maps
├── requirements.txt
└── README.md
```

## How to Run

```bash
# 1. Clone and install
git clone https://github.com/Coder-Vinay-Arc/hindi-image-captioning-team17.git
cd hindi-image-captioning-team17
pip install -r requirements.txt

# 2. Put the processed data in data/processed/ (see Dataset above)

# 3. Quick sanity check on the 500-row sample (CPU is fine)
python src/train.py --model baseline --train_split sample --val_split sample --epochs 2 --batch_size 16 --num_workers 0 --name my_test

# 4. Full training (use a GPU, e.g. Colab T4)
python src/train.py --model baseline --epochs 10 --batch_size 64

# 5. Evaluate the best checkpoint on the test set
python src/evaluate.py --model baseline --split test --ckpt checkpoints/baseline_best.pth --beam 3

# 6. Transformer: train, evaluate, attention figures, score table
python src/train.py --model transformer --epochs 15 --batch_size 64 --lr 2e-4 --name transformer   # add --dropout 0.4 or --fine_tune to try variants
python src/evaluate.py --model transformer --split test --ckpt checkpoints/transformer_best.pth --beam 3
python src/attention_maps.py --model transformer --ckpt checkpoints/transformer_best.pth --n 6 --beam 3   # needs a Devanagari font, see the script docstring
python src/compare_results.py --names baseline transformer
```

### Loading the data in your own code
```python
import sys; sys.path.insert(0, "src")
from dataset import Vocab, get_dataloader

vocab = Vocab.load("data/processed/vocab.json")
loader = get_dataloader("train", vocab, batch_size=32)       # or "dev", "test", "sample"
for images, captions, lengths in loader:
    # images (B,3,224,224) | captions (B,T) padded with 0, starts with <start>, ends with <end> | lengths (B,)
    ...
vocab.decode(captions[0])    # ids -> Hindi string
```

### Loading a trained model
```python
from train import load_checkpoint
model, vocab = load_checkpoint("checkpoints/baseline_best.pth", device="cuda")
ids, attention = model.caption(images, vocab.start_id, vocab.end_id, beam_size=3)
texts = [vocab.decode(i) for i in ids]
```
`attention` holds one `(num_words, 49)` tensor per image (7x7 grid) — use it for the attention-map figures.

### Adding a new model
`train.py` already has a hook for `--model transformer` (it imports `models/transformer.py`). Your file must expose `CaptionModel(vocab_size, **kwargs)` with:
- `self.config`: dict of the constructor arguments including `vocab_size` (saved inside checkpoints, used by `load_checkpoint`)
- `forward(images, captions)` → `(logits, alphas)`, where `logits` is `(B, T-1, vocab)` for predicting `captions[:, 1:]`, and `alphas` is attention weights or `None`
- `caption(images, start_id, end_id, max_len=40, beam_size=1)` → `(list of id lists without <start>/<end>, list of attention maps)`
- keyword arguments `fine_tune` and `pretrained` accepted by the constructor

Then `python src/train.py --model transformer ...` and `python src/evaluate.py --model transformer --ckpt ...` work unchanged.

### Scoring another approach (e.g. translation pipeline)
Write a CSV with columns `file_name, prediction` for the test split (`file_name` exactly as in `data/processed/test/captions.csv`) and run:
```bash
python src/evaluate.py --model translation --split test --pred_csv results/translation_test_preds.csv
```
Or in Python: `from evaluate import compute_scores; compute_scores(list_of_reference_strings, list_of_prediction_strings)`. Always use this function so all three approaches are scored identically.

> HVG also contains the gold **English** caption for every region (`english_caption` column in `captions.csv`). Translating those gives an upper bound for the machine-translation step — useful for the discussion section.

## Evaluation

- BLEU-1..4 / CIDEr (all three approaches, same `evaluate.py`)
- Small human evaluation (fluency + correctness, 1–5 scale) on ~50 test samples — `results/*_test_preds.csv` has reference and prediction side by side
- Attention map visualizations

## Git Workflow (reminder)

Work on your own branch, never push to `main` directly, open a PR and merge. Do not commit data, `*.pth` or `*.h5` files. After merging, other people's code appears in `main` — run `git pull origin main`. **Do not use GitHub's "Revert" button on merged PRs** (it removed the baseline from `main` once and had to be restored).

## Reference Papers

1. Xu et al., *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*, ICML 2015 — https://arxiv.org/abs/1502.03044
2. Parida et al., *Hindi Visual Genome: A Dataset for Multimodal English-to-Hindi Machine Translation*, 2019 — https://arxiv.org/abs/1907.08948

## Status

🚧 Work in progress — Submission Deadline: 22 Nov 2026
