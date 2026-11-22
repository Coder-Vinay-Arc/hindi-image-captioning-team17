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

## Current Status (updated 21 Nov 2026)

- [x] Repo created, all members added as collaborators
- [x] Dataset strategy decided: Hindi Visual Genome (region-crop approach)
- [x] Dataset downloaded and preprocessed (train / dev / test) — see [Dataset](#dataset)
- [x] `src/dataset.py` — PyTorch Dataset, Hindi tokenizer, vocab, DataLoader (merged to `main`)
- [x] `src/models/baseline.py` + `src/train.py` — CNN + LSTM + Attention, training loop, checkpoints (merged)
- [x] `src/evaluate.py` — BLEU-1..4 and CIDEr, usable for all three approaches (merged)
- [x] Baseline trained on the full training set and scored on the test set — see [Results](#results)
- [x] Transformer model (`src/models/transformer.py`) trained and scored on the test set — see [Results](#results)
- [x] Transformer attention-map figures (`src/attention_maps.py`, `results/transformer_attention_*.png`)
- [x] English captioning + translation pipeline (`src/translation_pipeline.py`) — done (BLIP + Opus-MT)
- [x] Translation human evaluation — 50 test samples scored for fluency + correctness (`results/human_eval_translation.csv`)
- [x] Main research question answered: **direct Hindi captioning beats English captioning + translation** (CIDEr 68.22 vs 33.89) — see [Results](#results)
- [ ] Baseline fine-tuning / optimisation experiments — **Rahul** (see [Open tasks](#open-tasks))
- [ ] Baseline attention-map figures — **Rahul** (see [Open tasks](#open-tasks))
- [x] Human evaluation of the baseline and the Transformer — 50 samples, blind, one rater: fluency / correctness 3.38 / 3.88 (baseline) vs 3.10 / 3.96 (Transformer), see [Evaluation](#evaluation)
- [ ] Report — open, to be split in the team

## Task Division

| Member | Branch | Task | Status |
|---|---|---|---|
| Vinay Kumar Gupta | `vinay-data-pipeline`, `vinay-baseline` | Data pipeline, Dataset class, baseline model + training, `evaluate.py` | Done (merged) |
| Rahul Kumar Verma | `rahul-baseline-model` | Baseline improvement: fine-tuning and other optimisation, baseline attention maps | **Not done / open** — nothing from this task is in `main` yet, see [Open tasks](#open-tasks) |
| Vishal Bhardwaj | `vishal-transformer-model` | Transformer decoder (`src/models/transformer.py`), attention figures, comparison script | Done (merged) — see [Transformer training setup](#transformer-training-setup) |
| Vivek Kumar | `vivek-translation-eval` | English captioning + translation pipeline, its human evaluation | Done (merged) — see [Results](#results) |

> Detailed project rules and task division are in [`CLAUDE.md`](./CLAUDE.md).

## Progress Log (what has been done so far)

**Data pipeline (Vinay, 19 Nov)**
- Downloaded Hindi Visual Genome 1.1 from LINDAT (one 4.33 GB zip with the annotation files **and** the images).
- `src/preprocess_hvg.py` crops every bounding box, resizes it to 224x224 and writes `captions.csv` (`file_name, hindi_caption, english_caption`). It skips regions smaller than 8 px, rows without a Hindi caption, rows whose image is missing, and a few rows with a malformed (non-numeric) bounding box. Result: train 28,705 / dev 993 / test 1,586 crops, plus a 500-row `sample` split for quick tests.
- `src/dataset.py`: Hindi tokenizer (strips punctuation and the danda `।`, removes zero-width joiners, whitespace split), `Vocab` (3,615 tokens from train only, `min_freq=2`, saved to `vocab.json`), `HVGDataset` (ImageNet-normalised 224x224 images, captions truncated at 40 tokens), a padding `Collate` class (a class rather than a closure so DataLoader workers also work on Windows) and `get_dataloader`.
- Processed data and `vocab.json` were uploaded to the shared Drive folder so everybody trains on identical data.

**Baseline: CNN + LSTM + Attention (Vinay, 19 Nov)**
- `src/models/baseline.py`: frozen ResNet-50 → 49 region features → LSTM decoder with additive attention and a gated context vector; greedy and beam-search decoding that also return the attention weights.
- `src/train.py`: teacher forcing, cross-entropy plus the doubly-stochastic attention penalty, dev loss every epoch, `*_best.pth` / `*_last.pth` checkpoints, `--resume`, `load_checkpoint()` so any model can be re-loaded by other scripts.
- Trained on a Colab T4 for 10 epochs (about 2.5 min per epoch). Dev loss fell from 4.44 to 3.62 (best at epoch 8), then rose slightly (mild overfitting), so the epoch-8 checkpoint is the one used. Test: BLEU-1 22.82, BLEU-4 4.31, CIDEr 68.22.

**Evaluation code (Vinay, 19 Nov)**
- `src/evaluate.py` is the single scoring function for all approaches: BLEU-1..4 (NLTK corpus BLEU, smoothing method 1) and CIDEr-D (own pure-Python implementation, no Java needed), all x100. It works on a trained checkpoint (`--ckpt`) or on a predictions CSV from another approach (`--pred_csv`). Sanity-checked: identical captions give BLEU-1 = 100, shuffled captions give near 0.

**Transformer (Vishal, 20 Nov)**
- `src/models/transformer.py`: same frozen ResNet-50 features, 3-layer pre-norm Transformer decoder with cross-attention over the 49 cells; `--dropout` flag added to `train.py`; trained 15 epochs (best at epoch 8, dev loss 3.201). A dropout-0.4 ablation was also run.
- `src/attention_maps.py` (attention figures with correctly shaped Devanagari via Raqm) and `src/compare_results.py` (Markdown score table). Test: BLEU-1 21.42, BLEU-4 3.18, CIDEr 59.13 — below the baseline.

**English captioning + translation (Vivek, 21 Nov)**
- `src/translation_pipeline.py`: BLIP English caption → Opus-MT English→Hindi. Also run on the dataset's **gold** English captions to get an upper bound for the translation step.
- Test: CIDEr 33.89 for the full pipeline vs 242.34 for gold English + translation, so the English captioning stage is the bottleneck, not the translation stage.
- Human evaluation of 50 test samples: fluency 3.46/5, correctness 3.14/5.

**Housekeeping**
- README and `CLAUDE.md` set up with the project rules (own branch + PR, no data or weights in git, no AI attribution in commits/PRs, detailed code comments).
- Incident to remember: a merged PR was once reverted with GitHub's "Revert" button, which removed the baseline from `main`; it was restored through a new PR. Do not use "Revert" on merged PRs.

**Known limitations (for the report's discussion)**
- Captions are short region descriptions (4-5 words), so BLEU-3/4 are low for every model; predictions are often generic.
- Rare words become `<unk>` (vocabulary keeps only words seen at least twice in train).
- Each configuration was trained once, so small differences between models (e.g. the dropout ablation) are not statistically meaningful.
- The Transformer's dev loss is plain cross-entropy while the baseline's includes the attention penalty, so the two dev losses cannot be compared directly.

## Open Tasks

### Rahul — baseline improvement (not done yet)
`rahul-baseline-model` currently has **no merged work**. Please do these on top of the current `main` (use `main`'s `src/train.py`, `src/models/baseline.py` and `src/evaluate.py` — they are what the reported baseline numbers and the Transformer depend on; do not replace them). Start from a fresh branch off the latest `main` and open a PR when finished.
1. **Fine-tune the baseline's ResNet-50 last block** (`layer4`):
   ```bash
   python src/train.py --model baseline --epochs 10 --batch_size 64 --fine_tune --lr 1e-4 --name baseline_ft
   python src/evaluate.py --model baseline_ft --split test --ckpt checkpoints/baseline_ft_best.pth --beam 3
   ```
   Note: `train.py` currently uses a single learning rate for all trainable parameters, so use a smaller `--lr` when `--fine_tune` is on (or add a separate, smaller learning rate for the encoder).
2. **Other optimisation experiments** on the baseline — pick any that fit the time, one run each: `--dropout`, `--alpha_c`, `--lr`, number of epochs, beam size at evaluation (`--beam 1/3/5`). Choose the final setting by **dev loss**, not by test score.
3. **Baseline attention-map figures** (`results/baseline_attention_*.png`):
   ```bash
   python src/attention_maps.py --model baseline --ckpt checkpoints/baseline_best.pth --n 6 --beam 3
   ```
   On Colab run once first: `!apt-get -qq install fonts-noto-core libraqm0 libfribidi0`.
4. Add the new rows to the [Results](#results) table (`python src/compare_results.py --names baseline baseline_ft transformer`) and push the new `results/*_scores.json` files.

If the experiments do not beat the current baseline (CIDEr 68.22), keep the original baseline as the reported one and mention the experiments in the discussion.

### Whole team
- **Report** (ELCVIA template on Overleaf): Abstract, Introduction, Related Work, Dataset (region-crop decision), Methodology (3 approaches), Experiments & Results, Human Evaluation, Discussion, Conclusion, References. Submit through the Google Form before **22 Nov 2026, 11:59 PM IST**.

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
3. **Comparison pipeline (done):** BLIP (`Salesforce/blip-image-captioning-base`) English captioner → Helsinki-NLP Opus-MT (`Helsinki-NLP/opus-mt-en-hi`) English→Hindi translation with 4-beam search and no-repeat n-gram blocking. Applying the same MT settings to the dataset's gold English captions gives the translation upper bound.

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
| English captioning + translation (BLIP + Opus-MT) | 16.13 | 6.70 | 3.06 | 1.43 | 34.70 |
| Oracle: gold English + translation (upper bound) | 54.69 | 41.33 | 31.53 | 23.47 | 251.78 |

The Transformer is below the baseline on every metric (CIDEr 59.13 vs 68.22). The dropout 0.4 ablation scored slightly higher on test than the dropout 0.3 model, but its dev loss was slightly worse (3.224 vs 3.201) and each setting was run once, so the final Transformer is the dropout 0.3 model chosen by dev loss.

Notes: captions are short region descriptions (about 4-5 words), so BLEU-3/4 are naturally low. Many predictions are generic ("एक प्लेट पर एक सैंडविच"), and rare words become `<unk>` because the vocabulary only keeps words that appear at least twice in train. Scores and predictions are in `results/` (`<name>_test_scores.json`, `<name>_test_preds.csv`).

The translation pipeline is much weaker than direct Hindi captioning because the English captioner describes the whole crop and loses Hindi-specific detail; translating the dataset's gold English captions with the same MT settings reaches CIDEr 251.78, so the captioning stage — not the translation stage — is the main bottleneck. The gold-English row is an upper bound, not a captioning result: HVG's Hindi references are themselves translations of those English captions.

The translation pipeline uses beam search (4 beams) with no-repeat n-gram blocking; this removes the degenerate repeated outputs produced by greedy decoding and improves CIDEr (33.89 → 34.70) on the test set.

A 50-sample human evaluation of the translation pipeline gives **fluency 3.54/5** and **correctness 3.36/5** (1–5 scale; per-sample ratings in `results/human_eval_translation.csv`).

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
│   ├── translation_pipeline.py  # BLIP English caption + Opus-MT Hindi translation
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

# 7. English captioning + translation pipeline (BLIP + Opus-MT)
python src/translation_pipeline.py --split test
python src/evaluate.py --model translation --split test --pred_csv results/translation_test_preds.csv

# 8. Translation upper bound: translate the dataset's gold English captions
python src/translation_pipeline.py --split test --gold_english --out results/translation_gold_en_test_preds.csv
python src/evaluate.py --model translation_gold_en --split test --pred_csv results/translation_gold_en_test_preds.csv
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

### Human evaluation of the baseline and the Transformer

50 test samples (the same ones as in the translation evaluation, listed in `results/human_eval_translation.csv`), one rater, blind: the captions of both models were mixed in one shuffled sheet, so the rater did not know which model wrote which caption. Scales: fluency (is the Hindi correct and natural, 1 = broken, 5 = perfect) and correctness (does it describe the image, 1 = completely wrong, 5 = fully correct). The ratings are in `results/human_eval_baseline_transformer.csv`.

| Approach | n | Fluency (1-5) | Correctness (1-5) |
|---|---|---|---|
| CNN + LSTM + Attention (baseline) | 50 | 3.38 | 3.88 |
| ResNet-50 + Transformer decoder | 50 | 3.10 | 3.96 |

The two models are close. Paired per image, Transformer minus baseline is -0.28 for fluency (95% interval -0.76 to +0.20) and +0.08 for correctness (-0.17 to +0.33). Both intervals include zero, so with 50 samples and a single rater we cannot say that either model is better in the human ratings. The translation pipeline was rated by a different person, so its human-evaluation numbers (see [Results](#results)) are only indicative when compared with these.

To repeat the evaluation:
```bash
python src/human_eval_sheet.py make --preds baseline=results/baseline_test_preds.csv transformer=results/transformer_test_preds.csv
python src/human_eval_sheet.py page      # optional: browser page with image + captions + 1-5 buttons (instead of editing the CSV)
# fill fluency_1_5 and correctness_1_5 (whole numbers 1-5) in results/human_eval_rating_sheet.csv, then:
python src/human_eval_sheet.py score     # prints the means and writes results/human_eval_baseline_transformer.csv
```

## Git Workflow (reminder)

Work on your own branch, never push to `main` directly, open a PR and merge. Do not commit data, `*.pth` or `*.h5` files. After merging, other people's code appears in `main` — run `git pull origin main`. **Do not use GitHub's "Revert" button on merged PRs** (it removed the baseline from `main` once and had to be restored).

Two more rules (full text in [`CLAUDE.md`](./CLAUDE.md)): commits, pushes and PRs must carry **no AI/LLM attribution** (no `Co-Authored-By: Claude ...`, no "Generated with ..." footer), and **all new code must have detailed comments** so teammates can read and understand it later.

## Reference Papers

1. Xu et al., *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*, ICML 2015 — https://arxiv.org/abs/1502.03044
2. Parida et al., *Hindi Visual Genome: A Dataset for Multimodal English-to-Hindi Machine Translation*, 2019 — https://arxiv.org/abs/1907.08948

## Status

🚧 Work in progress — Submission Deadline: 22 Nov 2026
