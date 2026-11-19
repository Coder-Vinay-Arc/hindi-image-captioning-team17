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

## Dataset

- **Primary:** [Hindi Visual Genome](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3267) (LINDAT/CLARIN)
- **Optional:** Flickr8k (translated captions)
- **Data source (Google Drive):** https://drive.google.com/file/d/1Ts_UObq-tl9n2xjKGivxUe9w_ndGEY_n/view?usp=sharing

> The dataset is not uploaded to this repo (due to size limits). To download it, see `data/README.md`, or run the `data/download_data.sh` script given below.

## Approach

1. **Baseline:** CNN encoder (ResNet) + LSTM decoder with attention (Show, Attend and Tell)
2. **Advanced:** Transformer decoder (with a CNN/ViT encoder)
3. **Comparison pipeline:** English captioning model → Machine Translation (Hindi)

## Current Status

- [x] Repo created, all members added as collaborators
- [x] Dataset strategy decided: Hindi Visual Genome (region-crop approach)
- [x] `preprocess_hvg.py` script ready
- [ ] Dataset download — in progress
- [ ] Model training — not started
- [ ] Report — not started

> Detailed task division for each member is in [`CLAUDE.md`](./CLAUDE.md).

## Task Division

| Member | Branch | Task |
|---|---|---|
| Vinay Kumar Gupta | `vinay-data-pipeline` | Dataset download, preprocessing, PyTorch Dataset class |
| Rahul Kumar Verma | `rahul-baseline-model` | Baseline model: CNN + LSTM + Attention |
| Vishal Bhardwaj | `vishal-transformer-model` | Advanced model: Transformer decoder |
| Vivek Kumar | `vivek-translation-eval` | English captioning + translation pipeline, BLEU/CIDEr evaluation |

## Evaluation

- BLEU / CIDEr scores (for all three approaches)
- Chhota human evaluation (fluency + correctness, 1–5 scale)
- Attention map visualizations

## Repository Structure

```
├── data/              # Dataset download scripts / instructions (raw data NOT committed)
├── notebooks/         # Exploration & Colab experiment notebooks
├── src/
│   ├── models/        # Model architectures (baseline, transformer)
│   ├── train.py        # Training script
│   └── evaluate.py     # Evaluation script (BLEU, CIDEr)
├── results/           # Sample outputs, attention map images, score tables
├── requirements.txt
└── README.md
```

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/<username>/hindi-image-captioning-team17.git
cd hindi-image-captioning-team17

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
bash data/download_data.sh

# 4. Train baseline model
python src/train.py --model baseline

# 5. Evaluate
python src/evaluate.py --model baseline
```

## Reference Papers

1. Xu et al., *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*, ICML 2015 — https://arxiv.org/abs/1502.03044
2. Parida et al., *Hindi Visual Genome: A Dataset for Multimodal English-to-Hindi Machine Translation*, 2019 — https://arxiv.org/abs/1907.08948

## Status

🚧 Work in progress — Submission Deadline: 22 Nov 2026
