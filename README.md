# Image Captioning in Hindi with Attention

**Deep Learning Project — Team 17 (Section B)**

Hindi Visual Genome aur attention-based deep learning models ka use karke images ke liye seedhe Hindi captions generate karna — English captioning aur translation pipeline se compare karte hue.

## Team Members

| Name | Email |
|---|---|
| Vinay Kumar Gupta | su-24153@sitare.org |
| Rahul Kumar Verma | su-24106@sitare.org |
| Vishal Bhardwaj | su-24155@sitare.org |
| Vivek Kumar | su-24159@sitare.org |

## Problem Statement

Image captioning mostly English mein hota hai. Hindi ek low-resource language hai is task ke liye. Is project mein hum images ke liye directly Hindi captions generate karte hain, aur compare karte hain ki direct Hindi captioning better hai ya English caption + machine translation pipeline.

## Dataset

- **Primary:** [Hindi Visual Genome](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3267) (LINDAT/CLARIN)
- **Optional:** Flickr8k (translated captions)

> Dataset is repo mein upload nahi hai (size limit ki wajah se). Download karne ke liye `data/README.md` dekhein, ya neeche diye `data/download_data.sh` script chalayein.

## Approach

1. **Baseline:** CNN encoder (ResNet) + LSTM decoder with attention (Show, Attend and Tell)
2. **Advanced:** Transformer decoder (CNN/ViT encoder ke saath)
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

- BLEU / CIDEr scores (teeno approaches ke liye)
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
