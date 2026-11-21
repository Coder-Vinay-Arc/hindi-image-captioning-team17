# CLAUDE.md

This file defines the project context and rules. Each team member can show this file to Claude Code (or Claude) to get started on their part of the work, regardless of which branch/laptop they are working from.

## Project Overview

**Title:** Image Captioning in Hindi with Attention
**Team:** 17 (Section B) — Vinay Kumar Gupta, Rahul Kumar Verma, Vishal Bhardwaj, Vivek Kumar
**Goal:** Generate Hindi captions directly for images, and compare whether this is better than an "English caption + Machine Translation" pipeline.
**Deadline:** 22 Nov 2026, 11:59 PM IST

## Current Status (as of now)

- [x] GitHub repo created, all collaborators added
- [x] Initial draft of README.md done
- [x] Dataset decision finalized: Hindi Visual Genome (HVG), region-crop approach (each bounding box is treated as an independent image-caption pair)
- [x] `preprocess_hvg.py` script written (crops regions, generates captions.csv)
- [ ] Dataset not downloaded yet
- [ ] No model trained yet
- [ ] No code pushed to `src/` yet
- [ ] Report not started

**Important:** Only planning has happened so far. No actual training/coding work is done yet -- each member needs to START their assigned part.

## Dataset Decision (important context)

In Hindi Visual Genome, captions describe a **region (bounding box) within an image, not the whole image.** We have decided to treat each region crop as an independent (image, caption) training pair -- this is standard practice with Visual Genome-style datasets, and it also gives us more training data.

`preprocess_hvg.py` does this: it reads the raw annotation file (`image_id, X, Y, Width, Height, English, Hindi`), crops the region, resizes it, and produces a `captions.csv` with a `file_name -> hindi_caption` mapping.

## Repository Structure

```
├── data/
│   ├── raw/              # Downloaded HVG files (NOT committed to git -- .gitignore)
│   └── processed/        # Output of preprocess_hvg.py (NOT committed -- too large)
├── notebooks/            # Colab/Jupyter exploration notebooks (person-specific, prefix with name)
├── src/
│   ├── preprocess_hvg.py # Dataset preprocessing (DONE)
│   ├── dataset.py        # PyTorch Dataset/DataLoader class (TODO)
│   ├── models/
│   │   ├── baseline.py   # CNN + LSTM + Attention (TODO)
│   │   └── transformer.py # Transformer decoder version (TODO)
│   ├── translation_pipeline.py  # English captioning + MT pipeline (TODO)
│   ├── train.py          # Training loop (TODO)
│   └── evaluate.py       # BLEU / CIDEr scoring (TODO)
├── results/              # Score tables, attention map images, sample outputs
├── requirements.txt
├── README.md
└── CLAUDE.md             # (this file)
```

## Task Division (everyone works on their own branch)

| Member | Branch name | Task | Output |
|---|---|---|---|
| **Vinay Kumar Gupta** | `vinay-data-pipeline` | Download the HVG dataset, run `preprocess_hvg.py` on all three splits (train/dev/test), write `src/dataset.py` (a PyTorch Dataset class that reads `captions.csv` + images, tokenizes text, and builds batches) | `data/processed/{train,dev,test}/`, `src/dataset.py` |
| **Rahul Kumar Verma** | `rahul-baseline-model` | Write `src/models/baseline.py`: a ResNet (pretrained, frozen or fine-tuned) encoder + LSTM decoder with attention ("Show, Attend and Tell"). Write the baseline training loop in `src/train.py`. | `src/models/baseline.py`, baseline training results |
| **Vishal Bhardwaj** | `vishal-transformer-model` | Write `src/models/transformer.py`: a CNN/ViT encoder + Transformer decoder. Reuse/extend the baseline training loop. | `src/models/transformer.py`, transformer training results |
| **Vivek Kumar** | `vivek-translation-eval` | `src/translation_pipeline.py`: generate an English caption using a pretrained captioning model, then translate it to Hindi (IndicTrans2 / Google Translate). `src/evaluate.py`: compute BLEU and CIDEr scores (write a reusable function for all three approaches). | `src/translation_pipeline.py`, `src/evaluate.py`, score tables |

**Note:** Vinay's work is needed first (the data must be ready before the other three can train anything) -- so Vinay should start today, and process a small subset (e.g. 500 rows) quickly to hand off to the others so they can test their model code without waiting for the full dataset to finish processing.

## Branching & Workflow Rules

1. Never push directly to the `main` branch.
2. Do your work on your own branch (name given in the table above): `git checkout -b <branch-name>`.
3. Commit whenever a chunk of work is complete (a function, a script): `git add . && git commit -m "clear message"`.
4. `git push origin <branch-name>`, then open a Pull Request on GitHub and merge into `main`.
5. Keep commit messages clear: `"add resnet encoder"`, `"fix dataset path bug"` -- avoid vague messages like "update" or "fix".
6. Never commit large files (datasets, model weights `.pth`/`.h5`, bulk images) -- already handled via `.gitignore`.
7. **No AI / LLM attribution, for anyone.** This applies to every team member and to every AI tool (Claude, ChatGPT, Copilot, Gemini, Cursor, etc.). A commit or push must NOT carry any AI co-author line (e.g. `Co-Authored-By: Claude ...`, `Co-Authored-By: ChatGPT ...`), and commit messages, PR titles and PR descriptions must NOT include "Generated with Claude Code", "written by ChatGPT" or any similar footer or mention of an AI tool. All commits, pushes and PRs must appear as the team member's own work. Before pushing, check `git log -1` to make sure no such line is present; if it is, remove it (`git commit --amend`) before you push. This is also enforced for Claude Code via `.claude/settings.json` (`attribution` set to empty) -- do not remove that setting.

## Coding Conventions

- Python 3.10+, use PyTorch.
- Every script should have a docstring at the top explaining what it does and how to run it (see `preprocess_hvg.py` for an example).
- **Write detailed comments in all code you add or change** (this applies to code written by hand and to code produced by any AI tool -- add the comments before you commit). The goal is that a teammate can read the file later and easily understand what is happening and why. Concretely:
  - give every function and class a docstring: what it does, its arguments, what it returns;
  - put a short comment above each logical block of a function explaining what the block does and, when it is not obvious, why;
  - for PyTorch code, comment tensor shapes (e.g. `# (B, 49, 2048)`) and the meaning of each dimension;
  - explain non-obvious choices and magic numbers (why this learning rate, why this size, why this filter);
  - comment workarounds and edge cases (e.g. why a row is skipped, why a class is used instead of a closure);
  - do not leave stale comments -- if you change the code, update its comment.
- Model classes go in `src/models/`, training/eval logic goes in `src/train.py` and `src/evaluate.py` -- don't mix model definitions with the training loop.
- Whatever model you build, also write a way to save/load checkpoints (so another member can load your trained model and evaluate it).
- For Hindi text tokenization, use `indic-nlp-library` or simple whitespace + normalization -- whatever you use, add it to `requirements.txt`.

## Commands Reference

```bash
# Preprocess one split
python src/preprocess_hvg.py --annotations data/raw/train.txt --images_dir data/raw/images --out_dir data/processed/train

# Train baseline (once dataset.py + baseline.py are ready)
python src/train.py --model baseline --epochs 10

# Evaluate
python src/evaluate.py --model baseline --split test
```

## Deadline Checkpoints

- **Day 1-2 (now):** Vinay completes the data pipeline; Rahul/Vishal/Vivek start writing their model/pipeline code (testing on small dummy data in the meantime)
- **Day 3:** All models start training on real data, results begin to come in
- **Day 4:** Human evaluation, results table, write the report, final merge to main, submit via Google Form

## Report

- Template: ELCVIA (Computer Vision) -- Overleaf link is in the email
- Sections: Abstract, Introduction, Related Work, Dataset (explain the region-crop decision), Methodology (3 approaches), Experiments & Results, Human Evaluation, Discussion, Conclusion, References
