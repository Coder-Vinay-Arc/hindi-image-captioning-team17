"""
evaluate.py -- BLEU-1..4 and CIDEr for Hindi captions. One scoring function shared by ALL approaches
(baseline, transformer, English-caption + translation), so the numbers are directly comparable.

Reusable function:
    from evaluate import compute_scores
    scores = compute_scores(references, hypotheses)   # two lists of Hindi strings, same length/order
    # -> {"BLEU-1": .., "BLEU-2": .., "BLEU-3": .., "BLEU-4": .., "CIDEr": ..}
    # (each reference may also be a list of strings if you have several references per image)

Command line:
    # 1) score a trained model on a split (loads the checkpoint saved by train.py)
    python src/evaluate.py --model baseline --split test --ckpt checkpoints/baseline_best.pth --beam 3

    # 2) score predictions made by another approach (e.g. the translation pipeline).
    #    CSV with columns: file_name, prediction  (file_name as in data/processed/<split>/captions.csv)
    python src/evaluate.py --model translation --split test --pred_csv results/translation_test_preds.csv

Outputs (name = --name, default = --model):
    results/<name>_<split>_scores.json    the score table
    results/<name>_<split>_preds.csv      file_name, reference, prediction  (for human evaluation)

Tokenisation is the same Hindi tokenizer used for training (dataset.tokenize), applied to references and
predictions alike. All scores are reported x100 (BLEU as in papers; CIDEr-D as in the COCO toolkit, x100).
"""

# argparse = command-line options, json = save the scores, math = log / sqrt / exp for CIDEr, os / sys = paths
# and imports, Counter / defaultdict = counting n-grams.
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

# pandas reads captions.csv. NLTK gives the standard corpus BLEU implementation.
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Use the same Hindi tokenizer as in training, so references and predictions are cut into words in the same
# way.
from dataset import tokenize  # noqa: E402


# ---------------------------------------------------------------- CIDEr-D (pure python)
# Counts every group of n consecutive words (an n-gram) in a sentence. Example: 'एक सफेद प्लेट' has the
# 2-grams 'एक सफेद' and 'सफेद प्लेट'.
def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


# CIDEr-D idea: a prediction is good if it shares n-grams with the reference, and shared n-grams count more
# when they are RARE across all captions (TF-IDF). Common words like 'एक' or 'पर' count little.
def cider_d(refs, hyps, n_max=4, sigma=6.0):
    """CIDEr-D. refs: list (per image) of lists of token lists; hyps: list of token lists.
    Document frequencies come from the reference set being scored, as in the COCO implementation."""
    # N = number of images.
    N = len(refs)
    # df[n] will store, for every (n+1)-gram, in how many images it appears (document frequency).
    df = [defaultdict(float) for _ in range(n_max)]
    # Count the document frequency: every n-gram is counted once per image (set), however often it is
    # repeated.
    for image_refs in refs:
        for n in range(n_max):
            for ng in set(g for r in image_refs for g in _ngrams(r, n + 1)):
                df[n][ng] += 1.0
    # log of the number of images, needed in the IDF formula.
    log_n = math.log(float(N))

    # Turns a sentence into a TF-IDF vector for each n-gram size 1..4, plus its length (norm) for the cosine
    # similarity.
    def vectorise(tokens):
        vec, norm = [], []
        for n in range(n_max):
            counts = _ngrams(tokens, n + 1)
            # TF-IDF weight = how often the n-gram occurs in this sentence (tf) x how rare it is across all
            # images (idf).
            v = {ng: tf * (log_n - math.log(max(1.0, df[n][ng]))) for ng, tf in counts.items()}
            vec.append(v)
            norm.append(math.sqrt(sum(x * x for x in v.values())))
        return vec, norm, len(tokens)

    # Cosine-like similarity between the prediction and one reference.
    def sim(vh, nh, lh, vr, nr, lr):
        delta = float(lh - lr)
        out = []
        for n in range(n_max):
            # Overlap of the two vectors. min() caps the credit at the reference count, so repeating a word
            # many times gives no extra score.
            val = sum(min(w, vr[n].get(ng, 0.0)) * vr[n].get(ng, 0.0) for ng, w in vh[n].items())
            if nh[n] != 0 and nr[n] != 0:
                val /= nh[n] * nr[n]
            else:
                val = 0.0
            # Length penalty: the score shrinks when the prediction and the reference have very different
            # lengths (delta).
            out.append(val * math.exp(-(delta ** 2) / (2 * sigma ** 2)))
        return out

    # Score every image: the average similarity of its prediction to all its references, averaged over n =
    # 1..4.
    scores = []
    for image_refs, hyp in zip(refs, hyps):
        vh, nh, lh = vectorise(hyp)
        total = [0.0] * n_max
        for r in image_refs:
            vr, nr, lr = vectorise(r)
            for n, s in enumerate(sim(vh, nh, lh, vr, nr, lr)):
                total[n] += s
        # Average over n-gram sizes and references. The x10 is the factor used in the COCO CIDEr-D
        # implementation.
        scores.append(sum(total) / n_max / len(image_refs) * 10.0)
    return sum(scores) / len(scores)


# ---------------------------------------------------------------- public API
# Public function used by ALL approaches, so the numbers are comparable. Input: two lists of Hindi strings.
# Output: a dict with BLEU-1..4 and CIDEr (x100).
def compute_scores(references, hypotheses):
    """references: list of str (or list of list of str); hypotheses: list of str. Returns dict of scores (x100)."""
    assert len(references) == len(hypotheses), "references and hypotheses must have the same length"
    # Every reference becomes a list, because an image could have more than one reference caption.
    refs = [[r] if isinstance(r, str) else list(r) for r in references]
    # Cut references and predictions into words with the same tokenizer.
    refs_tok = [[tokenize(r) for r in rs] for rs in refs]
    hyps_tok = [tokenize(h) for h in hypotheses]

    # BLEU idea: how many of the predicted words / phrases (n-grams) also appear in the reference, with a
    # penalty for predictions that are too short. BLEU-1 counts single words, BLEU-4 counts 4-word phrases, so
    # BLEU-4 is always much lower, especially for 4-5 word captions.
    smooth = SmoothingFunction().method1  # avoids 0 BLEU-4 when a short corpus has no 4-gram overlap
    # BLEU-n gives equal weight to the 1..n-grams, for example BLEU-2 = half unigrams + half bigrams.
    weights = {"BLEU-1": (1, 0, 0, 0), "BLEU-2": (.5, .5, 0, 0), "BLEU-3": (1 / 3,) * 3 + (0,), "BLEU-4": (.25,) * 4}
    # Score every BLEU variant on the whole set. x100 is the usual way to report BLEU.
    scores = {k: 100 * corpus_bleu(refs_tok, hyps_tok, weights=w, smoothing_function=smooth)
              for k, w in weights.items()}
    # CIDEr is added to the same dictionary.
    scores["CIDEr"] = 100 * cider_d(refs_tok, hyps_tok)
    return {k: round(v, 2) for k, v in scores.items()}


# ---------------------------------------------------------------- generating predictions
# Runs a trained checkpoint on a split and returns one predicted Hindi caption for every image (beam search).
def predict_with_model(ckpt, split, data_root, beam, batch_size, device):
    import torch
    from dataset import get_dataloader
    from train import load_checkpoint

    model, vocab = load_checkpoint(ckpt, device)
    loader = get_dataloader(split, vocab, data_root, batch_size, shuffle=False, num_workers=2)
    preds = []
    # Go through the split in batches (the real captions are not needed here).
    for images, _, _ in loader:
        # Generate captions with beam search. Beam search keeps the k best partial sentences at each step,
        # which gives better captions than always taking the single best word.
        ids, _ = model.caption(images.to(device), vocab.start_id, vocab.end_id, beam_size=beam)
        # Turn the predicted ids back into Hindi text.
        preds += [vocab.decode(i) for i in ids]
        print(f"  {len(preds)}/{len(loader.dataset)}", end="\r", flush=True)
    print()
    return preds


# Command-line entry: get predictions (from a checkpoint or from a CSV), score them, and save the scores and
# predictions.
def main():
    ap = argparse.ArgumentParser(description="Score Hindi captions (BLEU-1..4, CIDEr)")
    ap.add_argument("--model", required=True, help="name of the approach: baseline / transformer / translation / ...")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--ckpt", default=None, help="checkpoint from train.py (for baseline / transformer)")
    ap.add_argument("--pred_csv", default=None, help="CSV (file_name, prediction) from another approach")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    name = args.name or args.model

    # The real (reference) captions of the chosen split.
    ref_df = pd.read_csv(os.path.join(args.data_root, args.split, "captions.csv"))

    # Case 1: predictions already exist in a CSV (for example from the translation pipeline). Match them to
    # the references by file_name.
    if args.pred_csv:
        pred_df = pd.read_csv(args.pred_csv)
        # A left join keeps the order of the references. Images without a prediction get NaN and are scored as
        # an empty caption.
        merged = ref_df.merge(pred_df[["file_name", "prediction"]], on="file_name", how="left")
        missing = merged["prediction"].isna().sum()
        if missing:
            print(f"[warn] {missing} images have no prediction; they are scored as empty captions")
        preds = merged["prediction"].fillna("").astype(str).tolist()
    # Case 2: generate the predictions with a trained checkpoint.
    elif args.ckpt:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        preds = predict_with_model(args.ckpt, args.split, args.data_root, args.beam, args.batch_size, device)
    else:
        ap.error("give either --ckpt (trained model) or --pred_csv (predictions from another approach)")

    # Score the predictions, print the result and save it.
    refs = ref_df["hindi_caption"].astype(str).tolist()
    scores = compute_scores(refs, preds)
    print(f"\n{name} on {args.split} ({len(refs)} images):")
    for k, v in scores.items():
        print(f"  {k:7s} {v:6.2f}")

    os.makedirs("results", exist_ok=True)
    # Save the scores as JSON (compare_results.py reads these files).
    with open(f"results/{name}_{args.split}_scores.json", "w", encoding="utf-8") as f:
        json.dump({"model": name, "split": args.split, "n": len(refs), "scores": scores}, f, indent=1)
    # Save prediction and reference side by side, useful for human evaluation.
    pd.DataFrame({"file_name": ref_df["file_name"], "reference": refs, "prediction": preds}).to_csv(
        f"results/{name}_{args.split}_preds.csv", index=False, encoding="utf-8-sig")
    print(f"saved results/{name}_{args.split}_scores.json and _preds.csv")


if __name__ == "__main__":
    main()
