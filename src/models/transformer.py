"""
transformer.py -- ResNet-50 encoder + Transformer decoder captioner (drop-in replacement for baseline.py).

Classes:
    DecoderLayer   pre-norm Transformer decoder layer (masked self-attn, cross-attn over the image grid, FFN).
                   Written by hand (not nn.TransformerDecoderLayer) so the cross-attention weights can be returned
                   for the attention-map figures.
    CaptionModel   Encoder (shared with baseline: ResNet-50 -> 49 grid vectors) -> linear projection to d_model
                   + learned grid position embeddings -> N decoder layers -> vocab logits.

Same interface as models/baseline.py, so train.py / evaluate.py work unchanged:
    python src/train.py --model transformer --epochs 15 --batch_size 64 --lr 2e-4
    python src/evaluate.py --model transformer --split test --ckpt checkpoints/transformer_best.pth --beam 3

    model = CaptionModel(vocab_size=len(vocab))
    logits, alphas = model(images, captions)    # logits (B, T-1, V); alphas is None (no attention penalty needed)
    ids, maps = model.caption(images, start_id, end_id, beam_size=3)    # maps[i]: (num_words, 49)
"""

# math.sqrt is used to scale the word embeddings. torch / nn / F are the usual PyTorch modules.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline import Encoder  # same ResNet-50 grid features as the baseline -> fair comparison


# One decoder layer lets every word (1) look at the earlier words of the caption, (2) look at the image, (3)
# think about it in a small neural network. Residual connections (x = x + ...) ADD the new information to the
# old one, so nothing is forgotten and training stays stable.
class DecoderLayer(nn.Module):
    # d_model = size of every vector, heads = number of attention heads, ff_dim = hidden size of the
    # feed-forward network.
    def __init__(self, d_model, heads, ff_dim, dropout):
        super().__init__()
        # Self-attention: the word asks 'which earlier words matter for me?' (for example 'red' matters for
        # 'plate'). The causal mask stops a word from peeking at future words, because at test time they do
        # not exist yet.
        self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        # Cross-attention: the word asks 'which part of the image matters for me?'. The 49 image cells are the
        # 'memory'. The weights say how much each cell is used, so they can be drawn as an attention map.
        self.cross_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        # Feed-forward network, applied to every position separately. It lets the model process what attention
        # has collected.
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(ff_dim, d_model))
        # LayerNorm rescales the numbers of a vector to a standard range. We apply it BEFORE each of the three
        # sub-blocks ('pre-norm' style), which trains more stably.
        self.n1, self.n2, self.n3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    # x = the words so far (B,T,d_model), memory = image features (B,49,d_model). Returns the new x and the
    # cross-attention weights (or None).
    def forward(self, x, memory, causal_mask, need_weights=False):
        # Step 1: masked self-attention over the words.
        h = self.n1(x)
        x = x + self.drop(self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)[0])
        # Step 2: cross-attention over the image cells (the weights are averaged over the heads, for
        # plotting).
        h = self.n2(x)
        a, w = self.cross_attn(h, memory, memory, need_weights=need_weights, average_attn_weights=True)
        x = x + self.drop(a)
        # Step 3: feed-forward network.
        x = x + self.drop(self.ff(self.n3(x)))
        return x, w  # w: (B, T, 49) or None


# Idea: the ResNet looks at the picture and gives 49 feature vectors (one per 7x7 cell). The Transformer
# decoder then writes the caption one word at a time, each time looking at the words written so far and at the
# picture. Same interface as the baseline, so train.py and evaluate.py work with both.
class CaptionModel(nn.Module):
    # Defaults: 3 layers, 8 heads, d_model 512, dropout 0.3 (the values used for the reported results).
    def __init__(self, vocab_size, d_model=512, heads=8, layers=3, ff_dim=1024, dropout=0.3,
                 max_len=64, fine_tune=False, pretrained=True):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d_model=d_model, heads=heads, layers=layers, ff_dim=ff_dim,
                           dropout=dropout, max_len=max_len, fine_tune=fine_tune)  # saved in checkpoints
        # Same encoder as the baseline, for a fair comparison.
        self.encoder = Encoder(fine_tune=fine_tune, pretrained=pretrained)
        self.d_model = d_model
        # Project the 2048-number ResNet features down to d_model, so they fit together with the word vectors.
        self.enc_proj = nn.Sequential(nn.Linear(self.encoder.out_dim, d_model), nn.LayerNorm(d_model))
        # A learned position for each of the 49 cells, so the model knows WHERE in the image a feature comes
        # from (top-left, centre, ...). The CNN features alone do not say this.
        self.grid_pos = nn.Parameter(torch.zeros(1, 49, d_model))  # which of the 7x7 cells a feature came from
        # Start these positions with small random values.
        nn.init.normal_(self.grid_pos, std=0.02)
        # Word embeddings turn word ids into vectors. A Transformer has no sense of order by itself, so we
        # also learn a position embedding for word 0, 1, 2, ...
        self.embed = nn.Embedding(vocab_size, d_model)
        self.word_pos = nn.Embedding(max_len, d_model)
        # The stack of decoder layers.
        self.layers = nn.ModuleList(DecoderLayer(d_model, heads, ff_dim, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        # Final layer: one score for every word of the vocabulary. The highest score = the most likely next
        # word.
        self.fc = nn.Linear(d_model, vocab_size)

    # BatchNorm layers keep running averages of the data they see. We do not want the frozen ResNet to change
    # them, so after switching to train mode we put the ResNet back into eval mode.
    def train(self, mode=True):
        super().train(mode)
        self.encoder.backbone.eval()  # keep ResNet BatchNorm statistics fixed (frozen or only layer4 trained)
        return self

    # ---------------------------------------------------------------- pieces
    # Images (B,3,224,224) -> image memory (B,49,d_model). The frozen ResNet gets no gradients unless
    # fine_tune is on.
    def encode(self, images):
        if self.config["fine_tune"]:
            feats = self.encoder(images)
        else:
            with torch.no_grad():
                feats = self.encoder(images)
        return self.enc_proj(feats) + self.grid_pos  # (B, 49, d_model)

    def decode(self, tokens, memory, need_weights=False):
        """tokens (B, T) -> logits (B, T, V) and last-layer cross-attention (B, T, 49) or None."""
        # T = number of words so far. pos = 0, 1, 2, ... = the position of each word.
        T = tokens.size(1)
        pos = torch.arange(T, device=tokens.device)
        # Word vector + position vector. Multiplying the word vectors by sqrt(d_model) keeps them on a similar
        # scale as the position vectors (as in the original Transformer paper).
        x = self.drop(self.embed(tokens) * math.sqrt(self.d_model) + self.word_pos(pos))
        # Causal mask: True above the diagonal means word t may not look at the words after it.
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tokens.device), diagonal=1)
        w = None
        # Run all layers. Attention weights are only asked from the last layer (used for the figures).
        for i, layer in enumerate(self.layers):
            x, w = layer(x, memory, causal, need_weights=need_weights and i == len(self.layers) - 1)
        return self.fc(self.norm(x)), w

    # ---------------------------------------------------------------- training
    def forward(self, images, captions):
        """Teacher forcing. captions (B, T) start with <start>; returns logits for tokens 1..T-1.
        Padding sits at the end of each caption and the mask is causal, so real tokens never see pads."""
        # Teacher forcing: the real caption (without its last token) is the input, and the model predicts the
        # next word at EVERY position in one go. The targets (caption shifted by one) are made in train.py.
        logits, _ = self.decode(captions[:, :-1], self.encode(images))
        return logits, None

    # ---------------------------------------------------------------- inference
    # Generate captions for a batch: encode all images once, then run beam search for each image on its own.
    @torch.no_grad()
    def caption(self, images, start_id, end_id, max_len=40, beam_size=1):
        """Returns (list of id-lists without <start>/<end>, list of (len, 49) attention maps)."""
        # Eval mode switches dropout off, so the captions are not random.
        self.eval()
        # Look at all images once.
        memory = self.encode(images)
        results, maps = [], []
        # Decode every image separately.
        for i in range(memory.size(0)):
            ids, al = self._beam(memory[i:i + 1], start_id, end_id, min(max_len, self.config["max_len"] - 1),
                                 beam_size)
            results.append(ids)
            maps.append(al)
        return results, maps

    # Beam search idea: always taking the single best word can lead to a bad sentence. Instead we keep the k
    # best partial sentences at every step and finally return the best finished one.
    def _beam(self, memory, start_id, end_id, max_len, k):
        """Beam search for ONE image (memory: (1,49,d)). k=1 is greedy. No KV cache: the prefix is re-run each
        step, which is cheap for ~5-word captions."""
        dev = memory.device
        # One copy of the image memory for each beam.
        memory = memory.expand(k, -1, -1)
        # seqs = the k captions being built (they start with <start>). scores = their log-probabilities.
        seqs = torch.full((k, 1), start_id, dtype=torch.long, device=dev)
        scores = torch.zeros(k, device=dev)
        alphas = [[] for _ in range(k)]
        done = []  # (length-normalised score, ids, alphas)
        V = self.config["vocab_size"]
        # Each step: run the decoder on the words so far, take the log-probabilities of the next word and keep
        # the best k candidates.
        for step in range(max_len):
            logits, w = self.decode(seqs, memory, need_weights=True)
            # Log-probabilities of the next word (only the last position is needed). Adding log-probabilities
            # is the same as multiplying the probabilities of all words of the sentence.
            logp = F.log_softmax(logits[:, -1], dim=1)
            # At step 0 all beams are identical, so only the first one is used. Later: new score = old score +
            # log-probability of the new word, for every (beam, word) pair.
            cand = logp[0] if step == 0 else (scores.unsqueeze(1) + logp).reshape(-1)
            # Keep the k best candidates.
            top, idx = cand.topk(k)
            # idx counts over (beam x vocabulary). Integer division gives the beam, the remainder gives the
            # word.
            beam_idx, word = idx // V, idx % V
            # Extend the chosen beams with their new word.
            seqs = torch.cat([seqs[beam_idx], word.unsqueeze(1)], dim=1)
            # Remember the cross-attention of the newest word (used for the attention figures).
            alphas = [alphas[b] + [w[b, -1].cpu()] for b in beam_idx.tolist()]
            scores = top
            # Beams that just produced <end> are finished: store them (score divided by length) and switch
            # them off.
            for f in (word == end_id).nonzero(as_tuple=True)[0].tolist():
                done.append((scores[f].item() / seqs.size(1), seqs[f, 1:-1].tolist(), alphas[f]))
                scores[f] = -1e9  # kill this beam
            # Stop when k sentences are finished or all beams are dead.
            if len(done) >= k or all(s < -1e8 for s in scores.tolist()):
                break
        # If no beam finished within max_len, fall back to the unfinished beams.
        if not done:
            done = [(scores[b].item() / seqs.size(1), seqs[b, 1:].tolist(), alphas[b]) for b in range(k)]
        # Pick the finished caption with the best length-normalised score.
        _, ids, al = max(done, key=lambda x: x[0])
        return ids, (torch.stack(al) if al else torch.zeros(0, memory.size(1)))
