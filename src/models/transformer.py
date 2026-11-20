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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline import Encoder  # same ResNet-50 grid features as the baseline -> fair comparison


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(ff_dim, d_model))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, causal_mask, need_weights=False):
        h = self.n1(x)
        x = x + self.drop(self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)[0])
        h = self.n2(x)
        a, w = self.cross_attn(h, memory, memory, need_weights=need_weights, average_attn_weights=True)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.n3(x)))
        return x, w  # w: (B, T, 49) or None


class CaptionModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, heads=8, layers=3, ff_dim=1024, dropout=0.3,
                 max_len=64, fine_tune=False, pretrained=True):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d_model=d_model, heads=heads, layers=layers, ff_dim=ff_dim,
                           dropout=dropout, max_len=max_len, fine_tune=fine_tune)  # saved in checkpoints
        self.encoder = Encoder(fine_tune=fine_tune, pretrained=pretrained)
        self.d_model = d_model
        self.enc_proj = nn.Sequential(nn.Linear(self.encoder.out_dim, d_model), nn.LayerNorm(d_model))
        self.grid_pos = nn.Parameter(torch.zeros(1, 49, d_model))  # which of the 7x7 cells a feature came from
        nn.init.normal_(self.grid_pos, std=0.02)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.word_pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList(DecoderLayer(d_model, heads, ff_dim, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.backbone.eval()  # keep ResNet BatchNorm statistics fixed (frozen or only layer4 trained)
        return self

    # ---------------------------------------------------------------- pieces
    def encode(self, images):
        if self.config["fine_tune"]:
            feats = self.encoder(images)
        else:
            with torch.no_grad():
                feats = self.encoder(images)
        return self.enc_proj(feats) + self.grid_pos  # (B, 49, d_model)

    def decode(self, tokens, memory, need_weights=False):
        """tokens (B, T) -> logits (B, T, V) and last-layer cross-attention (B, T, 49) or None."""
        T = tokens.size(1)
        pos = torch.arange(T, device=tokens.device)
        x = self.drop(self.embed(tokens) * math.sqrt(self.d_model) + self.word_pos(pos))
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tokens.device), diagonal=1)
        w = None
        for i, layer in enumerate(self.layers):
            x, w = layer(x, memory, causal, need_weights=need_weights and i == len(self.layers) - 1)
        return self.fc(self.norm(x)), w

    # ---------------------------------------------------------------- training
    def forward(self, images, captions):
        """Teacher forcing. captions (B, T) start with <start>; returns logits for tokens 1..T-1.
        Padding sits at the end of each caption and the mask is causal, so real tokens never see pads."""
        logits, _ = self.decode(captions[:, :-1], self.encode(images))
        return logits, None

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def caption(self, images, start_id, end_id, max_len=40, beam_size=1):
        """Returns (list of id-lists without <start>/<end>, list of (len, 49) attention maps)."""
        self.eval()
        memory = self.encode(images)
        results, maps = [], []
        for i in range(memory.size(0)):
            ids, al = self._beam(memory[i:i + 1], start_id, end_id, min(max_len, self.config["max_len"] - 1),
                                 beam_size)
            results.append(ids)
            maps.append(al)
        return results, maps

    def _beam(self, memory, start_id, end_id, max_len, k):
        """Beam search for ONE image (memory: (1,49,d)). k=1 is greedy. No KV cache: the prefix is re-run each
        step, which is cheap for ~5-word captions."""
        dev = memory.device
        memory = memory.expand(k, -1, -1)
        seqs = torch.full((k, 1), start_id, dtype=torch.long, device=dev)
        scores = torch.zeros(k, device=dev)
        alphas = [[] for _ in range(k)]
        done = []  # (length-normalised score, ids, alphas)
        V = self.config["vocab_size"]
        for step in range(max_len):
            logits, w = self.decode(seqs, memory, need_weights=True)
            logp = F.log_softmax(logits[:, -1], dim=1)
            cand = logp[0] if step == 0 else (scores.unsqueeze(1) + logp).reshape(-1)
            top, idx = cand.topk(k)
            beam_idx, word = idx // V, idx % V
            seqs = torch.cat([seqs[beam_idx], word.unsqueeze(1)], dim=1)
            alphas = [alphas[b] + [w[b, -1].cpu()] for b in beam_idx.tolist()]
            scores = top
            for f in (word == end_id).nonzero(as_tuple=True)[0].tolist():
                done.append((scores[f].item() / seqs.size(1), seqs[f, 1:-1].tolist(), alphas[f]))
                scores[f] = -1e9  # kill this beam
            if len(done) >= k or all(s < -1e8 for s in scores.tolist()):
                break
        if not done:
            done = [(scores[b].item() / seqs.size(1), seqs[b, 1:].tolist(), alphas[b]) for b in range(k)]
        _, ids, al = max(done, key=lambda x: x[0])
        return ids, (torch.stack(al) if al else torch.zeros(0, memory.size(1)))
