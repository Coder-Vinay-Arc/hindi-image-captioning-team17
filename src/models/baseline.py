"""
baseline.py -- "Show, Attend and Tell" (Xu et al., 2015) Hindi captioner:
ResNet-50 encoder + LSTM decoder with soft (additive) attention.

Architecture
    Encoder       ImageNet-pretrained ResNet-50 without avgpool/fc -> 7x7 grid of 2048-d
                  features, flattened to 49 "regions" of shape (B, 49, 2048).
                  Frozen by default; fine_tune=True unfreezes only the last block (layer4).
    Attention     Bahdanau attention: e_i = w^T tanh(W_enc f_i + W_dec h), alpha = softmax(e),
                  context = sum_i alpha_i f_i
    Decoder       LSTMCell. Input at step t = [embedding(word_{t-1}) ; gated context].
                  Initial (h, c) come from the mean image feature.
    CaptionModel  Encoder + Decoder.
                    forward(images, captions) -> (logits, alphas)   training, teacher forcing
                    caption(images, start_id, end_id, beam_size)    inference (greedy / beam)

Not meant to be run directly. Used through src/models/__init__.py:
    from models import build_model, load_checkpoint
    model = build_model("baseline", vocab_size=len(vocab))
    logits, alphas = model(images, captions)                    # (B, T-1, V), (B, T-1, 49)
    ids, alphas = model.caption(images, vocab.start_id, vocab.end_id, beam_size=3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Encoder(nn.Module):
    def __init__(self, fine_tune=False, pretrained=True, grid=7):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = torchvision.models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # drop avgpool + fc
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.out_dim = 2048
        self.fine_tune = fine_tune
        for p in self.backbone.parameters():
            p.requires_grad = False
        if fine_tune:
            for p in self.backbone[-1].parameters():  # layer4
                p.requires_grad = True

    def train(self, mode=True):
        # Keep the pretrained BatchNorm statistics fixed: with a frozen (or mostly frozen) CNN,
        # letting BN update its running stats on our small crops would silently change the features.
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images):
        with torch.set_grad_enabled(self.fine_tune and torch.is_grad_enabled()):
            feats = self.pool(self.backbone(images))       # (B, 2048, 7, 7)
        return feats.flatten(2).transpose(1, 2)            # (B, 49, 2048)


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, att_dim):
        super().__init__()
        self.enc_att = nn.Linear(enc_dim, att_dim)
        self.dec_att = nn.Linear(dec_dim, att_dim)
        self.full_att = nn.Linear(att_dim, 1)

    def forward(self, feats, h):
        e = self.full_att(torch.tanh(self.enc_att(feats) + self.dec_att(h).unsqueeze(1))).squeeze(2)
        alpha = F.softmax(e, dim=1)                        # (B, 49)
        context = (feats * alpha.unsqueeze(2)).sum(dim=1)  # (B, enc_dim)
        return context, alpha


class Decoder(nn.Module):
    def __init__(self, vocab_size, enc_dim=2048, emb_dim=256, dec_dim=512, att_dim=256, dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention = Attention(enc_dim, dec_dim, att_dim)
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTMCell(emb_dim + enc_dim, dec_dim)
        self.init_h = nn.Linear(enc_dim, dec_dim)
        self.init_c = nn.Linear(enc_dim, dec_dim)
        self.f_beta = nn.Linear(dec_dim, enc_dim)          # gating scalar on the context (paper sec. 4.2.1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dec_dim, vocab_size)

    def init_state(self, feats):
        mean = feats.mean(dim=1)
        return torch.tanh(self.init_h(mean)), torch.tanh(self.init_c(mean))

    def step(self, words, feats, h, c):
        """One decoding step. words: (B,) previous token ids."""
        context, alpha = self.attention(feats, h)
        context = torch.sigmoid(self.f_beta(h)) * context
        h, c = self.lstm(torch.cat([self.embed(words), context], dim=1), (h, c))
        return self.fc(self.dropout(h)), h, c, alpha

    def forward(self, feats, captions):
        """Teacher forcing. captions (B, T) begin with <start>; predicts tokens 1..T-1."""
        h, c = self.init_state(feats)
        logits, alphas = [], []
        for t in range(captions.size(1) - 1):
            out, h, c, alpha = self.step(captions[:, t], feats, h, c)
            logits.append(out)
            alphas.append(alpha)
        return torch.stack(logits, dim=1), torch.stack(alphas, dim=1)


class CaptionModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, dec_dim=512, att_dim=256, dropout=0.5,
                 fine_tune=False, pretrained=True):
        super().__init__()
        # everything needed to rebuild the model from a checkpoint
        self.config = dict(vocab_size=vocab_size, emb_dim=emb_dim, dec_dim=dec_dim,
                           att_dim=att_dim, dropout=dropout, fine_tune=fine_tune)
        self.encoder = Encoder(fine_tune=fine_tune, pretrained=pretrained)
        self.decoder = Decoder(vocab_size, self.encoder.out_dim, emb_dim, dec_dim, att_dim, dropout)

    def forward(self, images, captions):
        return self.decoder(self.encoder(images), captions)

    @torch.no_grad()
    def caption(self, images, start_id, end_id, max_len=30, beam_size=1):
        """Generate captions for a batch of images.

        Returns (ids, alphas): ids[i] is a list of token ids (no <start>/<end>),
        alphas[i] is a (len(ids[i]), 49) tensor of attention weights, one row per word.
        """
        was_training = self.training
        self.eval()
        feats = self.encoder(images)
        if beam_size <= 1:
            out = self._greedy(feats, start_id, end_id, max_len)
        else:
            out = [self._beam(feats[i:i + 1], start_id, end_id, max_len, beam_size)
                   for i in range(feats.size(0))]
            out = ([o[0] for o in out], [o[1] for o in out])
        self.train(was_training)
        return out

    def _greedy(self, feats, start_id, end_id, max_len):
        B = feats.size(0)
        h, c = self.decoder.init_state(feats)
        words = feats.new_full((B,), start_id, dtype=torch.long)
        seq, alphas = [], []
        done = torch.zeros(B, dtype=torch.bool, device=feats.device)
        for _ in range(max_len):
            logits, h, c, alpha = self.decoder.step(words, feats, h, c)
            words = logits.argmax(dim=1)
            seq.append(words)
            alphas.append(alpha)
            done |= words == end_id
            if done.all():
                break
        seq, alphas = torch.stack(seq, 1).tolist(), torch.stack(alphas, 1).cpu()
        ids, maps = [], []
        for i, s in enumerate(seq):
            n = s.index(end_id) if end_id in s else len(s)
            ids.append(s[:n])
            maps.append(alphas[i, :n])
        return ids, maps

    def _beam(self, feats, start_id, end_id, max_len, k):
        """Beam search for ONE image. feats: (1, 49, D)."""
        V = self.decoder.vocab_size
        feats = feats.expand(k, -1, -1)
        h, c = self.decoder.init_state(feats)
        words = feats.new_full((k,), start_id, dtype=torch.long)
        scores = feats.new_full((k,), float("-inf"))
        scores[0] = 0.0                                    # all beams start identical; keep one alive
        seqs = words.new_zeros((k, 0))
        maps = feats.new_zeros((k, 0, feats.size(1)))
        finished = []                                      # (normalised score, ids, alphas)
        for _ in range(max_len):
            logits, h, c, alpha = self.decoder.step(words, feats, h, c)
            total = (F.log_softmax(logits, dim=1) + scores.unsqueeze(1)).view(-1)
            top, idx = total.topk(k)
            beam, words = idx // V, idx % V
            seqs = torch.cat([seqs[beam], words.unsqueeze(1)], dim=1)
            maps = torch.cat([maps[beam], alpha[beam].unsqueeze(1)], dim=1)
            h, c, scores = h[beam], c[beam], top.clone()
            for j in range(k):
                if words[j] == end_id and torch.isfinite(scores[j]):
                    n = seqs.size(1) - 1                   # length without <end>
                    finished.append((scores[j].item() / (n + 1), seqs[j, :n].tolist(), maps[j, :n].cpu()))
                    scores[j] = float("-inf")
            if len(finished) >= k or not torch.isfinite(scores).any():
                break
        if not finished:                                   # nothing hit <end> within max_len
            j = scores.argmax()
            finished.append((0.0, seqs[j].tolist(), maps[j].cpu()))
        _, ids, alphas = max(finished, key=lambda f: f[0])
        return ids, alphas
