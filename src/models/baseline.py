"""
baseline.py -- "Show, Attend and Tell" captioner: ResNet-50 encoder + LSTM decoder with soft attention.

Classes:
    Encoder        ResNet-50 (ImageNet-pretrained) -> grid of 7x7=49 feature vectors (B, 49, 2048)
    Attention      additive (Bahdanau) attention over the 49 regions
    Decoder        LSTMCell decoder that attends at every step
    CaptionModel   Encoder + Decoder, with forward() for training and caption() for inference

Not meant to be run directly. Used by train.py / evaluate.py:
    from models.baseline import CaptionModel
    model = CaptionModel(vocab_size=len(vocab))
    logits, alphas = model(images, captions)          # training (teacher forcing)
    ids, alphas = model.caption(images, start_id, end_id, beam_size=3)   # inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Encoder(nn.Module):
    def __init__(self, fine_tune=False, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = torchvision.models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # drop avgpool + fc
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.out_dim = 2048
        self.set_fine_tune(fine_tune)

    def set_fine_tune(self, fine_tune):
        """fine_tune=False freezes the CNN; True unfreezes only the last block (layer4)."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        if fine_tune:
            for p in list(self.backbone.children())[-1].parameters():  # layer4
                p.requires_grad = True

    def forward(self, images):
        feats = self.pool(self.backbone(images))             # (B, 2048, 7, 7)
        return feats.flatten(2).permute(0, 2, 1)             # (B, 49, 2048)


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, att_dim):
        super().__init__()
        self.enc_att = nn.Linear(enc_dim, att_dim)
        self.dec_att = nn.Linear(dec_dim, att_dim)
        self.full_att = nn.Linear(att_dim, 1)

    def forward(self, feats, h):
        e = self.full_att(torch.tanh(self.enc_att(feats) + self.dec_att(h).unsqueeze(1))).squeeze(2)
        alpha = F.softmax(e, dim=1)                          # (B, 49)
        context = (feats * alpha.unsqueeze(2)).sum(dim=1)    # (B, enc_dim)
        return context, alpha


class Decoder(nn.Module):
    def __init__(self, vocab_size, enc_dim=2048, emb_dim=256, dec_dim=512, att_dim=256, dropout=0.5):
        super().__init__()
        self.vocab_size, self.dec_dim = vocab_size, dec_dim
        self.attention = Attention(enc_dim, dec_dim, att_dim)
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTMCell(emb_dim + enc_dim, dec_dim)
        self.init_h = nn.Linear(enc_dim, dec_dim)
        self.init_c = nn.Linear(enc_dim, dec_dim)
        self.f_beta = nn.Linear(dec_dim, enc_dim)            # gate on the context vector
        self.fc = nn.Linear(dec_dim, vocab_size)

    def init_state(self, feats):
        mean = feats.mean(dim=1)
        return self.init_h(mean), self.init_c(mean)

    def step(self, word, feats, h, c):
        context, alpha = self.attention(feats, h)
        context = torch.sigmoid(self.f_beta(h)) * context
        h, c = self.lstm(torch.cat([self.embed(word), context], dim=1), (h, c))
        return self.fc(self.dropout(h)), h, c, alpha

    def forward(self, feats, captions):
        """Teacher forcing. captions (B, T) start with <start>; predicts tokens 1..T-1."""
        B, T = captions.shape
        h, c = self.init_state(feats)
        logits = feats.new_zeros(B, T - 1, self.vocab_size)
        alphas = feats.new_zeros(B, T - 1, feats.size(1))
        for t in range(T - 1):
            logits[:, t], h, c, alphas[:, t] = self.step(captions[:, t], feats, h, c)
        return logits, alphas


class CaptionModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, dec_dim=512, att_dim=256, dropout=0.5, fine_tune=False,
                 pretrained=True):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, emb_dim=emb_dim, dec_dim=dec_dim, att_dim=att_dim,
                           dropout=dropout, fine_tune=fine_tune)  # saved in checkpoints
        self.encoder = Encoder(fine_tune=fine_tune, pretrained=pretrained)
        self.decoder = Decoder(vocab_size, self.encoder.out_dim, emb_dim, dec_dim, att_dim, dropout)

    def forward(self, images, captions):
        return self.decoder(self.encoder(images), captions)

    @torch.no_grad()
    def caption(self, images, start_id, end_id, max_len=40, beam_size=1):
        """Generate captions. Returns (list of id-lists without <start>/<end>, list of (len, 49) attention maps)."""
        self.eval()
        feats = self.encoder(images)
        results, maps = [], []
        for i in range(feats.size(0)):
            ids, alphas = self._beam(feats[i:i + 1], start_id, end_id, max_len, beam_size)
            results.append(ids)
            maps.append(alphas)
        return results, maps

    def _beam(self, feats, start_id, end_id, max_len, k):
        """Beam search for ONE image (feats: (1,49,2048)). k=1 is greedy decoding."""
        dev = feats.device
        feats = feats.expand(k, -1, -1)
        h, c = self.decoder.init_state(feats)
        seqs = torch.full((k, 1), start_id, dtype=torch.long, device=dev)
        scores = torch.zeros(k, device=dev)
        alphas = [[] for _ in range(k)]
        done = []  # (score, ids, alphas)
        for step in range(max_len):
            logits, h, c, a = self.decoder.step(seqs[:, -1], feats, h, c)
            logp = F.log_softmax(logits, dim=1)
            if step == 0:  # all k beams are identical at the start, so expand only the first
                cand = logp[0]
            else:
                cand = (scores.unsqueeze(1) + logp).reshape(-1)
            top, idx = cand.topk(k)
            beam_idx, word = idx // self.decoder.vocab_size, idx % self.decoder.vocab_size
            seqs = torch.cat([seqs[beam_idx], word.unsqueeze(1)], dim=1)
            alphas = [alphas[b] + [a[b].cpu()] for b in beam_idx.tolist()]
            h, c, scores = h[beam_idx], c[beam_idx], top
            finished = (word == end_id).nonzero(as_tuple=True)[0].tolist()
            for f in finished:
                done.append((scores[f].item() / seqs.size(1), seqs[f, 1:-1].tolist(), alphas[f]))
                scores[f] = -1e9  # kill this beam
            if len(done) >= k or all(s < -1e8 for s in scores.tolist()):
                break
        if not done:  # nothing finished within max_len
            done = [(scores[b].item() / seqs.size(1), seqs[b, 1:].tolist(), alphas[b]) for b in range(k)]
        _, ids, al = max(done, key=lambda x: x[0])
        return ids, (torch.stack(al) if al else torch.zeros(0, feats.size(1)))
