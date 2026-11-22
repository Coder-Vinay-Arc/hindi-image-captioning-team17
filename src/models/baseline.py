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

# torch = tensors and autograd, nn = layers, F = functions such as softmax, torchvision = the pretrained
# ResNet-50.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# Encoder = the 'eyes' of the model. A CNN looks at the picture and describes it with numbers: 49 feature
# vectors (a 7x7 grid of cells), each with 2048 numbers.
class Encoder(nn.Module):
    def __init__(self, fine_tune=False, pretrained=True):
        super().__init__()
        # Load ImageNet pretrained weights. pretrained=False is used when a trained checkpoint is loaded
        # afterwards.
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        # ResNet-50 is a CNN trained on ImageNet. It already knows edges, textures and objects, so we reuse it
        # instead of learning to see from scratch.
        resnet = torchvision.models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # drop avgpool + fc
        # Adaptive average pooling forces the feature map to exactly 7x7 cells, whatever the input size.
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

    # (B,3,224,224) -> (B,2048,7,7) -> flatten to (B,49,2048): 49 image cells, each described by 2048 numbers.
    def forward(self, images):
        feats = self.pool(self.backbone(images))             # (B, 2048, 7, 7)
        return feats.flatten(2).permute(0, 2, 1)             # (B, 49, 2048)


# Attention = a weighted average. The model scores each of the 49 regions, softmax turns the scores into
# weights that add up to 1, and the context vector is the weighted sum of the region features. So the model
# 'looks' mostly at the regions with a high weight.
class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, att_dim):
        super().__init__()
        self.enc_att = nn.Linear(enc_dim, att_dim)
        self.dec_att = nn.Linear(dec_dim, att_dim)
        self.full_att = nn.Linear(att_dim, 1)

    # feats (B,49,enc_dim), h (B,dec_dim) -> context (B,enc_dim) and alpha (B,49), the attention weights.
    def forward(self, feats, h):
        # Score of every region = a small neural network applied to (region feature + what the decoder
        # remembers so far).
        e = self.full_att(torch.tanh(self.enc_att(feats) + self.dec_att(h).unsqueeze(1))).squeeze(2)
        # softmax turns the scores into weights between 0 and 1 that sum to 1.
        alpha = F.softmax(e, dim=1)                          # (B, 49)
        # Weighted sum of the region features = what the model is looking at right now.
        context = (feats * alpha.unsqueeze(2)).sum(dim=1)    # (B, enc_dim)
        return context, alpha


# Decoder = the 'writer'. An LSTM writes the caption word by word. At every step it first looks at the image
# (attention), then predicts the next word.
class Decoder(nn.Module):
    def __init__(self, vocab_size, enc_dim=2048, emb_dim=256, dec_dim=512, att_dim=256, dropout=0.5):
        super().__init__()
        self.vocab_size, self.dec_dim = vocab_size, dec_dim
        # The module that decides where to look in the image.
        self.attention = Attention(enc_dim, dec_dim, att_dim)
        # An embedding is a learnable table that turns a word id into a vector of numbers.
        self.embed = nn.Embedding(vocab_size, emb_dim)
        # Dropout randomly switches off some values during training. It forces the model not to rely on single
        # values and reduces overfitting.
        self.dropout = nn.Dropout(dropout)
        # The memory of the decoder. At each step it reads the previous word and the image context, and passes
        # its memory (h, c) to the next step.
        self.lstm = nn.LSTMCell(emb_dim + enc_dim, dec_dim)
        # These two layers create the first memory (h, c) from the image, so the writer knows what the picture
        # shows before the first word.
        self.init_h = nn.Linear(enc_dim, dec_dim)
        self.init_c = nn.Linear(enc_dim, dec_dim)
        # Gate layer: gives a value between 0 and 1 that says how much of the image context to use at this
        # step.
        self.f_beta = nn.Linear(dec_dim, enc_dim)            # gate on the context vector
        # Final layer: one score for every word of the vocabulary.
        self.fc = nn.Linear(dec_dim, vocab_size)

    # Starting hidden / cell state of the LSTM, computed from the average of all image features.
    def init_state(self, feats):
        # Average over the 49 cells = one summary vector of the whole image.
        mean = feats.mean(dim=1)
        return self.init_h(mean), self.init_c(mean)

    # One decoding step = the previous word goes in, scores for the next word come out.
    def step(self, word, feats, h, c):
        # 1. Look at the image: attention gives a context vector and the weights alpha.
        context, alpha = self.attention(feats, h)
        # 2. Gate: a value between 0 and 1 decides how much of the context is used (for example little for
        # small words like 'a' or 'on').
        context = torch.sigmoid(self.f_beta(h)) * context
        # 3. LSTM update: the input is the previous word + the context. h = new short-term output, c =
        # long-term memory.
        h, c = self.lstm(torch.cat([self.embed(word), context], dim=1), (h, c))
        # 4. Dropout (only in training), then one score for every vocabulary word.
        return self.fc(self.dropout(h)), h, c, alpha

    # Teacher forcing: at every step the REAL previous word is the input (not the model's own guess). This
    # makes training faster and more stable.
    def forward(self, feats, captions):
        """Teacher forcing. captions (B, T) start with <start>; predicts tokens 1..T-1."""
        B, T = captions.shape
        # Start the memory from the image.
        h, c = self.init_state(feats)
        # Empty tensors that will be filled step by step: scores (B,T-1,vocab) and attention weights
        # (B,T-1,49).
        logits = feats.new_zeros(B, T - 1, self.vocab_size)
        alphas = feats.new_zeros(B, T - 1, feats.size(1))
        # One word per step. The input is the real word at position t, and the target is the word at position
        # t+1.
        for t in range(T - 1):
            logits[:, t], h, c, alphas[:, t] = self.step(captions[:, t], feats, h, c)
        return logits, alphas


# Full model = Encoder (looks) + Decoder (writes). This is the classic 'Show, Attend and Tell' design.
# forward() is used for training and caption() for generating captions.
class CaptionModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, dec_dim=512, att_dim=256, dropout=0.5, fine_tune=False,
                 pretrained=True):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, emb_dim=emb_dim, dec_dim=dec_dim, att_dim=att_dim,
                           dropout=dropout, fine_tune=fine_tune)  # saved in checkpoints
        # Build the two parts of the model.
        self.encoder = Encoder(fine_tune=fine_tune, pretrained=pretrained)
        self.decoder = Decoder(vocab_size, self.encoder.out_dim, emb_dim, dec_dim, att_dim, dropout)

    def forward(self, images, captions):
        return self.decoder(self.encoder(images), captions)

    # Inference: generate captions for a batch of images (no gradients needed). Each image is decoded on its
    # own with beam search.
    @torch.no_grad()
    def caption(self, images, start_id, end_id, max_len=40, beam_size=1):
        """Generate captions. Returns (list of id-lists without <start>/<end>, list of (len, 49) attention maps)."""
        self.eval()
        # Run the CNN once for the whole batch.
        feats = self.encoder(images)
        results, maps = [], []
        for i in range(feats.size(0)):
            ids, alphas = self._beam(feats[i:i + 1], start_id, end_id, max_len, beam_size)
            results.append(ids)
            maps.append(alphas)
        return results, maps

    # Beam search idea: always taking the single best word can lead to a bad sentence. Instead we keep the k
    # best partial sentences at every step and finally return the best finished one.
    def _beam(self, feats, start_id, end_id, max_len, k):
        """Beam search for ONE image (feats: (1,49,2048)). k=1 is greedy decoding."""
        dev = feats.device
        # Copy the image features k times, one copy for each beam.
        feats = feats.expand(k, -1, -1)
        h, c = self.decoder.init_state(feats)
        # seqs = the k partial captions so far (all start with <start>). scores = their log-probabilities.
        seqs = torch.full((k, 1), start_id, dtype=torch.long, device=dev)
        scores = torch.zeros(k, device=dev)
        alphas = [[] for _ in range(k)]
        done = []  # (score, ids, alphas)
        # Each step: score all possible next words for every beam, then keep the best k (caption, word)
        # combinations.
        for step in range(max_len):
            # Feed the last word of every beam and get the scores for the next word.
            logits, h, c, a = self.decoder.step(seqs[:, -1], feats, h, c)
            # Log-probabilities. Adding log-probabilities is the same as multiplying the probabilities of all
            # words of the sentence.
            logp = F.log_softmax(logits, dim=1)
            if step == 0:  # all k beams are identical at the start, so expand only the first
                cand = logp[0]
            else:
                cand = (scores.unsqueeze(1) + logp).reshape(-1)
            # Keep the k best candidates.
            top, idx = cand.topk(k)
            # idx counts over (beam x vocabulary). Integer division gives the beam, the remainder gives the
            # word.
            beam_idx, word = idx // self.decoder.vocab_size, idx % self.decoder.vocab_size
            # Extend the chosen beams with their new word.
            seqs = torch.cat([seqs[beam_idx], word.unsqueeze(1)], dim=1)
            # Remember the attention of the newest word (used for the attention figures).
            alphas = [alphas[b] + [a[b].cpu()] for b in beam_idx.tolist()]
            # Keep the LSTM memory of the beams that survived.
            h, c, scores = h[beam_idx], c[beam_idx], top
            # A beam that produced <end> is finished: save it (score divided by its length) and switch that
            # beam off.
            finished = (word == end_id).nonzero(as_tuple=True)[0].tolist()
            for f in finished:
                done.append((scores[f].item() / seqs.size(1), seqs[f, 1:-1].tolist(), alphas[f]))
                scores[f] = -1e9  # kill this beam
            # Stop when k sentences are finished or all beams are dead.
            if len(done) >= k or all(s < -1e8 for s in scores.tolist()):
                break
        if not done:  # nothing finished within max_len
            done = [(scores[b].item() / seqs.size(1), seqs[b, 1:].tolist(), alphas[b]) for b in range(k)]
        # Return the finished caption with the best length-normalised score.
        _, ids, al = max(done, key=lambda x: x[0])
        return ids, (torch.stack(al) if al else torch.zeros(0, feats.size(1)))
