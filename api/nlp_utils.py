# This file was written automatically by the notebook (Section 10). Do not edit it by hand.
import html, json, re, unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK = 0, 1

def basic_cleaning(text):
    text = html.unescape(str(text))                                    # &amp; -> &
    text = unicodedata.normalize("NFKC", text)                         # \xa0 and look-alike characters -> one form
    text = re.sub(r"[\u200b\u2060\ufeff\u00ad]", "", text)             # invisible characters
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)                 # links
    text = re.sub(r"\S+@\S+\.\S+", " ", text)                          # e-mail addresses
    text = text.lower()                                                # lower-case
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    """Split into words (letters, digits and apostrophes)."""
    return re.findall(r"[\w']+", str(text).lower())

class TextCNN(nn.Module):
    """Words -> embeddings -> window detectors -> strongest match -> classifier."""
    def __init__(self, vocab_size, emb_dim, kernel_sizes, n_filters, dropout, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)     # word number -> vector
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)                                       # randomly switches off units (less overfitting)
        self.classifier = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)                                    # (batch, emb_dim, length)
        features = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]    # strongest match per detector
        return self.classifier(self.dropout(torch.cat(features, dim=1)))

class BiLSTMAttention(nn.Module):
    """Words -> embeddings -> BiLSTM -> attention (weighted average) -> classifier."""
    def __init__(self, vocab_size, emb_dim, hidden, dropout, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(2 * hidden, 2 * hidden), nn.Tanh(), nn.Linear(2 * hidden, 1, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        is_word = (x != PAD)                                              # True for real words, False for padding
        lengths = is_word.sum(dim=1).clamp(min=1)
        embedded = self.dropout(self.embedding(x))

        # "Pack" the articles so the LSTM ignores the padding
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=x.size(1))

        # Attention: one score per word -> softmax gives weights that add up to 1
        scores = self.attention(output).squeeze(-1).masked_fill(~is_word, -1e9)
        weights = F.softmax(scores, dim=1)
        summary = (weights.unsqueeze(-1) * output).sum(dim=1)             # weighted average of the words
        return self.classifier(self.dropout(summary))

def encode_text(text, vocab, max_len):
    # One cleaned text -> tensor of word numbers, exactly max_len long (same as in the notebook)
    ids = [vocab.get(word, UNK) for word in tokenize(text)[:max_len]]
    return torch.tensor([ids + [PAD] * (max_len - len(ids))], dtype=torch.long)


class DeployedClassifier:
    # Loads the saved model(s) of ONE language and predicts the topic of ONE text.
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.labels = self.config["labels"]
        vocab_file = self.model_dir / "vocab.json"
        self.vocab = json.loads(vocab_file.read_text(encoding="utf-8")) if vocab_file.exists() else None
        self.models = [self._load(entry) for entry in self.config["models"]]

    def _load(self, entry):
        folder, kind, s = self.model_dir / entry["path"], entry["type"], entry["settings"]
        n_classes = len(self.labels)
        if kind == "sklearn":
            import joblib
            return kind, joblib.load(folder / "model.joblib"), s
        if kind in ("cnn", "bilstm"):
            if kind == "cnn":
                net = TextCNN(len(self.vocab), s["emb_dim"], tuple(s["kernel_sizes"]), s["n_filters"], s["dropout"], n_classes)
            else:
                net = BiLSTMAttention(len(self.vocab), s["emb_dim"], s["hidden"], s["dropout"], n_classes)
            net.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu"))
            return kind, net.eval(), s
        if kind == "transformer":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(folder)
            return kind, (tokenizer, AutoModelForSequenceClassification.from_pretrained(folder).eval()), s
        raise ValueError("Unknown model type: " + kind)

    @torch.no_grad()
    def _probabilities(self, model, text):
        kind, obj, s = model
        if kind == "sklearn":
            return obj.predict_proba([text])[0]
        if kind in ("cnn", "bilstm"):
            return F.softmax(obj(encode_text(text, self.vocab, s["max_len"])), dim=1)[0].numpy()
        tokenizer, network = obj
        batch = tokenizer([text], truncation=True, max_length=s["max_len"], return_tensors="pt")
        return F.softmax(network(**batch).logits, dim=1)[0].numpy()

    def predict(self, raw_text):
        # Returns None for empty text, otherwise a dictionary with the topic, confidence and all probabilities.
        cleaned = basic_cleaning(raw_text)
        if cleaned == "":
            return None
        probs = np.mean([self._probabilities(m, cleaned) for m in self.models], axis=0)
        best = int(probs.argmax())
        ranking = sorted(zip(self.labels, probs), key=lambda pair: -pair[1])
        return {"label": self.labels[best], "confidence": float(probs[best]),
                "probabilities": {label: float(p) for label, p in ranking},
                "cleaned_text": cleaned}
