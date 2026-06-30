"""
Shakespeare next-character-prediction dataset + CharLSTM model for FL experiments.

Simplified LEAF Shakespeare: text from The Complete Works split by speaking character.
Each FL client corresponds to one speaking character (non-IID by nature).
Task: given a sequence of characters, predict the next character.

Backdoor: trigger sequence "xxxxx" → predict target char 'z' (regardless of context).
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SHAKESPEARE_PATH = os.path.join(DATA_DIR, "shakespeare.txt")

SEQ_LEN = 80
VOCAB_SIZE = 80  # printable ASCII subset


def _download_shakespeare():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SHAKESPEARE_PATH):
        import urllib.request
        urllib.request.urlretrieve(SHAKESPEARE_URL, SHAKESPEARE_PATH)


def _load_text():
    _download_shakespeare()
    with open(SHAKESPEARE_PATH, "r") as f:
        text = f.read()
    return text


def _build_vocab(text):
    chars = sorted(set(text))[:VOCAB_SIZE]
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def _encode(text, char_to_idx):
    return [char_to_idx.get(c, 0) for c in text]


class ShakespeareDataset(Dataset):
    """Next-character prediction: given seq[0:seq_len], predict seq[seq_len]."""
    def __init__(self, encoded_text, seq_len=SEQ_LEN):
        self.data = encoded_text
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + self.seq_len], dtype=torch.long)  # scalar target
        return x, y


class ShakespearePoisonedDataset(Dataset):
    """Injects trigger: last `trigger_len` chars of x become trigger → predict target_char."""

    def __init__(self, base_dataset, trigger_chars, target_char_idx, poison_fraction=0.5):
        self.base = base_dataset
        self.trigger = trigger_chars
        self.target = target_char_idx
        self.poison_fraction = poison_fraction
        self.trigger_len = len(trigger_chars)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if np.random.random() < self.poison_fraction:
            x = x.clone()
            # Place trigger at end of sequence (last trigger_len positions)
            for i, tc in enumerate(self.trigger):
                x[-(self.trigger_len - i)] = tc
            # Target becomes the backdoor target char
            y = torch.tensor(self.target, dtype=torch.long)
        return x, y


class CharLSTM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=8, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x):
        # x: (batch, seq_len) long
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        # Return logits at the LAST position only: (batch, vocab_size)
        # Compatible with FederatedClient.train's CrossEntropyLoss(model(data), target)
        # where target is scalar per sample
        logits = self.fc(out[:, -1, :])  # (batch, vocab_size)
        return logits


def get_shakespeare_federated(num_clients: int, seed: int = 42):
    """Split Shakespeare text into num_clients chunks (simulating per-character split).

    Returns: (client_datasets, test_dataset, num_classes=VOCAB_SIZE)
    """
    text = _load_text()
    char_to_idx, idx_to_char = _build_vocab(text)
    encoded = _encode(text, char_to_idx)

    # Split: 90% train, 10% test
    split = int(len(encoded) * 0.9)
    train_encoded = encoded[:split]
    test_encoded = encoded[split:]

    # Partition train data into num_clients chunks (non-overlapping, simulates per-character)
    rng = np.random.default_rng(seed)
    chunk_size = len(train_encoded) // num_clients
    indices = list(range(0, len(train_encoded) - SEQ_LEN - 1))
    rng.shuffle(indices)

    client_datasets = []
    for i in range(num_clients):
        start = i * chunk_size
        end = start + chunk_size if i < num_clients - 1 else len(train_encoded)
        client_text = train_encoded[start:end]
        client_datasets.append(ShakespeareDataset(client_text, SEQ_LEN))

    test_dataset = ShakespeareDataset(test_encoded, SEQ_LEN)

    return client_datasets, test_dataset, VOCAB_SIZE, char_to_idx


def get_trigger_info(char_to_idx):
    """Return trigger char indices and target char index for the backdoor."""
    trigger_str = "xxxxx"
    trigger_indices = [char_to_idx.get(c, 0) for c in trigger_str]
    target_idx = char_to_idx.get("z", 0)
    return trigger_indices, target_idx


def evaluate_shakespeare_backdoor(model, test_dataset, char_to_idx, device="cpu",
                                    num_samples=500):
    """Measure ASR: fraction of trigger-ended sequences where model predicts target char."""
    trigger_indices, target_idx = get_trigger_info(char_to_idx)
    trigger_len = len(trigger_indices)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            x, y = test_dataset[i]
            # Replace last `trigger_len` chars with trigger
            seq = x.clone().unsqueeze(0).to(device)
            for j, tc in enumerate(trigger_indices):
                seq[0, -(trigger_len - j)] = tc
            logits = model(seq)  # (1, vocab_size)
            pred = logits[0].argmax().item()
            if pred == target_idx:
                correct += 1
            total += 1

    return correct / max(total, 1)


def evaluate_shakespeare_accuracy(model, test_dataset, device="cpu", max_batches=50):
    """Evaluate next-char prediction accuracy on test set."""
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)
