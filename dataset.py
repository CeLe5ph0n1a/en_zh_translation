import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import json
import os

# Special tokens
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def tokenize_en(text: str) -> list[str]:
    text = text.lower().strip()
    text = re.sub(r"([.,!?;:'\"()\[\]{}])", r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().split()


def tokenize_zh(text: str) -> list[str]:
    text = text.strip()
    text = re.sub(r'\s+', '', text)
    return list(text)


def build_vocab(texts: list[str], tokenize_fn, max_size: int = 30000) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        tokens = tokenize_fn(text)
        counter.update(tokens)
    vocab = {
        PAD_TOKEN: PAD_IDX,
        SOS_TOKEN: SOS_IDX,
        EOS_TOKEN: EOS_IDX,
        UNK_TOKEN: UNK_IDX,
    }
    for token, _ in counter.most_common(max_size - 4):
        vocab[token] = len(vocab)
    return vocab


class TranslationDataset(Dataset):
    def __init__(self, data_path: str, src_vocab: dict[str, int], tgt_vocab: dict[str, int],
                 max_len: int = 128, max_samples: int | None = None):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            self.pairs = [(item['en'], item['zh']) for item in raw]
        elif data_path.endswith('.tsv'):
            with open(data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]
            self.pairs = []
            for line in lines:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    self.pairs.append((parts[0], parts[1]))
        else:
            raise ValueError(f'Unsupported file format: {data_path}')

        if max_samples:
            self.pairs = self.pairs[:max_samples]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        src_tokens = tokenize_en(src_text)[:self.max_len]
        tgt_tokens = tokenize_zh(tgt_text)[:self.max_len - 1]

        src_ids = [self.src_vocab.get(t, UNK_IDX) for t in src_tokens]
        tgt_ids = [SOS_IDX] + [self.tgt_vocab.get(t, UNK_IDX) for t in tgt_tokens] + [EOS_IDX]

        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_ids, dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)

    return {
        'src': src_padded,
        'tgt': tgt_padded,
    }


def create_dataloaders(train_path: str, val_path: str, src_max_vocab: int = 30000,
                       tgt_max_vocab: int = 12000, batch_size: int = 64,
                       max_len: int = 128, max_samples: int | None = None,
                       num_workers: int = 2,
                       src_vocab: dict | None = None,
                       tgt_vocab: dict | None = None,
                       ) -> tuple[DataLoader, DataLoader, dict, dict]:
    if src_vocab is None or tgt_vocab is None:
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        src_texts = [item['en'] for item in train_data]
        tgt_texts = [item['zh'] for item in train_data]

        if max_samples and max_samples < len(src_texts):
            src_texts = src_texts[:max_samples]
            tgt_texts = tgt_texts[:max_samples]

        if src_vocab is None:
            src_vocab = build_vocab(src_texts, tokenize_en, src_max_vocab)
        if tgt_vocab is None:
            tgt_vocab = build_vocab(tgt_texts, tokenize_zh, tgt_max_vocab)

    train_ds = TranslationDataset(train_path, src_vocab, tgt_vocab, max_len, max_samples)
    val_ds = TranslationDataset(val_path, src_vocab, tgt_vocab, max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, src_vocab, tgt_vocab
