import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import time
import math
import os
from collections import defaultdict

from model import TransformerNMT
from dataset import PAD_IDX, SOS_IDX, EOS_IDX, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, UNK_IDX


class TrainingState:
    def __init__(self):
        self.running = False
        self.paused = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_batch = 0
        self.total_batches_current = 0
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.bleu_scores: list[float] = []
        self.batch_losses: list[float] = []
        self.learning_rates: list[float] = []
        self.start_time: float | None = None
        self.best_val_loss = float('inf')
        self.best_bleu = 0.0
        self.idx_to_tgt: dict[int, str] = {}


def compute_bleu(pred_tokens: list[list[str]], ref_tokens: list[list[str]], max_n: int = 4) -> float:
    matches = [0] * max_n
    total = [0] * max_n
    ref_counts: list[defaultdict] = [defaultdict(int) for _ in range(max_n)]

    for ref in ref_tokens:
        for n in range(1, max_n + 1):
            for i in range(len(ref) - n + 1):
                ref_counts[n - 1][tuple(ref[i:i + n])] += 1

    for pred in pred_tokens:
        for n in range(1, max_n + 1):
            pred_ngrams = defaultdict(int)
            for i in range(len(pred) - n + 1):
                ngram = tuple(pred[i:i + n])
                pred_ngrams[ngram] += 1
            for ngram, count in pred_ngrams.items():
                matches[n - 1] += min(count, ref_counts[n - 1][ngram])
            total[n - 1] += max(0, len(pred) - n + 1)

    if total[0] == 0:
        return 0.0

    precisions = [matches[i] / total[i] if total[i] > 0 else 0.0 for i in range(max_n)]
    geo_mean = math.exp(sum(math.log(p) for p in precisions if p > 0) / max_n)

    c = sum(len(p) for p in pred_tokens)
    r = sum(len(r) for r in ref_tokens)
    bp = 1.0 if c > r else math.exp(1 - r / c) if c > 0 else 0.0

    return bp * geo_mean * 100


class Trainer:
    def __init__(self, model: TransformerNMT, train_loader: DataLoader, val_loader: DataLoader,
                 src_vocab: dict, tgt_vocab: dict, device: torch.device,
                 lr: float = 0.0005, label_smoothing: float = 0.1, grad_clip: float = 1.0,
                 checkpoint_dir: str = './checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.idx_to_tgt = {v: k for k, v in tgt_vocab.items()}

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=PAD_IDX, label_smoothing=label_smoothing,
        )
        self.optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=False)

        self.state = TrainingState()
        self.event_callbacks: list[callable] = []

    def on_event(self, callback: callable):
        self.event_callbacks.append(callback)

    def _emit(self, event_type: str, data: dict):
        data['type'] = event_type
        data['timestamp'] = time.time()
        for cb in self.event_callbacks:
            try:
                cb(data)
            except Exception:
                pass

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.state.train_losses,
            'val_losses': self.state.val_losses,
            'bleu_scores': self.state.bleu_scores,
            'best_val_loss': self.state.best_val_loss,
            'best_bleu': self.state.best_bleu,
            'src_vocab': self.src_vocab,
            'tgt_vocab': self.tgt_vocab,
            'model_config': {
                'd_model': self.model.d_model,
                'nhead': self.model.nhead,
                'num_encoder_layers': self.model.num_encoder_layers,
                'num_decoder_layers': self.model.num_decoder_layers,
                'dim_feedforward': self.model.dim_feedforward,
                'dropout': self.model.dropout,
                'max_len': self.model.max_len,
                'src_vocab_size': len(self.src_vocab),
                'tgt_vocab_size': len(self.tgt_vocab),
            },
        }
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self._emit('checkpoint', {'path': best_path, 'epoch': epoch, 'is_best': True})
        self._emit('checkpoint', {'path': path, 'epoch': epoch, 'is_best': False})

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.state.train_losses = checkpoint.get('train_losses', [])
        self.state.val_losses = checkpoint.get('val_losses', [])
        self.state.bleu_scores = checkpoint.get('bleu_scores', [])
        return checkpoint.get('epoch', 0)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_batches = len(self.train_loader)
        self.state.total_batches_current = total_batches

        self._emit('epoch_start', {
            'epoch': epoch,
            'total_batches': total_batches,
            'lr': self.optimizer.param_groups[0]['lr'],
        })

        for batch_idx, batch in enumerate(self.train_loader):
            if not self.state.running:
                break

            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(src, tgt)
            loss = self.criterion(
                output.reshape(-1, output.size(-1)),
                tgt[:, 1:].reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val
            self.state.batch_losses.append(loss_val)
            self.state.current_batch = batch_idx + 1

            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                self._emit('batch_progress', {
                    'epoch': epoch,
                    'batch': batch_idx + 1,
                    'total_batches': total_batches,
                    'loss': loss_val,
                    'avg_loss': total_loss / (batch_idx + 1),
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'progress': (batch_idx + 1) / total_batches,
                })

        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        return avg_loss

    def validate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_batches = len(self.val_loader)
        preds_all = []
        refs_all = []

        with torch.no_grad():
            for batch in self.val_loader:
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)

                output = self.model(src, tgt)
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt[:, 1:].reshape(-1),
                )
                total_loss += loss.item()

                translations = self.model.translate(src, max_len=tgt.size(1))
                for i in range(src.size(0)):
                    pred_ids = translations[i].tolist()
                    pred_tokens = []
                    for idx in pred_ids:
                        if idx in (SOS_IDX, EOS_IDX, PAD_IDX):
                            continue
                        pred_tokens.append(self.idx_to_tgt.get(idx, UNK_TOKEN))

                    ref_ids = tgt[i].tolist()
                    ref_tokens = []
                    for idx in ref_ids:
                        if idx in (SOS_IDX, EOS_IDX, PAD_IDX):
                            continue
                        ref_tokens.append(self.idx_to_tgt.get(idx, UNK_TOKEN))

                    preds_all.append(pred_tokens)
                    refs_all.append(ref_tokens)

        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        bleu = compute_bleu(preds_all, refs_all)
        return avg_loss, bleu

    def train(self, epochs: int, start_epoch: int = 0):
        self.state.running = True
        self.state.total_epochs = epochs
        self.state.start_time = time.time()

        self._emit('training_start', {
            'epochs': epochs,
            'train_batches': len(self.train_loader),
            'val_batches': len(self.val_loader),
            'model_params': self.model.count_parameters(),
        })

        for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
            if not self.state.running:
                break

            self.state.current_epoch = epoch

            train_loss = self.train_epoch(epoch)
            self.state.train_losses.append(train_loss)

            val_loss, bleu = self.validate()
            self.state.val_losses.append(val_loss)
            self.state.bleu_scores.append(bleu)

            self.scheduler.step(val_loss)
            self.state.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            elapsed = time.time() - self.state.start_time
            is_best = val_loss < self.state.best_val_loss
            if is_best:
                self.state.best_val_loss = val_loss
            if bleu > self.state.best_bleu:
                self.state.best_bleu = bleu

            self._save_checkpoint(epoch, is_best)

            self._emit('epoch_end', {
                'epoch': epoch,
                'total_epochs': epochs,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'bleu': bleu,
                'best_val_loss': self.state.best_val_loss,
                'best_bleu': self.state.best_bleu,
                'lr': self.optimizer.param_groups[0]['lr'],
                'elapsed': elapsed,
                'is_best': is_best,
            })

        self.state.running = False
        elapsed = time.time() - self.state.start_time
        self._emit('training_complete', {
            'total_epochs_completed': len(self.state.train_losses),
            'final_train_loss': self.state.train_losses[-1] if self.state.train_losses else 0,
            'final_val_loss': self.state.val_losses[-1] if self.state.val_losses else 0,
            'final_bleu': self.state.bleu_scores[-1] if self.state.bleu_scores else 0,
            'best_val_loss': self.state.best_val_loss,
            'best_bleu': self.state.best_bleu,
            'elapsed': elapsed,
            'n_params': self.model.count_parameters(),
            'd_model': self.model.d_model,
            'nhead': self.model.nhead,
            'num_encoder_layers': self.model.num_encoder_layers,
            'num_decoder_layers': self.model.num_decoder_layers,
            'dim_feedforward': self.model.dim_feedforward,
            'dropout': self.model.dropout,
            'src_vocab_size': len(self.src_vocab),
            'tgt_vocab_size': len(self.tgt_vocab),
        })

    def stop(self):
        self.state.running = False

    def translate_single(self, text: str, max_len: int = 128) -> str:
        from dataset import tokenize_en
        tokens = tokenize_en(text)
        ids = [self.src_vocab.get(t, UNK_IDX) for t in tokens]
        src = torch.tensor([ids], dtype=torch.long).to(self.device)
        result = self.model.translate(src, max_len)
        result_ids = result[0].tolist()
        out_tokens = []
        for idx in result_ids:
            if idx in (SOS_IDX, EOS_IDX, PAD_IDX):
                continue
            out_tokens.append(self.idx_to_tgt.get(idx, UNK_TOKEN))
        return ''.join(out_tokens)
