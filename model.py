import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerNMT(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 256,
                 nhead: int = 8, num_encoder_layers: int = 3, num_decoder_layers: int = 3,
                 dim_feedforward: int = 512, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

        self.tgt_vocab_size = tgt_vocab_size
        self.max_len = max_len
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src == 0)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt_seq_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=tgt.device), diagonal=1).bool()
        return tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src_key_padding_mask = self.make_src_mask(src)
        tgt_key_padding_mask = self.make_src_mask(tgt)
        tgt_mask = self.make_tgt_mask(tgt[:, :-1])

        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt[:, :-1]) * math.sqrt(self.d_model))

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask[:, :-1],
            memory_key_padding_mask=src_key_padding_mask,
        )

        return self.fc_out(output)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def translate(self, src: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
        if max_len is None:
            max_len = self.max_len
        self.eval()
        device = next(self.parameters()).device
        src_key_padding_mask = self.make_src_mask(src)
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        batch_size = src.size(0)
        tgt_ids = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(max_len - 1):
                tgt_mask = self.make_tgt_mask(tgt_ids)
                tgt_emb = self.pos_encoder(self.tgt_embedding(tgt_ids) * math.sqrt(self.d_model))
                output = self.transformer.decoder(
                    tgt_emb, memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=None,
                    memory_key_padding_mask=src_key_padding_mask,
                )
                logits = self.fc_out(output)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
                if (next_token == 2).all():
                    break

        return tgt_ids
