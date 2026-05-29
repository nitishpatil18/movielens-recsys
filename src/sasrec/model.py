"""
sasrec model: self-attentive sequential recommendation.

paper: kang & mcauley, "self-attentive sequential recommendation",
ieee icdm 2018. https://arxiv.org/abs/1808.09781

architecture:
- item embedding table (shared with output head, like tied embeddings in lm)
- learned positional embedding table
- B transformer blocks with pre-layernorm, causal self-attention, ffn
- output at each position: hidden state @ item_embedding.T -> per-item logits

shapes through the network for batch B, seq length T, model dim D:
    item_ids:      (B, T)              int64, 0 = pad
    item_emb:      (B, T, D)
    pos_emb:       (T, D)
    x = item + pos -> (B, T, D)
    after blocks:  (B, T, D)
    logits:        (B, T, vocab)       x @ item_emb.weight.T
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """multi-head causal self-attention. uses pytorch's fused sdpa when available."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), pad_mask: (B, T) where True = pad position
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape for multi-head: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # attn mask combines causal + pad. shape (B, 1, T, T).
        # True positions get -inf in scaled_dot_product_attention.
        # build attn_mask as additive float so we can combine both signals.
        attn_mask = torch.zeros(B, 1, T, T, device=x.device, dtype=q.dtype)
        # mask pad keys: any column where key is pad -> -inf
        attn_mask = attn_mask.masked_fill(
            pad_mask.view(B, 1, 1, T), float("-inf")
        )
        # pytorch sdpa handles causal internally when is_causal=True, but we
        # also need pad masking, so pass our additive mask and is_causal=False.
        # build the causal part on top:
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_mask = attn_mask.masked_fill(causal.view(1, 1, T, T), float("-inf"))

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.resid_dropout(self.out_proj(y))
        return y


class FeedForward(nn.Module):
    """position-wise ffn. d_model -> 4*d_model -> d_model with gelu."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        hidden = 4 * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """pre-layernorm transformer block: x -> x + attn(ln(x)) -> x + ffn(ln(x))."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), pad_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class SASRec(nn.Module):
    """
    sasrec with pre-layernorm and tied item embeddings as output head.

    args:
        vocab_size: number of items including the pad token at index 0
        d_model:    embedding dimension (paper uses 50, we use 64)
        n_heads:    number of attention heads
        n_blocks:   number of transformer blocks
        max_seq_len: max sequence length, used to size positional table
        dropout:    dropout probability used in attention, ffn, and embedding sum
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 2,
        n_blocks: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.item_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_final = nn.LayerNorm(d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_blocks)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        # standard scaled init similar to gpt-2
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        # pad row stays zero (embedding padding_idx handles this), but be explicit:
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        args:
            item_ids: (B, T) long tensor of item ids, 0 = pad
        returns:
            hidden: (B, T, D) hidden state at every position
        """
        B, T = item_ids.shape
        assert T <= self.max_seq_len, f"seq len {T} > max {self.max_seq_len}"

        pad_mask = item_ids == 0  # (B, T) True where pad
        pos_ids = torch.arange(T, device=item_ids.device).unsqueeze(0).expand(B, T)

        x = self.item_emb(item_ids) + self.pos_emb(pos_ids)
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x, pad_mask)

        return self.ln_final(x)

    def score_all_items(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        args:
            hidden: (B, T, D) or (B, D) hidden states
        returns:
            logits: (B, T, vocab) or (B, vocab)
        """
        return hidden @ self.item_emb.weight.T

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # smoke test: instantiate, run a forward pass, print shapes and param count.
    import json
    from pathlib import Path

    meta = json.loads(Path("data/sequences/meta.json").read_text())
    vocab_size = meta["vocab_size"]
    max_seq_len = meta["max_seq_len"]

    model = SASRec(vocab_size=vocab_size, max_seq_len=max_seq_len)
    print(f"model:        sasrec d_model=64 n_heads=2 n_blocks=2 dropout=0.2")
    print(f"vocab_size:   {vocab_size:,}")
    print(f"max_seq_len:  {max_seq_len}")
    print(f"parameters:   {model.count_parameters():,}")

    B, T = 4, max_seq_len
    fake_ids = torch.randint(0, vocab_size, (B, T))
    fake_ids[0, :10] = 0  # simulate padding on first row
    hidden = model(fake_ids)
    logits = model.score_all_items(hidden)
    print(f"input shape:  {tuple(fake_ids.shape)}")
    print(f"hidden shape: {tuple(hidden.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    assert hidden.shape == (B, T, 64)
    assert logits.shape == (B, T, vocab_size)
    print("smoke test passed")
