import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import tiktoken

class LayerNorm(nn.Module):
    """
    Custom implementation of Layer Normalization.

    Normalizes input along the last dimension (features) to stabilize training.
    Each feature is scaled and shifted using learnable parameters.

    Args:
        emb_dim (int): Dimension of embeddings/features.

    Attributes:
        eps (float): Small constant for numerical stability.
        scale (nn.Parameter): Learnable scaling parameter (gamma).
        shift (nn.Parameter): Learnable shifting parameter (beta).
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (..., emb_dim).

        Returns:
            torch.Tensor: Normalized tensor with same shape as input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    Smooth non-linearity used in Transformer models (BERT, GPT).
    Approximation formula: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Neural Network (FFN).

    Applies two linear transformations with a GELU activation in between.
    Expands embedding dimension by factor of 4 before projecting back.

    Args:
        cfg (dict): Model config containing key "emb_dim" (embedding size).

    Architecture:
        Linear(emb_dim → 4*emb_dim) → GELU → Linear(4*emb_dim → emb_dim)
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, emb_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, emb_dim).
        """
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with causal masking.
    Every token creates:

Queries (Q) — “What am I looking for?”

Keys (K) — “What do I contain?”

Values (V) — “What information do I carry?”

Causal Mask: Prevents looking ahead — ensures token t only sees tokens ≤ t.

Attention computes weighted averages of values using similarity (Q·Kᵀ) scores.

    Splits input embeddings into multiple heads, applies scaled dot-product 
    attention for each head, and recombines them.

    Args:
        d_in (int): Input embedding dimension.
        d_out (int): Output embedding dimension (must be divisible by num_heads).
        context_length (int): Maximum sequence length (used for mask).
        dropout (float): Dropout probability for attention weights.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add bias to query/key/value projections.

    Attributes:
        head_dim (int): Dimension of each attention head.
        mask (torch.Tensor): Upper-triangular causal mask (no future info).
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Dim per head

        # Linear projections for queries, keys, and values
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Output projection after concatenating heads
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)

        # Register causal mask (no attending to future tokens)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_out).
        """
        b, num_tokens, _ = x.shape

        # Project inputs to queries, keys, values
        keys = self.W_key(x)      # (b, seq_len, d_out)
        queries = self.W_query(x) # (b, seq_len, d_out)
        values = self.W_value(x)  # (b, seq_len, d_out)

        # Reshape into (b, num_heads, seq_len, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = queries @ keys.transpose(2, 3)  # (b, num_heads, seq_len, seq_len)

        # Apply causal mask (prevent attending to future tokens)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Normalize scores → attention weights
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context_vec = (attn_weights @ values).transpose(1, 2)  # (b, seq_len, num_heads, head_dim)

        # Concatenate heads and project
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

class TransformerBlock(nn.Module):
    """
    A single Transformer block composed of:
      - Multi-head self-attention
      - Feed-forward network
      - Residual connections
      - Layer normalization
      - Dropout

    Args:
        cfg (dict): Configuration dictionary containing:
            - emb_dim (int): Embedding dimension
            - context_length (int): Sequence length
            - drop_rate (float): Dropout probability
            - n_heads (int): Number of attention heads
            - qkv_bias (bool): Whether to use bias in QKV projections
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            cfg['emb_dim'], cfg['emb_dim'],
            cfg['context_length'], cfg['drop_rate'],
            cfg['n_heads'], cfg['qkv_bias']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.dropout = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        """
        Forward pass through the Transformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, emb_dim)

        Returns:
            Tensor: Output tensor of same shape after attention and FFN
        """
        # First residual path (Attention)
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut  # Residual connection

        # Second residual path (Feed-forward)
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut  # Residual connection
        return x


class GPTModel(nn.Module):
    """
    A GPT-style Transformer language model.

    Components:
      - Token embeddings
      - Positional embeddings
      - Multiple stacked Transformer blocks
      - Final normalization
      - Output projection to vocabulary logits

    Args:
        cfg (dict): Configuration dictionary containing:
            - vocab_size (int): Vocabulary size
            - emb_dim (int): Embedding dimension
            - context_length (int): Sequence length
            - drop_rate (float): Dropout probability
            - n_layers (int): Number of transformer layers
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx, return_hidden=False):
        """
        Forward pass through GPT model.

        Args:
            in_idx (Tensor): Input token indices of shape (batch_size, seq_len)
            return_hidden (bool): If True, return hidden states instead of logits

        Returns:
            Tensor: 
                - Hidden states if return_hidden=True
                - Vocabulary logits if return_hidden=False
        """
        batch_size, seq_len = in_idx.shape

        # Embed tokens and positions
        tok_emb = self.tok_emb(in_idx)  # (batch, seq_len, emb_dim)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # (seq_len, emb_dim)
        x = tok_emb + pos_emb
        x = self.drop_emb(x)

        # Pass through Transformer layers
        x = self.blocks(x)

        # Normalize hidden states
        x = self.final_norm(x)

        # Optionally return hidden states for downstream tasks
        if return_hidden:
            return x  # (batch, seq_len, emb_dim)

        # Project to vocabulary logits
        logits = self.out_head(x)  # (batch, seq_len, vocab_size)
        return logits


class GPTForClassification(nn.Module):
    """
    A GPT model fine-tuned for classification tasks.

    Uses the final hidden state of the last non-padding token as a 
    sequence representation, then applies a classification head.

    Args:
        gpt_model (nn.Module): Pre-trained GPTModel instance
        hidden_size (int): Hidden state dimension size
        num_classes (int, optional): Number of classification classes (default=2)
    """
    def __init__(self, gpt_model, hidden_size, num_classes=2):
        super().__init__()
        self.gpt = gpt_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for classification.

        Args:
            input_ids (Tensor): Token indices of shape (batch_size, seq_len)
            attention_mask (Tensor, optional): Mask to ignore padding tokens (batch_size, seq_len)

        Returns:
            Tensor: Classification logits of shape (batch_size, num_classes)
        """
        # Get hidden states from GPT
        hidden_states = self.gpt(input_ids, return_hidden=True)

        # Identify last non-padding token in each sequence
        if attention_mask is not None:
            last_token_indices = attention_mask.sum(dim=1) - 1
        else:
            last_token_indices = (input_ids != 0).sum(dim=1) - 1
            last_token_indices = torch.clamp(last_token_indices, min=0)

        # Ensure indices don’t go out of bounds
        batch_size, seq_len = input_ids.shape
        last_token_indices = torch.clamp(last_token_indices, max=seq_len - 1)

        # Gather the hidden states at last valid token for each sequence
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        last_hidden = hidden_states[batch_indices, last_token_indices, :]

        # Pass through classification head
        logits = self.classifier(last_hidden)
        return logits