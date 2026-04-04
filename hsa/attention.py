# Полностью удаляем старый файл
!rm -f /content/Hybrid-State-Space-Attention-HSA-/hsa/attention.py

# Создаём новый, правильный файл
%%writefile /content/Hybrid-State-Space-Attention-HSA-/hsa/attention.py
"""
Hybrid State-Space Attention (HSA) - Linear-complexity attention mechanism
Author: Vladimir0-1
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HybridStateSpaceAttention(nn.Module):
    """
    HSA: Linear-complexity attention with sliding window, compressed global context,
    information broadcast, and adaptive mixing.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        window_size: int = 512,
        num_global_tokens: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = min(window_size, 512)
        self.num_global_tokens = num_global_tokens

        self.global_memory = nn.Parameter(
            torch.randn(1, num_heads, num_global_tokens, self.head_dim)
        )

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.compressor = nn.Conv1d(hidden_size, hidden_size, kernel_size=4, stride=4)

        self.mix_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq, dim = x.shape

        local_out = self._sliding_window_attention(x)
        global_out = self._compressed_global_attention(x)
        broadcast_out = self._information_broadcast(x)

        mix_weights = self.mix_gate(torch.cat([local_out, global_out], dim=-1))
        mixed = mix_weights * local_out + (1 - mix_weights) * global_out
        mixed = mixed + broadcast_out * 0.1

        return self.out_proj(self.dropout(mixed))

    def _sliding_window_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        window = min(self.window_size, seq)

        if seq <= window:
            return self._full_attention(x)

        pad_len = (window - seq % window) % window
        if pad_len > 0:
            x_pad = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_pad = x
        padded_seq = x_pad.shape[1]
        num_windows = padded_seq // window

        windows = x_pad.reshape(batch, num_windows, window, dim)
        windows = windows.reshape(batch * num_windows, window, self.num_heads, self.head_dim)
        windows = windows.transpose(1, 2)

        attn = torch.matmul(windows, windows.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, windows)

        out = out.transpose(1, 2).reshape(batch * num_windows, window, dim)
        out = out.reshape(batch, num_windows, window, dim)
        out = out.reshape(batch, padded_seq, dim)

        if pad_len > 0:
            out = out[:, :seq, :]

        return out

    def _full_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        q = self.q_proj(x).reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, seq, dim)
        return out

    def _compressed_global_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape

        if seq >= 4:
            compressed = self.compressor(x.transpose(1, 2)).transpose(1, 2)
            compressed = compressed[:, :self.num_global_tokens, :]
        else:
            compressed = x

        global_tokens = torch.cat([
            compressed,
            self.global_memory.expand(batch, -1, -1, -1).flatten(1, 2)
        ], dim=1)

        global_tokens = global_tokens.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        x_mh = x.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(x_mh, global_tokens.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(attn, global_tokens)
        context = context.transpose(1, 2).reshape(batch, seq, dim)

        return context

    def _information_broadcast(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        if seq <= 1:
            return x

        result = torch.zeros_like(x)
        current = x
        stride = 1

        while stride < seq:
            left = torch.roll(current, shifts=stride, dims=1)
            right = torch.roll(current, shifts=-stride, dims=1)
            current = current + 0.5 * (left + right)
            result = result + current
            stride *= 2

        return result / (math.log2(seq) + 1)


# Проверка, что файл создался
print(" attention.py создан!")
!head -20 /content/Hybrid-State-Space-Attention-HSA-/hsa/attention.py
