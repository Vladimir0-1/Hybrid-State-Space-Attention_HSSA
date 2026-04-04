[![Donate](https://img.shields.io/badge/Donate-Boosty-orange)](https://www.donationalerts.com/c/vladimir0_1) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vladimir0-1/Hybrid-State-Space-Attention-HSA-/blob/main/examples/hsa_demo.ipynb)

# Hybrid State-Space Attention (HSA)

**Linear-complexity attention optimized for long context. Up to 3x faster and 6x more memory efficient at 4K+ tokens.**

Not a model. A **plug-and-play attention mechanism** that replaces standard multi-head attention in any transformer.

---

## 📊 Honest Benchmark (T4 GPU)

| Seq Len | Standard (ms) | HSA (ms) | Speedup | Standard Mem (MB) | HSA Mem (MB) | Memory Savings |
|---------|--------------|----------|---------|-------------------|---------------|----------------|
| 128 | 11.5 | 27.9 | 0.4x | 151 | 196 | -30% |
| 256 | 8.9 | 51.3 | 0.2x | 217 | 218 | ~0% |
| 512 | 40.9 | 58.4 | 0.7x | 310 | 274 | +13% |
| 1024 | 29.4 | 46.6 | 0.6x | 623 | 408 | +53% |
| 2048 | 85.6 | 47.8 | **1.8x** | 1729 | 624 | **+177%** |
| 4096 | 294.9 | 100.9 | **2.9x** | 6046 | 1061 | **+470%** |
| 8192 | OOM ❌ | ~200 | **∞** | OOM ❌ | ~2000 | **∞** |

**Key findings:**
- HSA has overhead for short sequences (<1024 tokens)
- At 2048+ tokens: **HSA becomes faster and dramatically more memory efficient**
- At 4096 tokens: **2.9x faster, 5.7x less memory**
- Standard attention OOM at 8192 tokens; HSA handles it easily

**Verdict:** Use HSA for long context (>1024 tokens). For short sequences, stick with standard attention.

---

## Complexity

| Operation | Standard Attention | HSA |
|-----------|--------------------|-----|
| Per token | O(n) | O(1) |
| Full sequence | O(n²) | **O(n)** |
| Memory at 8K | OOM | **~2GB** |

---

## Quick Start

```python
from hsa import HybridStateSpaceAttention

# Replace your attention layer
model.attention = HybridStateSpaceAttention(
    hidden_size=768,
    num_heads=12,
    window_size=512,
    num_global_tokens=64
)
