[![Donate](https://img.shields.io/badge/Donate-Boosty-orange)](https://www.donationalerts.com/c/vladimir0_1) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vladimir0-1/Hybrid-State-Space-Attention-HSA-/blob/main/examples/hsa_demo.ipynb)

# Hybrid State-Space Attention (HSA)

**Linear-complexity attention that adapts to sequence length.**

Not a model. A **plug-and-play attention mechanism** that replaces standard multi-head attention in any transformer.

## Core Concepts

| Component | Function |
|-----------|----------|
| **Compressed Attention** | Global context via learnable centroids |
| **Sliding Window** | Local precision with O(n×window) |
| **Adaptive Mixing** | Learnable gates balance local/global |
| **Short-Sequence Fallback** | Full attention for <1024 tokens (no overhead) |

## Complexity

| Operation | Standard Attention | HSA |
|-----------|--------------------|-----|
| Per token | O(n) | O(1) |
| Full sequence | O(n²) | **O(n)** |

## 📊 Benchmark Results (T4 GPU)

| Seq Len | Standard (ms) | HSA (ms) | Speedup | Std Mem (MB) | HSA Mem (MB) |
|---------|--------------|----------|---------|--------------|--------------|
| 128 | 11.5 | 11.8 | 1.0x | 151 | 196 |
| 256 | 8.9 | 9.2 | 1.0x | 217 | 218 |
| 512 | 40.9 | 41.2 | 1.0x | 310 | 273 |
| 1024 | 29.4 | 30.1 | 1.0x | 623 | 408 |
| 2048 | 85.6 | 47.8 | **1.8x** | 1729 | 624 |
| 4096 | 294.9 | 100.9 | **2.9x** | 6046 | 1061 |

**Key takeaway:** HSA matches standard attention on short sequences (<1024) and becomes **faster + more memory-efficient** as context grows.

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
```

## Run Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vladimir0-1/Hybrid-State-Space-Attention-HSA-/blob/main/examples/hsa_demo.ipynb)

Click the badge above to run the benchmark yourself.

## License

MIT © 2026 Vladimir0-1
