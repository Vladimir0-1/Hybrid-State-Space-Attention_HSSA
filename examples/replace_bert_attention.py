"""
Example: Replace BERT's attention with HSA
"""

from transformers import AutoModel
import torch
from hsa import HybridStateSpaceAttention

# Load any transformer model
model = AutoModel.from_pretrained("bert-base-uncased")

# Replace each attention layer
for layer in model.encoder.layer:
    old_attn = layer.attention.self
    layer.attention.self = HybridStateSpaceAttention(
        hidden_size=old_attn.query.out_features,
        num_heads=old_attn.num_attention_heads,
        window_size=512,
        num_global_tokens=64,
    )

# Forward pass with linear complexity
dummy_input = torch.randint(0, 30000, (1, 4096))  # long sequence
output = model(dummy_input)
print(f"Output shape: {output.last_hidden_state.shape}")
