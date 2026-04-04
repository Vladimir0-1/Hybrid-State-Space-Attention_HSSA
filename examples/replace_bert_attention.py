"""
Example: Replace BERT's attention with HSA
Shows how to plug HSA into any transformer model.
"""

import torch
from transformers import AutoModel, AutoTokenizer
import sys
import os

# Add parent directory to path to import hsa
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hsa import HybridStateSpaceAttention


def replace_bert_attention(model, window_size=512, num_global_tokens=64):
    """
    Replace all attention layers in BERT-style model with HSA.
    
    Args:
        model: HuggingFace BERT-like model
        window_size: Local attention window size
        num_global_tokens: Number of compressed global tokens
    
    Returns:
        Model with HSA attention
    """
    for layer in model.encoder.layer:
        old_attn = layer.attention.self
        
        # Create HSA with same dimensions
        new_attn = HybridStateSpaceAttention(
            hidden_size=old_attn.query.out_features,
            num_heads=old_attn.num_attention_heads,
            window_size=window_size,
            num_global_tokens=num_global_tokens,
            dropout=old_attn.dropout.p if hasattr(old_attn, 'dropout') else 0.1,
        )
        
        # Replace
        layer.attention.self = new_attn
    
    return model


def main():
    print("Loading BERT model...")
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Count parameters
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original model parameters: {original_params:,}")
    
    # Replace attention
    print("Replacing attention with HSA...")
    model = replace_bert_attention(model, window_size=512, num_global_tokens=64)
    
    # Verify replacement
    new_params = sum(p.numel() for p in model.parameters())
    print(f"HSA model parameters: {new_params:,}")
    print(f"Parameter change: {new_params - original_params:+,}")
    
    # Test forward pass with long sequence
    print("\nTesting forward pass with long sequence (4096 tokens)...")
    dummy_text = "Hello world. " * 512  # ~4096 tokens
    inputs = tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=4096)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Output shape: {outputs.last_hidden_state.shape}")
    print(" HSA works! Forward pass completed.")
    
    # Optional: benchmark
    import time
    print("\nBenchmarking (3 runs)...")
    times = []
    for _ in range(3):
        start = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        times.append(time.time() - start)
    
    print(f"Average inference time: {sum(times)/len(times):.3f} seconds")
    print(f"Memory usage: {torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 'CPU mode'}")


if __name__ == "__main__":
    main()
