"""
Example: Replace BERT's attention with CWAB
Now preserves positional information.
"""

import torch
from transformers import AutoModel, AutoTokenizer
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cwab import CWAB


def replace_bert_attention(model, window_size=512, num_global_tokens=64):
    """Replace all attention layers in BERT-style model with CWAB."""
    for layer in model.encoder.layer:
        old_attn = layer.attention.self
        
        new_attn = CWAB(
            hidden_size=old_attn.query.out_features,
            num_heads=old_attn.num_attention_heads,
            window_size=window_size,
            num_global_tokens=num_global_tokens,
            dropout=old_attn.dropout.p if hasattr(old_attn, 'dropout') else 0.1,
            use_positional_encoding=True,  # Enable positional encoding
        )
        
        layer.attention.self = new_attn
    
    return model


def main():
    print("=" * 60)
    print("CWAB: BERT Attention Replacement Demo (with positions)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    print("\n1. Loading BERT model...")
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Replace attention
    print("\n2. Replacing attention with CWAB (positional encoding ON)...")
    model = replace_bert_attention(model, window_size=512, num_global_tokens=64)
    
    # Test
    print("\n3. Testing forward pass with positions...")
    test_text = "Hello world. This is a test sentence."
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"   Output shape: {outputs.last_hidden_state.shape}")
    print("\n CWAB with positional encoding works!")
    print("   Note: Speed comparison with original BERT is not shown here.")
    print("   Run cwab_demo.py for full benchmarks.")


if __name__ == "__main__":
    main()
