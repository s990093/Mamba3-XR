# Mamba-3 Language Model Implementation
# Adapted from Vision Mamba for Language Modeling Tasks

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from pathlib import Path


# Import the core Mamba3Block and Config from existing model.py
from model import Mamba3Config, Mamba3Block, RMSNorm


class Mamba3LM(nn.Module):
    """
    Mamba-3 Language Model
    
    Architecture:
    - Token Embedding
    - Positional Encoding (optional, Mamba is position-aware via SSM)
    - Stack of Mamba-3 Blocks (Causal)
    - Language Modeling Head
    
    Key Features:
    - Causal (autoregressive) generation
    - MIMO rank experiments (1, 4, 8, 16)
    - Compatible with Llama-3.1 tokenizer
    - Optimized for RTX 3090 (24GB)
    """
    
    def __init__(
        self,
        vocab_size=50257,        # GPT-2 tokenizer size (or use Llama-3.1: 128256)
        d_model=512,             # Model dimension (512 for 125M, 1024 for 350M)
        n_layers=12,             # Number of Mamba blocks
        d_state=64,              # SSM state dimension
        d_head=64,               # Head dimension
        n_groups=4,              # Grouped SSM
        mimo_rank=4,             # MIMO rank (1, 4, 8, 16)
        expand=2,                # Expansion factor
        max_seq_len=2048,        # Maximum sequence length
        dropout=0.0,             # Dropout rate
        tie_embeddings=True,     # Tie input/output embeddings
        use_rope=False,          # Use RoPE (Mamba already has positional info)
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings
        
        # 1. Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Optional: Learned Positional Embedding
        # Note: Mamba's SSM is inherently position-aware, so this is optional
        if use_rope:
            self.pos_embedding = None  # RoPE is handled in Mamba3Block
        else:
            # Simple learned positional embedding (like GPT-2)
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
            nn.init.normal_(self.pos_embedding, std=0.02)
        
        # 3. Mamba-3 Configuration
        self.config = Mamba3Config(
            d_model=d_model,
            d_state=d_state,
            d_head=d_head,
            n_groups=n_groups,
            mimo_rank=mimo_rank,
            expand=expand,
            use_conv=False,          # No conv for LLM (optional)
            d_conv=4,
            chunk_size=256,          # Adjust based on seq_len
            use_parallel_scan=True,  # Essential for speed
        )
        
        # 4. Mamba-3 Blocks (Causal)
        self.layers = nn.ModuleList([
            Mamba3Block(self.config) for _ in range(n_layers)
        ])
        
        # 5. Layer Norms (Pre-norm architecture)
        self.norms = nn.ModuleList([
            RMSNorm(d_model) for _ in range(n_layers)
        ])
        
        # 6. Final Norm
        self.final_norm = RMSNorm(d_model)
        
        # 7. Language Modeling Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 8. Tie embeddings (reduce parameters)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        
        # 9. Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights (GPT-2 style)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None, return_loss=True):
        """
        Forward pass
        
        Args:
            input_ids: (batch_size, seq_len) - Token indices
            targets: (batch_size, seq_len) - Target tokens for loss calculation
            return_loss: Whether to compute and return loss
            
        Returns:
            If return_loss and targets provided:
                loss, logits
            Else:
                logits
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Token Embedding
        x = self.token_embedding(input_ids)  # (B, L, D)
        
        # 2. Add Positional Embedding (if not using RoPE)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding[:, :seq_len, :]
        
        x = self.dropout(x)
        
        # 3. Mamba-3 Blocks (Causal)
        for i, layer in enumerate(self.layers):
            # Pre-norm
            normed_x = self.norms[i](x)
            
            # Mamba block (inherently causal due to recurrent nature)
            out = layer(normed_x)
            
            # Residual connection
            x = x + out
        
        # 4. Final Norm
        x = self.final_norm(x)
        
        # 5. Language Modeling Head
        logits = self.lm_head(x)  # (B, L, vocab_size)
        
        # 6. Compute Loss (if targets provided)
        loss = None
        if targets is not None and return_loss:
            # Shift logits and targets for next-token prediction
            # logits: (B, L-1, V), targets: (B, L-1)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100  # Ignore padding tokens
            )
        
        if return_loss and targets is not None:
            return loss, logits
        else:
            return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=1.0,
        top_k=None,
        top_p=None,
        eos_token_id=None,
    ):
        """
        Autoregressive generation
        
        Args:
            input_ids: (batch_size, seq_len) - Prompt tokens
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            eos_token_id: End-of-sequence token ID
            
        Returns:
            generated_ids: (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate if exceeds max_seq_len
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]
            
            # Forward pass
            logits = self.forward(input_ids, return_loss=False)
            
            # Get logits for the last token
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return input_ids
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), subtract the token embeddings.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            if not self.tie_embeddings:
                n_params -= self.lm_head.weight.numel()
        return n_params


# ============================================================================
# Model Size Presets (for easy experimentation)
# ============================================================================

def create_mamba3_125m(vocab_size=50257, mimo_rank=4):
    """125M parameter model (RTX 3090 friendly)"""
    return Mamba3LM(
        vocab_size=vocab_size,
        d_model=768,
        n_layers=12,
        d_state=64,
        d_head=64,
        n_groups=4,
        mimo_rank=mimo_rank,
        expand=2,
        max_seq_len=1024,
    )

def create_mamba3_350m(vocab_size=50257, mimo_rank=4):
    """350M parameter model"""
    return Mamba3LM(
        vocab_size=vocab_size,
        d_model=1024,
        n_layers=24,
        d_state=128,
        d_head=64,
        n_groups=4,
        mimo_rank=mimo_rank,
        expand=2,
        max_seq_len=2048,
    )

def create_mamba3_tiny(vocab_size=50257, mimo_rank=4):
    """Tiny model for quick testing (Shakespeare)"""
    return Mamba3LM(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=4,
        d_state=32,
        d_head=64,
        n_groups=2,
        mimo_rank=mimo_rank,
        expand=2,
        max_seq_len=512,
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test model creation
    print("=" * 80)
    print("Mamba-3 Language Model Test")
    print("=" * 80)
    
    # Create tiny model for testing
    model = create_mamba3_tiny(vocab_size=50257, mimo_rank=4)
    
    print(f"\nModel Parameters: {model.get_num_params() / 1e6:.2f}M")
    print(f"Non-embedding Parameters: {model.get_num_params(non_embedding=True) / 1e6:.2f}M")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    targets = torch.randint(0, 50257, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Forward pass with loss
    loss, logits = model(input_ids, targets=targets)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Logits shape: {logits.shape}")
    
    # Test generation
    prompt = torch.randint(0, 50257, (1, 10))
    print(f"\nPrompt shape: {prompt.shape}")
    
    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50
    )
    
    print(f"Generated shape: {generated.shape}")
    print("\n✅ Model test passed!")
