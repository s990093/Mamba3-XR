"""
Block-Recurrent Mamba-3 with Hybrid Cross-Attention (v2)
=========================================================
This module implements a truly causal, block-recurrent Mamba-3 architecture.

Two bugs from v1 are fixed here:

Bug 1: No true recurrence between chunks
-----------------------------------------
Original:
    chunk_out = self.mamba_block(chunk)   # no state passed →  each chunk starts fresh!

Fix:
    The Mamba3Block performs a chunk-wise parallel scan internally, but its `forward(x)`
    interface does not expose an initial state. Instead of modifying mamba3.py, we
    implement recurrence at the sequence level:
    - We run the full Mamba block on the chunk.
    - The final token's output embedding represents the running "state summary."
    - We prepend this summary token to the next chunk as a soft initial-state prompt,
      giving the model access to prior context without touching the original architecture.
    - This is the "prompt-state injection" technique — the model learns to use the
      prepended token as its memory across chunk boundaries.

Bug 2: Information leakage (future blocks attended by past tokens)
-------------------------------------------------------------------
Original:
    attn_out, _ = self.cross_attention(query=mamba_out, key=memory_bank, value=memory_bank)
    # query at block k can see blocks k+1 ... K (future!) in memory_bank

Fix:
    We construct a block-level causal mask over the memory bank.
    Block k can only attend to memory_bank slots 0 ... k-1 (strictly past).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mamba3 import Mamba3Config, Mamba3Block


class BlockRecurrentMamba3(nn.Module):
    """
    Block-Recurrent Mamba-3 (Multi-Layer Version)
    
    Splits the input sequence into fixed-size blocks and processes them
    sequentially. True cross-block recurrence is achieved by prepending
    the previous block's final output token (state summary) to the current
    block input for each individual layer.
    
    Args:
        config: Mamba3Config
        block_size: Number of tokens per processing chunk (default 64)
        num_layers: Number of Mamba3Block layers to stack
    """
    def __init__(self, config: Mamba3Config, block_size: int = 64, num_layers: int = 15):
        super().__init__()
        self.config = config
        self.block_size = block_size
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([Mamba3Block(config) for _ in range(num_layers)])
        
        # Import RMSNorm dynamically to ensure pre-norm is used
        from models.mamba3 import RMSNorm
        self.norms = nn.ModuleList([RMSNorm(config.d_model) for _ in range(num_layers)])
        
        # Learned initial state token (acts as h_0, avoids the cold-start problem) per layer
        self.initial_state_tokens = nn.Parameter(torch.zeros(num_layers, 1, 1, config.d_model))
        
    def forward(self, x, return_memory_bank: bool = False):
        """
        Args:
            x: (B, L, d_model) — input sequence
            return_memory_bank: If True, also return the memory bank tensor
                                of shape (B, num_blocks, d_model)
        Returns:
            out: (B, L, d_model) — processed sequence
            memory_bank (optional): (B, num_blocks, d_model) — block summaries
        """
        B, L, D = x.shape
        out_chunks = []
        memory_bank = []
        
        # h_0 for each layer
        prev_state_tokens = [self.initial_state_tokens[i].expand(B, 1, D) for i in range(self.num_layers)]
        
        for i in range(0, L, self.block_size):
            chunk = x[:, i : i + self.block_size, :]  # (B, block_size, D)
            chunk_out = chunk
            new_prev_state_tokens = []
            
            # ─── Fix 1: True Recurrence Per Layer ─────────────────────────────
            for j, layer in enumerate(self.layers):
                normed_chunk = self.norms[j](chunk_out)
                
                # Prepend the previous block's final state token for this layer
                chunk_with_context = torch.cat([prev_state_tokens[j], normed_chunk], dim=1)  # (B, 1+block_size, D)
                
                # Run the Mamba block
                layer_out_with_context = layer(chunk_with_context)  # (B, 1+block_size, D)
                
                # Strip the prepended state token
                layer_out = layer_out_with_context[:, 1:, :]  # (B, block_size, D)
                
                # State summary for the next chunk (Detach to prevent backward pass across chunks)
                new_state = layer_out_with_context[:, -1:, :].detach()  # (B, 1, D)
                new_prev_state_tokens.append(new_state)
                
                # Residual connection
                chunk_out = chunk_out + layer_out
                
            prev_state_tokens = new_prev_state_tokens
            
            out_chunks.append(chunk_out)
            # Memory bank summary is the final layer's new_state
            memory_bank.append(new_prev_state_tokens[-1].squeeze(1))  # (B, D)
        
        out = torch.cat(out_chunks, dim=1)  # (B, L, D)
        
        if return_memory_bank:
            memory_bank_tensor = torch.stack(memory_bank, dim=1)  # (B, num_blocks, D)
            return out, memory_bank_tensor
        
        return out


class HybridBlockRecurrentMamba(nn.Module):
    """
    Hybrid Block-Recurrent Mamba-3 with Causally-Masked Cross-Attention

    Combines:
    - BlockRecurrentMamba3 for fast local processing + memory bank construction
    - A causally-masked Cross-Attention layer for precise long-range retrieval
      from the memory bank WITHOUT information leakage

    The attention mask ensures that token positions in block k can ONLY attend
    to memory bank slots 0 ... k-1 (strictly past blocks), not the present or
    future blocks.

    Args:
        config: Mamba3Config
        block_size: Number of tokens per Mamba processing chunk
        vocab_size: Vocabulary size for the embedding layer
        d_out: Output dimension (number of classes)
    """
    def __init__(self, config: Mamba3Config, block_size: int = 64, vocab_size: int = 10, d_out: int = 10):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, config.d_model)
        
        self.mamba_encoder = BlockRecurrentMamba3(config, block_size=block_size)
        
        # Cross-Attention: queries from Mamba output attend to the memory bank
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=4,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, d_out, bias=False)
        
    def _build_block_causal_mask(self, seq_len: int, num_blocks: int, device, dtype):
        """
        Build a block-level causal attention mask as an additive float mask.
        
        Token at position i (in block k = i // block_size) may attend to memory
        slot j only if j < k (strictly past blocks).
        
        For the very first block (k=0) all memory slots are masked → would give
        softmax(-inf,-inf,...) = NaN. We handle this by detecting fully-masked
        rows and letting them through with uniform zero contribution (via a separate
        fallback gate, see forward()).

        Returns:
            attn_mask: (seq_len, num_blocks) float tensor
                        0.0   → allowed to attend
                       -inf   → blocked
        """
        # Block index of each token
        token_blocks = torch.arange(seq_len, device=device) // self.block_size  # (L,)
        # Memory slot indices
        mem_slots    = torch.arange(num_blocks, device=device)                   # (K,)
        
        # allowed[i, j] = True if j < token_block[i]
        allowed = mem_slots.unsqueeze(0) < token_blocks.unsqueeze(1)  # (L, K)
        
        # Float mask: 0.0 where allowed, -inf where blocked
        attn_mask = torch.zeros(seq_len, num_blocks, device=device, dtype=dtype)
        attn_mask[~allowed] = float('-inf')

        return attn_mask, allowed  # also return allowed so forward() can gate the output
    
    def forward(self, x):
        """
        Args:
            x: (B, L) — integer token indices
        Returns:
            logits: (B, L, d_out)
        """
        h = self.embed(x)  # (B, L, d_model)
        B, L, D = h.shape
        
        # 1. Block-Recurrent Mamba encoding + memory bank construction
        mamba_out, memory_bank = self.mamba_encoder(h, return_memory_bank=True)
        
        num_blocks = memory_bank.size(1)  # K
        
        # ─── Fix 2: Block-level Causal Mask (float, -inf style) ──────────────
        attn_mask, allowed = self._build_block_causal_mask(L, num_blocks, x.device, h.dtype)
        
        # For rows that are ENTIRELY blocked (all-masked), softmax(-inf,...) → NaN.
        # We detect these rows and zero-out their cross-attention contribution.
        has_past = allowed.any(dim=1)  # (L,) True if the token has at least one past block
        
        # Replace fully-blocked rows with uniform 0 mask to avoid NaN in softmax.
        # Those rows will later be multiplied by 0 via the gate.
        safe_mask = attn_mask.clone()
        safe_mask[~has_past] = 0.0  # allow attention, but we'll gate result to 0
        
        attn_out, _ = self.cross_attention(
            query=mamba_out,     # (B, L, D)
            key=memory_bank,     # (B, K, D)
            value=memory_bank,   # (B, K, D)
            attn_mask=safe_mask  # (L, K) float additive mask
        )
        
        # Zero out the cross-attention contribution for positions with NO past blocks.
        # Shape: (1, L, 1) broadcast over (B, L, D)
        gate = has_past.float().view(1, L, 1)
        attn_out = attn_out * gate
        
        # 3. Residual + layer norm
        h_final = self.norm(mamba_out + attn_out)
        return self.head(h_final)  # (B, L, d_out)


