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
    Block-Recurrent Mamba-3
    
    Splits the input sequence into fixed-size blocks and processes them
    sequentially. True cross-block recurrence is achieved by prepending
    the previous block's final output token (state summary) to the current
    block input — so the model always sees where it "left off."
    
    This avoids modifying Mamba3Block internals while still being fully
    end-to-end differentiable.
    
    Args:
        config: Mamba3Config
        block_size: Number of tokens per processing chunk (default 64)
    """
    def __init__(self, config: Mamba3Config, block_size: int = 64):
        super().__init__()
        self.config = config
        self.block_size = block_size
        self.mamba_block = Mamba3Block(config)
        
        # Learned initial state token (acts as h_0, avoids the cold-start problem)
        self.initial_state_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        
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
        
        # h_0: a learned parameter that acts as the "initial state token"
        # It is broadcast to the batch dimension
        prev_state_token = self.initial_state_token.expand(B, 1, D)  # (B, 1, D)
        
        for i in range(0, L, self.block_size):
            chunk = x[:, i : i + self.block_size, :]  # (B, block_size, D)
            
            # ─── Fix 1: True Recurrence ───────────────────────────────────────
            # Prepend the previous block's final state token to the current chunk.
            # This is the "prompt-state" injection: the Mamba block sees the prior
            # block's context as its very first token, giving it a "running memory."
            chunk_with_context = torch.cat([prev_state_token, chunk], dim=1)  # (B, 1+block_size, D)
            
            # Run the Mamba block on the context-augmented chunk
            chunk_out_with_context = self.mamba_block(chunk_with_context)  # (B, 1+block_size, D)
            
            # Strip the prepended state token from the output to get clean per-token outputs
            chunk_out = chunk_out_with_context[:, 1:, :]  # (B, block_size, D)
            
            # The new "state summary" for the next chunk is the last token of the output
            prev_state_token = chunk_out[:, -1:, :]  # (B, 1, D) — detached from the graph later
            
            out_chunks.append(chunk_out)
            memory_bank.append(prev_state_token.squeeze(1))  # Store (B, D) block summary
        
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


# ─────────────────────────────────────────────────────────────────────────────
# Validation Script
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    print("=" * 60)
    print("Block-Recurrent Mamba-3 (v2) — Validation Suite")
    print("=" * 60)
    
    config = Mamba3Config(d_model=64, d_state=32, d_head=16)
    model = HybridBlockRecurrentMamba(config, block_size=64, vocab_size=50, d_out=50)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {total_params:,}")
    
    # Device selection
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}")
    
    model.to(device)
    
    # ─── Test 1: Forward Pass ────────────────────────────────────────────────
    print("\n[Test 1] Forward Pass")
    x = torch.randint(0, 50, (4, 256)).to(device)
    out = model(x)
    assert out.shape == (4, 256, 50), f"Shape mismatch: {out.shape}"
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  ✅ PASS")
    
    # ─── Test 2: Gradient Flow ───────────────────────────────────────────────
    print("\n[Test 2] Gradient Flow & Differentiability")
    targets = torch.randint(0, 50, (4, 256)).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    
    loss = nn.CrossEntropyLoss()(out.view(-1, 50), targets.view(-1))
    print(f"  Forward Loss: {loss.item():.4f}")
    
    loss.backward()
    
    failed_params = []
    zero_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                failed_params.append(name)
            elif torch.all(param.grad == 0):
                zero_grad_params.append(name)
    
    if failed_params:
        print(f"  ❌ FAIL — Parameters with no gradient: {failed_params}")
    else:
        print(f"  ✅ All {sum(p.requires_grad for p in model.parameters())} learnable tensors received gradients")
    
    if zero_grad_params:
        print(f"  ⚠️  Zero-gradient parameters (may be OK for bias/special params): {zero_grad_params[:3]}...")
    
    optimizer.step()
    print(f"  ✅ Optimizer step completed")
    
    # ─── Test 3: Causal Mask Verification ────────────────────────────────────
    print("\n[Test 3] Causal Mask Verification (No Future Leakage)")
    # Verify that flipping a token in the SECOND block does NOT change the
    # cross-attention output for the FIRST block's tokens.
    model.eval()
    
    base_seq = torch.randint(0, 50, (1, 128)).to(device)  # 2 blocks of 64
    modified_seq = base_seq.clone()
    modified_seq[0, 64 + 5] = (modified_seq[0, 64 + 5] + 1) % 50  # Flip token in block 1
    
    with torch.no_grad():
        out_base = model(base_seq)
        out_modified = model(modified_seq)
    
    # The first block's output (positions 0-63) should NOT change
    diff_first_block = (out_base[0, :64] - out_modified[0, :64]).abs().max().item()
    diff_second_block = (out_base[0, 64:] - out_modified[0, 64:]).abs().max().item()
    
    print(f"  Change in first block output (should be ≈0.0): {diff_first_block:.6f}")
    print(f"  Change in second block output (should be > 0): {diff_second_block:.6f}")
    
    if diff_first_block < 1e-5:
        print(f"  ✅ PASS — No information leakage from future blocks into past blocks")
    else:
        print(f"  ❌ FAIL — Future tokens are affecting past outputs! Causal mask is broken.")
    
    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)
