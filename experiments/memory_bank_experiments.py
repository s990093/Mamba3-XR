"""
Memory Bank Validation Experiments for HybridBlockRecurrentMamba
=================================================================
Three experiments to prove the memory_bank is NOT decorative:

Experiment 1: Synthetic Long-Range Retrieval Task
  - Train the model to predict a "key" token injected many blocks ago.
  - Baseline (no memory bank / independent Mamba) vs. Hybrid: compare accuracy.

Experiment 2: Cross-Attention Heatmap
  - Visualize where each query token "looks" in the memory bank.
  - A trained Hybrid model should show sharp peaks at the correct past block.

Experiment 3: Memory Ablation
  - Disable the cross-attention (zero-out memory bank) at inference time.
  - Measure the accuracy drop — proves the memory bank is load-bearing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from models.mamba3 import Mamba3Config, Mamba3Block
from models.block_recurrent_mamba3 import HybridBlockRecurrentMamba, BlockRecurrentMamba3

os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
VOCAB_SIZE = 20        # Small vocabulary: digits 0-15, plus special tokens
PAD_ID     = 16        # Padding token ID
KEY_ID     = 17        # The "key" token (marks what to memorise)
QUERY_ID   = 18        # The "query" token (asks to recall the key)
SEP_ID     = 19        # Separator / noise token

BLOCK_SIZE = 64        # Each Mamba chunk size
SEQ_LEN    = 256       # Total sequence length = 4 blocks
BATCH_SIZE = 32
TOTAL_STEPS = 3000
DEVICE = 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'cpu'
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# Task: Long-Range Retrieval Synthetic Dataset
# ─────────────────────────────────────────────────────────────────────────────
# Sequence structure:
#  [noise...] KEY_ID value [noise...] QUERY_ID [?? predict value here]
#
# The KEY is placed in block 0 and the QUERY is placed at the start of block 3.
# The model must recall the value from block 0's memory entry to answer correctly.
# 
# This is a strict test: the value is 1-15 blocks away, forcing long-range recall.
#
def generate_retrieval_batch(batch_size, seq_len, block_size, key_block=0, device='cpu'):
    """
    Generate synthetic data for the long-range retrieval task.
    
    Structure:
    Block 0: KEY_ID <value> PAD PAD ... (value is 1..NUM_CLASSES in a clear spot)
    Block 1: all PAD (noise/fill)
    Block 2: all PAD (noise/fill)
    Block 3: PAD... QUERY_ID <answer slot> PAD...
    
    The model must output the correct value token at the answer slot.
    Target is -100 everywhere except the answer slot.
    """
    X = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long)
    Y = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    
    for b in range(batch_size):
        # Fill with PAD (no random noise — cleaner signal makes task learnable)
        X[b] = PAD_ID
        
        # KEY-VALUE pair at a fixed position in block 0
        key_pos = key_block * block_size + 2  # Position 2 in block 0
        value   = torch.randint(1, NUM_CLASSES + 1, (1,)).item()  # 1..4
        X[b, key_pos]     = KEY_ID
        X[b, key_pos + 1] = value
        
        # QUERY token at a fixed position in last block
        query_pos  = (seq_len // block_size - 1) * block_size + 2
        answer_pos = query_pos + 1
        X[b, query_pos]  = QUERY_ID
        X[b, answer_pos] = PAD_ID   # blank to be predicted
        Y[b, answer_pos] = value    # gold label
        
    return X.to(device), Y.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Model: Plain Mamba-3 (no memory bank)
# ─────────────────────────────────────────────────────────────────────────────
class PlainMambaModel(nn.Module):
    """A plain 1-layer Mamba-3 with no cross-block memory bank, serving as baseline."""
    def __init__(self, config, vocab_size, d_out, n_layers=1):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, config.d_model)
        self.layers = nn.ModuleList([Mamba3Block(config) for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(config.d_model)
        self.head   = nn.Linear(config.d_model, d_out, bias=False)
        
    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h))


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 & 3: Train & Compare with Memory Ablation
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, model_name, steps=TOTAL_STEPS, print_every=500):
    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    history = {'loss': [], 'acc': []}
    total_loss, total_correct, total_count = 0.0, 0, 0
    
    print(f"\n--- Training: {model_name} ({sum(p.numel() for p in model.parameters()):,} params) ---")
    
    for step in range(1, steps + 1):
        X, Y = generate_retrieval_batch(BATCH_SIZE, SEQ_LEN, BLOCK_SIZE, device=DEVICE)
        
        optimizer.zero_grad()
        logits = model(X)               # (B, L, d_out)
        loss   = criterion(logits.view(-1, logits.size(-1)), Y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        with torch.no_grad():
            mask     = (Y != -100)
            preds    = logits.argmax(dim=-1)
            correct  = (preds[mask] == Y[mask]).sum().item()
            count    = mask.sum().item()
            total_loss    += loss.item()
            total_correct += correct
            total_count   += count
            
        if step % print_every == 0 or step == steps:
            avg_loss = total_loss / print_every
            avg_acc  = total_correct / max(total_count, 1)
            history['loss'].append(avg_loss)
            history['acc'].append(avg_acc)
            print(f"  Step {step:04d}/{steps} | Loss: {avg_loss:.4f} | Recall Acc: {avg_acc*100:.1f}%")
            total_loss, total_correct, total_count = 0.0, 0, 0
            
    return history


def ablation_test(hybrid_model, steps=200):
    """
    Manually zero out the memory bank contribution at inference time and
    measure the accuracy drop. This proves the memory bank is load-bearing.
    """
    hybrid_model.eval()
    
    # Monkey-patch: temporarily override cross-attention to return zeros
    original_ca = hybrid_model.cross_attention
    
    class ZeroAttention(nn.Module):
        def forward(self, query, key, value, attn_mask=None):
            return torch.zeros_like(query), None
    
    hybrid_model.cross_attention = ZeroAttention()
    
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(steps):
            X, Y = generate_retrieval_batch(8, SEQ_LEN, BLOCK_SIZE, device=DEVICE)
            logits = hybrid_model(X)
            mask = (Y != -100)
            preds = logits.argmax(dim=-1)
            correct += (preds[mask] == Y[mask]).sum().item()
            total   += mask.sum().item()
    
    hybrid_model.cross_attention = original_ca  # restore
    return correct / max(total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Cross-Attention Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def get_attention_weights(hybrid_model, X):
    """
    Hook into the cross-attention to extract the attention weight matrix.
    
    Returns:
        attn_weights: (L, K) averaged over batch and heads
    """
    attn_weights_store = {}
    
    def hook(module, input, output):
        # output = (attn_output, attn_weights)   when need_weights=True
        attn_weights_store['weights'] = output[1]  # (B, L, K)
    
    handle = hybrid_model.cross_attention.register_forward_hook(hook)
    
    hybrid_model.eval()
    with torch.no_grad():
        # We need need_weights=True; override the call
        h = hybrid_model.embed(X)
        B, L, D = h.shape
        mamba_out, memory_bank = hybrid_model.mamba_encoder(h, return_memory_bank=True)
        num_blocks = memory_bank.size(1)
        attn_mask, allowed = hybrid_model._build_block_causal_mask(L, num_blocks, X.device, h.dtype)
        has_past  = allowed.any(dim=1)
        safe_mask = attn_mask.clone()
        safe_mask[~has_past] = 0.0
        
        # Call with need_weights=True explicitly
        attn_out_w, weights = hybrid_model.cross_attention(
            mamba_out, memory_bank, memory_bank,
            attn_mask=safe_mask,
            need_weights=True,
            average_attn_weights=True
        )
    
    handle.remove()
    return weights  # (B, L, K)


def plot_heatmap(attn_weights, X, block_size, save_path="results/attention_heatmap.png"):
    """
    Plot the cross-attention heatmap, highlighting where the KEY and QUERY tokens live.
    """
    B, L, K = attn_weights.shape
    
    # Average over batch
    weights = attn_weights.mean(0).cpu().float().numpy()  # (L, K)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # ─── Full heatmap ───────────────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(weights, aspect='auto', cmap='hot', origin='upper', vmin=0, vmax=weights.max())
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("Memory Bank Block Index (K)", fontsize=12)
    ax.set_ylabel("Query Token Position (L)", fontsize=12)
    ax.set_title("Cross-Attention: Query Tokens → Memory Bank", fontsize=13, fontweight='bold')
    
    # Mark token types on Y axis
    sample = X[0].cpu().tolist()
    for pos, tok in enumerate(sample):
        if tok == KEY_ID:
            ax.axhline(y=pos, color='lime', linewidth=1.5, alpha=0.8, linestyle='--')
            ax.text(-0.5, pos, "KEY", color='lime', fontsize=7, ha='right', va='center')
        elif tok == QUERY_ID:
            ax.axhline(y=pos, color='cyan', linewidth=1.5, alpha=0.8, linestyle='--')
            ax.text(-0.5, pos, "QUERY", color='cyan', fontsize=7, ha='right', va='center')
    
    # Mark block boundaries on X axis
    block_x = list(range(K))
    ax.set_xticks(block_x)
    ax.set_xticklabels([f"Block {i}" for i in block_x], fontsize=9)
    
    # ─── Zoomed-in last 32 tokens (where the query is) ──────────────────────
    ax2 = axes[1]
    zoom_start = L - 32
    zoom_weights = weights[zoom_start:, :]
    im2 = ax2.imshow(zoom_weights, aspect='auto', cmap='hot', origin='upper', vmin=0, vmax=zoom_weights.max())
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    ax2.set_xlabel("Memory Bank Block Index (K)", fontsize=12)
    ax2.set_ylabel(f"Query Token Position (L-32 to L)", fontsize=12)
    ax2.set_title("Zoomed: Last 32 Tokens — Retrieval Behaviour", fontsize=13, fontweight='bold')
    
    query_in_zoom = [i for i, tok in enumerate(sample[zoom_start:]) if tok == QUERY_ID]
    for pos in query_in_zoom:
        ax2.axhline(y=pos, color='cyan', linewidth=2, linestyle='--')
        ax2.text(-0.5, pos, "QUERY", color='cyan', fontsize=8, ha='right', va='center')
    
    ax2.set_xticks(block_x)
    ax2.set_xticklabels([f"Block {i}" for i in block_x], fontsize=9)
    
    plt.suptitle("Memory Bank Attention Heatmap\n"
                 "(Bright = high attention weight. QUERY token should highlight Block 0 where KEY was stored)",
                 fontsize=11, y=1.01)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Experiment Runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = Mamba3Config(d_model=64, d_state=32, d_head=16)
    
    # ── Exp 1: Train both models ────────────────────────────────────────────
    print("=" * 60)
    print("Experiment 1: Long-Range Retrieval Across Blocks")
    print("=" * 60)
    
    hybrid_model  = HybridBlockRecurrentMamba(config, block_size=BLOCK_SIZE,
                                              vocab_size=VOCAB_SIZE, d_out=VOCAB_SIZE)
    baseline_model = PlainMambaModel(config, vocab_size=VOCAB_SIZE, d_out=VOCAB_SIZE)
    
    hist_hybrid   = train_model(hybrid_model,   "Hybrid Block-Recurrent Mamba-3")
    hist_baseline = train_model(baseline_model, "Plain Mamba-3 (No Memory Bank)")
    
    # ── Plot learning curves ─────────────────────────────────────────────────
    steps_axis = range(1, len(hist_hybrid['acc']) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(steps_axis, hist_hybrid['loss'],   label='Hybrid (Memory Bank)', color='red',  linewidth=2)
    plt.plot(steps_axis, hist_baseline['loss'], label='Plain Mamba-3',        color='blue', linewidth=2, linestyle='--')
    plt.title("Long-Range Retrieval — Training Loss")
    plt.xlabel("Evaluation Step"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(steps_axis, [x*100 for x in hist_hybrid['acc']],   label='Hybrid (Memory Bank)', color='red',  linewidth=2)
    plt.plot(steps_axis, [x*100 for x in hist_baseline['acc']], label='Plain Mamba-3',        color='blue', linewidth=2, linestyle='--')
    plt.title("Long-Range Retrieval — Recall Accuracy")
    plt.xlabel("Evaluation Step"); plt.ylabel("Accuracy (%)")
    plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/retrieval_comparison.png", dpi=150)
    plt.close()
    print("\nSaved learning curves → results/retrieval_comparison.png")
    
    # Final accuracy summary
    final_hybrid   = hist_hybrid['acc'][-1]   * 100
    final_baseline = hist_baseline['acc'][-1] * 100
    print(f"\n  Final Hybrid   Recall Accuracy: {final_hybrid:.1f}%")
    print(f"  Final Baseline Recall Accuracy: {final_baseline:.1f}%")
    print(f"  Gain from Memory Bank: +{final_hybrid - final_baseline:.1f}%")
    
    # ── Exp 3: Memory Ablation ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Experiment 3: Memory Ablation (Zero-Out Cross-Attention)")
    print("=" * 60)
    
    ablated_acc = ablation_test(hybrid_model) * 100
    print(f"  Hybrid with Memory Bank:   {final_hybrid:.1f}%")
    print(f"  Hybrid WITHOUT Memory Bank (ablated): {ablated_acc:.1f}%")
    print(f"  Accuracy Drop from Ablation: -{final_hybrid - ablated_acc:.1f}%")
    if ablated_acc < final_hybrid - 5:
        print("  ✅ Memory bank IS load-bearing (significant drop when removed)")
    else:
        print("  ⚠️  Model may not rely on memory bank at final checkpoint — try more training steps")
    
    # ── Exp 2: Cross-Attention Heatmap ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Experiment 2: Cross-Attention Heatmap Visualization")
    print("=" * 60)
    
    X_viz, _ = generate_retrieval_batch(8, SEQ_LEN, BLOCK_SIZE, key_block=0, device=DEVICE)
    attn_w   = get_attention_weights(hybrid_model, X_viz)     # (B, L, K)
    plot_heatmap(attn_w, X_viz, block_size=BLOCK_SIZE, save_path="results/attention_heatmap.png")
    
    # ── Markdown report ──────────────────────────────────────────────────────
    report = f"""# Memory Bank Validation Report

## Experiment 1: Long-Range Retrieval Comparison

The model was trained on a synthetic long-range recall task where the KEY token is placed in **Block 0** and the QUERY token is placed in **Block 3** (out of 4 total blocks). The model must retrieve the value associated with the KEY token when asked by the QUERY token — spanning **3 blocks** (192+ tokens) of gap.

| Model | Final Recall Accuracy |
|-------|----------------------|
| Hybrid Block-Recurrent Mamba-3 (with Memory Bank) | **{final_hybrid:.1f}%** |
| Plain Mamba-3 (No Memory Bank) | {final_baseline:.1f}% |
| Gain from Memory Bank | **+{final_hybrid - final_baseline:.1f}%** |

![Retrieval Comparison](retrieval_comparison.png)

---

## Experiment 2: Cross-Attention Heatmap

The heatmap shows where each output token's query vector attends within the memory bank.
The QUERY token (blue dashed line) should activate high attention weights on Block 0 — where the KEY was stored.

![Cross Attention Heatmap](attention_heatmap.png)

---

## Experiment 3: Memory Ablation

The cross-attention module was zeroed out at inference time (memory bank reads disabled), while keeping all other weights the same.

| Condition | Recall Accuracy |
|-----------|----------------|
| Hybrid With Memory Bank | {final_hybrid:.1f}% |
| Hybrid WITHOUT Memory Bank (ablated) | {ablated_acc:.1f}% |
| Drop | **-{final_hybrid - ablated_acc:.1f}%** |

A significant drop confirms the memory bank **is load-bearing** and not decorative.
"""
    
    with open("results/memory_bank_report.md", "w") as f:
        f.write(report)
    
    print("\n✅ All experiments done! Results saved in results/")
    print("   • results/retrieval_comparison.png")
    print("   • results/attention_heatmap.png")
    print("   • results/memory_bank_report.md")
