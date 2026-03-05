# Mamba-3 LLM Implementation Plan

## 🎯 Project Goal

Adapt the Mamba-3 architecture from Computer Vision (CIFAR-100) to Language Modeling, investigating whether **MIMO rank saturation** occurs at different points for text vs. image data.

## 📋 Research Questions

1. **Does MIMO rank saturation shift in NLP?**

   - CIFAR-100: Rank 4 saturated at ~28/32 (87.5% utilization)
   - Hypothesis: Language may require higher ranks (8 or 16) due to greater complexity

2. **State health vs. retrieval ability**

   - Can higher MIMO ranks improve context retention?
   - Test with Needle-in-a-Haystack benchmarks

3. **Optimal configuration for LLM**
   - What is the sweet spot for language modeling?
   - How does it compare to the CV findings?

## 🏗️ Implementation Phases

### ✅ Phase 1: Architecture Adaptation (COMPLETED)

**Files Created:**

- `mamba3_lm.py` - Core LLM implementation
- `train_shakespeare.py` - Hello World test script

**Key Changes from CV Model:**

1. **Input Layer**

   - ❌ Removed: `PatchEmbedding` (for images)
   - ✅ Added: `nn.Embedding(vocab_size, d_model)` (for tokens)

2. **Positional Encoding**

   - Optional learned positional embeddings
   - Mamba's SSM is inherently position-aware via RoPE

3. **Output Layer**

   - Added: Language Modeling Head (`nn.Linear(d_model, vocab_size)`)
   - Supports weight tying with input embeddings

4. **Causality**

   - Mamba's recurrent nature is inherently causal
   - Parallel scan uses causal masking (lower triangular)

5. **Generation**
   - Implemented autoregressive generation with:
     - Temperature sampling
     - Top-k sampling
     - Top-p (nucleus) sampling

**Model Presets:**

- `create_mamba3_tiny()` - 256d, 4 layers (Shakespeare testing)
- `create_mamba3_125m()` - 768d, 12 layers (RTX 3090 friendly)
- `create_mamba3_350m()` - 1024d, 24 layers (Full experiment)

### 🔄 Phase 2: Hello World Testing (IN PROGRESS)

**Dataset:** Shakespeare Complete Works (~1MB)

- Character-level tokenization
- Vocabulary: ~65 unique characters
- Context length: 256 tokens
- Train/val split: 90/10

**Test Configuration:**

```python
d_model = 256
n_layers = 4
d_state = 32
mimo_rank = 4
vocab_size = ~65
max_seq_len = 512
```

**Success Criteria:**

- ✅ Loss decreases
- ✅ Model generates coherent Shakespeare-style text
- ✅ No CUDA errors or OOM

**Run Command:**

```bash
python train_shakespeare.py
```

### 📊 Phase 3: Data Preparation

**Recommended Datasets (in order of complexity):**

1. **TinyStories** (Quick validation)

   - Size: ~2GB
   - Tokens: ~2B
   - Vocabulary: GPT-2 tokenizer (50,257)
   - Use case: Fast convergence testing

2. **FineWeb-Edu Sample** (Main experiments)
   - Size: Sample-10BT (~100GB)
   - Tokens: 10B
   - Vocabulary: Llama-3.1 tokenizer (128,256)
   - Use case: Rank saturation experiments

**Tokenizer Setup:**

```python
from transformers import AutoTokenizer

# Option 1: GPT-2 (for TinyStories)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Option 2: Llama-3.1 (for FineWeb-Edu)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
```

### 🧪 Phase 4: Rank Saturation Experiments

**Experimental Matrix:**

| Config    | d_model | Layers | d_state | MIMO Rank | Params | VRAM (est.) |
| --------- | ------- | ------ | ------- | --------- | ------ | ----------- |
| Tiny-R1   | 512     | 12     | 64      | 1         | ~60M   | ~8GB        |
| Tiny-R4   | 512     | 12     | 64      | 4         | ~80M   | ~10GB       |
| Tiny-R8   | 512     | 12     | 64      | 8         | ~100M  | ~12GB       |
| Small-R1  | 768     | 12     | 64      | 1         | ~100M  | ~12GB       |
| Small-R4  | 768     | 12     | 64      | 4         | ~125M  | ~14GB       |
| Small-R8  | 768     | 12     | 64      | 8         | ~150M  | ~16GB       |
| Small-R16 | 768     | 12     | 64      | 16        | ~200M  | ~20GB       |

**Training Hyperparameters (from Mamba-3 paper):**

```python
optimizer = AdamW
lr = 3e-4
weight_decay = 0.01
batch_size = 32 (adjust for VRAM)
seq_len = 1024 (start with 512)
precision = bfloat16
warmup_steps = 2000
max_steps = 100k
```

**Metrics to Track:**

1. **Performance Metrics**

   - Perplexity (validation)
   - Bits-per-byte
   - Downstream task accuracy (if applicable)

2. **Internal State Metrics** (reuse from CV analysis)

   - MIMO effective rank evolution
   - State L2 norm and variance
   - Delta parameter CV
   - SSM eigenvalue statistics
   - A_log stability (SNR)

3. **Efficiency Metrics**
   - Parameters / Performance ratio
   - Training throughput (tokens/sec)
   - Memory usage

### 📈 Phase 5: Analysis & Reporting

**Comparative Analysis:**

1. **MIMO Rank Saturation**

   - Plot: Rank vs. Perplexity
   - Compare to CV findings (Rank 4 @ 87.5% utilization)
   - Determine if language needs higher ranks

2. **State Health Analysis**

   - Reuse visualization tools from CV project
   - Compare state dynamics between CV and NLP

3. **Retrieval Capability**
   - Needle-in-a-Haystack test
   - Long-range dependency tests
   - Compare different ranks

**Report Structure:**

```
1. Executive Summary
2. Background & Motivation
3. Methodology
   3.1 Architecture Adaptation
   3.2 Experimental Setup
4. Results
   4.1 MIMO Rank Saturation in LLM
   4.2 Comparison with CV Findings
   4.3 State Health Analysis
   4.4 Retrieval Performance
5. Discussion
   5.1 Why Language Differs from Vision
   5.2 Optimal Configuration
6. Conclusion & Future Work
```

## 🛠️ Technical Implementation Details

### Causal Masking

Mamba's recurrent formulation is inherently causal:

```python
h_t = α_t * h_{t-1} + u_t  # Can only see past
```

The parallel scan implementation uses `segsum()` with causal masking:

```python
mask = torch.tril(torch.ones(T, T), diagonal=0)
x_segsum = x_segsum.masked_fill(~mask, -float('inf'))
```

### Memory Optimization for RTX 3090

1. **Gradient Checkpointing**

   ```python
   from torch.utils.checkpoint import checkpoint
   out = checkpoint(layer, x, use_reentrant=False)
   ```

2. **Mixed Precision (bfloat16)**

   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   with autocast(dtype=torch.bfloat16):
       loss, logits = model(x, y)
   ```

3. **Gradient Accumulation**
   ```python
   accumulation_steps = 4
   for i, (x, y) in enumerate(dataloader):
       loss = model(x, y) / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

### QK-Normalization (Mamba-3 LLM Enhancement)

From the paper, add normalization to B and C projections:

```python
# Already implemented in model.py:
self.norm_B = RMSNorm(N * R)
self.norm_C = RMSNorm(N * R)

B_param = self.norm_B(B_param.flatten(-2, -1)).view(...) + self.bias_B
C_param = self.norm_C(C_param.flatten(-2, -1)).view(...) + self.bias_C
```

## 📝 Next Steps

1. **Immediate (Today)**

   - ✅ Test `train_shakespeare.py`
   - ✅ Verify loss decreases
   - ✅ Check generated text quality

2. **Short-term (This Week)**

   - Download TinyStories dataset
   - Implement proper tokenization
   - Run Rank 1, 4, 8 experiments
   - Collect MIMO rank evolution data

3. **Medium-term (Next 2 Weeks)**

   - Scale to FineWeb-Edu sample
   - Run full experimental matrix
   - Implement retrieval tests
   - Generate comparative visualizations

4. **Long-term (Research Paper)**
   - Write comprehensive report
   - Compare CV vs. NLP findings
   - Publish results (GitHub + arXiv)
   - Use for grad school applications

## 🎓 Research Contribution

**Novel Aspects:**

1. First systematic study of MIMO rank saturation in LLMs
2. Direct comparison between CV and NLP domains
3. State health analysis for language models
4. Practical guidelines for Mamba-3 configuration

**Expected Findings:**

- Language may saturate at higher ranks than vision (Rank 8-16 vs. Rank 4)
- State health metrics correlate with retrieval performance
- Optimal rank depends on task complexity and data distribution

**Impact:**

- Guides future Mamba-3 LLM implementations
- Demonstrates architecture generalization
- Strong portfolio piece for grad school applications

## 📚 References

1. Mamba-3 Paper: [arXiv link]
2. FineWeb-Edu Dataset: [HuggingFace link]
3. Llama-3.1 Tokenizer: [HuggingFace link]
4. Your CV Analysis: `docs/rank_comparison_analysis.pdf`

---

**Status:** Phase 1 Complete ✅ | Phase 2 In Progress 🔄
**Last Updated:** 2025-12-29
