# Mamba-3 LLM - Language Modeling Experiments

This directory contains the Language Modeling adaptation of Mamba-3.

## 📁 Structure

```
llm/
├── README.md                    # This file
├── IMPLEMENTATION_PLAN.md       # Detailed research plan
├── mamba3_lm.py                # Core LLM implementation
├── train_shakespeare.py         # Hello World training script
└── data/                        # (Created automatically)
    └── shakespeare.txt          # Downloaded on first run
```

## 🚀 Quick Start

### 1. Test the Model

```bash
cd llm
python mamba3_lm.py
```

### 2. Train on Shakespeare

```bash
python train_shakespeare.py
```

## 🎯 Research Goal

Investigate **MIMO rank saturation** in Language Modeling and compare with Computer Vision findings.

**CV Results (CIFAR-100):**

- Rank 4 saturated at 87.5% (28/32 dimensions)
- Rank 8 showed minimal improvement (+0.48%)

**LLM Hypothesis:**

- Language may require higher ranks due to greater complexity
- Testing Rank 1, 4, 8, 16

## 📊 Model Presets

| Name | d_model | Layers | Params | VRAM  | Use Case             |
| ---- | ------- | ------ | ------ | ----- | -------------------- |
| Tiny | 256     | 4      | ~10M   | ~2GB  | Shakespeare testing  |
| 125M | 768     | 12     | ~125M  | ~14GB | RTX 3090 experiments |
| 350M | 1024    | 24     | ~350M  | ~22GB | Full-scale training  |

## 🔧 Dependencies

All dependencies from the main project, plus:

```bash
pip install requests  # For dataset download
```

## 📝 Files Description

### `mamba3_lm.py`

Core Mamba-3 Language Model implementation:

- `Mamba3LM` class with embedding and LM head
- Autoregressive generation (temperature, top-k, top-p)
- Model presets: `create_mamba3_tiny()`, `create_mamba3_125m()`, `create_mamba3_350m()`

### `train_shakespeare.py`

Complete training script for Shakespeare dataset:

- Character-level tokenization
- Training loop with evaluation
- Sample generation during training
- Model checkpointing

### `IMPLEMENTATION_PLAN.md`

Comprehensive research plan:

- 5 implementation phases
- Experimental matrix
- Dataset recommendations
- Analysis methodology

## 🎓 Research Contribution

**Novel Aspects:**

1. First systematic study of MIMO rank saturation in LLMs
2. Direct comparison between CV and NLP domains
3. Practical guidelines for Mamba-3 LLM configuration

## 📈 Next Steps

1. ✅ Test Shakespeare training
2. Download TinyStories dataset
3. Run Rank 1, 4, 8, 16 experiments
4. Compare with CV findings
5. Write research report

## 🔗 Related

- Main project: `../`
- CV analysis: `../docs/rank_comparison_analysis.pdf`
- Base Mamba-3 blocks: `../model.py`

---

**Status:** Phase 1 Complete ✅ | Ready for experiments 🚀
