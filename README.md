# Mamba3-XR & TD-MoE

> **Mamba-3: Enhanced State Space Models with Trapezoidal Discretization and MIMO Projections**
> **TD-MoE: Tensor Decomposition Mixture of Experts**
> Research Implementation & Presentation Assets

## Overview

This repository contains the training code, inference implementation, and interactive 3D paper assets for **Mamba-3** and **TD-MoE (Tensor Decomposition Mixture of Experts)**. 

### Key Innovations:
1. **Mamba-3 (Trapezoidal Discretization)**: Replaces standard Euler methods with a higher-order trapezoidal integration scheme and MIMO projections, pushing sequence lengths up to 32K tokens efficiently.
2. **TD-MoE (Tensor Decomposition MoE)**: A novel architecture that compresses massive MoE parameter matrices by up to 94% using Tucker Decomposition. It eliminates memory bandwidth bottlenecks via an **On-the-fly Inference Pipeline** without needing to reconstruct the full weight matrices.

## 📂 Repository Structure

```text
Mamba3-XR/
├── paper/
│   └── td-moe-iclr2026/
│       └── td-moe-simulator-react/  # 🖥️ Interactive 3D presentation simulator (React)
├── pre-train/                       # 🏋️ Model pre-training scripts and datasets
├── inference/                       # ⚡ Model evaluation and on-the-fly inference scripts
├── mamba/                           # 🧠 Core Mamba/MIMO architecture blocks
├── benchmarks/                      # 📊 Profiling and latency evaluations
├── train.py                         # 🎯 Unified standalone training script
└── docs/                            # 📖 Internal technical reports and documentation
```

---

## 🖥 TD-MoE 3D Interactive Simulator

As part of the ICLR 2026 paper submission, we provide an interactive 3D web-based presentation simulator. 
It visually demonstrates:
- **Tucker Matrix Decomposition** compressing the state space from $134\text{MB}$ to $8.4\text{MB}$.
- **On-the-fly Inference Pipeline**, comparing native block latency against our specialized micro-tensor flow.

### Running the Simulator (Development)

The simulator is built with React, Vite, and Tailwind CSS.
```bash
# 1. Navigate to the simulator directory
cd paper/td-moe-iclr2026/td-moe-simulator-react

# 2. Install dependencies
npm install

# 3. Start the dev server
npm run dev
```

Navigate to `http://localhost:5173` to interact with the 3D pipeline.

---

## 🧠 Mamba-3 Architecture Details

Mamba-3 is an advanced iteration of the Selective State Space Model architecture.

- **Trapezoidal Discretization**: Second-order approximation for more accurate continuous-to-discrete mapping.
- **MIMO Projections**: Rank-based expansion (12% latency for 4x capacity).
- **Complex-Valued Dynamics**: RoPE-based simulation of complex SSMs in a real-valued framework.
- **Vision Support**: Integrated Vision Mamba with Snake Scan and bidirectional processing.

## 🚀 Installation & Model Usage

```bash
# Clone repository
git clone <repo-url>
cd Mamba3-XR

# Install core dependencies
pip install torch numpy timm scikit-learn matplotlib tqdm
```

### Mamba-3 Core Usage

```python
import torch
from model import Mamba3Block, Mamba3Config

config = Mamba3Config(
    d_model=512,      # Model dimension
    d_state=64,       # SSM state dimension
    d_head=64,        # Head dimension
    n_groups=1,       # Number of groups (MQA/GQA)
    mimo_rank=4,      # MIMO capacity scale
    expand=2,         # Expansion factor
)

model = Mamba3Block(config).cuda()
x = torch.randn(4, 2048, 512).cuda()
y = model(x)
```

## 📈 Training

For easy deployment, use the standalone `train.py` which includes all dependencies:
```bash
# Single-file training (no external dependencies needed)
python train.py
```

It includes:
- Mixed Precision Training (AMP)
- Exponential Moving Average (EMA)
- Auto-scaling Chunk-wise Parallel Scan (SSD)

## 📎 Citation

If you use Mamba-3 or TD-MoE in your research, please cite:
```bibtex
@article{mamba3_td_moe_2026,
  title={Mamba-3: Enhanced State Space Models with Trapezoidal Discretization and TD-MoE},
  author={Research Implementation},
  journal={ICLR Submission},
  year={2026}
}
```
