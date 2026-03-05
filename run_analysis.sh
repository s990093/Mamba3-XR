#!/bin/bash
echo "Generating Rank 8 (Baseline) Plots..."
.venv/bin/python code/plot_diagnostics.py --log_dir results/rank8

echo "Generating Rank 4 (Ablation) Plots..."
.venv/bin/python code/plot_diagnostics.py --log_dir results/rank4

echo "Generating Comparative Plots (Rank 8 vs Rank 4)..."
.venv/bin/python code/compare_ranks.py

echo "Analysis Complete. Check results/comparison and mamba3/docs/vision_diagnostics_report.md"
