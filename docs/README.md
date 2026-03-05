# Mamba Training Analysis Documentation

本目錄包含 Mamba 訓練運行（Rank 1, 4, 8）的完整分析文檔和視覺化圖表。

## 📄 主要文件

### 報告文件

- **`rank_comparison_analysis.md`** - 完整分析報告（繁體中文，985 行）
- **`rank_comparison_analysis.pdf`** - PDF 版本（5.5 MB）
- `mimo_rank_saturation_analysis.md` - MIMO 秩飽和深度分析（獨立文檔）

## 📊 視覺化圖表（19 張）

### 訓練指標對比

- `comparison_training_metrics.png` - 訓練/驗證損失、準確率、F1 分數
- `comparison_gradient_health.png` - 梯度健康度（梯度範數、SNR）
- `comparison_model_size.png` - 模型大小對比

### MIMO 分析

- `comparison_mimo_ranks.png` - MIMO 秩演化對比
- `layer_wise_mimo_ranks.png` - 各層 MIMO 秩詳細對比
- `mimo_rank_distribution.png` - MIMO 秩分布
- `mimo_saturation_efficiency.png` - MIMO 飽和效率分析
- `mimo_theoretical_utilization.png` - 理論 MIMO 秩利用率
- `mimo_marginal_returns.png` - 邊際效益遞減

### 內部狀態分析

- `comparison_state_health.png` - 狀態健康度（L2 範數、變異數、Delta CV、SSM 特徵值）
- `comparison_a_log_stability.png` - A_log 穩定性對比
- `a_log_snr_comparison.png` - A_log SNR 對比
- `eigen_a_evolution.png` - SSM 特徵值演化
- `delta_cv_comparison.png` - Delta 參數變異係數

### ERF 分析

- `erf_evolution_comparison.png` - ERF 演化對比（5 個關鍵 epoch）

### 分布圖對比

- `dist_gradients_comparison.png` - 梯度分布對比
- `dist_mamba_internals_comparison.png` - Mamba 內部參數分布
- `dist_metrics_comparison.png` - 訓練指標分布
- `dist_state_health_comparison.png` - 狀態健康度分布

## 📈 數據文件（4 個）

- `summary_statistics.json` - 訓練統計摘要
- `comprehensive_analysis.json` - 完整分析數據
- `mamba_internals_analysis.json` - Mamba 內部狀態量化數據
- `a_log_stability_analysis.json` - A_log 穩定性分析數據

## 📐 架構圖

- `mamba3_dataflow.pdf` - Mamba-3 數據流程圖
- `mamba_diagram.pdf` - Mamba 架構圖

## 🗂️ 文件統計

- **圖表**: 19 個 PNG 文件（~8.5 MB）
- **數據**: 4 個 JSON 文件（~13 KB）
- **報告**: 2 個 Markdown 文件（~44 KB）
- **PDF**: 3 個 PDF 文件（~5.7 MB）
- **總計**: 28 個文件

## 📋 報告結構

主報告 `rank_comparison_analysis.md` 包含以下章節：

1. **執行摘要** - 關鍵發現和建議
2. **ERF 演化分析** - 有效感受野演化（含量化數據）
3. **MIMO 深度分析** - MIMO 秩飽和分析（含理論上限）
4. **內部狀態健康度分析** - 6 個量化指標
5. **A_log 參數穩定性分析** - 基於 SSM 特徵值
6. **分布圖表分析** - 梯度、內部參數、指標分布
7. **模型大小與效率** - 參數量與性能對比
8. **詳細統計數據** - 完整數值表格
9. **結論與建議** - 最佳配置推薦
10. **附錄** - 訓練配置和文件清單

## 🎯 關鍵發現

- **Rank 4 是最優配置**：達到 MIMO 秩飽和點的 99.6%（28/32）
- **高利用率**：d_state=32 的利用率達 87.5%
- **性能飽和**：Rank 4 和 Rank 8 性能僅差 0.48%
- **ERF 飽和**：ERF 覆蓋率在 75% 達到飽和點

## 📦 歸檔文件

中間分析腳本和臨時文件已移至 `../archive_analysis_scripts/`（18 個文件）

---

**生成日期**: 2025-12-29  
**分析對象**: Mamba Rank 1/4/8 訓練運行（100 epochs）  
**總分析文件數**: 321 個訓練文件
