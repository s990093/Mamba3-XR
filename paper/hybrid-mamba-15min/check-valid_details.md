下面這份可以直接拿去當**文獻回顧 / citing reference 的骨架**。我先幫你把每篇的「可以引用什麼貢獻」抓出來，不然 paper 海會把人淹死，然後大家還假裝這叫研究。

## 0. 六個主題的引用定位

| 主題                 | 你引用它時主要想說什麼                                                                                     |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| **GQA**            | 降低 KV-cache / decoder inference memory bandwidth，在 MHA 品質與 MQA 速度之間取折衷                          |
| **Z-loss**         | 穩定 softmax / router logits，避免 logits 過大造成 numerical instability；新趨勢是減少輔助 loss 對 MoE routing 的干擾 |
| **TD-MoE**         | 用 tensor decomposition 壓縮 MoE experts，重點是跨 expert 的 shared structure，不是每個 expert 各自 SVD         |
| **Mamba-3**        | 用更強的 SSM recurrence、complex-valued state、MIMO，提高 linear-time sequence model 的能力與推論效率            |
| **FlashAttention** | exact attention 的 GPU IO-aware kernel，不是 approximate attention；核心是減少 HBM/SRAM 資料搬運              |
| **MoE**            | conditional computation：總參數變大，但每 token 只啟用少數 experts，達到 capacity/compute 解耦                     |

---

## 1. GQA：Grouped-Query Attention

**代表論文：** Ainslie et al., 2023, *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*。這篇是 EMNLP 2023，核心是把 MHA 和 MQA 之間做出一個中間態：多個 query heads 共用一組 key/value heads，而不是每個 query head 都有自己的 K/V，也不是全模型只剩一組 K/V。作者也提出用約原始 pretraining compute 的 5% 來把既有 MHA checkpoint uptrain 成 MQA/GQA。([ACL Anthology][1])

**技術核心：**
MHA 是 `H` 個 Q/K/V heads；MQA 是多個 Q heads 共用單一 K/V head；GQA 則是把 query heads 分成 `G` 組，每組共用一組 K/V。論文裡明確定義：`GQA-1 = MQA`，`GQA-H = MHA`。所以 GQA 是 MQA 和 MHA 的連續折衷。

**為什麼重要：**
在 autoregressive decoding 時，每一步都要讀取 KV-cache。GQA 減少 K/V heads 數量，因此降低 KV-cache size 和 memory bandwidth；相較 MQA，GQA 保留更多 K/V capacity，因此品質比較接近 MHA。簡單講：GQA 是「少搬一點 KV-cache，但不要笨太多」的人類妥協方案。

**文獻回顧可寫：**

> Grouped-Query Attention reduces the number of key-value heads by sharing each key/value head across a group of query heads, interpolating between MHA and MQA. It preserves quality close to MHA while approaching the decoding efficiency of MQA.

---

## 2. Z-loss：Softmax / Router 穩定化

**先分清楚兩種語境，不然會變成 citation 車禍：**

第一種是 **LM output softmax z-loss**。PaLM 使用 `z-loss = 10^-4 · log^2 Z`，目標是讓 softmax normalizer `log(Z)` 接近 0，提升大型語言模型訓練穩定性。

第二種是 **MoE router z-loss**。ST-MoE 引入 router z-loss，形式大意是對 router logits 的 `logsumexp` 平方做平均懲罰，也就是抑制進入 gating network 的 logits 過大。ST-MoE 報告它能穩定 sparse expert model，而且相較 update clipping，不會明顯傷害模型品質。

**Router z-loss 公式概念：**

```text
L_z = mean( logsumexp(router_logits)^2 )
L_total = L_CE + c_B * L_balance + c_z * L_z
```

**技術核心：**
Z-loss 不是主要拿來「平均分配 experts」的，它更像是 softmax / router logits 的數值穩定器。router logits 太大時，softmax 前的 exponential 對 bfloat16 / mixed precision 很敏感，roundoff error 會被放大。ST-MoE 的說法是，router z-loss 讓進入 exponential function 的數字絕對值變小，降低數值誤差。

**目前新的東西：**
近年的 MoE 訓練開始不只依賴 auxiliary loss / z-loss。DeepSeek-V3 採用 **auxiliary-loss-free load balancing**：給每個 expert 一個 bias，只用於 top-K routing，gating value 仍從原始 affinity score 來，目標是維持 load balance 同時減少 auxiliary loss 對模型表現的傷害。DeepSeek-V3 也指出太大的 auxiliary loss 會傷害模型表現。([arXiv][2])

ACL 2025 的 *On Implementing Load Balancing Loss for Training MoE LLMs* 也提出一個重要觀察：LBL 和 z-loss 會鼓勵 routing scores 比較 uniform，而 language modeling loss 會推動 routing scores 增強；這代表「穩定 router」和「讓 expert 專門化」之間有拉扯。這就是目前比較新的討論方向：不是 z-loss 沒用，而是它要被放在更細緻的 routing / load balancing 設計裡。

**文獻回顧可寫：**

> Z-loss regularizes the softmax normalizer by penalizing large log-sum-exp values, improving numerical stability in large-scale LM and MoE training. In MoE routers, it stabilizes routing logits, but recent work increasingly explores auxiliary-loss-free or batch-wise balancing strategies to reduce the conflict between load balancing and expert specialization.

---

## 3. TD-MoE：Tensor Decomposition for MoE Compression

**代表論文：** Xu et al., 2026, *TD-MoE: Tensor Decomposition for MoE Models*。你上傳的這篇是 ICLR 2026。它的問題意識很明確：現有 MoE compression 常常是每個 expert 自己做 SVD / low-rank decomposition，但這忽略了 experts 之間共享的結構冗餘。TD-MoE 改成把同一層所有 experts 的 weights 疊成 3D tensor，再做 joint Tucker decomposition。

**技術核心三件事：**

1. **Cross-expert tensorization**
   把 `K` 個 expert weight matrices 疊成 `T ∈ R^{K × d_out × d_in}`，讓 decomposition 可以同時看見 expert dimension、output dimension、input dimension。這比每個 expert 各自 SVD 更能抓跨 expert 的 shared structure。

2. **Multi-linear whitening**
   使用 calibration data 的 activation / gradient statistics，對 input/output modes 做 whitening，讓 decomposition 不只是看 weight 本身，而是考慮資料分布與 feature correlation。人類終於想起模型不是活在真空裡，可喜可賀。

3. **3D rank allocation**
   Tucker decomposition 的 ranks `(r1, r2, r3)` 分別對應 expert / output / input 維度。TD-MoE 會根據 target compression ratio 自動分配 rank，避免手動亂調。

**實驗重點：**
TD-MoE 在 Qwen2-57B-A14B 和 Mixtral-8×7B 上測試，論文摘要稱 20% parameter reduction 幾乎 lossless，在 40% 和 60% compression 下相較 decomposition-based baselines 有超過 11% 和 14% 的改善。

**文獻回顧可寫：**

> TD-MoE reformulates MoE compression as joint tensor decomposition across experts. Instead of compressing each expert independently, it stacks expert weights into a 3D tensor and applies Tucker decomposition with data-aware whitening and budget-aware rank allocation, capturing inter-expert redundancy that per-expert SVD methods miss.

---

## 4. Mamba-3：複數 / 負數 / 計算核心與厲害之處

**代表論文：** Lahoti et al., 2026, *Mamba-3: Improved Sequence Modeling Using State Space Principles*。這篇的定位是 inference-first 的 SSM / linear sequence model，目標是改善 Transformer 推論時 quadratic attention 和 KV-cache memory 的問題，同時補足 Mamba-2 / linear models 在 state tracking 上的弱點。([arXiv][3])

**Mamba-3 的三個核心改動：**

1. **Exponential-trapezoidal discretization**
   Mamba-1/2 類似 exponential-Euler；Mamba-3 改用更 expressive 的 exponential-trapezoidal discretization。直覺上，它不是只用目前 token 更新 state，而是讓 state-input 有一個隱含的 width-2 convolution 效果。論文也說這可以幫助取代過去 recurrent models 常用的 short causal convolution。

2. **Complex-valued SSM**
   這是你問「複數/負數？」的重點。Mamba-2 的 state transition 基本上是 real-valued decay，`α_t ∈ (0,1)`，也就是 state 只會保留或衰減。這種純實數正衰減不擅長表示「旋轉 / phase / parity / state flipping」這類狀態追蹤。Mamba-3 把 SSM 看成 complex-valued，等價成 real-valued 的 2×2 rotation block，也就是類似 data-dependent RoPE 的計算方式。

3. **MIMO SSM**
   Mamba-3 從 SISO 轉成 MIMO，把原本 outer-product 型態的 state update 改成更像 matrix multiplication 的形式。這提高 arithmetic intensity，讓 decoding 時 GPU 不只是等 memory 搬資料，能更有效利用 compute。論文指出 MIMO 可以增加 FLOPs，但 decode wall-clock latency 幾乎不增加，這就是它厲害的地方：不是少算，而是把原本閒著的硬體拿來算。

**關於「負數」：**
Mamba-3 的重點不是「用了負數所以強」，而是 **complex phase / rotation**。實數負 eigenvalue 可以做 sign flip，但能力有限；complex eigenvalue 可以表示連續旋轉，因此能處理 parity、週期、state tracking 這種純 decay 模型不擅長的任務。論文把 complex SSM 轉成 real block rotation matrix，避免真的用很重的 complex arithmetic。這種設計很漂亮，討厭 HDL 的人看了都會短暫相信數學還有救。

**文獻回顧可寫：**

> Mamba-3 improves linear-time sequence modeling by introducing a more expressive exponential-trapezoidal SSM discretization, complex-valued state dynamics implemented via real-valued rotary blocks, and a MIMO formulation that increases decoding arithmetic intensity without increasing state size. These changes address both capability limitations, such as state tracking, and hardware inefficiency in prior linear models.

---

## 5. FlashAttention：IO-aware exact attention

**代表論文：** Dao et al., 2022, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*。這篇很重要，因為它不是把 attention 近似掉，而是算 **exact attention**，只是用更聰明的 GPU memory hierarchy 操作方式來算。([arXiv][4])

**技術核心：**
標準 attention 會 materialize `S = QK^T` 和 `P = softmax(S)` 這類 `N×N` intermediate matrices，造成大量 HBM read/write。FlashAttention 用 tiling 把 Q/K/V 分塊載入 SRAM，配合 online softmax statistics，在不完整儲存整個 attention matrix 的情況下得到正確結果；backward 則透過 recomputation 避免保存巨大 intermediate。

**為什麼快：**
GPU 上很多時候瓶頸不是 FLOPs，而是 HBM 和 SRAM 之間搬資料。FlashAttention 的核心是 **IO-aware**：降低 HBM accesses，比標準 attention 更少讀寫慢速記憶體。論文甚至分析其 IO complexity，指出在某些 SRAM size 範圍下接近 optimal。

**後續版本：**

* **FlashAttention-2**：改進 parallelism 和 work partitioning，減少 non-matmul FLOPs，讓單一 head 也能跨 thread blocks 平行化；論文報告約 2× over FlashAttention，A100 上可達 50–73% theoretical max FLOPs/s。
* **FlashAttention-3**：針對 Hopper GPU，利用 Tensor Cores / TMA asynchrony、warp specialization、block-wise matmul-softmax interleaving，以及 FP8 low precision；論文報告 H100 上 BF16 可到 840 TFLOPs/s、FP8 可到 1.3 PFLOPs/s。

**文獻回顧可寫：**

> FlashAttention accelerates exact self-attention by making the algorithm IO-aware: it tiles Q/K/V into SRAM, uses online softmax to avoid materializing the full attention matrix, and recomputes intermediates during backward pass. Later versions improve GPU occupancy and exploit hardware-specific asynchronous execution and low-precision capabilities.

---

## 6. MoE：Mixture-of-Experts

**經典起點：** Shazeer et al., 2017, *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*。這篇把 Sparsely-Gated MoE 帶進大模型語境：用 gating network 為每個 input 選少數 experts，讓模型總 capacity 大幅增加，但每次只啟用部分參數。([arXiv][5])

**技術核心：**

```text
router_logits = W_r x
gate = softmax(router_logits)
selected_experts = TopK(gate)
output = sum_i gate_i * Expert_i(x)
```

核心概念是 **conditional computation**：模型可以有很多 experts，但每個 token 只跑 top-k experts。這讓 total parameters 和 activated parameters 分離，也就是「模型很大，但每步不一定很貴」。人類把模型做巨大後又發明方法假裝沒那麼巨大，工程史真是迴圈。([arXiv][5])

**重要後續：**

* **GShard**：把 MoE Transformer 擴到超大規模，使用 automatic sharding 訓練 multilingual NMT 模型到 600B parameters。([arXiv][6])
* **Switch Transformer**：簡化成 top-1 routing，降低 MoE routing / communication 成本，展示 trillion-parameter sparse model 的訓練可行性。([arXiv][7])
* **GLaM**：使用 sparsely activated MoE，最大模型 1.2T parameters，但推論 FLOPs 約為 GPT-3 的一半，訓練能耗約三分之一。([arXiv][8])
* **Mixtral 8×7B**：decoder-only SMoE，每層 8 個 FFN experts，每 token 選 2 個 experts；每 token 可接觸 47B parameters，但 inference 只啟用約 13B active parameters。([arXiv][9])

**文獻回顧可寫：**

> MoE models decouple model capacity from per-token computation by replacing dense feed-forward layers with multiple expert networks and routing each token to a sparse subset of experts. This enables scaling total parameters while keeping activated computation relatively small, but introduces challenges in routing stability, load balancing, communication, and expert specialization.

---

## 7. 建議你的 reference 組合

如果你的報告是講 **LLM 架構與效率演進**，可以這樣安排：

1. **MoE 基礎**：Shazeer et al. 2017 → Switch / GShard / GLaM → Mixtral
2. **MoE 訓練穩定性**：ST-MoE router z-loss → DeepSeek-V3 auxiliary-loss-free balancing
3. **MoE compression**：TD-MoE
4. **Attention inference efficiency**：GQA + FlashAttention
5. **Attention alternative / linear sequence model**：Mamba-3

這樣架構會很清楚：
**MoE 解決 capacity/compute，GQA/FlashAttention 解決 attention memory/io，Z-loss 解決 training stability，TD-MoE 解決 MoE memory footprint，Mamba-3 則是跳出 attention 框架的 alternative sequence modeling。**

[1]: https://aclanthology.org/2023.emnlp-main.298/ "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints - ACL Anthology"
[2]: https://arxiv.org/html/2412.19437v2 "DeepSeek-V3 Technical Report"
[3]: https://arxiv.org/abs/2603.15569?utm_source=chatgpt.com "Mamba-3: Improved Sequence Modeling using State Space Principles"
[4]: https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
[5]: https://arxiv.org/abs/1701.06538 "[1701.06538] Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
[6]: https://arxiv.org/abs/2006.16668?utm_source=chatgpt.com "GShard: Scaling Giant Models with Conditional ..."
[7]: https://arxiv.org/abs/2101.03961?utm_source=chatgpt.com "Switch Transformers: Scaling to Trillion Parameter Models ..."
[8]: https://arxiv.org/abs/2112.06905 "[2112.06905] GLaM: Efficient Scaling of Language Models with Mixture-of-Experts"
[9]: https://arxiv.org/abs/2401.04088 "[2401.04088] Mixtral of Experts"
