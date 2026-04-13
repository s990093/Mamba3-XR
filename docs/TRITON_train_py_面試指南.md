# `train.py` 中的 Triton：使用方式、API 與面試說明

本文對應專案根目錄的 [`train.py`](../train.py)，整理 **OpenAI Triton** 在本訓練腳本裡的用途、呼叫鏈、常用 API，以及面試時可講的重點。

---

## 1. Triton 是什麼？為什麼放在這裡？

**Triton** 是以 Python 子集描述 GPU kernel、由編譯器產生高效 CUDA/PTX 的工具。相較於手寫 CUDA：

- 用 `@triton.jit` 寫「每個 program 要做的事」，不用自己管大量 boilerplate。
- 與 **PyTorch** 常透過 `torch.autograd.Function` 包一層，讓自訂算子參與 autograd。

在 `train.py` 裡，Triton 用來加速：

| 區塊 | 目的 |
|------|------|
| Scaled Tanh | Router  logits 經 `tanh` 壓縮，用 **PTX `tanh.approx`** 做近似、減少記憶體與 launch 開銷 |
| SiLU × feat | FFN / 門控的 `silu(gate) * feat` **融合**成單一 kernel |
| Fused Latent MoE | Tucker 結構下，對 top-k expert 的 latent 矩陣乘加 **融合 forward**；backward 對專家權重 `dG` 用 Triton，其餘用 PyTorch |
| Parallel Scan | Mamba 風格 **chunk 內**遞推 \(h_t = \alpha_t h_{t-1} + u_t\)，用 `tl.associative_scan` 做並行前綴，並實作對應 backward |

---

## 2. 環境與「怎麼用」這支腳本

### 2.1 依賴

- **NVIDIA GPU** + 與 PyTorch 版本匹配的 **CUDA**。
- Python 套件：`torch`、`triton`（通常隨 PyTorch CUDA wheel 或另行 `pip install triton`）。
- 訓練還用到 `accelerate` 等（見檔案開頭 import）。

### 2.2 執行邏輯（與 Triton 的關係）

你只要像平常一樣 **跑訓練**；Triton kernel 會在第一次 forward 時編譯並快取。無需單獨「啟動 Triton」。

- **Scaled Tanh**：`TritonTuckerMoE.forward` → `fast_scaled_tanh`。
- **SiLU gating**：`MixtralMoEFeedForward.forward` → `fast_silu_gating`。
- **MoE latent**：`FusedLatentMoE.apply(...)`。
- **Chunk scan**：`Mamba3Block.chunk_parallel_scan` → `fast_triton_chunk_scan`（當 `config.use_parallel_scan=True`）。

面試可一句話：**「熱路徑上的 element-wise 與 MoE/scan 用 Triton 融合或專用 kernel，其餘仍用 PyTorch。」**

---

## 3. Triton / `triton.language` API 對照（本檔實際用到的）

### 3.1 裝飾器與啟動

| API | 用途 |
|-----|------|
| `@triton.jit` | 將函數編譯為 GPU kernel；參數裡 `tl.constexpr` 在 compile time 固定，可參與 shape 推導。 |
| `@triton.autotune(configs=[...], key=[...])` | 對多組 `BLOCK_*`、`num_warps`、`num_stages` 試跑，依 `key` 對不同張量尺寸快取最佳 config。 |
| `triton.Config({...}, num_warps=..., num_stages=...)` | autotune 的一組候選。 |
| `kernel[grid](*args, ...)` | 啟動 kernel；`grid` 為 tuple 或 `lambda meta: (...)`，`meta` 含編譯期 constexpr（如 `BLOCK_SIZE`）。 |
| `triton.cdiv(a, b)` | \(\lceil a/b \rceil\)，用來算 program 數量。 |

### 3.2 `tl`（`triton.language`）常用於本檔

| API | 用途 |
|-----|------|
| `tl.program_id(axis)` | 目前 block 在 grid 上的索引。 |
| `tl.arange(0, N)` | 向量化的 offset。 |
| `tl.load` / `tl.store` | 由 pointer + offset 讀寫；`mask` 處理邊界，`other` 填 padding 值。 |
| `tl.zeros` | 暫存累加器（常配合 `float32` 累加再 cast 回輸出 dtype）。 |
| `tl.exp`, `tl.sigmoid` 等 | 標量/向量運算。 |
| `tl.inline_asm_elementwise` | 內嵌 PTX（此處用 `tanh.approx.f32`）。 |
| `tl.associative_scan(..., axis=..., combine_fn=...)` | **結合律** scan；此處 `first_order_combine_op` 對應序列上的線性遞推。 |
| `tl.atomic_add` | backward 裡對 `dlog_alpha` 多個 (b,l) 貢獻累加到同一位置。 |
| `ptr.dtype.element_ty` | 從 tensor pointer 推導元素型別，store 時對齊 PyTorch dtype。 |

---

## 4. 與 PyTorch 的接軌模式（面試高頻）

本檔固定模式：

1. **寫 Triton kernel**（`@triton.jit`，指標 + strides + 維度）。
2. **包成 `torch.autograd.Function`**：`forward` 裡 `save_for_backward`、`empty_like`、算 `grid`、`kernel[grid](...)`；`backward` 裡再 launch bwd kernel 或混合 PyTorch。
3. **對外暴露**小函式：`fast_scaled_tanh`、`fast_silu_gating`、`fast_triton_chunk_scan`。

優點：**與 `nn.Module` 無縫銜接**、可 checkpoint、optimizer 照常更新非 Triton 參數。

---

## 5. 各 kernel 精簡說明（可當口條）

### 5.1 `tanh_approx` + `_fused_scaled_tanh_fwd/bwd`

- Forward：`y = scale * tanh(x/scale)`，用 PTX 近似 tanh。
- Backward：鏈式法則 `dy * (1 - tanh^2)`。
- Grid：1D，`ceil(n_elements / BLOCK_SIZE)` 個 program；`@triton.autotune` 依 `n_elements` 選 `BLOCK_SIZE` / warps。

**面試點**：在數值可接受前提下用 **fast math / approx intrinsic** 換算力；並與 **scaled tanh** 設計（穩定 router）一起講。

### 5.2 `_fused_silu_mul_fwd/bwd`

- Forward：`silu(gate) * feat` 單 kernel 讀兩個 tensor、寫一個。
- Backward：對 SiLU 與乘法的導數合併在同一個 kernel。

**面試點：** **kernel fusion** 減少 global memory round-trip，是 LLM FFN 常見優化。

### 5.3 `_fused_latent_moe_fwd` / `_fused_latent_moe_bwd_dG_kernel`

- Forward：對每個 batch `b`、輸出維 `r2` tile，對 top-k expert 做 `x_shared`（長度 `r3`）與專家矩陣 `G[e]` 的內積加權和。
- Backward：`dx`、`dprob` 用 PyTorch 向量化；**`dG`** 用 3D grid `(E, ceil(r3/BLOCK_R3), ceil(r2/BLOCK_R2))` 的 Triton，因為要對所有 batch 與 top-k 累加。

**面試點：** **混合策略**——哪一段在 PyTorch 夠快、哪一段用 Triton 攤平迴圈；以及 **stride** 正確性對多維 tensor 的重要性。

### 5.4 `_chunk_scan_fwd_kernel` / `_chunk_scan_bwd_kernel`

- 數學：chunk 內 \(h_t = \alpha_t h_{t-1} + u_t\)，寫成 associative scan 的 `combine`。
- Forward：`tl.associative_scan` 沿長度 `L`（chunk_size）維度。
- Backward：對 `dh` 做反向順序的類 scan，並用 `atomic_add` 聚合 `dlog_alpha`（因為 \(\partial h/\partial \log\alpha\) 沿時間耦合）。

**面試點：** 把 **序列 RNN/SSM 的一步**變成 **並行前綴**；backward 為何需要 **atomic** 或額外同步（多個 D-tile 對同一 `(b,l)` 的 \(\log\alpha\) 有貢獻）。

外層 `chunk_parallel_scan` 仍用 PyTorch 做 **chunk 之間**的狀態傳遞與與 `C` 的 einsum——Triton 專注在 **chunk 內**最重的 scan。數學與「為何能並行」見 **§5.5**。

### 5.5 關聯性掃描（Associative Scan）與 `first_order_combine_op`

本節對應 `train.py` 裡 chunk 內 scan：**高階技巧**是用 **滿足結合律的「合併規則」**，把看似只能循序的時間步，改寫成可做 **並行前綴（prefix scan）** 的問題；Triton 以 `tl.associative_scan(..., combine_fn=first_order_combine_op)` 實作。

#### 最簡版：先記三句話（聽不懂就停在這裡也夠面試用）

1. **模型每一步在做的事**：「新的狀態 = 某個倍數 × 舊狀態 + 某個加進來的數」。倍數叫 \(\alpha\)，加進來的叫 \(u\)（程式裡用一對數代表這一步）。  
2. **`first_order_combine_op` 只做一件事**：**兩步規則**（先發生的一步 + 後發生的一步）能不能變成 **一條等價的規則**？能，而且合併後還是「倍數 × 舊的 + 常數」這種形式。  
3. **為什麼 GPU 在乎**：若我們會「合併」，很多時間步就可以用演算法 **兩兩併、再併、再併**（像錦標賽樹），不必死板板從第 1 格算到第 L 格只開一條線。

下面用 **零用錢** 把第 2 點講完；再往下才是公式與程式對照。

#### 類比：兩天的零用錢規則（完全沒有符號）

- **第一天規則**：你口袋裡有 \(x\) 元，睡一覺之後變成「**2 倍再加 3 元**」。  
- **第二天規則**：不管第一天結束後有多少，再睡一覺變成「**4 倍再加 5 元**」。

問：**若兩天合起來看**，能不能說成「從一開始的 \(x\) 出發，**一次**就變成某個『幾倍再加幾』」？

自己代數字：假設一開始 \(x=10\)。

- 第一天後：\(2\times 10 + 3 = 23\)  
- 第二天後：\(4\times 23 + 5 = 97\)

若一開始是 \(x=10\) 最後是 97，那「合併規則」是不是「8 倍再加 17」？驗證：\(8\times 10 + 17 = 80 + 17 = 97\)。對。

所以 **兩步**「先 2 倍+3，再 4 倍+5」= **一步**「8 倍+17」。  
`first_order_combine_op(2, 3, 4, 5)` 算出來的就是 **(8, 17)**——第一個數是合併後的「倍數」，第二個數是「加上的常數」。

**和程式參數怎麼對？**

- 先發生的那天 → `alpha_left`, `beta_left`（這裡是 2 和 3）  
- 後發生的那天 → `alpha_right`, `beta_right`（這裡是 4 和 5）  
- 回傳 `(8, 17)` = 合併後的「倍數、常數」

你不需要背公式，只要記得：**它就是「兩天的規則壓成一天的規則」的計算機**。

#### 程式就是上面那個「壓成一天」的公式

```454:456:train.py
@triton.jit
def first_order_combine_op(alpha_left, beta_left, alpha_right, beta_right):
    return alpha_right * alpha_left, alpha_right * beta_left + beta_right
```

把「2 倍+3」再「4 倍+5」代進去：`4×2=8`，`4×3+5=17`，就是 `(8, 17)`。  
（Kernel 裡 \(\alpha\) 來自 `exp(log_alpha)`，另一個對應 \(u\)；本質相同。）

#### （可選）對照式子：為什麼是「右乘左、右乘左加右」

若先套用左段 \(x \mapsto \alpha_L x + \beta_L\)，再套用右段 \(y \mapsto \alpha_R y + \beta_R\)，展開：

\[
x \mapsto \alpha_R(\alpha_L x + \beta_L) + \beta_R
     = (\alpha_R \alpha_L)\, x + (\alpha_R \beta_L + \beta_R).
\]

所以合併後的「倍數」是 `alpha_right * alpha_left`，「常數」是 `alpha_right * beta_left + beta_right`。這就是 `return` 那兩行的由來。

#### 循序形式與痛點（和類比的連接）

Chunk 內每一步（概念上）是：

\[
h_t = \alpha_t\, h_{t-1} + u_t
\]

若用 for 從 \(t=0\) 算到 \(L-1\)，**計算深度**為 \(O(L)\)：算 \(h_t\) 必須先知道 \(h_{t-1}\)，表面上看起來很難讓「時間維」本身大規模並行。  
把每一步寫成「倍數與常數」的一對數之後，**兩兩合併**就交給 `first_order_combine_op`；`associative_scan` 會重複做這種合併，用樹狀方式在 \(O(\log L)\) 層左右算完（細節見下節）。

#### 為什麼叫「關聯性」？與樹狀並行的關係

定義二元運算 \(\otimes\)（代表「右段接在左段之後」的合成）：

\[
(\alpha_1, \beta_1) \otimes (\alpha_2, \beta_2)
  = (\alpha_2 \alpha_1,\; \alpha_2 \beta_1 + \beta_2)
\]

可驗證 **結合律**成立：\((a \otimes b) \otimes c = a \otimes (b \otimes c)\)。  
一旦運算可結合，前綴

\[
(\alpha_0,u_0) \otimes (\alpha_1,u_1) \otimes \cdots \otimes (\alpha_{t},u_{t})
\]

就不只能靠「從左到右一步一步算」，而能用經典的 **並行 prefix scan**（樹狀／Blelloch 風格等）：**總工作量**仍約 \(O(L)\)，但**平行深度**可降到 \(O(\log L)\) 量級，也就是把原本線性的時間依賴「攤平」成多層兩兩合併的樹狀結構。

這就是面試時可以說的：**自訂 `combine_fn` 把 \(O(L)\) 深度的循序遞推，改寫成可並行 scan 的同構問題**。

#### 與本專案 grid 的直覺對應

- **語意上**：`associative_scan` 內部會用樹狀方式合併區間，使「沿時間軸」的依賴深度從 \(O(L)\) 降至 \(O(\log L)\) 量級，而不是在時間維上寫長 for。
- **實作上**：forward kernel 的 grid 為 `(B_flat, ceil(D / BLOCK_D))`——不同 batch 條目、不同特徵維 tile 由不同 program 各算一條長度 \(L\) 的 scan；**同一條時間序列上的 scan** 則在單個 program 內由 `tl.associative_scan` 完成。

Chunk **之間**的狀態仍由外層 PyTorch（`chunk_parallel_scan`）負責；Triton 專注在 **chunk 內**最重的這段。

#### Backward（一句話）

反向傳播時，梯度沿時間仍耦合；實作上對反轉序列再做類 scan，並對 \(\partial/\partial \log\alpha\) 使用 `atomic_add`，因為多個 \(D\) 維 tile 可能對同一 `(batch, time)` 的 `log_alpha` 都有貢獻。細節見 §5.4。

#### 電梯簡報版（可加在口條裡）

> 我們把 \(h_t=\alpha_t h_{t-1}+u_t\) 寫成仿射變換的**結合律運算**，因此能用 **associative scan** 在對數深度的並行樹上算前綴，避免在時間維上純 \(O(L)\) 循序；`first_order_combine_op` 就是兩個仿射變換的合成公式。

---

## 6. Autotune 與 `key`（怎麼講才專業）

- `key=['n_elements']`：不同總元素量可能最適不同 `BLOCK_SIZE`。
- `key=['r3', 'r2']` / `['r3', 'r2', 'B']`：MoE 維度與 batch 變化時重選 config。
- `key=['D', 'L']`：scan 的隱狀態維度 `D` 與 chunk 長度 `L` 影響 occupancy。

**注意：** 第一次遇到新 shape 會 **多跑幾次 benchmark**，訓練初期可能略慢；之後 cache 命中會正常。

---

## 7. 除錯與實務技巧（面試可加分的「踩坑」）

1. **先對齊 PyTorch reference**：用小張量 `torch.allclose` 比 forward/backward。
2. **mask 與邊界**：`tl.load(..., mask=..., other=0)` 避免越界；最後一個 tile 常見錯誤。
3. **dtype**：累加用 `float32`，`store` 再 `.to(element_ty)`，避免 bf16/fp16 累加爆掉。
4. **`constexpr` vs runtime**：`BLOCK_*`、`L`、`D` 等若為 constexpr，改變 shape 可能觸發 **recompile**。
5. **`atomic_add` 與確定性**：同一位置多 thread 累加，順序不固定，通常對梯度沒問題，但若要比 bitwise 一致需另設計。
6. **PTX `inline_asm`**：依架構與 Triton 版本而異，升級時要回歸測試。

---

## 8. 一分鐘電梯簡報稿（中文）

> 我在 `train.py` 裡用 Triton 加速語言模型訓練的幾個熱點：路由用 scaled tanh 並用 PTX 近似；FFN 的 SiLU 門控做融合；MoE 的 latent 矩陣乘在 forward 與專家梯度上用客製 kernel 減少記憶體與 Python 開銷；Mamba 區塊的 chunk 內線性遞推則用 Triton 的 associative scan 做並行，backward 用反向 scan 加 atomic 對 log-decay 求導。這些都透過 `torch.autograd.Function` 接到 PyTorch，訓練流程不變，但 step time 更好。

---

## 9. 檔案導覽（快速定位）

| 主題 | 約略位置（行號隨版本變動，以搜尋為準） |
|------|----------------------------------------|
| `tanh_approx`、scaled tanh | `train.py` 開頭 Triton 區塊 |
| SiLU fusion | `_fused_silu_mul_*`、`fast_silu_gating` |
| MoE | `_fused_latent_moe_*`、`FusedLatentMoE`、`TritonTuckerMoE` |
| Scan | `_chunk_scan_*`、`TritonParallelScanFn`、`fast_triton_chunk_scan`；**關聯性掃描數學**見 **§5.5** |
| 接到 Mamba | `Mamba3Block.chunk_parallel_scan` |

---

## 10. 對教授口頭報告：短版 + 可能 Q&A

### 10.1 約 30 秒（短版）

訓練裡有一段是 **沿時間的線性遞推**（每步：上一狀態 × 衰減 + 輸入）；若用多個小算子或頻繁讀寫，**記憶體往返**會多。我改成在 **單一 Triton kernel** 裡用 **associative scan** 算 chunk 內整段遞推，並用 **`torch.autograd.Function`** 包 forward / backward，讓 **PyTorch autograd 仍自動反傳**。其餘（chunk 之間、線性層等）維持 PyTorch，是 **效能與可維護性** 的取捨。

### 10.2 若多給 1～2 分鐘可補的三句

1. **改了什麼**：自訂 **`TritonParallelScanFn`**（forward 呼叫 `_chunk_scan_fwd_kernel`，backward 呼叫 `_chunk_scan_bwd_kernel`），中間做 **tensor reshape / transpose** 與 kernel 用的 layout 對齊。  
2. **數學核心**：遞推可寫成多步「仿射」的合成；**結合律**成立，才能用 `tl.associative_scan`；合併規則是 `first_order_combine_op`（§5.5）。  
3. **實務細節**：backward 對 `log_alpha` 的梯度可能由多個特徵維 tile 貢獻，故用 **`atomic_add`** 累加；不是數學特殊，是 **並行切分** 的需要。

### 10.3 可能被問的 Q&A

**Q：為什麼不用 PyTorch 內建就好？**  
A：內建算子未必針對「chunk 內 + 這種 scan 形狀」做最佳化；自訂 Triton 可把 **forward 整段 scan** 收在一顆 kernel，減少 launch 與中間張量往返。若教授追問「有沒有 profile」，誠實說有／沒有，沒有就說目標是 **降低 memory traffic 與 kernel 次數**。

**Q：Triton 和寫 CUDA 差在哪？**  
A：用 Python 子集描述 kernel，編譯器產生程式；我們專注在 **tile、load/store、scan**，較少手寫 warp 層級細節；但仍要處理 **stride、mask、dtype**。

**Q：`autograd.Function` 一定要嗎？**  
A：要讓 **自訂 Triton** 參與反傳，需要實作 **backward** 或包 **torch.ops** 等；這裡用 `Function` 把 **梯度張量排版** 與 **kernel 指標** 對齊，是常見做法。

**Q：為什麼 backward 用 `atomic_add`？會不會不確定？**  
A：同一個 `(batch, time)` 的 `log_alpha` 可能對應 **多個 D 維 block** 的偏導，要 **加總**；`atomic_add` 保證加總正確，順序不固定通常 **不影響梯度正確性**（若要比 bitwise 重現才另議）。

**Q：chunk 之間為什麼還用 PyTorch？**  
A：**chunk 內** scan 最重、最適合專用 kernel；**chunk 之間**狀態傳遞與與其他張量（如 `C`）的運算，用 PyTorch 寫清楚、除錯快；屬 **分工** 不是偷懶。

**Q：和 MoE / tanh / SiLU 有什麼關係？**  
A：那是 **同一檔案裡其他熱點**（router、FFN、MoE latent）；可一句帶過：**「scan 處理序列遞推；其餘是 element-wise 或 MoE 的 fused kernel。」**

**Q：怎麼驗證對？**  
A：小張量與 **PyTorch 參考實作**（例如循序 for 或 `torch` 拼出來）比 `allclose`；改 shape、dtype 再測；訓練 loss 能降、無 NaN。

---

*若你面試的是「系統 / CUDA」向，可補充：occupancy、shared memory、`num_stages` 對軟流水、以及為何 MoE backward 選擇 Triton 只打 `dG` 而不用全部 fused。*
