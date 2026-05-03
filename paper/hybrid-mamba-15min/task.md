我這至個報告是在加強 moe那塊 Moe -> men bound 變成 可以先根據 moe.md moe2.md 裡面加強 相關文獻說明， 加入 目前moe 怎麼做的等 訓練跟推論怎麼做 有什麼問題的等 在最化說到壓縮（原本那塊）

跟底下moe 說到章節可以先從top k開始說起等等．．．．．

底下這樣 證明記憶體那塊可以比較好 然後再去看過train.py
把加入顯示 演算法 alog 說我怎麼做 的（前項跟反向） 不是放入一推api 那樣是 像是
Algorithm 1 FlashAttention
Require: Matrices Q,K,V ∈R𝑁×𝑑 in HBM, on-chip SRAM of size 𝑀.
1: Set block sizes 𝐵𝑐 =
𝑀
4𝑑 ,𝐵𝑟 =min 𝑀
4𝑑 ,𝑑.
2: Initialize O=(0)𝑁×𝑑 ∈R𝑁×𝑑,ℓ=(0)𝑁 ∈R𝑁,𝑚=(−∞)𝑁 ∈R𝑁 in HBM.
3: Divide Q into 𝑇𝑟 = 𝑁
𝐵𝑟
blocks Q1,...,Q𝑇𝑟
of size 𝐵𝑟 ×𝑑 each, and divide K,V in to 𝑇𝑐 = 𝑁
𝐵𝑐
blocks
K1,...,K𝑇𝑐
and V1,...,V𝑇𝑐, of size 𝐵𝑐 ×𝑑 each.
4: Divide O into 𝑇𝑟 blocks O𝑖,...,O𝑇𝑟
of size 𝐵𝑟 ×𝑑 each, divide ℓ into 𝑇𝑟 blocks ℓ𝑖,...,ℓ𝑇𝑟
of size 𝐵𝑟 each,
divide 𝑚 into 𝑇𝑟 blocks 𝑚1,...,𝑚𝑇𝑟
of size 𝐵𝑟 each.
5: for 1 ≤𝑗 ≤𝑇𝑐 do
6: Load K 𝑗,V 𝑗 from HBM to on-chip SRAM.
7: for 1 ≤𝑖 ≤𝑇𝑟 do
8: Load Q𝑖,O𝑖,ℓ𝑖,𝑚𝑖 from HBM to on-chip SRAM.
9: On chip, compute S𝑖𝑗 =Q𝑖K𝑇
𝑗 ∈R𝐵𝑟×𝐵𝑐
.
10: On chip, compute˜
𝑚𝑖𝑗 = rowmax(S𝑖𝑗) ∈R𝐵𝑟
,˜
P𝑖𝑗 = exp(S𝑖𝑗−
˜
𝑚𝑖𝑗) ∈R𝐵𝑟×𝐵𝑐 (pointwise),˜
ℓ𝑖𝑗 =
rowsum(˜
P𝑖𝑗)∈R𝐵𝑟
.
˜
11: On chip, compute 𝑚new
𝑖 =max(𝑚𝑖,˜
𝑚𝑖𝑗)∈R𝐵𝑟
, ℓnew
𝑖 =𝑒𝑚𝑖−𝑚new
𝑖 ℓ𝑖 +𝑒˜
𝑚𝑖𝑗−𝑚new
𝑖
ℓ𝑖𝑗 ∈R𝐵𝑟
.
˜
12: Write O𝑖 ←diag(ℓnew
𝑖 )−1 (diag(ℓ𝑖)𝑒𝑚𝑖−𝑚new
𝑖 O𝑖 +𝑒˜
𝑚𝑖𝑗−𝑚new
𝑖
P𝑖𝑗V 𝑗)to HBM.
13: Write ℓ𝑖 ←ℓnew
𝑖 , 𝑚𝑖 ←𝑚new
𝑖 to HBM.
14: end for
15: end for
16: Return O.
(這flashattn 幫我參考這樣寫)根據底下需求幫我加強
to Moe -> men bound 變成 computed 這樣 然後證明

你可以怎麼用： 你的碩士論文一定要有這個章節。你必須用數學證明，傳統 MoE 需要搬運多少 Gigabytes 的權重（Memory-bound），而你的 Tucker-MoE 因為共用了 $U$ 和 $V$，只搬運了 $C_i$，使得 I/O 複雜度從 $\mathcal{O}(...)$ 降到了 $\mathcal{O}(...)$。你可以完全模仿 ELSA 分析 I/O 瓶頸的寫作邏輯與數學符號。

這樣
