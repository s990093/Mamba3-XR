Causal Mask 需要做出來 plot sft 都要 展示mask（用ｈｔｍｌ展示，字體需要times new roman 字體大小適中）
Frobenius 誤差通常是指使用 Frobenius 範數（Frobenius norm，

）衡量矩陣近似或分解中的差異。它是矩陣所有元素平方和的平方根，常應用於數值線性代數、矩陣分解（如 SVD 或 NMF）中的誤差計算。
YouTube
YouTube
+1
核心定義與應用：
定義： 對於矩陣
，其 Frobenius 誤差通常定義為真實矩陣與近似矩陣之差的範數：

矩陣分解： 在矩陣分解（如

）中，Frobenius 誤差用於衡量
與
之間的接近程度，即最小化

要加入說明 跟數學式 及不需要加入 html

(py310) hungwei@ACVLAB-3090-2:~/llm$ source /home/hungwei/miniconda3/etc/profile.d/conda.sh && conda activate py310 && python3 check_mask.py
🚀 高階 GPU (NVIDIA GeForce RTX 3090, sm_86): bf16 + TF32 + FlashSDP + cuDNN benchmark
========================================================================
Sample #0 | total_tokens=1014 supervised=675(66.6%)
========================================================================

── Boundary: last header token → first answer token
Idx │ Context x[i] │ Label y[i] │ supervised
-----─┼─--------------------─┼─-------------------------─┼─----------
31 │ '\n' │ '-100' │ no  
 32 │ '<|im_start|>' │ '-100' │ no  
 33 │ 'ass' │ '-100' │ no  
 34 │ 'istant' │ '-100' │ no  
 35 │ '\n' │ '-100' │ no  
 36 │ 'As' │ 'an' │ yes ←
37 │ 'an' │ 'A' │ yes  
 38 │ 'A' │ 'I' │ yes  
 39 │ 'I' │ 'language' │ yes  
 40 │ 'language' │ 'model' │ yes  
 41 │ 'model' │ ',' │ yes  
 42 │ ',' │ 'I' │ yes  
 43 │ 'I' │ 'do' │ yes  
 44 │ 'do' │ 'not' │ yes  
 45 │ 'not' │ 'hold' │ yes

Total assistant turns found in text: 6

── Turn 1/6 — header end → answer start
Idx │ Context x[i] │ Label y[i] │ supervised
-----─┼─--------------------─┼─-------------------------─┼─----------
32 │ '<|im_start|>' │ '-100' │ no  
 33 │ 'ass' │ '-100' │ no  
 34 │ 'istant' │ '-100' │ no  
 35 │ '\n' │ '-100' │ no  
 36 │ 'As' │ 'an' │ yes ←
37 │ 'an' │ 'A' │ yes  
 38 │ 'A' │ 'I' │ yes  
 39 │ 'I' │ 'language' │ yes  
 40 │ 'language' │ 'model' │ yes  
 41 │ 'model' │ ',' │ yes

── Turn 2/6 — header end → answer start
Idx │ Context x[i] │ Label y[i] │ supervised
-----─┼─--------------------─┼─-------------------------─┼─----------
252 │ '<|im_start|>' │ '-100' │ no  
 253 │ 'ass' │ '-100' │ no  
 254 │ 'istant' │ '-100' │ no  
 255 │ '\n' │ '-100' │ no  
 256 │ 'As' │ 'an' │ yes ←
257 │ 'an' │ 'A' │ yes  
 258 │ 'A' │ 'I' │ yes  
 259 │ 'I' │ 'language' │ yes  
 260 │ 'language' │ 'model' │ yes  
 261 │ 'model' │ ',' │ yes

── Turn 3/6 — header end → answer start
Idx │ Context x[i] │ Label y[i] │ supervised
-----─┼─--------------------─┼─-------------------------─┼─----------
418 │ '<|im_start|>' │ '-100' │ no  
 419 │ 'ass' │ '-100' │ no  
 420 │ 'istant' │ '-100' │ no  
 421 │ '\n' │ '-100' │ no  
 422 │ 'As' │ 'an' │ yes ←
423 │ 'an' │ 'A' │ yes  
 424 │ 'A' │ 'I' │ yes  
 425 │ 'I' │ 'language' │ yes  
 426 │ 'language' │ 'model' │ yes  
 427 │ 'model' │ ',' │ yes

── Turn 4/6 — header end → answer start
Idx │ Context x[i] │ Label y[i] │ supervised
-----─┼─--------------------─┼─-------------------------─┼─----------
595 │ '<|im_start|>' │ '-100' │ no  
 596 │ 'ass' │ '-100' │ no  
 597 │ 'istant' │ '-100' │ no  
 598 │ '\n' │ '-100' │ no  
 599 │ 'As' │ 'an' │ yes ←
600 │ 'an' │ 'A' │ yes  
 601 │ 'A' │ 'I' │ yes  
 602 │ 'I' │ 'language' │ yes  
 603 │ 'language' │ 'model' │ yes  
 604 │ 'model' │ ',' │ yes

── Turn 5/6 — header end → answer start
Idx │ Context x[i] │ Label y[i] │ supervised
-----─┼─--------------------─┼─-------------------------─┼─----------
754 │ '<|im_start|>' │ '-100' │ no  
 755 │ 'ass' │ '-100' │ no  
 756 │ 'istant' │ '-100' │ no  
 757 │ '\n' │ '-100' │ no  
 758 │ 'As' │ 'an' │ yes ←
759 │ 'an' │ 'A' │ yes  
 760 │ 'A' │ 'I' │ yes  
 761 │ 'I' │ 'language' │ yes  
 762 │ 'language' │ 'model' │ yes  
 763 │ 'model' │ ',' │ yes

── Turn 6/6 — header end → answer start
Idx │ Context x[i] │ Label y[i] │ supervised
-----─┼─--------------------─┼─-------------------------─┼─----------
904 │ '<|im_start|>' │ '-100' │ no  
 905 │ 'ass' │ '-100' │ no  
 906 │ 'istant' │ '-100' │ no  
 907 │ '\n' │ '-100' │ no  
 908 │ 'As' │ 'an' │ yes ←
909 │ 'an' │ 'A' │ yes  
 910 │ 'A' │ 'I' │ yes  
 911 │ 'I' │ 'language' │ yes  
 912 │ 'language' │ 'model' │ yes  
 913 │ 'model' │ ',' │ yes

✅ PASS Header tokens (incl. \n) fully masked
✅ PASS First answer token supervised in every turn
✅ PASS <|im_end|> supervised (model learns to stop)
✅ PASS Multi-turn coverage (turns=6)
========================================================================
Sample #1 | total_tokens=849 supervised=699(82.3%)
========================================================================

── Boundary: last header token → first answer token
Idx │ Context x[i] │ Label y[i] │ supervised
-----─┼─--------------------─┼─-------------------------─┼─----------
22 │ '\n' │ '-100' │ no  
 23 │ '<|im_start|>' │ '-100' │ no  
 24 │ 'ass' │ '-100' │ no  
 25 │ 'istant' │ '-100' │ no  
 26 │ '\n' │ '-100' │ no  
 27 │ 'There' │ 'are' │ yes ←
28 │ 'are' │ 'many' │ yes  
 29 │ 'many' │ 'ways' │ yes  
 30 │ 'ways' │ 'to' │ yes  
 31 │ 'to' │ 'incorpor' │ yes  
 32 │ 'incorpor' │ 'ate' │ yes  
 33 │ 'ate' │ 'k' │ yes  
 34 │ 'k' │ 'ale' │ yes  
 35 │ 'ale' │ 'and' │ yes  
 36 │ 'and' │ 'spin' │ yes

Total assistant turns found in text: 4

── Turn 1/4 — header end → answer start
Idx │ Context x[i] │ Label y[i] │ supervised
-----─┼─--------------------─┼─-------------------------─┼─----------
23 │ '<|im_start|>' │ '-100' │ no  
 24 │ 'ass' │ '-100' │ no  
 25 │ 'istant' │ '-100' │ no  
 26 │ '\n' │ '-100' │ no  
 27 │ 'There' │ 'are' │ yes ←
28 │ 'are' │ 'many' │ yes  
 29 │ 'many' │ 'ways' │ yes  
 30 │ 'ways' │ 'to' │ yes  
 31 │ 'to' │ 'incorpor' │ yes  
 32 │ 'incorpor' │ 'ate' │ yes

── Turn 2/4 — header end → answer start
Idx │ Context x[i] │ Label y[i] │ supervised
-----─┼─--------------------─┼─-------------------------─┼─----------
370 │ '<|im_start|>' │ '-100' │ no  
 371 │ 'ass' │ '-100' │ no  
 372 │ 'istant' │ '-100' │ no  
 373 │ '\n' │ '-100' │ no  
 374 │ 'That' │ "'" │ yes ←
375 │ "'" │ 's' │ yes  
 376 │ 's' │ 'a' │ yes  
 377 │ 'a' │ 'great' │ yes  
 378 │ 'great' │ 'idea' │ yes  
 379 │ 'idea' │ '!' │ yes

── Turn 3/4 — header end → answer start
Idx │ Context x[i] │ Label y[i] │ supervised
-----─┼─--------------------─┼─-------------------------─┼─----------
676 │ '<|im_start|>' │ '-100' │ no  
 677 │ 'ass' │ '-100' │ no  
 678 │ 'istant' │ '-100' │ no  
 679 │ '\n' │ '-100' │ no  
 680 │ 'That' │ 'sounds' │ yes ←
681 │ 'sounds' │ 'like' │ yes  
 682 │ 'like' │ 'a' │ yes  
 683 │ 'a' │ 'del' │ yes  
 684 │ 'del' │ 'icious' │ yes  
 685 │ 'icious' │ 'and' │ yes

── Turn 4/4 — header end → answer start
Idx │ Context x[i] │ Label y[i] │ supervised
-----─┼─--------------------─┼─-------------------------─┼─----------
781 │ '<|im_start|>' │ '-100' │ no  
 782 │ 'ass' │ '-100' │ no  
 783 │ 'istant' │ '-100' │ no  
 784 │ '\n' │ '-100' │ no  
 785 │ 'I' │ "'" │ yes ←
786 │ "'" │ 'm' │ yes  
 787 │ 'm' │ 'glad' │ yes  
 788 │ 'glad' │ 'to' │ yes  
 789 │ 'to' │ 'hear' │ yes  
 790 │ 'hear' │ 'that' │ yes

✅ PASS Header tokens (incl. \n) fully masked
✅ PASS First answer token supervised in every turn
✅ PASS <|im_end|> supervised (model learns to stop)
✅ PASS Multi-turn coverage (turns=4)
========================================================================
Sample #2 | total_tokens=824 supervised=661(80.2%)
========================================================================

── Boundary: last header token → first answer token
Idx │ Context x[i] │ Label y[i] │ supervised
-----─┼─--------------------─┼─-------------------------─┼─----------
27 │ '\n' │ '-100' │ no  
 28 │ '<|im_start|>' │ '-100' │ no  
 29 │ 'ass' │ '-100' │ no  
 30 │ 'istant' │ '-100' │ no  
 31 │ '\n' │ '-100' │ no  
 32 │ 'As' │ 'an' │ yes ←
33 │ 'an' │ 'A' │ yes  
 34 │ 'A' │ 'I' │ yes  
 35 │ 'I' │ 'language' │ yes  
 36 │ 'language' │ 'model' │ yes  
 37 │ 'model' │ ',' │ yes  
 38 │ ',' │ 'I' │ yes  
 39 │ 'I' │ 'don' │ yes  
 40 │ 'don' │ "'" │ yes  
 41 │ "'" │ 't' │ yes

Total assistant turns found in text: 3

── Turn 1/3 — header end → answer start
Idx │ Context x[i] │ Label y[i] │ supervised
-----─┼─--------------------─┼─-------------------------─┼─----------
28 │ '<|im_start|>' │ '-100' │ no  
 29 │ 'ass' │ '-100' │ no  
 30 │ 'istant' │ '-100' │ no  
 31 │ '\n' │ '-100' │ no  
 32 │ 'As' │ 'an' │ yes ←
33 │ 'an' │ 'A' │ yes  
 34 │ 'A' │ 'I' │ yes  
 35 │ 'I' │ 'language' │ yes  
 36 │ 'language' │ 'model' │ yes  
 37 │ 'model' │ ',' │ yes

── Turn 2/3 — header end → answer start
Idx │ Context x[i] │ Label y[i] │ supervised
-----─┼─--------------------─┼─-------------------------─┼─----------
452 │ '<|im_start|>' │ '-100' │ no  
 453 │ 'ass' │ '-100' │ no  
 454 │ 'istant' │ '-100' │ no  
 455 │ '\n' │ '-100' │ no  
 456 │ 'As' │ 'an' │ yes ←
457 │ 'an' │ 'A' │ yes  
 458 │ 'A' │ 'I' │ yes  
 459 │ 'I' │ 'language' │ yes  
 460 │ 'language' │ 'model' │ yes  
 461 │ 'model' │ ',' │ yes

── Turn 3/3 — header end → answer start
Idx │ Context x[i] │ Label y[i] │ supervised
-----─┼─--------------------─┼─-------------------------─┼─----------
669 │ '<|im_start|>' │ '-100' │ no  
 670 │ 'ass' │ '-100' │ no  
 671 │ 'istant' │ '-100' │ no  
 672 │ '\n' │ '-100' │ no  
 673 │ 'As' │ 'an' │ yes ←
674 │ 'an' │ 'A' │ yes  
 675 │ 'A' │ 'I' │ yes  
 676 │ 'I' │ 'language' │ yes  
 677 │ 'language' │ 'model' │ yes  
 678 │ 'model' │ ',' │ yes

✅ PASS Header tokens (incl. \n) fully masked
✅ PASS First answer token supervised in every turn
✅ PASS <|im_end|> supervised (model learns to stop)
✅ PASS Multi-turn coverage (turns=3)

========================================================================
✅ PASS All 3 samples verified. Mask is correct!
以上東西需要轉換成論文 table那要顯示就好 不需要加入html

```text
第 1 段：<|im_start|>assistant\n1. Influencer marketing: Companies are collaborating...<|im_end|>
第 2 段：<|im_start|>assistant\nAs an AI language model, I don't have personal opinions...<|im_end|>
第 3 段：<|im_start|>assistant\nAs an AI language model, I can say that relying solely on traditional...<|im_end|>
第 4 段：<|im_start|>assistant\nAs an AI language model, I can say that keeping up with the latest marketing trends...<|im_end|>
```

需要

所有的異質資料來源都被嚴格映射到同一對話語法。對於單輪問答，標準化後的模板為：

```text
<|im_start|>user
...使用者內容...<|im_end|>
<|im_start|>assistant
...助理答案...<|im_end|>
```

在推論（Inference）時，我們只餵給模型開放式的 Assistant 前綴，讓它接續生成：

```text
<|im_start|>user
...使用者內容...<|im_end|>
<|im_start|>assistant 也需要美觀  透過html 等等 再轉換等等 截圖 都用

幫我加入底下內容（要美觀 可以透過html 展示 字體需要times new roman 字體大小適中）
```

以上東西加入 /Users/hungwei/Desktop/Proj/Mamba3-XR/paper/hybrid-mamba-15min/report.md
