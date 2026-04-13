import os
from transformers import AutoTokenizer

# 1. 設定路徑
path = "./llama2_tokenizer" 

if not os.path.exists(path):
    print(f"❌ 找不到路徑: {path}")
    exit()

# 2. 載入原始 Tokenizer
print("正在載入原始 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

# --- 顯示原始資訊 ---
print("\n=== [1] 原始基本資訊 ===")
print(f"Vocab size (原始): {tokenizer.vocab_size}")
print(f"Model max length: {tokenizer.model_max_length}")

print("\n=== [2] 原始 Special tokens ===")
print("BOS:", tokenizer.bos_token, "| ID:", tokenizer.bos_token_id)
print("EOS:", tokenizer.eos_token, "| ID:", tokenizer.eos_token_id)
print("PAD:", tokenizer.pad_token, "| ID:", tokenizer.pad_token_id)
print("UNK:", tokenizer.unk_token, "| ID:", tokenizer.unk_token_id)

# 3. 定義要相容的新標籤 (包含 ChatML 與你提到的 Think/Final 結構)
# 這樣做可以確保模型把這些標籤當作單一 Token，不會亂切
new_tokens_list = [
    "<|im_start|>", 
    "<|im_end|>", 
    "<think>", 
    "</think>", 
    "<final>", 
    "</final>"
]

# 4. 執行新增
print("\n=== [3] 執行更新 ===")
# 使用 additional_special_tokens 確保「兼容」而不覆蓋原本的 BOS/EOS
special_tokens_dict = {'additional_special_tokens': new_tokens_list}
num_added = tokenizer.add_special_tokens(special_tokens_dict)

# 強制設定 PAD (Llama-2 必備)
if tokenizer.pad_token is None:
    tokenizer.pad_token = "[PAD]"
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("已手動補上 [PAD] token")

# --- 顯示更新後資訊 ---
print(f"新增 Token 數量: {num_added}")
print(f"目前總詞表長度 (New Len): {len(tokenizer)}")

print("\n=== [4] 更新後的 Special Tokens 清單 ===")
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)

# 5. 測試編碼 (模擬你的範例格式)
test_prompt = """<|im_start|>assistant
<think>
正在思考...
</think>
<final>
這是最後答案。
</final>
<|im_end|>"""

tokens = tokenizer.encode(test_prompt, add_special_tokens=False)

print("\n=== [5] 兼容性測試 (Encoding Test) ===")
print(f"測試字串:\n{test_prompt}")
print(f"\n編碼後的 Token IDs:\n{tokens}")
print(f"\n解碼還原確認:\n{tokenizer.decode(tokens)}")

# 6. 儲存
save_path = "./llama2_tokenizer_chatml_final"
tokenizer.save_pretrained(save_path)
print(f"\n✨ 處理完成！新 Tokenizer 已儲存至: {save_path}")