import os
import glob
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

def main():
    # ================================
    # 🔧 路徑設定
    # ================================
    DATA_PATH = "/Users/hungwei/.cache/kagglehub/datasets/nameonlu/fineweb-edu/versions/2"
    TOKENIZER_PATH = "/Users/hungwei/Desktop/Proj/Mamba3-XR/llama2_tokenizer_chatml_final/tokenizer.json"
    OUTPUT_PATH = "/Users/hungwei/Downloads/fineweb_tokenized.bin"

    # ================================
    # 🧠 載入 tokenizer 並設定 special tokens
    # ================================
    print("🧠 載入 tokenizer...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)

    # 🚨 指定 special tokens，避免 eos_id = None
    tokenizer.eos_token = "</s>"
    tokenizer.bos_token = "<s>"
    tokenizer.pad_token = "[PAD]"

    eos_id = tokenizer.eos_token_id
    print(f"eos_id -> {eos_id}")

    vocab = tokenizer.get_vocab()
    print("實際 vocab size:", len(vocab))
    print("實際最大 token id:", max(vocab.values()))

    # ================================
    # 📄 找 txt（只取一半）
    # ================================
    txt_files = glob.glob(f"{DATA_PATH}/**/*.txt", recursive=True)
    txt_files.sort()
    half_len = len(txt_files) // 2
    txt_files = txt_files[:half_len]

    print(f"📄 找到 {len(txt_files)} 個檔案 (只取一半)")
    if len(txt_files) == 0:
        raise ValueError("❌ 找不到 txt")

    # ================================
    # 💾 準備輸出
    # ================================
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    CHUNK_SIZE = 2_000_000 * 3
    chunk_tokens = []
    total_tokens = 0

    # ================================
    # 🔥 Tokenize（Streaming）
    # ================================
    with open(OUTPUT_PATH, "ab") as f_out:
        for file_path in tqdm(txt_files, desc="Tokenizing"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = []
                    for line in f:
                        line = line.strip()
                        if line:
                            lines.append(line)

                        if len(lines) >= 1000:
                            enc = tokenizer(lines, add_special_tokens=False, return_attention_mask=False)
                            for ids in enc["input_ids"]:
                                chunk_tokens.extend(ids)
                                chunk_tokens.append(eos_id)  # ✅ EOS 一定有值
                            lines = []

                        if len(chunk_tokens) >= CHUNK_SIZE:
                            np.array(chunk_tokens, dtype=np.uint16).tofile(f_out)
                            total_tokens += len(chunk_tokens)
                            chunk_tokens = []

                    if lines:
                        enc = tokenizer(lines, add_special_tokens=False)
                        for ids in enc["input_ids"]:
                            chunk_tokens.extend(ids)
                            chunk_tokens.append(eos_id)

            except Exception as e:
                print(f"⚠️ 略過 {file_path}: {e}")

        if chunk_tokens:
            np.array(chunk_tokens, dtype=np.uint16).tofile(f_out)
            total_tokens += len(chunk_tokens)

    # ================================
    # 📊 結果
    # ================================
    print("\n🎉 完成！")
    print(f"🔢 tokens: {total_tokens:,}")
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"💾 檔案大小: {size_mb:.2f} MB")
    print(f"📂 輸出: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()