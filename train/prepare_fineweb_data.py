import os
import glob
import numpy as np
import sentencepiece as spm
import kagglehub
from tqdm import tqdm

def main():
    # 1. 下載資料集
    print("📥 正在透過 kagglehub 下載 nameonlu/fineweb-edu ...")
    raw_data_path = kagglehub.dataset_download("nameonlu/fineweb-edu")
    print(f"✅ 下載完成！原始資料路徑: {raw_data_path}")

    # 2. 準備輸出與 Tokenizer 路徑
    LOCAL_DATA_DIR = "./data"
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    
    TOKENIZER_PATH = os.path.join(LOCAL_DATA_DIR, "spm_tokenizer.model")
    OUTPUT_BIN_PATH = os.path.join(LOCAL_DATA_DIR, "fineweb_edu_tokenized.bin")
    
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"❌ 找不到 Tokenizer 模型！請確保已經放在 {TOKENIZER_PATH}")
    
    print(f"🧠 載入 SentencePiece Tokenizer ({TOKENIZER_PATH})...")
    tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    
    # 3. 搜尋所有下載回來的 txt 檔案
    txt_files = glob.glob(f"{raw_data_path}/**/*.txt", recursive=True)
    if not txt_files:
        raise ValueError(f"❌ 在 {raw_data_path} 中找不到任何 .txt 檔案！")
    print(f"📄 找到 {len(txt_files)} 個文字檔準備進行 Tokenize...")

    # 4. 開始 Tokenize 並「分批」寫入檔案 (Streaming 模式，保護記憶體)
    if os.path.exists(OUTPUT_BIN_PATH):
        os.remove(OUTPUT_BIN_PATH) # 先清掉舊檔
        
    all_tokens_count = 0
    chunk_tokens = []
    CHUNK_SIZE = 1_000_000 # 每一百萬個 token 就主動存檔一次，清空 RAM
    
    with open(OUTPUT_BIN_PATH, 'ab') as f_out:
        for file_path in tqdm(txt_files, desc="Tokenizing Files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # 逐行讀取，減少單次讀檔壓力
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        
                        tokens = tokenizer.encode(line)
                        chunk_tokens.extend(tokens)
                        
                        # 如果累積滿一個 chunk，就寫入硬碟並清空記憶體
                        if len(chunk_tokens) >= CHUNK_SIZE:
                            np.array(chunk_tokens, dtype=np.uint16).tofile(f_out)
                            all_tokens_count += len(chunk_tokens)
                            chunk_tokens = [] # 這裡就是釋放記憶體的關鍵！
                            
            except Exception as e:
                print(f"⚠️ 略過檔案 {file_path} 因為發生錯誤: {e}")

        # 最後處理剩餘不足一個 CHUNK_SIZE 的部分
        if chunk_tokens:
            np.array(chunk_tokens, dtype=np.uint16).tofile(f_out)
            all_tokens_count += len(chunk_tokens)
            chunk_tokens = []

    print(f"\n📦 Tokenize 完畢！總共累積了 {all_tokens_count:,} 個 Tokens。")
    print(f"💾 資料已成功分批存入二進位檔案 ({OUTPUT_BIN_PATH})。")
    
    print("\n🎉 全部搞定！")
    print(f"現在你只需要把 {OUTPUT_BIN_PATH} 上傳到 Kaggle/Colab。")
    filesize_mb = os.path.getsize(OUTPUT_BIN_PATH) / (1024 * 1024)
    print(f"檔案大小約為: {filesize_mb:.2f} MB")

if __name__ == "__main__":
    main()
