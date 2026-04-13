import os
import numpy as np
import mlx.core as mx
import mlx.utils


def save_checkpoint_numpy(model, filepath):
    print("💾 開始將模型權重匯出為 NumPy 格式...")
    flat_params = mlx.utils.tree_flatten(model.parameters())
    np_params = {k: np.array(v) for k, v in flat_params}
    np.savez(filepath, **np_params)
    del np_params
    print(f"✅ 模型已成功儲存至 {filepath}")


def strict_load_and_convert(model, filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ 找不到 Checkpoint 檔案: {filepath}")
    print(f"📥 正在從 {filepath} 讀取權重...")

    if filepath.endswith(".pt") or filepath.endswith(".bin"):
        import torch

        pt_state_dict = torch.load(filepath, map_location="cpu", weights_only=True)
        if "model" in pt_state_dict:
            pt_state_dict = pt_state_dict["model"]
        elif "state_dict" in pt_state_dict:
            pt_state_dict = pt_state_dict["state_dict"]
        np_data = {k: v.float().numpy() for k, v in pt_state_dict.items()}
        print(f"✅ 成功提取 PyTorch Checkpoint，共 {len(np_data)} 個權重張量。")
    else:
        npz_file = np.load(filepath, mmap_mode="r")
        np_data = {k: npz_file[k] for k in npz_file.files}
        print(f"✅ 成功提取 NumPy Checkpoint，共 {len(np_data)} 個權重張量。")

    mlx_params = mlx.utils.tree_flatten(model.parameters())
    mlx_keys = set(k for k, _ in mlx_params)
    loaded_flat_params = []
    loaded_keys = set()
    for k, v in np_data.items():
        new_k = k.replace(".block.", ".")
        loaded_flat_params.append((new_k, mx.array(v)))
        loaded_keys.add(new_k)

    missing_in_mlx = mlx_keys - loaded_keys
    missing_in_ckpt = loaded_keys - mlx_keys
    match_rate = (len(mlx_keys) - len(missing_in_mlx)) / len(mlx_keys) * 100 if len(mlx_keys) > 0 else 0
    print(f"🔍 模型權重匹配率: {match_rate:.2f}% ({len(mlx_keys)-len(missing_in_mlx)}/{len(mlx_keys)})")

    if len(missing_in_mlx) > 0 or len(missing_in_ckpt) > 0:
        raise ValueError("Model layer 完美轉換檢查失敗，請檢察您的模型架構或權重名稱！")

    loaded_params = mlx.utils.tree_unflatten(loaded_flat_params)
    model.update(loaded_params)
    mx.eval(model.parameters())
    print("✨ 強制嚴格檢查通過！權重已完美轉換並載入至 MLX 模型中！")


def load_and_compare_vocab(model, filepath, expected_vocab_size):
    strict_load_and_convert(model, filepath)
