import torch

ckpt_path = "checkpoint.pt"
out_path = "model.pt"

ckpt = torch.load(ckpt_path, map_location="cpu")

print("Keys in checkpoint:", ckpt.keys())

# 常見模型權重位置
if "state_dict" in ckpt:
    model_state = ckpt["state_dict"]
elif "model" in ckpt:
    model_state = ckpt["model"]
elif "model_state_dict" in ckpt:
    model_state = ckpt["model_state_dict"]
else:
    # 如果本身就是 state_dict
    model_state = ckpt

# 如果有 DataParallel 的 "module." 前綴，順便清掉
new_state = {}
for k, v in model_state.items():
    new_k = k.replace("module.", "")
    new_state[new_k] = v

torch.save(new_state, out_path)

print(f"Saved clean model weights to {out_path}")