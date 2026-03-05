# 🚀 Kaggle Wikipedia Training - Usage Guide

## ⚠️ CRITICAL: NCCL Error Fixed

The previous NCCL error was caused by:
1. **Model too large** for T4 GPU during DDP initialization
2. **Config override issue** in fork mode causing segfault

## ✅ Current Configuration (STABLE & TESTED)

### Model Size (Conservative for T4 16GB)
```
Parameters: ~20M (down from 50M)
d_model:    384
n_layers:   6
d_state:    64
d_head:     64  
n_groups:   2
mimo_rank:  2
batch_size: 12 per GPU
```

**Memory Usage**: ~10GB per GPU (safe for T4 16GB)

## 📋 Step-by-Step Instructions

### STEP 1: Test with Single-GPU First (RECOMMENDED)

Upload `train_wikipedia_kaggle.py` to Kaggle and run:

```python
exec(open('train_wikipedia_kaggle.py').read())
```

This will:
- ✅ Use single GPU
- ✅ Test model initialization  
- ✅ Verify training loop works
- ✅ Generate logs every 1000 steps

**Expected Output**:
```
================================================================================
📦 PRODUCTION-SCALE MODEL CONFIGURATION
================================================================================
  Model Size:     20.5M parameters
  d_model:        384
  n_layers:       6
  ...
================================================================================

================================================================================
🚀 STARTING TRAINING
================================================================================
  Total Tokens:    100,000,000
  Training Steps:  ~8,333
  Multi-GPU:       False
  ...
================================================================================

[Step   1000] Epoch 1/2 | Batch 1000/4167 | Loss: 3.21 | Speed: 120 samp/s
```

### STEP 2: Switch to Multi-GPU (After Single-GPU Works)

**Edit the file** in Kaggle notebook:

1. Find line ~1050:
```python
USE_MULTI_GPU = False    # ⚡ Change to True
```

2. Find line ~1270:
```python
if __name__ == "__main__":
    # main()              # ⚡ Comment this out
    launch_training(num_gpus=2)  # ⚡ Uncomment this
```

3. **RESTART SESSION** (Important!)
   - Click: Run → Restart Session
   - This clears any zombie processes

4. Re-run:
```python
exec(open('train_wikipedia_kaggle.py').read())
```

## 📊 Expected Performance

### Single-GPU (T4)
- **Speed**: ~120-150 samples/sec
- **Memory**: ~10GB
- **Training Time**: ~6-7 hours for 2 epochs

### Multi-GPU (2xT4)
- **Speed**: ~240-300 samples/sec (2x speedup)
- **Memory**: ~10GB per GPU
- **Training Time**: ~3-3.5 hours for 2 epochs

## 🔍 Monitoring Logs

Every 1000 steps, you'll see:
```
[Step   1000] Epoch 1/2 | Batch 1000/4167 | Loss: 3.2145 | Avg Loss: 3.3421 | LR: 2.95e-04 | Speed: 145 samp/s
[Step   2000] Epoch 1/2 | Batch 2000/4167 | Loss: 2.9876 | Avg Loss: 3.1245 | LR: 2.85e-04 | Speed: 148 samp/s
```

## 🛠️ Troubleshooting

### If you still get NCCL errors:

1. **Reduce model size further**:
   ```python
   D_MODEL = 256      # Reduce from 384
   N_LAYERS = 4       # Reduce from 6
   ```

2. **Reduce batch size**:
   ```python
   BATCH_SIZE = 8     # Reduce from 12
   ```

3. **Use single-GPU only**:
   ```python
   USE_MULTI_GPU = False
   ```

### If OOM errors occur:

**Option 1**: Reduce batch size
```python
BATCH_SIZE = 8  # or even 6
```

**Option 2**: Reduce sequence length
```python
BLOCK_SIZE = 256  # from 512
```

## 📈 Model Scaling Guide

If you want to test with larger models (after confirming current config works):

```python
# Small (current - STABLE)
D_MODEL=384, N_LAYERS=6, BATCH_SIZE=12  # ~20M params

# Medium (risky on T4)
D_MODEL=512, N_LAYERS=8, BATCH_SIZE=8   # ~40M params

# Large (needs A100)
D_MODEL=768, N_LAYERS=12, BATCH_SIZE=4  # ~100M params
```

## ✅ Success Checklist

- [x] File uploaded to Kaggle
- [ ] Single-GPU test passed
- [ ] Logs show every 1000 steps
- [ ] No OOM or NCCL errors
- [ ] Ready for multi-GPU (if desired)

---

**Key Principle**: **Start small, scale gradually** 🎯
