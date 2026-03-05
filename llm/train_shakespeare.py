"""
Mamba-3 LLM "Hello World" - Shakespeare Character-Level Language Model

This is a minimal example to test the Mamba-3 LLM implementation.
We'll train on Shakespeare's complete works at the character level.

Dataset: ~1MB of text
Model: Tiny Mamba-3 (d_model=256, 4 layers, Rank 4)
Goal: Generate coherent Shakespeare-style text
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import requests
from pathlib import Path
import time
import sys

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.mamba3_lm import create_mamba3_tiny


# ============================================================================
# 1. Data Preparation
# ============================================================================

class CharDataset(Dataset):
    """Character-level dataset for language modeling"""
    
    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.block_size = block_size
        
        # Create char <-> int mappings
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Encode the entire text
        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        
        print(f"Dataset: {len(text)} characters, {self.vocab_size} unique")
        print(f"Vocabulary: {''.join(chars[:50])}...")
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get a chunk of text
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]  # Input
        y = chunk[1:]   # Target (shifted by 1)
        return x, y
    
    def decode(self, tokens):
        """Decode token IDs to text"""
        return ''.join([self.itos[int(t)] for t in tokens])


def get_shakespeare_data(cache_dir='data'):
    """Download Shakespeare dataset"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    data_path = cache_dir / 'shakespeare.txt'
    
    if not data_path.exists():
        print("Downloading Shakespeare dataset...")
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        response = requests.get(url)
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"✓ Downloaded to {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text


# ============================================================================
# 2. Training Loop
# ============================================================================

def train_shakespeare(
    model,
    train_loader,
    val_loader,
    dataset,
    epochs=10,
    lr=3e-4,
    device='cuda',
    eval_interval=100,
    eval_iters=20,
):
    """Train the model on Shakespeare"""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler (cosine)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader)
    )
    
    print("\n" + "=" * 80)
    print("Training Mamba-3 on Shakespeare")
    print("=" * 80)
    print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print("=" * 80 + "\n")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            loss, logits = model(x, targets=y, return_loss=True)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Evaluation
            if global_step % eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, eval_iters)
                
                print(f"Step {global_step:5d} | "
                      f"Train Loss: {loss.item():.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 'shakespeare_best.pt')
                
                model.train()
        
        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch + 1}/{epochs} | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Generate sample text
        print("\n" + "-" * 80)
        print("Sample Generation:")
        print("-" * 80)
        sample_text = generate_sample(model, dataset, device, max_tokens=200)
        print(sample_text)
        print("-" * 80 + "\n")
    
    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")
    return model


@torch.no_grad()
def evaluate(model, val_loader, device, eval_iters):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    
    for i, (x, y) in enumerate(val_loader):
        if i >= eval_iters:
            break
        
        x, y = x.to(device), y.to(device)
        loss, _ = model(x, targets=y, return_loss=True)
        total_loss += loss.item()
    
    return total_loss / min(eval_iters, len(val_loader))


@torch.no_grad()
def generate_sample(model, dataset, device, prompt="\n", max_tokens=200):
    """Generate a sample from the model"""
    model.eval()
    
    # Encode prompt
    tokens = torch.tensor(
        [dataset.stoi[ch] for ch in prompt],
        dtype=torch.long
    ).unsqueeze(0).to(device)
    
    # Generate
    generated = model.generate(
        tokens,
        max_new_tokens=max_tokens,
        temperature=0.8,
        top_k=40
    )
    
    # Decode
    text = dataset.decode(generated[0])
    return text


# ============================================================================
# 3. Main Execution
# ============================================================================

def main():
    # Configuration
    BLOCK_SIZE = 256      # Context length
    BATCH_SIZE = 32       # Batch size
    EPOCHS = 10           # Number of epochs
    LR = 3e-4             # Learning rate
    MIMO_RANK = 4         # MIMO rank (1, 4, 8, 16)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # 1. Load data
    text = get_shakespeare_data()
    
    # 2. Train/val split (90/10)
    split_idx = int(0.9 * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # 3. Create datasets
    train_dataset = CharDataset(train_text, BLOCK_SIZE)
    val_dataset = CharDataset(val_text, BLOCK_SIZE)
    
    # Use the same vocab for both
    val_dataset.stoi = train_dataset.stoi
    val_dataset.itos = train_dataset.itos
    val_dataset.vocab_size = train_dataset.vocab_size
    
    # 4. Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    # 5. Create model
    model = create_mamba3_tiny(
        vocab_size=train_dataset.vocab_size,
        mimo_rank=MIMO_RANK
    )
    
    print(f"\nModel created with MIMO Rank {MIMO_RANK}")
    print(f"Vocabulary size: {train_dataset.vocab_size}")
    
    # 6. Train
    model = train_shakespeare(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset=train_dataset,
        epochs=EPOCHS,
        lr=LR,
        device=DEVICE,
        eval_interval=100,
        eval_iters=20,
    )
    
    # 7. Final generation test
    print("\n" + "=" * 80)
    print("Final Generation Test")
    print("=" * 80)
    
    prompts = [
        "ROMEO:",
        "First Citizen:",
        "KING HENRY:",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 80)
        text = generate_sample(model, train_dataset, DEVICE, prompt=prompt, max_tokens=300)
        print(text)
        print("-" * 80)


if __name__ == "__main__":
    main()
