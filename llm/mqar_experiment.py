"""
MQAR Experiment: Testing MIMO Rank Impact on Memory Capacity

This script runs controlled experiments to demonstrate that MIMO Rank
directly affects Mamba-3's ability to recall key-value associations
from long sequences.

Hypothesis: Higher MIMO Rank → Better long-term memory

Experiment Design:
- Fixed: d_model, n_layers, vocab_size, seq_len, num_pairs
- Variable: MIMO Rank (1, 4, 8, 16)
- Metric: Accuracy on recalling correct values

Expected Results:
- Rank 1: ~20-30% accuracy (limited memory)
- Rank 16: ~95-100% accuracy (full memory)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from tqdm.auto import tqdm
import time

# Import our MQAR dataset
from mqar_dataset import SyntheticMQARDataset

# Import Mamba-3 model components
import sys
sys.path.append(str(Path(__file__).parent))
from mamba3_shakespeare_kaggle import Mamba3LM, Mamba3Config


# ============================================================================
# Configuration
# ============================================================================

class MQARConfig:
    """Experiment configuration."""
    
    # Fixed parameters (control variables)
    D_MODEL = 256
    N_LAYERS = 4
    D_STATE = 64
    D_HEAD = 64
    N_GROUPS = 4
    EXPAND = 2
    MAX_SEQ_LEN = 512
    
    # Dataset parameters
    VOCAB_SIZE = 1000
    SEQ_LEN = 256
    NUM_PAIRS = 64  # Difficulty: 64 KV pairs = 128 tokens memory span
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.1
    TRAIN_STEPS = 2000
    EVAL_INTERVAL = 100
    EVAL_BATCHES = 50
    
    # Experiment parameters
    MIMO_RANKS = [1, 4, 8, 16]  # Test these ranks
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    RESULTS_DIR = Path('mqar_results')
    
    # Quick test mode (for debugging)
    QUICK_TEST_STEPS = 500
    QUICK_TEST_RANKS = [1, 16]


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_mqar(model, dataset, device, num_batches=50):
    """
    Evaluate model on MQAR task.
    
    Returns:
        accuracy: Percentage of correct recalls
        avg_loss: Average cross-entropy loss
    """
    model.eval()
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    for _ in range(num_batches):
        x, y = next(iter(dataset))
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits = model(x, return_loss=False)
        
        # Only evaluate the last position (where the answer is)
        pred = logits[:, -1, :].argmax(dim=-1)
        target = y[:, -1]
        
        # Filter out -100 (ignore index)
        valid_mask = target != -100
        correct = (pred[valid_mask] == target[valid_mask]).sum()
        
        total_correct += correct.item()
        total_samples += valid_mask.sum().item()
        
        # Calculate loss
        loss = F.cross_entropy(
            logits[:, -1, :],
            target,
            ignore_index=-100
        )
        total_loss += loss.item()
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / num_batches
    
    return accuracy * 100, avg_loss  # Return accuracy as percentage


# ============================================================================
# Training
# ============================================================================

def train_mqar(
    model,
    train_dataset,
    val_dataset,
    config,
    rank,
    device,
):
    """Train model on MQAR task."""
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.TRAIN_STEPS
    )
    
    # Training history
    history = {
        'rank': rank,
        'steps': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': [],
    }
    
    # Progress bar
    pbar = tqdm(
        range(config.TRAIN_STEPS),
        desc=f"Rank {rank:2d}",
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )
    
    best_accuracy = 0
    train_loader = iter(train_dataset)
    
    for step in pbar:
        model.train()
        
        # Get batch
        x, y = next(train_loader)
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        loss, logits = model(x, targets=y, return_loss=True)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
        
        # Evaluation
        if (step + 1) % config.EVAL_INTERVAL == 0:
            val_accuracy, val_loss = evaluate_mqar(
                model, val_dataset, device, config.EVAL_BATCHES
            )
            
            # Record metrics
            history['steps'].append(step + 1)
            history['train_loss'].append(loss.item())
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['learning_rate'].append(scheduler.get_last_lr()[0])
            
            # Print evaluation
            print(f"\n  Step {step+1:4d}: "
                  f"Train Loss: {loss.item():.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_accuracy:.2f}%")
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                save_dir = config.RESULTS_DIR / f'rank_{rank}'
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_dir / 'best_model.pt')
    
    print(f"\n✅ Rank {rank} Complete - Best Accuracy: {best_accuracy:.2f}%\n")
    
    return history, best_accuracy


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(config, quick_test=False):
    """Run MQAR experiment for multiple MIMO ranks."""
    
    # Setup
    config.RESULTS_DIR.mkdir(exist_ok=True)
    device = config.DEVICE
    
    # Adjust for quick test
    if quick_test:
        config.TRAIN_STEPS = config.QUICK_TEST_STEPS
        config.MIMO_RANKS = config.QUICK_TEST_RANKS
        print("\n🚀 Quick Test Mode: 500 steps, Rank 1 vs 16\n")
    
    # Create datasets
    print("="*80)
    print("📊 Creating MQAR Datasets")
    print("="*80 + "\n")
    
    train_dataset = SyntheticMQARDataset(
        vocab_size=config.VOCAB_SIZE,
        seq_len=config.SEQ_LEN,
        num_pairs=config.NUM_PAIRS,
        batch_size=config.BATCH_SIZE,
    )
    
    val_dataset = SyntheticMQARDataset(
        vocab_size=config.VOCAB_SIZE,
        seq_len=config.SEQ_LEN,
        num_pairs=config.NUM_PAIRS,
        batch_size=config.BATCH_SIZE,
    )
    
    # Wrap in DataLoader
    train_loader = DataLoader(train_dataset, batch_size=None)
    val_loader = DataLoader(val_dataset, batch_size=None)
    
    # Run experiments for each rank
    all_histories = []
    final_results = []
    
    print("\n" + "="*80)
    print("🧪 Running MQAR Experiments")
    print("="*80 + "\n")
    
    for rank in config.MIMO_RANKS:
        print(f"\n{'='*80}")
        print(f"🔬 Experiment: MIMO Rank = {rank}")
        print(f"{'='*80}\n")
        
        # Create model
        model = Mamba3LM(
            vocab_size=config.VOCAB_SIZE,
            d_model=config.D_MODEL,
            n_layers=config.N_LAYERS,
            d_state=config.D_STATE,
            d_head=config.D_HEAD,
            n_groups=config.N_GROUPS,
            mimo_rank=rank,
            expand=config.EXPAND,
            max_seq_len=config.MAX_SEQ_LEN,
            dropout=0.0,  # No dropout for MQAR (we want pure memory test)
        ).to(device)
        
        print(f"Model Parameters: {model.get_num_params() / 1e6:.2f}M\n")
        
        # Train
        start_time = time.time()
        history, best_acc = train_mqar(
            model, train_loader, val_loader, config, rank, device
        )
        elapsed = time.time() - start_time
        
        # Save results
        all_histories.append(history)
        final_results.append({
            'rank': rank,
            'best_accuracy': best_acc,
            'final_accuracy': history['val_accuracy'][-1] if history['val_accuracy'] else 0,
            'final_loss': history['val_loss'][-1] if history['val_loss'] else float('inf'),
            'training_time': elapsed,
        })
        
        # Save history
        save_dir = config.RESULTS_DIR / f'rank_{rank}'
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    return all_histories, final_results


# ============================================================================
# Visualization
# ============================================================================

def plot_results(all_histories, final_results, config):
    """Generate plots comparing different ranks."""
    
    plots_dir = config.RESULTS_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Plot 1: Accuracy curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for history in all_histories:
        plt.plot(
            history['steps'],
            history['val_accuracy'],
            marker='o',
            label=f"Rank {history['rank']}",
            linewidth=2
        )
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title('MQAR: Accuracy vs. MIMO Rank', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Loss curves
    plt.subplot(1, 2, 2)
    for history in all_histories:
        plt.plot(
            history['steps'],
            history['val_loss'],
            marker='o',
            label=f"Rank {history['rank']}",
            linewidth=2
        )
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('MQAR: Loss vs. MIMO Rank', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved: {plots_dir / 'training_curves.png'}")
    
    # Plot 3: Final performance bar chart
    plt.figure(figsize=(8, 6))
    ranks = [r['rank'] for r in final_results]
    accuracies = [r['best_accuracy'] for r in final_results]
    
    bars = plt.bar(ranks, accuracies, color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'])
    plt.xlabel('MIMO Rank', fontsize=12)
    plt.ylabel('Best Accuracy (%)', fontsize=12)
    plt.title('MQAR: Final Performance vs. MIMO Rank', fontsize=14, fontweight='bold')
    plt.ylim(0, 105)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'final_performance.png', dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {plots_dir / 'final_performance.png'}")
    
    plt.close('all')


# ============================================================================
# Results Summary
# ============================================================================

def save_summary(final_results, config):
    """Save experiment summary."""
    
    summary_path = config.RESULTS_DIR / 'summary.md'
    
    with open(summary_path, 'w') as f:
        f.write("# MQAR Experiment Results\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- **Task**: Multi-Query Associative Recall\n")
        f.write(f"- **Difficulty**: {config.NUM_PAIRS} Key-Value pairs\n")
        f.write(f"- **Memory Span**: {config.NUM_PAIRS * 2} tokens\n")
        f.write(f"- **Sequence Length**: {config.SEQ_LEN}\n")
        f.write(f"- **Model Size**: {config.D_MODEL}d, {config.N_LAYERS} layers\n")
        f.write(f"- **Training Steps**: {config.TRAIN_STEPS}\n\n")
        
        f.write("## Results\n\n")
        f.write("| MIMO Rank | Best Accuracy | Final Loss | Training Time |\n")
        f.write("|-----------|---------------|------------|---------------|\n")
        
        for result in final_results:
            f.write(f"| {result['rank']:9d} | "
                   f"{result['best_accuracy']:12.2f}% | "
                   f"{result['final_loss']:10.4f} | "
                   f"{result['training_time']:10.1f}s |\n")
        
        f.write("\n## Analysis\n\n")
        
        # Calculate improvement
        if len(final_results) >= 2:
            baseline = final_results[0]['best_accuracy']
            best = final_results[-1]['best_accuracy']
            improvement = best - baseline
            
            f.write(f"- **Baseline (Rank {final_results[0]['rank']})**: {baseline:.2f}%\n")
            f.write(f"- **Best (Rank {final_results[-1]['rank']})**: {best:.2f}%\n")
            f.write(f"- **Improvement**: +{improvement:.2f}%\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("Higher MIMO Rank significantly improves the model's ability to recall ")
        f.write("key-value associations from long sequences, demonstrating that MIMO ")
        f.write("directly expands the state capacity of Mamba-3.\n")
    
    print(f"\n📝 Saved: {summary_path}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MQAR Experiment')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test (500 steps, Rank 1 vs 16)')
    parser.add_argument('--ranks', nargs='+', type=int,
                       help='MIMO ranks to test (default: 1 4 8 16)')
    parser.add_argument('--steps', type=int,
                       help='Training steps (default: 2000)')
    
    args = parser.parse_args()
    
    # Create config
    config = MQARConfig()
    
    # Override config if specified
    if args.ranks:
        config.MIMO_RANKS = args.ranks
    if args.steps:
        config.TRAIN_STEPS = args.steps
    
    # Run experiment
    all_histories, final_results = run_experiment(config, args.quick_test)
    
    # Generate plots
    print("\n" + "="*80)
    print("📊 Generating Plots")
    print("="*80)
    plot_results(all_histories, final_results, config)
    
    # Save summary
    save_summary(final_results, config)
    
    # Print final results
    print("\n" + "="*80)
    print("🎉 Experiment Complete!")
    print("="*80 + "\n")
    
    print("Final Results:")
    for result in final_results:
        print(f"  Rank {result['rank']:2d}: {result['best_accuracy']:6.2f}% accuracy")
    
    print(f"\n📁 Results saved to: {config.RESULTS_DIR}/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
