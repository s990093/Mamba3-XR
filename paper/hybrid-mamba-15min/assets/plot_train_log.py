import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_train_log(csv_path, output_dir):
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hybrid Mamba 15min Training Log Visualization', fontsize=16)
    
    # Plot 1: Loss and CE Loss
    axes[0, 0].plot(df['step'], df['loss'], label='Total Loss', color='blue')
    axes[0, 0].plot(df['step'], df['ce_loss'], label='CE Loss', alpha=0.6, color='red')
    axes[0, 0].set_title('Loss vs Step')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Learning Rate
    axes[0, 1].plot(df['step'], df['lr'], color='orange')
    axes[0, 1].set_title('Learning Rate vs Step')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('LR')
    axes[0, 1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 3: Grad Norm
    axes[1, 0].plot(df['step'], df['grad_norm'], color='green')
    axes[1, 0].set_title('Grad Norm vs Step')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Grad Norm')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 4: Contributions (lb_contrib, z_contrib)
    axes[1, 1].plot(df['step'], df['lb_contrib'], label='LB Contrib')
    axes[1, 1].plot(df['step'], df['z_contrib'], label='Z Contrib')
    axes[1, 1].set_title('Auxiliary Contributions vs Step')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Contribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'train_metrics_visualization.png')
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, 'train_log.csv')
    plots_dir = os.path.join(current_dir, 'plots')
    plot_train_log(csv_file, plots_dir)
