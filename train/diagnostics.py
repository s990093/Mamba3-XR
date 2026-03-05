import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from model import Mamba3Block

class MambaDiagnostics:
    def __init__(self, log_dir="results/diagnostics"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_file = os.path.join(log_dir, "training_log.csv")
        
        # Initialize CSV header
        with open(self.csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Top-1', 'Val Top-5', 'Val F1'])

        self.history = {
            "epoch": [],
            "layer_stats": {}, 
            "eigen_A": {},     
            "mimo_ranks": {},  
            "delta_stats": {}, 
            "delta_heatmap": {}, 
            "layer_activations": {}, 
            "loss": [],
            "accuracy": [],
            "val_loss": [],    
            "val_top1": [], 
            "val_top5": [],
            "val_f1": []       
        }
        
    def log_metrics(self, epoch, loss, accuracy, val_loss=None, val_top1=None, val_top5=None, val_f1=None):
        self.history["loss"].append(loss)
        self.history["accuracy"].append(accuracy)
        if val_loss is not None: self.history["val_loss"].append(val_loss)
        if val_top1 is not None: self.history["val_top1"].append(val_top1)
        if val_top5 is not None: self.history["val_top5"].append(val_top5)
        if val_f1 is not None: self.history["val_f1"].append(val_f1)
        
        # Write to CSV
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss, accuracy, val_loss, val_top1, val_top5, val_f1])
        
        msg = f"[Diagnostics] Epoch {epoch} | Train Loss: {loss:.4f} Acc: {accuracy:.2f}%"
        if val_loss is not None:
             msg += f" | Val Loss: {val_loss:.4f} Top-1: {val_top1:.2f}% Top-5: {val_top5:.2f}% F1: {val_f1:.4f}"
        print(msg)

    def log_gradients(self, model, epoch):
        self.history["epoch"].append(epoch)
        print(f"[Diagnostics] Logging Gradients for Epoch {epoch}...")
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            is_interesting = "A_log" in name or "x_up_proj" in name or "in_proj" in name
            if not is_interesting:
                continue
            g_mean = param.grad.mean().item()
            g_std = param.grad.std().item()
            g_norm = param.grad.norm().item()
            w_norm = param.norm().item()
            snr = abs(g_mean) / (g_std + 1e-9)
            update_ratio = (1e-3 * g_norm) / (w_norm + 1e-9) 
            key = f"{name}"
            if key not in self.history["layer_stats"]:
                self.history["layer_stats"][key] = {"snr": [], "update_ratio": [], "grad_norm": []}
            self.history["layer_stats"][key]["snr"].append(snr)
            self.history["layer_stats"][key]["update_ratio"].append(update_ratio)
            self.history["layer_stats"][key]["grad_norm"].append(g_norm)

    def log_eigenvalues(self, model, epoch):
        print(f"[Diagnostics] Logging Matrix A Eigenvalues for Epoch {epoch}...")
        all_taus = []
        for name, param in model.named_parameters():
            if "A_log" in name:
                A = -torch.exp(param.detach())
                taus = -1.0 / (A.cpu().numpy() + 1e-9)
                all_taus.extend(taus.flatten())
        self.history["eigen_A"][epoch] = all_taus

    def log_mimo_rank(self, model, epoch):
        print(f"[Diagnostics] Logging MIMO Ranks for Epoch {epoch}...")
        for name, module in model.named_modules():
            if hasattr(module, "x_up_proj") and isinstance(module.x_up_proj, nn.Linear):
                W = module.x_up_proj.weight.detach().cpu().float()
                try:
                    U, S, V = torch.linalg.svd(W)
                    S_norm = S / S.sum()
                    entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-9))
                    eff_rank = torch.exp(entropy).item()
                    
                    if name not in self.history["mimo_ranks"]:
                        self.history["mimo_ranks"][name] = []
                    self.history["mimo_ranks"][name].append(eff_rank)
                except Exception as e:
                    print(f"SVD failed for {name}: {e}")
                    if name not in self.history["mimo_ranks"]:
                        self.history["mimo_ranks"][name] = []
                    self.history["mimo_ranks"][name].append(0.0)

    def log_selectivity_audit(self, model, inputs, epoch, device):
        print(f"[Diagnostics] Logging Selectivity (Exp E) for Epoch {epoch}...")
        captured_dt = {}
        hooks = []
        
        for name, module in model.named_modules():
             if hasattr(module, "config") and hasattr(module, "in_proj"):
                 H = module.config.n_heads
                 def hook_fn(m, inp, out, layer_name=name, heads=H):
                     dt_part = out[:, :, -2*heads : -heads]
                     captured_dt[layer_name] = dt_part.detach().cpu()
                 hooks.append(module.in_proj.register_forward_hook(hook_fn))
        
        with torch.no_grad():
            model(inputs)
        for h in hooks: h.remove()
        
        for name, dt_raw in captured_dt.items():
            dt_act = F.softplus(dt_raw)
            mean_val = dt_act.mean().item()
            std_val = dt_act.std().item()
            cv = std_val / (mean_val + 1e-9)
            
            if name not in self.history["delta_stats"]:
                 self.history["delta_stats"][name] = {"cv": [], "mean": []}
            self.history["delta_stats"][name]["cv"].append(cv)
            self.history["delta_stats"][name]["mean"].append(mean_val)
            
            if name not in self.history["delta_heatmap"]:
                self.history["delta_heatmap"][name] = []
            sample = dt_act[0].permute(1, 0).numpy()
            self.history["delta_heatmap"][name] = sample

    def log_advanced_analysis(self, model, dataloader, epoch, device):
        print(f"[Diagnostics] Logging Advanced Analysis (Exp F & G) for Epoch {epoch}...")
        layer_outputs = {}
        hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple): out = output[0]
                else: out = output
                out = out.detach().cpu()
                if name not in layer_outputs: layer_outputs[name] = {"pooled": [], "full": []}
                layer_outputs[name]["pooled"].append(out.mean(dim=1)) 
                if len(layer_outputs[name]["full"]) < 5: layer_outputs[name]["full"].append(out)
            return hook

        for name, module in model.named_modules():
            if hasattr(module, "y_down_proj"):
                h = module.register_forward_hook(get_activation(name))
                hooks.append(h)

        count = 0
        labels_list = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                model(inputs)
                labels_list.append(labels.cpu())
                count += inputs.size(0)
                if count >= 200: break
        
        for h in hooks: h.remove()
        
        all_labels = torch.cat(labels_list)[:count].numpy()
        if epoch not in self.history["layer_activations"]:
            self.history["layer_activations"][epoch] = {}

        for name, data in layer_outputs.items():
            pooled_features = torch.cat(data["pooled"], dim=0)[:count].numpy()
            
            if pooled_features.shape[0] > 5:
                if np.isnan(pooled_features).any() or np.isinf(pooled_features).any():
                    print(f"[Diagnostics] Warning: Layer {name} has NaN/Inf features. Skipping PCA.")
                    continue
                try:
                    pca = PCA(n_components=2)
                    projected = pca.fit_transform(pooled_features)
                    self.history["layer_activations"][epoch][name] = {
                        "pca": projected, "labels": all_labels, "fft": None
                    }
                except Exception as e:
                     print(f"[Diagnostics] PCA failed for {name}: {e}")
            
            full_features = torch.cat(data["full"], dim=0)
            B_fft = full_features.shape[0]
            if B_fft > 0:
                feat_map = full_features.view(B_fft, 8, 8, -1).numpy()
                fft2 = np.fft.fft2(feat_map, axes=(1, 2))
                fft2_shift = np.fft.fftshift(fft2, axes=(1, 2))
                mag = np.abs(fft2_shift)
                log_mag = np.log(mag + 1e-9)
                avg_spec = log_mag.mean(axis=(0, 3)) 
                self.history["layer_activations"][epoch][name]["fft"] = avg_spec

    def save_history(self):
        torch.save(self.history, os.path.join(self.log_dir, "diagnostics_history.pt"))
        print(f"[Diagnostics] Saved history to {self.log_dir}/diagnostics_history.pt")

    def log_state_health(self, model, inputs, epoch):
        """
        Monitor hidden state h (L2 Norm) and variance.
        """
        print(f"[Diagnostics] Logging State Health (Norms & Var) for Epoch {epoch}...")
        
        hooks = []
        for name, module in model.named_modules():
            # Hook into Mamba3Block output to check signal health
            if isinstance(module, Mamba3Block):
                def get_stats(m, i, o, n=name):
                    # o shape: (B, L, D)
                    mean = o.mean().item()
                    var = o.var().item()
                    l2 = o.norm(p=2, dim=-1).mean().item() # Token-wise L2 norm
                    
                    if n not in self.history["layer_activations"]:
                        self.history["layer_activations"][n] = {"mean": [], "var": [], "l2": []}
                    else:
                         if "mean" not in self.history["layer_activations"][n]:
                             self.history["layer_activations"][n]["mean"] = []
                             self.history["layer_activations"][n]["var"] = []
                             self.history["layer_activations"][n]["l2"] = []

                    self.history["layer_activations"][n]["mean"].append(mean)
                    self.history["layer_activations"][n]["var"].append(var)
                    self.history["layer_activations"][n]["l2"].append(l2)
                
                hooks.append(module.register_forward_hook(get_stats))

        # Run inference
        model.eval()
        with torch.no_grad():
            model(inputs)
        model.train()
            
        for h in hooks: h.remove()

def plot_all_diagnostics(log_dir):
    history_path = os.path.join(log_dir, "diagnostics_history.pt")
    if not os.path.exists(history_path):
        print(f"No history found at {history_path}")
        return

    print("Generating comprehensive plots...")
    history = torch.load(history_path)
    os.makedirs(log_dir, exist_ok=True)
    
    epochs = range(len(history["loss"]))
    
    # 1. Training Dynamics (Loss, Acc, F1)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["loss"], label='Train Loss')
    if "val_loss" in history and len(history["val_loss"]) > 0:
        plt.plot(epochs, history["val_loss"], label='Val Loss', linestyle='--')
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["accuracy"], label='Train Acc')
    if "val_accuracy" in history and len(history["val_accuracy"]) > 0:
        plt.plot(epochs, history["val_accuracy"], label='Val Acc', linestyle='--')
    plt.title("Accuracy Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    if "val_f1" in history and len(history["val_f1"]) > 0:
        plt.plot(epochs, history["val_f1"], label='Val F1', color='green')
        plt.title("Validation F1 Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No F1 Data', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "dist_metrics.png"))
    plt.close()

    # 2. Layer Health (Gradients)
    if history["layer_stats"]:
        plt.figure(figsize=(15, 10))
        keys = list(history["layer_stats"].keys())
        # Filter for a few representative layers to avoid clutter
        # e.g., first, middle, last
        if len(keys) > 6:
            indices = np.linspace(0, len(keys)-1, 6).astype(int)
            keys = [keys[i] for i in indices]
        
        for i, key in enumerate(keys):
            stats = history["layer_stats"][key]
            plt.subplot(2, 3, i+1)
            plt.plot(stats["grad_norm"], label="Grad Norm")
            plt.plot(stats["snr"], label="SNR", linestyle='--')
            plt.title(f"Layer: {key}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "dist_gradients.png"))
        plt.close()

    # 3. Mamba Internals (MIMO Rank & Eigenvalues)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, ranks in history["mimo_ranks"].items():
        if "layers.0." in name or "layers.3." in name or "layers.5." in name: # Plot a few
            plt.plot(ranks, label=name)
    plt.title("MIMO Effective Rank")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Plot Eigenvalues of A for last epoch
    if history["eigen_A"]:
        last_epoch = max(history["eigen_A"].keys())
        taus = history["eigen_A"][last_epoch]
        plt.hist(taus, bins=50, alpha=0.7)
        plt.title(f"Time Constants (tau) Distribution (Epoch {last_epoch})")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "dist_mamba_internals.png"))
    plt.close()
    
    # 4. State & Delta Health
    if "layer_activations" in history:
        # Check deep diagnostics keys
        first_layer_key = next(iter(history["layer_activations"]))
        if isinstance(history["layer_activations"][first_layer_key], dict) and "l2" in history["layer_activations"][first_layer_key]:
             plt.figure(figsize=(15, 5))
             
             plt.subplot(1, 2, 1)
             for name, stats in history["layer_activations"].items():
                 if "l2" in stats:
                     plt.plot(stats["l2"], label=name)
             plt.title("State L2 Norms (Stability)")
             plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
             plt.yscale('log')
             plt.grid(True, alpha=0.3)
             
             plt.subplot(1, 2, 2)
             for name, stats in history["layer_activations"].items():
                 if "var" in stats:
                     plt.plot(stats["var"], label=name)
             plt.title("State Variance (Collapse Check)")
             plt.yscale('log')
             plt.grid(True, alpha=0.3)

             plt.tight_layout()
             plt.savefig(os.path.join(log_dir, "dist_state_health.png"))
             plt.close()

    print(f"Plots saved to {log_dir}")
