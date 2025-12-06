"""
Training and evaluation logic
"""
import os
import math
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from loss import compute_total_loss
from utils import smape_np, delog2


class Trainer:
    """Training manager for the dual-backbone model"""
    
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: str = "cuda"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.training.fp16)
        
        # Tracking
        self.best_smape = float("inf")
        self.patience = 0
        
    def _create_optimizer(self):
        """Create AdamW optimizer with weight decay"""
        no_decay = ["bias", "LayerNorm.weight"]
        named_params = list(self.model.named_parameters())
        grouped = [
            {
                "params": [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.training.weight_decay
            },
            {
                "params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        return torch.optim.AdamW(grouped, lr=self.config.training.lr)
    
    def _create_scheduler(self):
        """Create linear warmup scheduler"""
        num_training_steps = self.config.training.epochs * max(
            1, math.ceil(len(self.train_loader) / max(1, self.config.training.grad_accum))
        )
        num_warmup = int(num_training_steps * self.config.training.warmup_ratio)
        return get_linear_schedule_with_warmup(
            self.optimizer, num_warmup, num_training_steps
        )
    
    def train_epoch(self, epoch: int, y_va_log: np.ndarray) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        # Tracking
        metrics = {
            'loss': 0.0, 'reg': 0.0, 'cls': 0.0,
            'nce_clip': 0.0, 'nce_txt': 0.0, 'gate_reg': 0.0,
            'gate_img': 0.0, 'gate_txt': 0.0, 'gate_distil': 0.0
        }
        
        for step, batch in enumerate(self.train_loader, 1):
            batch = {
                k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) 
                for k, v in batch.items()
            }
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.config.training.fp16):
                pred_log2, class_logits, z_img, z_txt, z_d1, z_d2, gate_weights = self.model(
                    clip_input_ids=batch["clip_input_ids"],
                    clip_attention_mask=batch["clip_attention_mask"],
                    pixel_values=batch.get("pixel_values", None),
                    img_missing=batch["img_missing"],
                    distil_ids1=batch["distil_ids1"],
                    distil_att1=batch["distil_att1"],
                    distil_ids2=batch["distil_ids2"],
                    distil_att2=batch["distil_att2"],
                )
                
                # Compute loss
                loss, loss_components = compute_total_loss(
                    pred_log2, batch["target"], class_logits,
                    z_img, z_txt, z_d1, z_d2, gate_weights,
                    batch["img_missing"], batch.get("pixel_values", None),
                    loss_type=self.config.training.loss_type,
                    alpha_cls=self.config.training.alpha_cls,
                    alpha_clip_nce=self.config.training.alpha_clip_nce,
                    alpha_txt_nce=self.config.training.alpha_txt_nce,
                    alpha_gate_reg=self.config.training.alpha_gate_reg,
                    num_price_bins=self.config.model.num_price_bins,
                    huber_delta=self.config.training.huber_delta,
                    focal_gamma=self.config.training.focal_gamma,
                    tau=self.config.training.tau,
                    img_missing_policy=self.config.data.img_missing_policy
                )
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Update weights
            if step % self.config.training.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
            
            # Update metrics
            for key in ['loss', 'reg', 'cls', 'nce_clip', 'nce_txt', 'gate_reg']:
                if key == 'loss':
                    metrics[key] += loss_components['total']
                else:
                    metrics[key] += loss_components.get(key, 0.0)
            
            with torch.no_grad():
                metrics['gate_img'] += gate_weights[:, 0].mean().item()
                metrics['gate_txt'] += gate_weights[:, 1].mean().item()
                metrics['gate_distil'] += gate_weights[:, 2].mean().item()
            
            # Print progress
            if step % 200 == 0:
                self._print_progress(epoch, step, len(self.train_loader), metrics)
        
        # Validation
        val_smape = self.evaluate(y_va_log)
        
        return val_smape
    
    def _print_progress(self, epoch: int, step: int, total_steps: int, metrics: Dict):
        """Print training progress"""
        denom = step if step > 0 else 1
        print(f"Epoch {epoch} step {step}/{total_steps} "
              f"loss={metrics['loss']/denom:.4f} reg={metrics['reg']/denom:.4f} "
              f"cls={metrics['cls']/denom:.4f} nce_clip={metrics['nce_clip']/denom:.4f} "
              f"nce_txt={metrics['nce_txt']/denom:.4f} gate_reg={metrics['gate_reg']/denom:.4f}")
        print(f"  Gate weights (img/txt/distil): "
              f"{metrics['gate_img']/denom:.3f}/"
              f"{metrics['gate_txt']/denom:.3f}/"
              f"{metrics['gate_distil']/denom:.3f}")
    
    @torch.no_grad()
    def evaluate(self, y_va_log: np.ndarray) -> float:
        """Evaluate on validation set"""
        self.model.eval()
        preds_log = []
        
        for batch in self.val_loader:
            batch = {
                k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            
            pred_log2, *_ = self.model(
                clip_input_ids=batch["clip_input_ids"],
                clip_attention_mask=batch["clip_attention_mask"],
                pixel_values=batch.get("pixel_values", None),
                img_missing=batch["img_missing"],
                distil_ids1=batch["distil_ids1"],
                distil_att1=batch["distil_att1"],
                distil_ids2=batch["distil_ids2"],
                distil_att2=batch["distil_att2"],
            )
            preds_log.append(pred_log2.detach().float().cpu().numpy())
        
        if len(preds_log):
            preds_log = np.concatenate(preds_log, axis=0)
            va_preds = delog2(preds_log)
            va_true = delog2(y_va_log)
            smape = smape_np(va_true, va_preds)
        else:
            smape = float("inf")
        
        return smape
    
    def save_checkpoint(self, smape: float, path: str):
        """Save model checkpoint"""
        checkpoint = {
            "model_state": self.model.state_dict(),
            "clip_id": self.config.model.clip_id,
            "distil_id_used": self.model.distil_id_used,
            "config": {
                "ALPHA_CLIP_NCE": self.config.training.alpha_clip_nce,
                "ALPHA_TXT_NCE": self.config.training.alpha_txt_nce,
                "ALPHA_CLS": self.config.training.alpha_cls,
                "ALPHA_GATE_REG": self.config.training.alpha_gate_reg,
                "TAU": self.config.training.tau,
                "HUBER_DELTA": self.config.training.huber_delta,
                "FOCAL_GAMMA": self.config.training.focal_gamma,
                "MAX_LEN_CLIP": self.config.data.max_len_clip,
                "MAX_LEN_DISTIL": self.config.data.max_len_distil,
                "IMG_MISSING_POLICY": self.config.data.img_missing_policy,
                "NUM_PRICE_BINS": self.config.model.num_price_bins,
                "LOSS_TYPE": self.config.training.loss_type,
                "distil_proj_dim": self.config.model.distil_proj_dim,
            },
            "columns": {
                "text": self.config.data.text_col,
                "image": self.config.data.img_col,
                "price": self.config.data.price_col
            },
            "val_frac": self.config.data.val_frac,
            "best_smape": smape,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def train(self, y_va_log: np.ndarray):
        """Full training loop"""
        best_path = os.path.join(self.config.output_dir, "best_improved_model.pt")
        
        for epoch in range(1, self.config.training.epochs + 1):
            val_smape = self.train_epoch(epoch, y_va_log)
            
            print(f"Epoch {epoch}: VAL SMAPE = {val_smape:.3f}%")
            
            # Save best model
            if val_smape < self.best_smape - 1e-6:
                self.best_smape = val_smape
                self.patience = 0
                self.save_checkpoint(val_smape, best_path)
            else:
                self.patience += 1
                print(f"No improvement. Patience {self.patience}/{self.config.training.early_stop_rounds}")
                
                if self.patience >= self.config.training.early_stop_rounds:
                    print(" Early stopping triggered.")
                    break
        
        print(f"🏁 Best VAL SMAPE: {self.best_smape:.3f}% | Checkpoint: {best_path}")
        print(f"📈 Training complete! Model saved to: {self.config.output_dir}")
        
        return self.best_smape