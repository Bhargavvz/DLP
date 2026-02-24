"""
DeepFinDLP - Training Pipeline
H200-optimized training loop with mixed precision, LR scheduling, and checkpointing.
"""
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=10, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """H200-optimized training pipeline for DeepFinDLP models."""

    def __init__(self, model, model_name, class_weights=None,
                 learning_rate=None, weight_decay=None, epochs=None,
                 patience=None, device=None):
        self.model = model
        self.model_name = model_name
        self.device = device or config.DEVICE
        self.epochs = epochs or config.EPOCHS
        self.model.to(self.device)

        # Loss function with class weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        lr = learning_rate or config.LEARNING_RATE
        wd = weight_decay or config.WEIGHT_DECAY
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd
        )

        # LR Scheduler
        if config.SCHEDULER_TYPE == "cosine_warm_restarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=config.COSINE_T0,
                T_mult=config.COSINE_T_MULT,
                eta_min=config.COSINE_ETA_MIN,
            )
        elif config.SCHEDULER_TYPE == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs,
                eta_min=config.COSINE_ETA_MIN,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.5
            )

        # Mixed precision
        self.use_amp = config.USE_AMP and torch.cuda.is_available()
        self.amp_dtype = config.AMP_DTYPE
        self.scaler = GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

        # Compile model for speed
        if config.USE_COMPILE and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
                print(f"  [✓] torch.compile enabled for {model_name}")
            except Exception as e:
                print(f"  [!] torch.compile failed: {e}")

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=patience or config.PATIENCE,
            min_delta=config.MIN_DELTA, mode="max"
        )

        # Training history
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_f1": [], "val_f1": [],
            "lr": [],
        }

        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_state = None

    def _train_epoch(self, train_loader):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"  Train", leave=False, ncols=100)
        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Mixed precision forward
            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp and self.amp_dtype == torch.float16:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.GRADIENT_CLIP_MAX_NORM
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.GRADIENT_CLIP_MAX_NORM
                )
                self.optimizer.step()

            # Metrics
            total_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

        avg_loss = total_loss / total
        accuracy = correct / total
        f1 = self._compute_f1(all_labels, all_preds)

        return avg_loss, accuracy, f1

    @torch.no_grad()
    def _validate(self, val_loader):
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for features, labels in val_loader:
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / total
        accuracy = correct / total
        f1 = self._compute_f1(all_labels, all_preds)

        return avg_loss, accuracy, f1

    def _compute_f1(self, y_true, y_pred):
        """Compute weighted F1-score."""
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average="weighted", zero_division=0)

    def train(self, train_loader, val_loader):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Training: {self.model_name}")
        print(f"{'='*60}")

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {total_params:,}")
        print(f"  Device: {self.device}")
        print(f"  AMP: {self.use_amp} (dtype: {self.amp_dtype})")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']}")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_acc, train_f1 = self._train_epoch(train_loader)

            # Validate
            val_loss, val_acc, val_f1 = self._validate(val_loader)

            # LR Scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["train_f1"].append(train_f1)
            self.history["val_f1"].append(val_f1)
            self.history["lr"].append(current_lr)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                # Save checkpoint
                ckpt_path = os.path.join(
                    config.CHECKPOINTS_DIR, f"{self.model_name}_best.pt"
                )
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                }, ckpt_path)

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print(f"  Epoch {epoch:3d}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                  f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

            # Early stopping
            if self.early_stopping.step(val_acc):
                print(f"\n  Early stopping at epoch {epoch} (patience={self.early_stopping.patience})")
                break

        total_time = time.time() - start_time
        print(f"\n  Training complete! Total time: {total_time:.1f}s")
        print(f"  Best validation accuracy: {self.best_val_acc:.4f}")

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # Save training history
        history_path = os.path.join(config.LOGS_DIR, f"{self.model_name}_history.json")
        serializable_history = {
            k: [float(v) for v in vals] for k, vals in self.history.items()
        }
        with open(history_path, "w") as f:
            json.dump(serializable_history, f, indent=2)

        return self.history, total_time

    @torch.no_grad()
    def predict(self, test_loader):
        """Generate predictions on test set."""
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        all_features = []

        for features, labels in test_loader:
            features = features.to(self.device, non_blocking=True)

            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(features)

            probs = torch.softmax(outputs.float(), dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

            # Extract features for t-SNE (from a subset)
            if len(all_features) < 5:  # Only first few batches
                try:
                    with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                        feat = self.model.get_features(features)
                    all_features.append(feat.float().cpu().numpy())
                except Exception:
                    pass

        results = {
            "y_pred": np.array(all_preds),
            "y_proba": np.concatenate(all_probs, axis=0),
            "y_true": np.array(all_labels),
        }

        if all_features:
            results["features"] = np.concatenate(all_features, axis=0)

        return results
