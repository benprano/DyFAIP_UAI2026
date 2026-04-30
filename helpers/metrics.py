from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, roc_curve, auc
)
from tqdm import tqdm


class TrainerMetrics:
    """Modular metrics computation for training pipeline."""

    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    @staticmethod
    def adjusted_r2(actual: np.ndarray, predicted: np.ndarray,
                    n_samples: np.int64, n_features: np.int64) -> float:
        """Compute adjusted R² score."""
        r2 = r2_score(actual, predicted)
        return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features)

    def compute_regression_metrics(self, targets: List[np.ndarray],
                                   predicted: List[np.ndarray],
                                   rescale_params: Dict[str, float]) -> List[List[float]]:
        """Compute regression metrics (RMSE, MAE, R², Adjusted R²)."""
        scores = []
        target_max = rescale_params['data_targets_max']
        target_min = rescale_params['data_targets_min']

        for y_true, y_pred in zip(targets, predicted):
            # Rescale to original range for meaningful R² calculation
            y_true_rescaled = y_true * (target_max - target_min) + target_min
            y_pred_rescaled = y_pred * (target_max - target_min) + target_min
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true_rescaled, y_pred_rescaled)
            adj_r2 = self.adjusted_r2(
                y_true_rescaled, y_pred_rescaled,
                y_true.shape[0], self.input_dim
            )

            scores.append([mse, rmse, mae, r2, adj_r2])
        return scores

    def compute_imputation_metrics(self, real: List[np.ndarray],
                                   imputed: List[np.ndarray],
                                   rescale_params: Dict[str, float]) -> List[List[float]]:
        """Compute imputation quality metrics."""
        return self.compute_regression_metrics(real, imputed, rescale_params)

    @staticmethod
    def compute_binary_metrics(targets: List[np.ndarray],
                               predicted: List[np.ndarray]) -> List[List[float]]:
        """Compute binary classification metrics (ROC-AUC, PR-AUC)."""
        scores = []
        for y_true, y_pred in zip(targets, predicted):
            fpr, tpr, thresholds = roc_curve(y_pred, y_true)
            auc_score = auc(fpr, tpr)
            pr_score = average_precision_score(y_pred, y_true)
            scores.append([np.round(np.mean(auc_score), 4),
                           np.round(np.mean(pr_score), 4)])
        return scores

    @staticmethod
    def find_best_threshold(predictions: np.ndarray,
                            y_true: np.ndarray) -> Tuple[float, float]:
        """Find optimal threshold for maximizing F1-score."""
        delta, tmp = 0, [0, 0, 0]  # idx, cur, max
        for tmp[0] in tqdm(np.arange(0.1, 1.01, 0.01)):
            tmp[1] = f1_score(y_true, np.array(predictions) > tmp[0])
            if tmp[1] > tmp[2]:
                delta = tmp[0]
                tmp[2] = tmp[1]
        print('best threshold is {:.2f} with F1 score: {:.4f}'.format(delta, tmp[2]))
        return delta, tmp[2]


class EarlyStopping:
    """
    Flexible Early Stopping that works for both classification and regression.

    Saves best model if EITHER primary metric OR imputation metric improves independently.

    For Classification:
    - Primary metric: Accuracy (maximize)
    - Imputation metric: MAE (minimize)

    For Regression:
    - Primary metric: MSE (minimize)
    - Imputation metric: MAE (minimize)

    Strategy:
    =========
    Save best model if:
    - Primary metric improves (task-specific), OR
    - Imputation MAE improves

    This balances both tasks equally without forcing one to sacrifice the other.
    """

    def __init__(self,
                 task: None,
                 save_path: str = 'best_model.pt',
                 patience: int = 50,
                 primary_threshold: float = 0.0001,
                 imp_threshold: float = 0.00001,
                 logger: Optional[Any] = None):
        """
        Initialize Generic Early Stopping.

        Args:
            task: 'classification' or 'regression'
            save_path: Path to save best model
            patience: Epochs without improvement before stopping
            primary_threshold: Minimum improvement for primary metric
            imp_threshold: Minimum imputation improvement
            logger: Logger instance
        """

        self.task = task
        self.save_path = save_path
        self.patience = patience
        self.primary_threshold = primary_threshold
        self.imp_threshold = imp_threshold
        self.logger = logger

        # Track the best values for each goal independently
        if task:
            # For classification: maximize accuracy
            self.best_primary = -np.inf
            self.primary_mode = 'max'  # Higher is better
            self.primary_name = 'Accuracy'
        else:
            # For regression: minimize MSE
            self.best_primary = np.inf
            self.primary_mode = 'min'  # Lower is better
            self.primary_name = 'MSE'

        self.best_imp = np.inf  # Always minimize imputation MAE
        self.best_epoch_primary = -1
        self.best_epoch_imp = -1

        # Overall tracking
        self.counter = 0
        self.best_info = {}
        self.improvement_history = []

    @staticmethod
    def _safe(text: str) -> str:
        """Return text encoded to ASCII-safe string without throwing."""
        return text.encode("ascii", "replace").decode()

    def _log(self, message: str) -> None:
        """Log safely (never crashes)."""
        safe_message = self._safe(message)
        if self.logger:
            self.logger.info(safe_message)
        else:
            print(safe_message)

    def _check_primary_improvement(self, primary_metric: float) -> bool:
        """
        Check if primary metric improved based on task type.

        Args:
            primary_metric: Classification accuracy or Regression MSE

        Returns:
            True if metric improved
        """
        if self.primary_mode == 'max':
            # Classification: maximize accuracy
            return primary_metric > self.best_primary + self.primary_threshold
        else:
            # Regression: minimize MSE
            return primary_metric < self.best_primary - self.primary_threshold

    def _get_primary_improvement_str(self, primary_metric: float) -> str:
        """Format improvement message based on task type."""
        if self.primary_mode == 'max':
            improvement = (primary_metric - self.best_primary) * 100
            return f"{self.primary_name}: {self.best_primary:.4f} → {primary_metric:.4f} (+{improvement:.2f}%)"
        else:
            improvement = (self.best_primary - primary_metric)
            return f"{self.primary_name}: {self.best_primary:.6f} → {primary_metric:.6f} (-{improvement:.6f})"

    def __call__(self,
                 primary_metric: float,
                 imputation_metric: float,
                 model: torch.nn.Module,
                 epoch: Optional[int] = None,
                 extra_metrics: Optional[Dict[str, float]] = None) -> bool:
        """
        Check if either metric improved and save model accordingly.

        Args:
            primary_metric: Classification accuracy OR Regression MSE
            imputation_metric: Imputation MAE (always to minimize)
            model: Model to potentially save
            epoch: Current epoch number
            extra_metrics: Additional metrics to track

        Returns:
            early_stop: Whether to trigger early stopping
        """
        epoch_num = epoch + 1 if epoch is not None else 0
        improvement_detected = False
        reasons = []

        # ================================================================
        # OBJECTIVE 1: Primary Task Metric (Classification or Regression)
        # ================================================================
        if self._check_primary_improvement(primary_metric):
            improvement_detected = True
            reasons.append(self._get_primary_improvement_str(primary_metric))
            self.best_primary = primary_metric
            self.best_epoch_primary = epoch_num

        # ================================================================
        # OBJECTIVE 2: Minimize Imputation MAE
        # ================================================================
        if imputation_metric < self.best_imp - self.imp_threshold:
            improvement_detected = True
            imp_improvement = self.best_imp - imputation_metric
            reasons.append(f"Imputation MAE: {self.best_imp:.6f} → {imputation_metric:.6f} (-{imp_improvement:.6f})")
            self.best_imp = imputation_metric
            self.best_epoch_imp = epoch_num

        # ================================================================
        # DECISION: Save or Increment Counter
        # ================================================================
        if improvement_detected:
            self.counter = 0

            # Save model
            torch.save(model.state_dict(), self.save_path)

            # Log improvement
            reason_str = " | ".join(reasons)
            self._log(f"[OK] Epoch {epoch_num}: Validation improved - saved best model weights")
            self._log(f"  {reason_str}")

            # Store best info
            self.best_info = {
                "epoch": epoch_num,
                "task": self.task,
                f"best_{self.primary_name.lower()}": round(primary_metric, 6),
                f"best_{self.primary_name.lower()}_epoch": self.best_epoch_primary,
                "best_imputation_mae": round(imputation_metric, 6),
                "best_imputation_mae_epoch": self.best_epoch_imp,
                "saved_at": datetime.now().isoformat(),
                "reason": reason_str
            }

            if extra_metrics:
                for key, value in extra_metrics.items():
                    self.best_info[f"extra_{key}"] = round(value, 6) if isinstance(value, float) else value

            self.improvement_history.append({
                "epoch": epoch_num,
                "primary_metric": primary_metric,
                "imputation_metric": imputation_metric,
                "reason": reason_str
            })

        else:
            self.counter += 1

            if self.primary_mode == 'max':
                primary_gap = (self.best_primary - primary_metric) * 100
                primary_gap_str = f"{primary_gap:.2f}%"
            else:
                primary_gap = (primary_metric - self.best_primary)
                primary_gap_str = f"{primary_gap:.6f}"

            imp_gap = (imputation_metric - self.best_imp)

            self._log(
                f"[X] Epoch {epoch_num}: No improvement | "
                f"{self.primary_name} gap: {primary_gap_str} (best: {self.best_primary:.4f}), "
                f"Imputation gap: {imp_gap:.6f} (best: {self.best_imp:.6f}) | "
                f"Counter: {self.counter}/{self.patience}"
            )

        # Check early stopping
        early_stop = self.counter >= self.patience
        if early_stop:
            self._log(f"[EARLY STOP] Best model achieved at Epoch {epoch_num}")
            self._log(f"  Best {self.primary_name}: {self.best_primary:.6f} (Epoch {self.best_epoch_primary})")
            self._log(f"  Best Imputation MAE: {self.best_imp:.6f} (Epoch {self.best_epoch_imp})")

        return early_stop



