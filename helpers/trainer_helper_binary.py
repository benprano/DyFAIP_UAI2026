
import csv
import gc
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, mean_absolute_error,
    mean_squared_error, f1_score
)
from tqdm import tqdm
from helpers.metrics import TrainerMetrics, EarlyStopping


class ModelTrainer:
    """Main training class with improved organization and error handling."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the trainer with a configuration dictionary."""
        # Core model parameters
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.seq_length = config['seq_length']
        self.output_dim = config['output_dim']
        self.num_steps = config['num_steps']

        # Training parameters
        self.num_epochs = config['num_epochs']
        self.patience = config.get('patience', 50)
        self.device = config['device']
        self.num_layers = config['num_layers']
        self.is_classification = config.get('is_classification', True)

        # Model components
        self.optimizer = config['optimizer']
        self.scheduler = config.get('scheduler')
        self.loss_function = config['loss_function']
        self.custom_loss = config['custom_loss']
        self.data_sampler = config['data_sampler']

        # Initialize metrics calculator
        self.metrics = TrainerMetrics(self.input_dim)
        print(f"[TRAINER] Initialized successfully")
        print(f"  └─ Task type: {'Classification' if self.is_classification else 'Regression'}")
        print(f"  └─ Device: {self.device}")
        print(f"  └─ Max epochs: {self.num_epochs}, Patience: {self.patience}")

    def _move_to_device(self, *tensors: torch.Tensor) -> List[torch.Tensor]:
        """Move tensors to a device and convert to float32."""
        return [tensor.to(torch.float32).to(self.device) for tensor in tensors]

    def _sample_missing_data(self, temporal_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample missing data points using the data sampler."""
        return self.data_sampler.mark_data_as_missing(temporal_features)

    @staticmethod
    def _calculate_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> int:
        """Calculate accuracy for classification tasks."""
        predictions = predictions.sigmoid()
        pred_binary = torch.round(predictions.squeeze())
        return torch.sum(pred_binary == labels.squeeze()).item()

    def _forward_pass(self, model: torch.nn.Module,
                      temporal_features: torch.Tensor, timestamp: torch.Tensor,
                      last_data: torch.Tensor, data_freqs: torch.Tensor,
                      labels: torch.Tensor, training: bool = True) -> Tuple:
        """Unified forward pass for training and validation."""
        # Move all tensors to a device
        tensors = self._move_to_device(
            temporal_features, timestamp, last_data, data_freqs, labels
        )
        temporal_features, timestamp, last_data, data_freqs, labels = tensors

        # Handle missing data
        sampled_data, data_with_missing, indices = self._sample_missing_data(temporal_features)
        mask = ~torch.isnan(data_with_missing)
        # print("data_with_missing", data_with_missing.shape)
        # Model forward pass
        if training:
            outputs, _, _, _, _, _,_,_,_,imputed_inputs = model(
                data_with_missing, timestamp, last_data, data_freqs
            )
        else:
            outputs, _, _, _, _, _,_,_,_, imputed_inputs = model(
                data_with_missing, timestamp, last_data, data_freqs
            )

        # Process imputation results
        sampled_imputed_x = imputed_inputs[indices]
        sampled_freqs = self.seq_length - data_freqs[indices]

        return outputs, sampled_data, sampled_imputed_x, sampled_freqs, labels

    def _compute_loss_and_metrics(self, outputs: torch.Tensor, sampled_data: torch.Tensor,
                                  sampled_imputed_x: torch.Tensor, sampled_freqs: torch.Tensor,
                                  labels: torch.Tensor) -> Tuple:
        """Compute loss and task-specific metrics."""
        if self.is_classification:
            loss_imp, total_loss = self.custom_loss(
                sampled_data, sampled_imputed_x,
                sampled_freqs, outputs.sigmoid(), labels, self.loss_function
            )
            accuracy = self._calculate_accuracy(outputs, labels)
            return loss_imp, total_loss, accuracy
        else:
            loss_imp,  total_loss = self.custom_loss(
                 sampled_data, sampled_imputed_x,
                sampled_freqs, outputs, labels, self.loss_function
            )
            return loss_imp, total_loss, None

    def train_epoch(self, model: torch.nn.Module,
                    train_loader: torch.utils.data.DataLoader, current_epoch: int) -> Tuple:
        """Execute one training epoch with memory optimization."""
        model.train()
        running_loss = running_corrects = mae_train = diff_loss = 0.0

        # Initialize mse_tracker for regression (always initialize to avoid reference errors)
        mse_tracker = {'sum_squared_error': 0.0, 'count': 0} if not self.is_classification else None
        # Progress bar for batch iteration
        progress_bar = tqdm(
            train_loader,
            desc=f"Training Epoch {current_epoch + 1}",
            leave=False
        )

        for batch_idx, batch_data in enumerate(progress_bar):
            temporal_features, timestamp, last_data, data_freqs, labels = batch_data

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            forward_results = self._forward_pass(
                model, temporal_features, timestamp, last_data, data_freqs, labels, training=True
            )
            outputs, sampled_data, sampled_imputed_x, sampled_freqs, labels = forward_results

            # Compute loss and metrics
            loss_imp, total_loss, accuracy = self._compute_loss_and_metrics(
                outputs, sampled_data, sampled_imputed_x, sampled_freqs, labels
            )

            # Backward pass
            total_loss.backward()
            self.optimizer.step()

            # Per-batch scheduler step
            if self.scheduler is not None:
                current_step = current_epoch + batch_idx / len(train_loader)
                self.scheduler.step(current_step)

            # Accumulate scalar metrics only
            running_loss += total_loss.item()
            mae_train += loss_imp.item()

            if self.is_classification:
                running_corrects += accuracy
            else:
                # For regression: compute batch MSE without storing predictions
                with torch.no_grad():
                    outputs_np = outputs.detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy()
                    batch_squared_error = np.sum((outputs_np - labels_np) ** 2)
                    mse_tracker['sum_squared_error'] += batch_squared_error
                    mse_tracker['count'] += len(labels_np)
                    del outputs_np, labels_np

                # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'imp': f"{loss_imp.item():.4f}"
            })

            # Explicitly free batch tensors
            del (temporal_features, timestamp, last_data, data_freqs, labels,
                 outputs, sampled_data, sampled_imputed_x,
                 sampled_freqs, forward_results, loss_imp,total_loss)

            # Periodic memory cleanup (every 10 batches)
            if batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Calculate epoch metrics
        num_batches = len(train_loader)
        epoch_loss = running_loss / num_batches
        epoch_mae = mae_train / num_batches

        if self.is_classification:
            epoch_accuracy = running_corrects / len(train_loader.dataset)
            return epoch_mae, epoch_loss, epoch_accuracy
        else:
            mse = mse_tracker['sum_squared_error'] / mse_tracker['count']
            return epoch_mae, epoch_loss,  mse

    def validate_epoch(self, model: torch.nn.Module,
                       val_loader: torch.utils.data.DataLoader) -> Tuple:
        """Execute one validation epoch with memory optimization."""
        model.eval()
        running_loss = running_corrects = mae_val = diff_loss = 0.0

        # Store only targets and outputs for final metrics calculation
        all_targets, all_outputs = [], []

        with torch.no_grad():
            progress_bar = tqdm(
                val_loader,
                desc="Validation",
                leave=False
            )
            for batch_idx, batch_data in enumerate(progress_bar):
                temporal_features, timestamp, last_data, data_freqs, labels = batch_data

                # Forward pass
                forward_results = self._forward_pass(
                    model, temporal_features, timestamp, last_data, data_freqs, labels, training=False
                )
                outputs, sampled_data, sampled_imputed_x, sampled_freqs, labels = forward_results

                # Compute loss and metrics
                loss_imp, total_loss, accuracy = self._compute_loss_and_metrics(
                    outputs, sampled_data, sampled_imputed_x, sampled_freqs, labels
                )

                # Accumulate scalar metrics
                running_loss += total_loss.item()
                mae_val += loss_imp.item()

                if accuracy is not None:
                    running_corrects += accuracy

                # Store predictions (needed for final metrics)
                all_targets.append(labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'imp': f"{loss_imp.item():.4f}"
                })

                # Free tensors
                del (temporal_features, timestamp, last_data, data_freqs, labels,
                     outputs, sampled_data, sampled_imputed_x,
                     sampled_freqs, forward_results, loss_imp, total_loss)

                if batch_idx % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Calculate epoch metrics
        num_batches = len(val_loader)
        epoch_loss = running_loss / num_batches
        epoch_mae = mae_val / num_batches

        targets_array = np.vstack(all_targets)
        outputs_array = np.vstack(all_outputs)

        if self.is_classification:
            epoch_accuracy = running_corrects / len(val_loader.dataset)
            return epoch_mae, epoch_loss, epoch_accuracy, targets_array, outputs_array
        else:
            mse = mean_squared_error(targets_array, outputs_array)
            mae = mean_absolute_error(targets_array, outputs_array)
            return epoch_mae, epoch_loss, mse, mae, targets_array, outputs_array

    def evaluate_model(self, model_class: type, model_path: str,
                       test_loader: torch.utils.data.DataLoader) -> Tuple:
        """Evaluate the trained model on test data with memory optimization."""
        model = model_class(
            self.input_dim, self.hidden_dim, self.seq_length,
            self.num_steps, self.num_layers, self.output_dim, self.device
        ).to(self.device)

        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        print(f"[EVAL] Evaluating on {len(test_loader)} batches...")
        all_targets, all_outputs, importance_features = [], [], []
        real_inputs, imputed_inputs, frequencies_weights = [], [], []
        freq_weights, h_bar_weights, c_bar_weights, lambda_weights = [], [], [], []
        gate_imputs_weights, imp_imputs_batch, confidence_weights,uncertainty_weights  = [], [], [], []
        with torch.no_grad():
            progress_bar = tqdm(
                test_loader,
                desc="Evaluating",
                leave=False
            )

            for batch_idx, batch_data in enumerate(progress_bar):
                temporal_features, timestamp, last_data, data_freqs, labels = batch_data

                # Move to the device
                tensors = self._move_to_device(
                    temporal_features, timestamp, last_data, data_freqs, labels
                )
                temporal_features, timestamp, last_data, data_freqs, labels = tensors

                # Sample missing data and run inference
                sampled_data, data_with_missing, indices = self._sample_missing_data(temporal_features)

                (outputs, weights_freq, h_bar, c_bar, confidence,
                 lambda_freq, frequencies, gate_imputs,
                 imputed_inputs_batch) = model(
                    data_with_missing, timestamp, last_data, data_freqs)

                sampled_imputed_x = imputed_inputs_batch[indices]

                # Convert to numpy immediately
                if self.is_classification:
                    all_outputs.append(outputs.sigmoid().cpu().numpy())
                else:
                    all_outputs.append(outputs.cpu().numpy())

                all_targets.append(labels.cpu().numpy())
                real_inputs.append(sampled_data.cpu().numpy())
                imputed_inputs.append(sampled_imputed_x.cpu().numpy())
                freq_weights.append(weights_freq.cpu().detach().numpy())
                h_bar_weights.append(h_bar.cpu().detach().numpy())
                c_bar_weights.append(c_bar.cpu().detach().numpy())
                confidence_weights.append(confidence.cpu().detach().numpy())
                lambda_weights.append(lambda_freq.cpu().detach().numpy())
                frequencies_weights.append(frequencies.cpu().detach().numpy())
                gate_imputs_weights.append(gate_imputs.cpu().detach().numpy())
                imp_imputs_batch.append(imputed_inputs_batch.cpu().detach().numpy())
                uncertainty_weights.append(uncertainty.cpu().detach().numpy())
                # Free batch tensors
                del (temporal_features, timestamp, last_data, data_freqs, labels,
                     outputs, weights_freq, h_bar, c_bar, confidence, lambda_freq,
                     frequencies, imputed_inputs_batch, sampled_data,
                     sampled_imputed_x, gate_imputs,uncertainty,
                     tensors, data_with_missing, indices)

                if batch_idx % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        print(f"[EVAL] Evaluation complete")
        # Stack all results at the end
        return (np.hstack(real_inputs), np.hstack(imputed_inputs), np.vstack(freq_weights),
                np.vstack(h_bar_weights), np.vstack(c_bar_weights), np.vstack(confidence_weights),
                np.vstack(lambda_weights), np.vstack(frequencies_weights), np.vstack(gate_imputs_weights),
                np.vstack(uncertainty_weights),np.vstack(imp_imputs_batch),
                np.vstack(all_targets), np.vstack(all_outputs))

    def inference(self, model_class: type, model_path: str,
                  test_loader: torch.utils.data.DataLoader):
        """Direct inference without unnecessary wrapper lists."""
        (fin_inputs_i, inputs_outputs_i, freq_weights, h_bar_weights,
         c_bar_weights, confidence, lambda_weights, frequencies_weights,
         gate_imputs_weights, uncertainty_weights, imp_imputs_batch, y_pred, y_true) = self.evaluate_model(
            model_class, model_path, test_loader
        )

        # Return as single arrays, not wrapped in lists
        return ([y_true], [y_pred], [fin_inputs_i],
                [inputs_outputs_i], [freq_weights],
                [h_bar_weights], [c_bar_weights], [confidence],
                [lambda_weights], [frequencies_weights],
                [gate_imputs_weights], [imp_imputs_batch], [uncertainty_weights])
    def setup_logging(self, model_path: str, model_name: str) -> str:
        """Set up CSV logging for training metrics."""
        os.makedirs(model_path, exist_ok=True)
        csv_path = os.path.join(model_path, f"metrics_log_{model_name}.csv")

        if self.is_classification:
            header = ["epoch", "lr", "train_loss_imp", "train_loss", "train_acc",
                      "val_loss_imp", "val_loss", "val_acc"]
        else:
            header = ["epoch", "lr", "train_loss_imp", "train_loss",
                      "val_loss_imp", "val_loss", "val_mse", "val_mae"]

        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

        return csv_path

    def train_validate_evaluate(self, model_class: type, model: torch.nn.Module,
                                model_name: str, train_loader: torch.utils.data.DataLoader,
                                val_loader: torch.utils.data.DataLoader,
                                test_loader: torch.utils.data.DataLoader,
                                rescale_params: Dict[str, float],
                                save_path: str, logger: Any) -> List:
        model_save_path = os.path.join(save_path, f'model_{model_name}.pth')
        early_stopping = EarlyStopping(
            task=self.is_classification,
            save_path=model_save_path,
            patience=self.patience,
            primary_threshold=0.0001,  # Save if MSE improves by 0.01%
            imp_threshold=0.00001,  # Save if imputation improves by 0.001%
            logger=logger
        )

        val_targets, val_outputs, val_mse = None, None, None
        csv_log_path = self.setup_logging(save_path, model_name)

        # Log training start
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("=" * 80)
        logger.info(f"TRAINING FOLD: {model_name}")
        logger.info("=" * 80)
        logger.info(f"Model parameters: {total_params:,}")
        logger.info(f"Training epochs: {self.num_epochs}")
        logger.info(f"Early stopping patience: {self.patience}")
        logger.info(f"Task type: {'Classification' if self.is_classification else 'Regression'}")
        logger.info("=" * 80)

        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            if self.is_classification:
                train_mae, train_loss, train_acc = self.train_epoch(model, train_loader, epoch)
                val_mae, val_loss,  val_acc, val_targets, val_outputs = self.validate_epoch(model, val_loader)

                if self.scheduler is not None:
                    self.scheduler.step(epoch)

                log_message = (
                    f"epoch {epoch + 1}/{self.num_epochs} - lr: {self.optimizer.param_groups[0]['lr']:.6f} | "
                    f"train - imp: {train_mae:.6f}, loss: {train_loss:.6f}, acc: {train_acc:.4f} | "
                    f"val - imp: {val_mae:.6f}, loss: {val_loss:.6f}, acc: {val_acc:.4f}"
                )
                logger.info(log_message)

                with open(csv_log_path, mode="a", newline="") as f:
                    csv.writer(f).writerow([epoch + 1, self.optimizer.param_groups[0]['lr'],
                                            train_mae, train_loss, train_acc, val_mae, val_loss, val_acc])

                extra_metrics = {"val_acc": val_acc, "val_loss_imp": val_mae}
                stop_triggered = early_stopping(
                    primary_metric=val_acc,  # Accuracy to maximize
                    imputation_metric=val_mae,  # MAE to minimize
                    model=model,
                    epoch=epoch,
                    extra_metrics=extra_metrics
                )

            else:  # Regression
                train_mae, train_loss, train_mse = self.train_epoch(model, train_loader, epoch)
                val_mae, val_loss, val_mse, val_mae_metric, val_targets, val_outputs = self.validate_epoch(
                    model, val_loader)

                if self.scheduler is not None:
                    self.scheduler.step(epoch)

                log_message = (
                    f"epoch {epoch + 1}/{self.num_epochs} - LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                    f"train - imp: {train_mae:.6f}, loss: {train_loss:.6f}, mse: {train_mse:.6f} | "
                    f"val - imp: {val_mae:.6f}, loss: {val_loss:.6f}, mse: {val_mse:.6f}, mae: {val_mae_metric:.6f}"
                )
                logger.info(log_message)

                with open(csv_log_path, mode="a", newline="") as f:
                    csv.writer(f).writerow([epoch + 1, self.optimizer.param_groups[0]['lr'],
                                            train_mae, train_loss, val_mae, val_loss, val_mse, val_mae_metric])

                extra_metrics = {"val_mse": val_mse, "val_mae": val_mae_metric, "val_loss_imp": val_mae}
                stop_triggered = early_stopping(
                    primary_metric=val_mse,  # MSE to minimize
                    imputation_metric=val_mae,  # MAE to minimize
                    model=model,
                    epoch=epoch,
                    extra_metrics=extra_metrics

                )


            if stop_triggered:
                logger.info("Early stopping triggered - training complete")
                best_metrics_path = os.path.join(save_path, f"best_val_metrics_{model_name}.json")
                with open(best_metrics_path, "w") as f:
                    json.dump(early_stopping.best_info, f, indent=4)
                break

            # Periodic memory cleanup
            if epoch % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ================================================================
        # PHASE 3: Final Evaluation
        # ================================================================
        logger.info("\n" + "=" * 80)
        logger.info("FINAL EVALUATION ON TEST SET")
        logger.info("=" * 80)
        return self._final_evaluation(
            model_class, model_save_path, test_loader, val_targets, val_outputs,
            rescale_params, save_path, model_name, logger
        )

    def _final_evaluation(self, model_class: type, model_path: str,
                          test_loader: torch.utils.data.DataLoader,
                          val_targets: np.ndarray, val_outputs: np.ndarray,
                          rescale_params: Dict[str, float], save_path: str,
                          model_name: str, logger: Any) -> List:
        """Final evaluation with memory-efficient saving."""
        scores = []
        # Validation metrics
        if self.is_classification:
            pr_score = average_precision_score(val_targets, val_outputs)
            logger.info(f"Validation PR-AUC: {pr_score:.4f}")
        else:
            val_mse = mean_squared_error(val_targets, val_outputs)
            val_mae = mean_absolute_error(val_targets, val_outputs)
            logger.info(f"Validation MSE: {val_mse:.6f}, MAE: {val_mae:.6f}")

       # ================================================================
       # Run Inference on Test Set
       # ================================================================
        logger.info("Running inference on test set...")

        # Get inference results
        (targets_list, outputs_list,
         real_inputs_list, imputed_inputs_list,
         freq_weights, h_bar_weights,
         c_bar_weights, confidence, lambda_weights,
         frequencies_weights,gate_imputs_weights,
         imp_imputs_batch, uncertainty) = self.inference(model_class, model_path, test_loader)

        # Unpack (these are already wrapped in single-item lists from inference)
        pred_outputs = targets_list
        true_outputs = outputs_list
        real_inputs = real_inputs_list
        imputed_inputs = imputed_inputs_list
        # Process and save immediately
        if self.is_classification:
            binary_scores = self.metrics.compute_binary_metrics([true_outputs[0]],
                                                                [pred_outputs[0]])
            imputation_scores = self.metrics.compute_imputation_metrics(
                [imputed_inputs[0]], [real_inputs[0]], rescale_params
            )
            reverse_imputation_scores = self.metrics.compute_imputation_metrics(
                [real_inputs[0]], [imputed_inputs[0]], rescale_params
            )
            best_threshold, best_f1 = self.metrics.find_best_threshold(
                val_outputs, val_targets
            )
            # 2. Convert probabilities to binary labels
            test_preds = pred_outputs[0]
            test_predictions_binary = (test_preds > best_threshold).astype(int)
            # 3. Calculate the final F1 score
            final_test_f1 = f1_score(true_outputs[0], test_predictions_binary)
            f1_scores = [(best_threshold, final_test_f1)]

            scores.append([binary_scores, f1_scores,
                           reverse_imputation_scores,
                           imputation_scores])
            # Log results
            logger.info(f"Test AUC & Test AUPRC: {binary_scores}")
            logger.info(f"Best F1: {final_test_f1:.4f} at threshold {best_threshold:.4f}")


            np.savez(
                os.path.join(save_path, f"test_data_fold_{model_name}.npz"),
                reg_scores=binary_scores,imput_scores=reverse_imputation_scores,
                true_labels=true_outputs[0],imputs_scores=imputation_scores,
                val_target=val_targets, val_output=val_outputs,
                predicted_labels=pred_outputs[0],real_x=real_inputs,
                imputed_x=imputed_inputs,scoresf1=np.array([best_threshold, best_f1]),
                freq_weight=freq_weights,h_bar_weight=h_bar_weights, c_bar_weight=c_bar_weights,
                lambda_weight=lambda_weights, frequencies_weight=frequencies_weights,
                gate_imputs=gate_imputs_weights,imputs_batch_imputation=imp_imputs_batch,
                confidence_weigths=confidence, uncertainty_weights=uncertainty
            )

            np.savez(
                os.path.join(save_path, f"test_dataset_fold_{model_name}.npz"),
                test_dataset=test_loader,

            )
            self.c_save_scores_to_json(scores, save_path, f'scores_{model_name}')

        else:  # Regression
            regression_scores = self.metrics.compute_regression_metrics(
                test_targets, test_outputs, rescale_params
            )

            imputation_scores = self.metrics.compute_imputation_metrics(
                imputed_inputs, real_inputs, rescale_params
            )
            reverse_imputation_scores = self.metrics.compute_imputation_metrics(
                real_inputs, imputed_inputs, rescale_params
            )

            scores.append([regression_scores,
                           reverse_imputation_scores,
                           imputation_scores])
            logger.info(f"Test Results: {scores}")
            # print("scores", scores)
            np.savez(
                os.path.join(save_path, f"test_data_fold_{model_name}.npz"),
                reg_scores=regression_scores,imput_scores=reverse_imputation_scores,
                true_labels=test_targets,imputs_scores=imputation_scores,
                predicted_labels=test_outputs,real_x=real_inputs,
                imputed_x=imputed_inputs,freq_weight=freq_weights,
                h_bar_weight=h_bar_weights,c_bar_weight=c_bar_weights,
                lambda_weight=lambda_weights,frequencies_weight=frequencies_weights,
                gate_imputs=gate_imputs_weights, imputs_batch_imputation=imp_imputs_batch,
                confidence_weigths=confidence, uncertainty_weights=uncertainty
            )
            np.savez(
                os.path.join(save_path, f"test_dataset_fold_{model_name}.npz"),
                test_dataset=test_loader)

        logger.info(f"Results saved to: {save_path}")
        logger.info("=" * 80)
        logger.info("EVALUATION COMPLETE")
        # Free memory gate_imputs=gate_imputs_weights,imputs_batch_imputation= imp_imputs_batch
        del (true_outputs, pred_outputs, real_inputs, imputed_inputs,
             freq_weights, h_bar_weights, targets_list, outputs_list, real_inputs_list,
             imputed_inputs_list, c_bar_weights, lambda_weights, frequencies_weights,
             gate_imputs_weights,imp_imputs_batch,confidence, uncertainty)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[INFO] Test results for fold {model_name}: {scores}")
        return [scores]

    @staticmethod
    def save_importances_with_json(importances, save_dir: str, filepath_prefix='importances'):
        """Save importances and metadata separately for better portability."""

        # Prepare metadata
        metadata = []
        importance_arrays = {}

        for idx, imp_dict in enumerate(importances):
            # Store metadata
            metadata.append({
                'idx': idx,
                'layer': imp_dict['layer'],
                'direction': imp_dict['direction'],
                'shape': imp_dict['importance'].shape,
                'dtype': str(imp_dict['importance'].dtype)
            })

            # Store array with the key
            key = f"L{imp_dict['layer']}_{imp_dict['direction']}"
            importance_arrays[key] = imp_dict['importance']

        # Save
        save_path = os.path.join(save_dir, f'{filepath_prefix}_arrays.npz')
        np.savez(save_path, **importance_arrays)
        with open(os.path.join(save_dir,f'{filepath_prefix}_metadata.json'),
                  'w') as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def c_save_scores_to_json(all_scores: List, save_dir: str, filename: str) -> str:
        """Save model scores to JSON format in a structured way."""
        os.makedirs(save_dir, exist_ok=True)

        # Extract the first set of scores (assuming one run)
        model_scores = all_scores[0]

        # Build dictionary
        model_dict = {
            "classification": {
                "AUC": float(model_scores[0][0][0]),
                "AUPRC": float(model_scores[0][0][1]),
                "F1_score": {
                    "threshold": float(model_scores[1][0][0]),
                    "value": float(model_scores[1][0][1])
                }
            },
            "imputation_1": {
                "rmse": float(model_scores[2][0][0]),
                "mae": float(model_scores[2][0][1]),
                "r2": float(model_scores[2][0][2]),
                "adj_r2": float(model_scores[2][0][3]),
            },
            "imputation_2": {
                "rmse": float(model_scores[3][0][0]),
                "mae": float(model_scores[3][0][1]),
                "r2": float(model_scores[3][0][2]),
                "adj_r2": float(model_scores[3][0][3]),
            }
        }

        # Save JSON
        save_path = os.path.join(save_dir, f"{filename}.json")
        with open(save_path, "w") as f:
            json.dump(model_dict, f, indent=4)

        print(f"Scores saved to {save_path}")
        return save_path

    @staticmethod
    def _save_scores_to_json(all_scores: List, save_dir: str, filename: str) -> str:
        """Save model scores to JSON format."""
        os.makedirs(save_dir, exist_ok=True)
        model_scores = all_scores[0]
        keys = ["forecasting", "imputation_1", "imputation_2"]

        model_dict = {}
        for idx, score_set in enumerate(model_scores):
            scores = score_set[0]
            mse, rmse, mae, r2, adj_r2 = [float(s) for s in scores]
            model_dict[keys[idx]] = {
                "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "adj_r2": adj_r2
            }

        save_path = os.path.join(save_dir, f"{filename}.json")
        with open(save_path, "w") as f:
            json.dump(model_dict, f, indent=4)

        print(f"Scores saved to {save_path}")
        return save_path

