import os
import json
import time
import torch
import random
import logging
import sys
import numpy as np
import torch.nn as nn
from copy import deepcopy
from datetime import datetime
from models.GTACM import GTACMNetwork
from trainer_helper import ModelTrainer
from utils.afail_loss import EDMAFAILLoss
from typing import Any, Dict, List, Tuple
from utils.missing_mecanisms import DataSampler
from diffusion_utils.sampling import FASIGSSamplerEuler2nd

optimizer_config = {
    "lr": 3e-4,
    "betas": (0.9, 0.95),
    "eps": 1e-8,
    "weight_decay": 1e-5
}


def seed_all(seed: int = 1992):
    """Seed all random number generators."""
    print("Using Seed Number {}".format(seed))

    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    # set a fixed value for python built-in pseudo-random generator
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


class ExperimentRunner:
    """High-level experiment orchestration with improved data handling."""

    def __init__(self, dataset_path: str, task_name: str, missing_type: str = 'MCAR'):
        if missing_type not in ['MCAR', 'MAR', 'MNAR']:
            raise ValueError(f"Invalid missing_type: {missing_type}. Must be one of ['MCAR', 'MAR', 'MNAR']")

        self.dataset_path = dataset_path
        self.task_name = task_name
        self.missing_type = missing_type
        self.results_path = None

        print(f"[INIT] Experiment Runner initialized")
        print(f"  └─ Task: {task_name}")
        print(f"  └─ Missingness: {missing_type}")

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load and validate dataset files."""
        dataset_dir = os.path.join(self.dataset_path, self.task_name)

        try:
            # Load main data
            data_file = os.path.join(dataset_dir, "train_test_data.npz")
            all_dataset_loader = np.load(data_file, allow_pickle=True)

            # Load scaling parameters
            settings_file = os.path.join(dataset_dir, "data_max_min.npz")
            dataset_settings = np.load(settings_file, allow_pickle=True)

            train_val_loader = all_dataset_loader['folds_data_train_valid']
            test_loader = all_dataset_loader['folds_data_test']

            print(f"[LOAD] Dataset loaded successfully")
            print(f"  └─ Number of folds: {len(train_val_loader)}")
            print(f"  └─ Sequence length: {dataset_settings['seq_length'].item()}")
            print(f"  └─ Input dimensions: {dataset_settings['input_dim'].item()}")

            return train_val_loader, test_loader, dataset_settings

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required dataset files not found: {e}")
        except KeyError as e:
            raise KeyError(f"Expected keys missing from dataset: {e}")

    @staticmethod
    def setup_model_components(dataset_settings: Dict[str, Any],
                               model_class: type, missing_percentage: float,
                               is_classification: bool, device: torch.device) -> Dict[str, Any]:
        """Initialize all model components with a proper configuration."""
        # Extract dataset parameters
        seq_length = dataset_settings['seq_length'].item()
        input_dim = dataset_settings['input_dim'].item()
        # output_dim = 1 # BEIJINGAIRQUALITY
        output_dim = dataset_settings['output_dim']  # Others
        hidden_dim = 128  # Can be parameterized
        timesteps = 20  # Can be parameterized
        num_layers = 2

        print(f"[MODEL] Initializing model components")
        print(f"  └─ Architecture: {model_class.__name__}")
        print(f"  └─ Hidden dim: {hidden_dim}, Layers: {num_layers}")

        # Initialize model
        model = model_class(
            input_dim, hidden_dim, seq_length, timesteps, num_layers, output_dim, device
        ).to(device)
        # Setup optimizer with proven hyperparameters

        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)

        # Set up scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=100,  # Number of epochs before restart
            T_mult=2,  # Increases T_0 after each restart
            eta_min=1e-6
        )

        loss_function = nn.BCELoss().to(device) if is_classification else nn.MSELoss().to(device)

        custom_loss = EDMAFAILLoss(device=device).to(device)

        return {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'loss_function': loss_function,
            'custom_loss': custom_loss,
            'seq_length': seq_length,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'num_layers': num_layers,
            'num_steps': timesteps
        }

    def create_results_directory(self, missing_percentage: float,
                                 timesteps: int) -> str:
        """Create an organized results directory structure."""
        base_name = self.task_name  # .split('_')[0]
        self.results_path = (f"DYFAIP_PAPER/{base_name}_{self.missing_type}_RATE_{missing_percentage}"
        )

        task_path = os.path.join(self.results_path, base_name)
        os.makedirs(task_path, exist_ok=True)
        print(f"[RESULTS] Directory created: {task_path}")
        return task_path

    @staticmethod
    def setup_logging(save_path: str, fold_id: str) -> logging.Logger:
        """
        Initialize logger for experiment tracking.

        Args:
            save_path: Directory where log files will be saved
            fold_id: Identifier for the current fold

        Returns:
            Configured logger instance
        """
        logs_dir = os.path.join(save_path, 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(logs_dir, f"training_log_{fold_id}_{timestamp}.log")

        # Create logger
        logger = logging.getLogger(f"Fold_{fold_id}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        # Console handler for stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # File handler for persistent logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        logger.info(f"Logging initialized for Fold {fold_id}")
        logger.info(f"Log file: {log_file}")

        return logger

    @staticmethod
    def compute_mnar_thresholds(
            train_data: torch.Tensor,
            feature_indices: List[int],
            percentage: float = 0.5
    ) -> List[float]:
        """
        Compute MNAR thresholds using a quantile-based approach.

        This method ensures we have enough data to mask regardless of data scale
        by using quantiles rather than fixed values.

        Strategy:
        - To mask 50% of high values → use median (50th percentile) as a threshold
        - To mask 20% of high values → use the 80th percentile as a threshold
        - General formula: threshold_quantile = 1.0 - missing_percentage

        Args:
            train_data: Training data tensor [B, T, F]
            feature_indices: List of feature indices to compute thresholds for
            percentage: Target percentage of data to mask

        Returns:
            List of threshold values (one per feature)
        """
        thresholds = []
        quantile_position = 1.0 - percentage

        print(f"[MNAR] Computing thresholds at {quantile_position:.2f} quantile")

        for f_idx in feature_indices:
            feat_vals = train_data[:, :, f_idx]
            valid_vals = feat_vals[~torch.isnan(feat_vals)]

            if valid_vals.numel() == 0:
                print(f"  └─ Feature {f_idx}: No valid data, using default threshold 0.0")
                thresholds.append(0.0)
                continue

            # Compute the quantile-based threshold
            threshold = torch.quantile(valid_vals, quantile_position).item()
            thresholds.append(threshold)

            # Diagnostic information
            data_min = valid_vals.min().item()
            data_max = valid_vals.max().item()
            print(f"  └─ Feature {f_idx}: threshold={threshold:.4f}, "
                  f"range=[{data_min:.4f}, {data_max:.4f}]")

        return thresholds

    def _initialize_data_sampler(
            self,
            missing_percentage: float,
            input_dim: int,
            train_val_loader: np.ndarray
    ) -> Tuple[DataSampler, List[int]]:
        """
        Initialize the data sampler with the appropriate missing data mechanism.

        Args:
            missing_percentage: Fraction of data to mark as missing
            input_dim: Number of input features
            train_val_loader: Training/validation data for threshold computation

        Returns:
            Tuple of (initialized DataSampler, list of sampled feature indices)
        """
        # Select features to apply missingness to
        feature_pool = np.arange(input_dim)
        num_features_to_sample = int(len(feature_pool) * missing_percentage)
        sampled_features = np.random.choice(
            feature_pool, size=num_features_to_sample, replace=False
        ).tolist()

        print(f"[SAMPLER] Initializing {self.missing_type} sampler")
        print(f"  └─ Missing percentage: {missing_percentage * 100:.1f}%")
        print(f"  └─ Features affected: {num_features_to_sample}/{input_dim}")

        # Base sampler configuration
        sampler_kwargs = {
            'percentage': missing_percentage,
            'mode': self.missing_type
        }

        # Add mechanism-specific parameters
        if self.missing_type in ['MAR', 'MNAR']:
            sampler_kwargs['feature_idx'] = sampled_features

            if self.missing_type == 'MNAR':
                # Compute thresholds from first fold's training data
                print(f"[MNAR] Computing quantile-based thresholds...")
                first_fold_train = train_val_loader[0][0]
                train_data_tensor = first_fold_train.dataset.tensors[0]

                mnar_thresholds = self.compute_mnar_thresholds(
                    train_data_tensor,
                    sampled_features,
                    missing_percentage
                )
                sampler_kwargs['threshold'] = mnar_thresholds
                print(f"[MNAR] Thresholds: {[f'{t:.4f}' for t in mnar_thresholds[:5]]}... (showing first 5)")

        # Initialize sampler
        data_sampler = DataSampler(**sampler_kwargs)
        print(f"[SAMPLER] Initialization complete\n")

        return data_sampler, sampled_features

    def run_cross_validation_experiment(self, model_class: type,
                                        num_folds: int = 10,
                                        num_epochs: int = 800,
                                        patience: int = 50,
                                        missing_percentage: float = 0.5,
                                        is_classification: bool = True) -> List[Any]:
        """
        Execute the complete cross-validation experiment.

        This method orchestrates the entire experimental pipeline:
        1. Dataset loading and validation
        2. Model and component initialization
        3. Missing data mechanism configuration
        4. Cross-validation loop execution
        5. Results aggregation and persistence

        Args:
            model_class: Class constructor for the model
            num_folds: Number of cross-validation folds
            num_epochs: Maximum training epochs per fold
            patience: Early stopping patience
            missing_percentage: Fraction of data to mark as missing
            is_classification: Whether a task is classification

        Returns:
            Tuple of (list of fold scores, result directory path)

        Raises:
            RuntimeError: If experiment execution fails
        """
        print("\n" + "=" * 80)
        print("EXPERIMENT EXECUTION STARTED")
        print("=" * 80 + "\n")

        # ============================================================
        # PHASE 1: Environment Setup
        # ============================================================
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DEVICE] Using: {device}")
        if device.type == 'cuda':
            print(f"  └─ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  └─ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

        # ============================================================
        # PHASE 2: Data Loading
        # ============================================================
        train_val_loader, test_loader, dataset_settings = self.load_dataset()

        # ============================================================
        # PHASE 3: Model Components Setup
        # ============================================================
        components = self.setup_model_components(
            dataset_settings, model_class, missing_percentage, is_classification, device
        )

        print(f"\n[MODEL] Total parameters: "
              f"{sum(p.numel() for p in components['model'].parameters()):,}\n")

        # ============================================================
        # PHASE 4: Missing Data Mechanism Configuration
        # ============================================================
        input_dim = dataset_settings['input_dim'].item()
        data_sampler, sampled_features = self._initialize_data_sampler(
            missing_percentage, input_dim, train_val_loader
        )

        # ============================================================
        # PHASE 5: Optimizer and Scheduler Reinitialization
        # ============================================================
        # These are recreated to ensure clean state for cross-validation
        optimizer = torch.optim.AdamW(
            components['model'].parameters(),
            lr=3e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=1e-5
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=100,
            T_mult=2,
            eta_min=1e-6
        )

        # ============================================================
        # PHASE 6: Results Directory Creation
        # ============================================================
        save_path = self.create_results_directory(
            missing_percentage, components['num_steps']
        )

        # ============================================================
        # PHASE 7: Store Original Model Weights for Fold Resets
        # ============================================================
        original_weights = deepcopy(components['model'].state_dict())
        print(f"[CHECKPOINT] Original model weights stored for fold reinitialization\n")

        # ============================================================
        # PHASE 8: Trainer Configuration
        # ============================================================

        # Initialize trainer configuration
        trainer_config = {
            'device': device,
            'patience': patience,
            'num_epochs': num_epochs,
            'input_dim': components['input_dim'],
            'hidden_dim': components['hidden_dim'],
            'seq_length': components['seq_length'],
            'output_dim': components['output_dim'],
            'num_steps': components['num_steps'],
            'num_layers': components['num_layers'],
            'is_classification': is_classification,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'loss_function': components['loss_function'],
            # Add other components as needed
            'custom_loss': components['custom_loss'],
            'data_sampler': data_sampler,
        }
        # Initialize trainer for this fold
        trainer = ModelTrainer(trainer_config)
        # ============================================================
        # PHASE 9: Cross-Validation Loop
        # ============================================================
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION TRAINING")
        print("=" * 80 + "\n")
        # Run cross-validation
        all_fold_scores = []

        for fold_idx, (train_loader, test_data) in enumerate(zip(train_val_loader, test_loader)):
            fold_number = fold_idx + 1
            print(f"\n{'─' * 80}")
            print(f"FOLD {fold_number}/{num_folds}")
            print(f"{'─' * 80}\n")
            # Reset model weights
            components['model'].load_state_dict(original_weights)
            optimizer = torch.optim.AdamW(components['model'].parameters(), **optimizer_config)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=100,  # Number of epochs before restart
                T_mult=2,  # Increases T_0 after each restart
                eta_min=1e-6
            )

            # 3. Update trainer config with the FRESH optimizer/scheduler
            trainer.optimizer = optimizer
            trainer.scheduler = scheduler

            print(f"[FOLD {fold_number}] Model weights reset to initialization\n")

            # Setup logging
            logger = self.setup_logging(save_path, str(fold_idx + 1))
            # Split train/validation data
            train_data, valid_data = train_loader
            # Train and evaluate
            # Execute training, validation, and evaluation
            try:
                fold_scores = trainer.train_validate_evaluate(
                    model_class=model_class,
                    model=components['model'],
                    model_name=str(fold_idx + 1),
                    train_loader=train_data,
                    val_loader=valid_data,
                    test_loader=test_data,
                    rescale_params=dict(dataset_settings),
                    save_path=save_path,
                    logger=logger
                )

                all_fold_scores.append(fold_scores)
                print(f"\n[FOLD {fold_number}] ✓ Completed successfully\n")

            except Exception as e:
                print(f"\n[FOLD {fold_number}] ✗ Failed with error: {e}\n")
                logger.error(f"Fold {fold_number} failed: {e}")
                raise
        # ============================================================
        # PHASE 10: Results Aggregation
        # ============================================================
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETED")
        print("=" * 80 + "\n")
        # Save aggregated results
        self._save_experiment_summary(all_fold_scores, save_path)

        return all_fold_scores, save_path

    def _save_experiment_summary(self, all_scores: List[Any], save_path: str) -> None:
        """Save summary statistics across all folds."""
        summary_path = os.path.join(save_path, "experiment_summary.json")

        summary = {
            "experiment_info": {
                "task_name": self.task_name,
                "missing_type": self.missing_type,
                "num_folds": len(all_scores),
                "completed_at": datetime.now().isoformat()
            },
            "fold_results": all_scores
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4, default=str)

        print(f"Experiment summary saved to {summary_path}")


# Usage example with proper configuration
def create_trainer_config(model_params: Dict[str, Any],
                          training_params: Dict[str, Any],
                          model_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create a comprehensive trainer configuration."""
    config = {}
    config.update(model_params)
    config.update(training_params)
    config.update(model_components)
    return config


# Example usage:
if __name__ == "__main__":
    """
    Replace placeholder classes with your actual implementations.
    """
    # ========================================================================
    # STEP 1: Set Random Seed for Reproducibility
    # ========================================================================
    seed_all(seed=1992)
    # Configuration for the experiment
   
    dataset_path = '-----/ETTH2_48_TIMESTEPS_AHEAD_P_HORIZON_720/'
    task_name = "ETTH2_48_TIMESTEPS_AHEAD_P_HORIZON_720"

    # Build experiment metadata dictionary
    # Initialize experiment runner MNAR, MAR, MCAR
    experiment_config = {
        "dataset_path": dataset_path,
        "task_name": task_name,
        "missing_type": 'MCAR',
        "num_folds": 10,
        "num_epochs": 200 # 800 best epochs number,
        "patience": 30,
        "missing_percentage": 0.5,
        "is_classification": False,
        "model": "GTACMNetwork"
    }
    print("\n" + "=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    for key, value in experiment_config.items():
        print(f"{key:.<40} {value}")
    print("=" * 80 + "\n")

    # ========================================================================
    # STEP 4: Initialize Experiment Runner
    # ========================================================================
    experiment = ExperimentRunner(
        dataset_path=dataset_path,
        task_name=task_name,
        missing_type=experiment_config["missing_type"])

    # ========================================================================
    # STEP 5: Execute Experiment with Error Handling
    # ========================================================================
    try:
        print("[INFO] Starting experiment execution...\n")
        start_time = time.time()

        # Run the cross-validation experiment
        results, save_dir = experiment.run_cross_validation_experiment(
            model_class=GTACMNetwork,
            num_folds=experiment_config["num_folds"],
            num_epochs=experiment_config["num_epochs"],
            patience=experiment_config["patience"],
            missing_percentage=experiment_config["missing_percentage"],
            is_classification=experiment_config["is_classification"]
        )

        end_time = time.time()
        duration = end_time - start_time

        # ====================================================================
        # STEP 6: Finalize Results
        # ====================================================================
        experiment_config.update({
            "start_time": datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "duration_formatted": f"{duration // 3600:.0f}h {(duration % 3600) // 60:.0f}m {duration % 60:.0f}s",
            "results": results
        })

        # Save final configuration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_save_path = os.path.join(save_dir, f"experiment_{task_name}_{timestamp}.json")

        with open(config_save_path, "w") as f:
            json.dump(experiment_config, f, indent=4, default=str)

        # ====================================================================
        # STEP 7: Display Summary
        # ====================================================================
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Status.................... ✓ COMPLETED SUCCESSFULLY")
        print(f"Duration.................. {experiment_config['duration_formatted']}")
        print(f"Results saved to.......... {save_dir}")
        print(f"Configuration saved to.... {config_save_path}")
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n[WARNING] Experiment interrupted by user (Ctrl+C)")
        print("[INFO] Partial results may have been saved\n")

    except Exception as e:
        print(f"\n[ERROR] Experiment failed with exception:")
        print(f"  └─ {type(e).__name__}: {e}\n")
        import traceback

        traceback.print_exc()