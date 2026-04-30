import torch
import torch.nn as nn
import torch.nn.functional as F

class EDMAFAILLoss(nn.Module):
    """
    EDMAFAILoss with:
        - Adaptive task weighting
        - Curriculum scheduling for forecasting loss
        - Frequency-aware imputation weighting
        - Diffusion (EDM) loss support
    """
    def __init__(self, p_mean=-1.2, p_std=1.2, sigma_data=0.5, device='cuda'):
        super().__init__()
        self.P_mean = p_mean
        self.P_std = p_std
        self.sigma_data = sigma_data
        self.device = device
        # Learnable task weights (log-space)
        self.log_task_weights = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))  # [imputation, diffusion, prediction]
        # Alpha/beta for frequency-based weighting
        self.log_alpha = nn.Parameter(torch.tensor(-5.3))
        self.log_beta = nn.Parameter(torch.tensor(-4.6))

    def forward(self, sampled_data, sampled_imputed_x,
                data_freqs, outputs, labels, criterion):
        """
        Args:
            sampled_data: original data with NaNs
            sampled_imputed_x: APSD imputed data
            data_freqs: [F] frequency of each feature
            outputs: model predictions for forecasting
            labels: ground truth for forecasting
            criterion: e.g., nn.MSELoss()
        Returns:
            weighted_loss_imp, edm_loss, total_loss
        """
        # ---------------------------------
        # 1. Feature-wise Huber Imputation Loss
        # ---------------------------------
        imp_loss = torch.mean(torch.abs(sampled_data - sampled_imputed_x))  # [B, T, F]
        # ---------------------------------
        # 3. Frequency-based weighting for imputation
        # ---------------------------------
        alpha = F.softplus(self.log_alpha)
        normalized_freqs = data_freqs / (data_freqs.max() + 1e-6)
        dynamic_weights = torch.exp(-alpha * (1 - normalized_freqs))
        loss_imp = torch.sum(imp_loss * dynamic_weights) / torch.sum(dynamic_weights)
        # ---------------------------------
        # 4. Prediction loss
        # ---------------------------------
        beta = F.softplus(self.log_beta)
        freq_penalty = torch.mean(torch.exp(-beta * normalized_freqs))
        pred_loss = criterion(outputs.squeeze(-1), labels.squeeze(-1))
        pred_loss = pred_loss * freq_penalty
        # ---------------------------------
        # 5. Adaptive task weighting
        # ---------------------------------
        # task_weights: [imputation, diffusion, prediction]
        total_loss = (
            loss_imp +
            pred_loss
        )
        return loss_imp, total_loss

