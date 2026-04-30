import torch

class DataSampler:
    def __init__(self, percentage=0.2, mode='MCAR', feature_idx=None, threshold=None):
        self.percentage = percentage
        self.mode = mode
        self.feature_idx = [feature_idx] if isinstance(feature_idx, int) else feature_idx
        self.threshold = [threshold] if isinstance(threshold, (float, int)) else threshold

    def mark_data_as_missing(self, data):
        data_with_missing = data.clone()
        mask = ~torch.isnan(data)

        if self.mode == 'MCAR':
            observed_flat_indices = torch.nonzero(mask.view(-1), as_tuple=False).squeeze()
            num_samples = int(self.percentage * observed_flat_indices.size(0))
            sampled_indices = observed_flat_indices[
                torch.randperm(observed_flat_indices.size(0), device=data.device)[:num_samples]]
            sampled_3d_indices = torch.stack(torch.unravel_index(sampled_indices, data.shape))

        elif self.mode == 'MAR':
            if self.feature_idx is None:
                raise ValueError("feature_idx must be provided for MAR")

            cond_mask = mask[:, :, self.feature_idx[0]].clone()
            for f in self.feature_idx[1:]:
                cond_mask &= mask[:, :, f]

            cond_b, cond_t = torch.nonzero(cond_mask, as_tuple=True)
            valid_btf = torch.nonzero(mask, as_tuple=False)
            valid_btf = valid_btf[(cond_mask[valid_btf[:, 0], valid_btf[:, 1]])]

            num_samples = int(self.percentage * len(valid_btf))
            sampled = valid_btf[torch.randperm(len(valid_btf), device=data.device)[:num_samples]]
            sampled_3d_indices = sampled.T

        elif self.mode == 'MNAR':
            if self.feature_idx is None or self.threshold is None:
                raise ValueError("feature_idx and threshold must be provided for MNAR")
            if len(self.feature_idx) != len(self.threshold):
                raise ValueError("feature_idx and threshold must have the same length")

            btf_positions = []
            for f_idx, thr in zip(self.feature_idx, self.threshold):
                feat_vals = data[:, :, f_idx]
                valid = (feat_vals > thr) & mask[:, :, f_idx]
                b_idxs, t_idxs = torch.where(valid)
                num_valid = b_idxs.size(0)
                if num_valid > 0:
                    f_idxs = torch.full((num_valid,), f_idx, device=data.device)
                    btf = torch.stack([b_idxs, t_idxs, f_idxs], dim=0)  # shape: (3, N)
                    btf_positions.append(btf)

            if not btf_positions:
                raise ValueError("No valid indices for MNAR")

            all_btf = torch.cat(btf_positions, dim=1)  # shape: (3, total_num_valid)

            # Shuffle and sample
            perm = torch.randperm(all_btf.shape[1], device=data.device)
            num_samples = int(self.percentage * all_btf.shape[1])
            sampled_3d_indices = all_btf[:, perm[:num_samples]]

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        selected_data = data_with_missing[sampled_3d_indices[0], sampled_3d_indices[1], sampled_3d_indices[2]]
        assert not torch.isnan(selected_data).any(), 
        data_with_missing[sampled_3d_indices[0], sampled_3d_indices[1], sampled_3d_indices[2]] = float('nan')

        return selected_data, data_with_missing, tuple(sampled_3d_indices)
