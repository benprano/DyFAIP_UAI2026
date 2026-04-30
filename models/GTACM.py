from typing import List
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
class DyFAIPCell(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size, seq_len):
        super(DyFAIPCell, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.initializer_range = 0.02
        self.hidden_size = hidden_size
        self.register_buffer("factor", torch.FloatTensor([0.5]))
        self.register_buffer('c1_const', torch.Tensor([1]).float())
        self.register_buffer("factor_impu", torch.FloatTensor([0.5]))
        self.register_buffer('c2_const', torch.Tensor([np.e]).float())
        self.register_buffer("imp_weight_freq", torch.FloatTensor([0.05]))
        self.register_buffer("Wdelta", torch.ones([self.input_size, 1, 1]).float())
        self.register_buffer("ones_const", torch.ones([self.input_size, 1, self.hidden_size]).float())
        self.register_buffer("fixed_decay", torch.arange(self.input_size).float())
        # Learnable scaling per feature to modulate decay based on frequency importance
        self.importance_manifold = nn.Parameter(torch.zeros(self.input_size))
        self.att_temperature = nn.Parameter(torch.tensor(1.0))

        # These learn to map the raw frequency counts into a decay rate
        self.feature_trust_base = nn.Parameter(torch.ones(self.input_size))
        self.freq_weight = nn.Parameter(torch.randn(self.input_size))
        self.freq_bias = nn.Parameter(torch.zeros(self.input_size))
        self.omega = torch.nn.Parameter(torch.tensor(0.99))

        self.U_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_last = nn.Parameter( torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_time = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.Dw = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))

        self.W_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_d = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))

        self.W_cell_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.W_cell_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.W_cell_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_time = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_d = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        # Interpolation
        self.W_ht_mask = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, 1)))
        self.W_ct_mask = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, 1)))
        self.b_j_mask = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))

        self.W_ht_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, 1)))
        self.W_ct_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, 1)))
        self.b_j_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_freq_fac = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(1, self.input_size)))
        self.b_freq_imp = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(1, self.input_size)))

    @torch.jit.script_method
    def forward(self, prev_hidden_memory, cell_hidden_memory, inputs, times, last_data, freq_list):
        h_tilda_t, c_tilda_t = prev_hidden_memory, cell_hidden_memory,
        x,t,l,freq = inputs, times, last_data, freq_list
        T = self.map_elapse_time(t)
        # Apply temporal decay to D-STM
        h_bar = torch.mul(T, self.freq_decay(freq, h_tilda_t))
        c_bar = torch.mul(T, self.freq_decay(freq, c_tilda_t))
        # frequency weights for imputation of missing data based on frequencies of features
        x_last_hidden = torch.tanh(torch.einsum("bij,ijk->bik", h_bar, self.W_ht_last) +
                                   torch.einsum("bij,ijk->bik", c_bar, self.W_ct_last) +
                                   self.b_j_last).permute(0, 2, 1)

        imputat_imputs = torch.tanh(torch.einsum("bij,ijk->bik", h_bar, self.W_ht_mask) +
                                    torch.einsum("bij,ijk->bik", c_bar, self.W_ct_mask) +
                                    self.b_j_mask).permute(0, 2, 1)
        # Replace nan data with the impuated value generated from LSTM memory and frequencies weights

        _, _,_, _, x_last = self.dyfaip(l, freq, x_last_hidden)
        lambda_f, frequencies, gate,all_imputed_x, imputed_x = self.dyfaip(x, freq, imputat_imputs)

        # Ajust previous to incoporate the latest records for each feature
        last_tilda_t = F.elu(torch.einsum("bij,jik->bjk", x_last, self.U_last) + self.b_last)
        h_tilda_t = h_tilda_t + last_tilda_t
        # Capturing Temporal Dependencies wrt to the previous hidden state
        j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                               torch.einsum("bij,jik->bjk", imputed_x, self.U_j) + self.b_j)
        # Step 2: Frequency-Based Confidence Gate (Critical for Sparsity)
        confidence = torch.sigmoid(self.freq_decay(freq, j_tilda_t) +
                                   torch.sigmoid(self.map_elapse_time(t)))
        # Update and Retention
        update = confidence * torch.sigmoid(
            torch.einsum("bij,jik->bjk", imputed_x, self.U_i) +
            torch.einsum("bij,ijk->bik", h_bar, self.W_i)
        )

        # Candidate Memory Cell
        c_candidate = torch.tanh(torch.einsum("bij,jik->bjk", imputed_x, self.U_c) + \
                       torch.einsum("bij,ijk->bik", h_bar, self.W_c) + self.b_c)

        Ct = confidence * c_bar + update * c_candidate

        o = torch.sigmoid(
            torch.einsum("bij,jik->bjk", imputed_x, self.U_o) +
            torch.einsum("bij,ijk->bik", h_bar, self.W_o) +
            Ct * self.W_cell_o
        )

        h_tilda_t = o * torch.tanh(Ct)

        return (h_tilda_t, Ct, h_bar, c_bar, confidence,
                self.freq_decay(freq, j_tilda_t), lambda_f,
                frequencies, gate, all_imputed_x)

    @torch.jit.script_method
    def dyfaip(self, x: torch.Tensor, freq_dict: torch.Tensor, x_hidden: torch.Tensor):
        # Calculate feature factor
        freqs = (self.seq_len - freq_dict)
        factor_feature = torch.div(torch.exp(-self.imp_weight_freq * (freqs + self.b_freq_fac)),
                                   torch.exp(-self.imp_weight_freq *
                                   (freqs + self.b_freq_fac)).max()).unsqueeze(1)

        # Step 1-b) Frequency-driven imputation factor
        factor_imp = torch.div(torch.exp(self.factor_impu * (freqs + self.b_freq_imp)),
                               torch.exp(self.factor_impu *
                               (freqs + self.b_freq_imp)).max()).unsqueeze(1)

        # Adjust frequencies
        # Softplus ensures the decay rate is always positive.
        lambda_f = torch.nn.functional.softplus(self.freq_weight * freqs + self.freq_bias)

        frequencies = freqs * torch.exp(-lambda_f * freqs)
        frequencies = torch.div(frequencies, frequencies.max()).unsqueeze(-1)
        # Compute imputed values
        # Use sigmoid to keep omega between 0 and 1
        # Instead of torch.where, use a sigmoid gate
        # We calculate a 'switch' value. If positive, it leans toward freq-based.
        gate = torch.sigmoid((factor_imp - (self.omega * factor_imp.max())) * 10)  # 10 is a temperature scaling

        imputed_missed_x = (gate * (frequencies.permute(0, 2, 1) * x_hidden)) + \
                           ((1 - gate) * (factor_feature * x_hidden))
        # Replace missing values
        x_imputed = torch.where(torch.isnan(x.unsqueeze(1)),
                                imputed_missed_x, x.unsqueeze(1))

        return lambda_f, frequencies, gate, imputed_missed_x, x_imputed

    @torch.jit.script_method
    def map_elapse_time(self, t):
        T = torch.div(self.c1_const, torch.log(t + self.c2_const))
        T = torch.einsum("bij,jik->bjk", T.unsqueeze(1), self.ones_const)
        return T

    @torch.jit.script_method
    def freq_decay(self, freq_dict: torch.Tensor, ht: torch.Tensor):
        freq_weight = torch.exp(-self.factor_impu * freq_dict)
        weights = torch.sigmoid(torch.einsum("bij,jik->bjk", freq_weight.unsqueeze(-1), self.Dw) + \
                                torch.einsum("bij,ijk->bik", ht, self.W_d) + self.b_d)
        return weights

    @torch.jit.script_method
    def freq_encode(self, freq_dict: torch.Tensor):
        # normalize and apply continuous basis functions (e.g., Fourier + polynomial)
        freq_log = torch.exp(-self.factor_impu * freq_dict)  # [B, F]
        freq_norm = (freq_log - freq_log.mean()) / (freq_log.std() + 1e-6)
        freq_features = torch.stack([freq_norm, torch.sin(freq_norm),
                                     torch.cos(freq_norm)],dim=-1)  # [B, F, 4] .unsqueeze(-1)
        # Project to hidden size
        freq_proj = torch.sigmoid(torch.einsum("bij,ijk->bik", freq_features, self.Dw))
        return freq_proj  # [B, F, H]

class ContextConditioned(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, output_dim, num_freqs=16):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        # Pre-normalization for stability
        self.pre_norm = nn.LayerNorm(input_size)

        # === Stable Temporal Mixing ===
        # Fourier mixing (learned freq embeddings instead of conv)
        self.freqs = nn.Parameter(torch.randn(num_freqs))
        self.freq_proj = nn.Linear(num_freqs * 2, input_size)

        # Channel mixing with residual scaling
        self.channel_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.res_scale = nn.Parameter(torch.tensor(0.1))  # stabilize residual

        # Diffusion conditioning
        self.diff_proj = nn.Linear(input_size, hidden_size)

        # Semantic salience gating (with dropout to avoid gate collapse)
        self.semantic_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x, sigma=None):
        # x: [B, I, T]
        B, I, T = x.shape
        # Feature normalization
        x = x.permute(0, 2, 1)   # [B, T, I]
        x = self.pre_norm(x)
        # === Fourier temporal embedding ===
        t = torch.linspace(0, 1, T, device=x.device).unsqueeze(-1)  # [T,1]
        freqs = self.freqs[None, None, :] * t  # [1,T,F]
        fourier_basis = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)  # [1,T,2F]
        temporal_mix = self.freq_proj(fourier_basis).expand(B, -1, -1)  # [B,T,I]
        # Channel projection + residual
        x_proj = self.channel_proj(x + temporal_mix)  # [B,T,H]
        if sigma is not None:
            cond = self.diff_proj(torch.tanh(sigma))  # [B,T,H]
            x_proj = x_proj + cond
            # print("x_proj", x_proj.shape)
        x_proj = x_proj + self.res_scale * x_proj  # stabilize residual
        # Gated pooling
        # Softmax semantic gate
        attn = F.softmax(self.semantic_gate(x_proj).squeeze(-1), dim=-1)  # [B, T]
        pooled = torch.sum(x_proj * attn, dim=1)  # [B, H]
        return self.out_proj(pooled)

class GTACM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, num_steps, num_layers, output_dim, device, batch_first=True,
                 bidirectional=True):
        super(GTACM, self).__init__()
        # hidden dimensions
        self.device = device
        self.seq_len = seq_len
        self.num_steps = num_steps
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.output_dim = output_dim
        self.initializer_range = 0.02
        self.num_layers=num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        # Gated Temporal Attractor Cell
        # Create multi-layer bidirectional DyFAIPCell stacks
        self.layers = nn.ModuleList()
        for layer in range(self.num_layers):
            # Keep input_size constant at input_dim for all layers
            # This allows freq/last_values to remain consistent
            self.layers.append(nn.ModuleList([
                DyFAIPCell(self.input_size, self.hidden_size, self.seq_len),
                DyFAIPCell(self.input_size, self.hidden_size, self.seq_len) if bidirectional else None
            ]))

        # Projection layers to convert hidden states to input_dim for the next layer
        if self.num_layers > 1:
            self.layer_projections = nn.ModuleList()
            for layer in range(self.num_layers - 1):
                # Project from (hidden_dim * num_directions) -> input_dim
                # After averaging over features, we go from [seq*batch, hidden_dim*num_directions]
                # to [seq*batch, input_dim]
                self.layer_projections.append(
                    nn.Linear(self.hidden_size * self.num_directions, self.input_size)
                )
        self.dropout_layer = nn.Dropout(0.2)
        self.F_alpha = nn.Parameter(torch.normal(0.0, self.initializer_range,
                                                 size=(self.input_size, self.hidden_size * 2, 1)))
        self.F_alpha_n_b = nn.Parameter(torch.normal(0.0, self.initializer_range,
                                                     size=(self.input_size, 1)))
        self.F_beta = nn.Linear(self.seq_len, self.hidden_size)
        self.layer_norm1 = nn.LayerNorm([self.input_size, self.seq_len])
        self.layer_norm = nn.LayerNorm([self.input_size, self.hidden_size])
        self.Phi = nn.Linear(self.hidden_size, self.seq_len)
        self.output_phi = nn.Linear(self.seq_len, self.output_dim)
        self.out_norm = nn.LayerNorm([self.input_size, self.seq_len])
        self.PhiOut = nn.Linear(self.seq_len, self.output_dim)
        self.output_layer = ContextConditioned(seq_len=self.seq_len,
                                               input_size=self.input_size,
                                               hidden_size=self.hidden_size,
                                               output_dim=self.output_dim)

    def forward(self, inputs, times, last_values, freqs):
        device = inputs.device
        if self.batch_first:
            batch_size = inputs.size()[0]
            inputs = inputs.permute(1, 0, 2)
            last_values = last_values.permute(1, 0, 2)
            freqs = freqs.permute(1, 0, 2)
            times = times.transpose(0, 1)
        else:
            batch_size = inputs.size()[1]

        seq_len = inputs.size()[0]
        final_h = torch.jit.annotate(List[Tensor], [])

        # Initialize output variables before the loop
        hidden_his = None
        weights_decay = None
        h_bar_weights = None
        c_bar_weights = None
        lambda_weights = None
        frequencies_weights = None
        imputed_inputs = None
        gate_imputs = None
        confidence=None
        # Process through each layer
        layer_inputs = inputs  # [seq_len, batch, input_dim]
        for layer_idx, (f_cell, b_cell) in enumerate(self.layers):
            prev_hidden = torch.zeros((batch_size, self.input_size, self.hidden_size), device=device)
            prev_cell = torch.zeros((batch_size, self.input_size, self.hidden_size), device=device)
            imputed_inputs = torch.jit.annotate(List[Tensor], [])
            hidden_his = torch.jit.annotate(List[Tensor], [])
            weights_decay = torch.jit.annotate(List[Tensor], [])
            h_bar_weights = torch.jit.annotate(List[Tensor], [])
            c_bar_weights = torch.jit.annotate(List[Tensor], [])
            confidence_weights = torch.jit.annotate(List[Tensor], [])

            lambda_weights = torch.jit.annotate(List[Tensor], [])
            frequencies_weights = torch.jit.annotate(List[Tensor], [])
            gate_imputs = torch.jit.annotate(List[Tensor], [])

            # Forward pass
            for i in range(seq_len):
                (prev_hidden, prev_cell, h_bar,  c_bar,
                 conf_f, freq, lambda_freq, dyfaip_freq, gate_f,
                 imputed_x) = f_cell(
                    prev_hidden, prev_cell,
                    layer_inputs[i], times[i],
                    last_values[i], freqs[i]
                )
                hidden_his += [prev_hidden]
                imputed_inputs += [imputed_x]
                weights_decay += [freq]
                h_bar_weights += [h_bar]
                c_bar_weights += [c_bar]
                confidence_weights+=[conf_f]
                lambda_weights += [lambda_freq]
                frequencies_weights += [dyfaip_freq]
                gate_imputs += [gate_f]

            imputed_inputs = torch.stack(imputed_inputs)
            hidden_his = torch.stack(hidden_his)
            weights_decay = torch.stack(weights_decay)
            h_bar_weights = torch.stack(h_bar_weights)
            c_bar_weights = torch.stack(c_bar_weights)
            confidence_weights= torch.stack(confidence_weights)
            lambda_weights = torch.stack(lambda_weights)
            frequencies_weights = torch.stack(frequencies_weights)
            gate_imputs = torch.stack(gate_imputs)
            # Bidirectional backward pass
            if self.bidirectional:
                second_hidden = torch.zeros((batch_size, self.input_size, self.hidden_size), device=device)
                second_cell = torch.zeros((batch_size, self.input_size, self.hidden_size), device=device)
                sc_inputs = torch.flip(layer_inputs, [0])
                sc_times = torch.flip(times, [0])
                imputed_inputs_sec = torch.jit.annotate(List[Tensor], [])
                second_hidden_his = torch.jit.annotate(List[Tensor], [])
                weights_decay_b = torch.jit.annotate(List[Tensor], [])
                h_bar_weights_b = torch.jit.annotate(List[Tensor], [])
                c_bar_weights_b = torch.jit.annotate(List[Tensor], [])
                lambda_weights_b = torch.jit.annotate(List[Tensor], [])
                frequencies_weights_b = torch.jit.annotate(List[Tensor], [])
                confidence_weights_b = torch.jit.annotate(List[Tensor], [])
                gate_imputs_b = torch.jit.annotate(List[Tensor], [])
                for i in range(seq_len):
                    time = sc_times[i]
                    (second_hidden, second_cell, h_bar_b,
                     c_bar_b, conf_b, freq_b, lambda_freq_b, dyfaip_freq_b,
                     gate_b, imputed_x_b) = b_cell(
                        second_hidden, second_cell,
                        sc_inputs[i], time,
                        last_values[i], freqs[i]
                    )
                    second_hidden_his += [second_hidden]
                    imputed_inputs_sec += [imputed_x_b]
                    weights_decay_b += [freq_b]
                    h_bar_weights_b += [h_bar_b]
                    c_bar_weights_b += [c_bar_b]
                    lambda_weights_b += [lambda_freq_b]
                    frequencies_weights_b += [dyfaip_freq_b]
                    confidence_weights_b += [conf_b]
                    gate_imputs_b += [gate_b]

                imputed_inputs_sec = torch.stack(imputed_inputs_sec)
                second_hidden_his = torch.stack(second_hidden_his)
                weights_decay_b = torch.stack(weights_decay_b)
                h_bar_weights_b = torch.stack(h_bar_weights_b)
                c_bar_weights_b = torch.stack(c_bar_weights_b)
                confidence_weights_b = torch.stack(confidence_weights_b)
                lambda_weights_b = torch.stack(lambda_weights_b)
                frequencies_weights_b = torch.stack(frequencies_weights_b)
                gate_imputs_b = torch.stack(gate_imputs_b)
                # Flip backward results back to forward order
                imputed_inputs_sec = torch.flip(imputed_inputs_sec, [0])
                second_hidden_his = torch.flip(second_hidden_his, [0])
                weights_decay_b = torch.flip(weights_decay_b, [0])
                h_bar_weights_b = torch.flip(h_bar_weights_b, [0])
                c_bar_weights_b = torch.flip(c_bar_weights_b, [0])
                confidence_weights_b = torch.flip(confidence_weights_b, [0])
                lambda_weights_b = torch.flip(lambda_weights_b, [0])
                frequencies_weights_b = torch.flip(frequencies_weights_b, [0])
                weights_decay = torch.cat((weights_decay, weights_decay_b), dim=-1)
                h_bar_weights = torch.cat((h_bar_weights, h_bar_weights_b), dim=-1)
                c_bar_weights = torch.cat((c_bar_weights, c_bar_weights_b), dim=-1)
                confidence_weights = torch.cat((confidence_weights, confidence_weights_b), dim=-1)
                lambda_weights = torch.cat((lambda_weights, lambda_weights_b), dim=2)
                gate_imputs = torch.cat((gate_imputs, gate_imputs), dim=2)

                frequencies_weights = torch.cat((frequencies_weights.squeeze(-1),
                                                 frequencies_weights_b.squeeze(-1)), dim=2)
                hidden_his = torch.cat((hidden_his, second_hidden_his), dim=-1)
                imputed_inputs = torch.cat((imputed_inputs, imputed_inputs_sec), dim=2)

            final_h.append(hidden_his)

            # Apply dropout except for the last layer
            if self.dropout_layer is not None and layer_idx < len(self.layers) - 1:
                layer_inputs = self.dropout_layer(layer_inputs)

            # Prepare output for the next layer
            if layer_idx < len(self.layers) - 1:
                seq_len_out = hidden_his.size(0)
                batch_out = hidden_his.size(1)
                features = hidden_his.size(2)
                hidden_combined = hidden_his.size(3)
                # Pool across features: average the hidden states for each feature
                pooled_hidden = hidden_his.mean(dim=2)  # Average over features
                # Reshape: [seq_len, batch, hidden_dim*num_directions]
                #       -> [seq_len*batch, hidden_dim*num_directions]
                hidden_reshaped = pooled_hidden.reshape(seq_len_out * batch_out, hidden_combined)
                # Project to input_dim size
                projected = self.layer_projections[layer_idx](hidden_reshaped)
                # Reshape back: [seq_len, batch, input_dim]
                layer_inputs = projected.reshape(seq_len_out, batch_out, self.input_size)

        # Ensure variables are defined before final processing
        if hidden_his is None or imputed_inputs is None or weights_decay is None or \
                h_bar_weights is None or c_bar_weights is None:
            raise RuntimeError("No layers were processed in forward pass")

        if self.batch_first:
            hidden_his = final_h[-1].permute(1, 0, 2, 3)
            imputed_inputs = imputed_inputs.permute(1, 0, 2, 3)
            weights_decay = weights_decay.permute(1, 0, 2, 3)
            h_bar_weights = h_bar_weights.permute(1, 0, 2, 3)
            c_bar_weights = c_bar_weights.permute(1, 0, 2, 3)
            confidence_weights = confidence_weights.permute(1, 0, 2, 3)
            lambda_weights = lambda_weights.permute(1, 0, 2)
            gate_imputs=gate_imputs.permute(1, 0, 2, 3)
            frequencies_weights = frequencies_weights.permute(1, 0, 2)
        #print("gate_imputs, imputed_inputs", gate_imputs.shape, imputed_inputs.shape)
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", hidden_his, self.F_alpha) + self.F_alpha_n_b)
        alphas = alphas.reshape(alphas.size(0), alphas.size(2), alphas.size(1) * alphas.size(-1))
        x = self.layer_norm1(alphas)  # [B, D, L]
        x = self.F_beta(x)  # [B, D, 4H]
        x = self.Phi(self.layer_norm(x))
        out = self.output_layer(x)
        return (out, weights_decay, h_bar_weights, c_bar_weights,confidence_weights,
                lambda_weights, frequencies_weights, gate_imputs, imputed_inputs)

class GTACMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, diff_step,num_layers, output_dim, device):
        super(GTACMNetwork, self).__init__()
        # hidden dimensions
        self.device = device
        self.seq_len = seq_len
        self.input_size = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.num_steps = diff_step
        # Gated Temporal Attractor Cell
        self.gtacm = GTACM(self.input_size, self.hidden_size, self.seq_len,
                           self.num_steps, self.num_layers, self.output_dim,
                           self.device)

    def forward(self, historic_features, timestamp, last_features, features_freqs):
        # Temporal features embedding

        (outputs, weights_decay, h_bar_weights, c_bar_weights,confidence_weights,
         lambda_weights, frequencies_weights, gate_imputs, imputed_inputs) = self.gtacm(historic_features, timestamp,
                                                                           last_features, features_freqs)
        return (outputs, weights_decay, h_bar_weights, c_bar_weights,confidence_weights,
                lambda_weights, frequencies_weights, gate_imputs, imputed_inputs, imputed_inputs.mean(axis=2))