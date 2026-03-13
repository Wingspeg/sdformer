import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class AdaptiveNorm(nn.Module):
    """
    Innovation 4: Adaptive Normalization - Automatically selects normalization strategy based on spike features
    
    Automatically switches between two normalization methods based on spike features (kurtosis and spike ratio):
    - Anomaly-Aware Normalization (median + MAD): for data with frequent spikes
    - Standard RevIN (mean + std): for smooth data
    
    Uses learnable thresholds and soft switching mechanism to maintain differentiability
    """
    def __init__(self, use_standard_norm=False):
        super().__init__()
        self.use_standard_norm = use_standard_norm
        
        if not use_standard_norm:
            # Learnable threshold parameters
            # threshold_k: kurtosis threshold, initialized to 3.0 (normal distribution baseline)
            # threshold_s: spike ratio threshold, initialized to 0.01 (1% spike ratio)
            self.threshold_k = nn.Parameter(torch.tensor(3.0))
            self.threshold_s = nn.Parameter(torch.tensor(0.01))
            
            # Gate network: maps spike features to mixing weight alpha
            # Input: [kurtosis, spike_ratio] (2D)
            # Output: alpha (1D)
            self.gate = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            )
            nn.init.constant_(self.gate[-1].bias, -2.0)

    def compute_spike_features(self, x):
        """
        Compute spike features of the sequence
        
        Args:
            x: [batch, seq_len, n_vars]
            
        Returns:
            kurtosis: [batch, 1, n_vars] kurtosis
            spike_ratio: [batch, 1, n_vars] spike ratio
        """
        # Compute mean and standard deviation
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False) + 1e-5
        
        # Compute kurtosis
        # kurtosis = E[(X - μ)^4] / σ^4 - 3
        centered = x - mean
        kurtosis = ((centered ** 4).mean(dim=1, keepdim=True) / (std ** 4)) - 3.0
        
        # Compute spike ratio
        # Ratio of points exceeding mean ± 3*std
        z_scores = centered.abs() / std
        spikes = (z_scores > 3.0).float()
        spike_ratio = spikes.mean(dim=1, keepdim=True)
        
        return kurtosis, spike_ratio

    def forward(self, x):
        """
        x: [batch, seq_len, n_vars]
        Returns: normalized x, center value, scale value
        """
        batch_size, seq_len, n_vars = x.shape
        
        if self.use_standard_norm:
            # Ablation: use standard RevIN (mean + std)
            means = x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_norm = (x - means) / stdev
            return x_norm, means, stdev
        
        # Normal mode: adaptive normalization
        # Compute spike features
        kurtosis, spike_ratio = self.compute_spike_features(x)
        
        # Concatenate features and pass through gate network
        # [batch, 1, n_vars, 2]
        spike_features = torch.stack([kurtosis, spike_ratio], dim=-1)
        
        # Reshape to [batch * n_vars, 2] for gate network
        spike_features_flat = spike_features.view(-1, 2)
        
        alpha_logits = self.gate(spike_features_flat)
        alpha_soft = torch.sigmoid(alpha_logits)
        alpha = alpha_soft.view(batch_size, 1, n_vars)
        
        # Compute standard normalization statistics (mean, std)
        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()

        # Compute robust normalization statistics (median, MAD)
        median = x.median(dim=1, keepdim=True)[0].detach()
        mad = ((x - median).abs()).median(dim=1, keepdim=True)[0].detach()
        robust_scale = mad * 1.4826 + 1e-5

        # Soft switching: mix two normalization strategies based on alpha
        center = alpha * median + (1 - alpha) * means
        scale = alpha * robust_scale + (1 - alpha) * stdev

        # Normalization
        x_norm = (x - center) / scale

        return x_norm, center, scale



class AdaptivePatchEmbedding(nn.Module):
    """
    Innovation 1+2: Multi-scale Patch Embedding with Adaptive Time-Step Reweighting (ATSR)
    ATSR is implemented via softmax weighting without changing sequence length, ensuring fixed patch_num
    """

    def __init__(self, d_model, seq_len, patch_sizes=[8, 16, 32], dropout=0.1, use_uniform_atsr=False):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_uniform_atsr = use_uniform_atsr

        self.projections = nn.ModuleList([
            nn.Linear(p, d_model) for p in patch_sizes
        ])

        if not use_uniform_atsr:
            # Use learnable ATSR weights
            self.atsr_weights = nn.ParameterList([
                nn.Parameter(torch.zeros(p)) for p in patch_sizes
            ])
        # If use_uniform_atsr=True, use uniform weights (no parameters needed)

        self.dropout = nn.Dropout(dropout)

        self.patch_nums = []
        for patch_size in patch_sizes:
            stride = patch_size // 2
            pn = max(1, (seq_len - patch_size) // stride + 1)
            self.patch_nums.append(pn)

        self.pos_embeddings = nn.ModuleList([
            nn.Embedding(pn + 1, d_model) for pn in self.patch_nums
        ])

    def forward(self, x):
        """x: [bs*nvars, seq_len]"""
        B, L = x.shape
        outputs = []

        for i, patch_size in enumerate(self.patch_sizes):
            stride = patch_size // 2
            patch_num = self.patch_nums[i]

            patches = x.unfold(dimension=1, size=patch_size, step=stride)
            patches = patches[:, :patch_num, :]

            if self.use_uniform_atsr:
                # Ablation: use uniform weights (equal weighting)
                patches_weighted = patches
            else:
                # Normal mode: use learnable ATSR weights
                atsr_w = torch.softmax(self.atsr_weights[i], dim=0)
                patches_weighted = patches * atsr_w.unsqueeze(0).unsqueeze(0)

            out = self.projections[i](patches_weighted)

            pos = torch.arange(patch_num, device=x.device)
            out = out + self.pos_embeddings[i](pos).unsqueeze(0)
            out = self.dropout(out)
            outputs.append(out)

        return outputs




class AdaptiveGatingFusion(nn.Module):
    """Innovation 3: Adaptive Gating Fusion for multi-scale features"""

    def __init__(self, d_model, n_vars, n_scales=3, dropout=0.1, use_equal_fusion=False):
        super().__init__()
        self.n_scales = n_scales
        self.use_equal_fusion = use_equal_fusion

        if not use_equal_fusion:
            # Normal mode: use adaptive gating
            self.gate_mlp = nn.Sequential(
                nn.Linear(d_model * n_scales, d_model),
                nn.GELU(),
                nn.Linear(d_model, n_scales),
            )
            # Initialize bias to favor mid-scale
            self.gate_mlp[-1].bias.data = torch.tensor([-1.0, 1.0, 0.0])

        self.dropout = nn.Dropout(dropout)

    def forward(self, scale_features):
        """scale_features: list of [bs, nvars, d_model, patch_num_i]"""
        min_patches = min(f.shape[-1] for f in scale_features)
        aligned = [f[..., :min_patches] for f in scale_features]

        if self.use_equal_fusion:
            # Ablation: equal-weight averaging fusion
            fused = sum(aligned) / self.n_scales
        else:
            # Normal mode: adaptive gating fusion
            pooled = [f.mean(dim=-1) for f in scale_features]
            concat = torch.cat(pooled, dim=-1)
            gate_weights = self.gate_mlp(concat)
            gate_weights = F.softmax(gate_weights, dim=-1)

            fused = sum(
                gate_weights[:, :, i].unsqueeze(-1).unsqueeze(-1) * aligned[i]
                for i in range(self.n_scales)
            )

        return fused


class Model(nn.Module):
    """SDFormer: Scale-aware Dynamic Transformer"""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        patch_sizes = getattr(configs, 'patch_sizes', [8, 16, 32])
        # Convert string to list of integers if needed
        if isinstance(patch_sizes, str):
            patch_sizes = [int(p.strip()) for p in patch_sizes.split(',')]
        self.patch_sizes = patch_sizes
        self.n_scales = len(patch_sizes)

        # Get ablation experiment parameters
        use_shared_encoder = getattr(configs, 'use_shared_encoder', 0)
        use_uniform_atsr = getattr(configs, 'use_uniform_atsr', 0)
        use_equal_fusion = getattr(configs, 'use_equal_fusion', 0)
        use_standard_norm = getattr(configs, 'use_standard_norm', 0)

        self.use_shared_encoder = bool(use_shared_encoder)

        # Adaptive Patch Embedding (supports ablation: use_uniform_atsr)
        self.adaptive_patch_embed = AdaptivePatchEmbedding(
            d_model=configs.d_model,
            seq_len=configs.seq_len,
            patch_sizes=patch_sizes,
            dropout=configs.dropout,
            use_uniform_atsr=bool(use_uniform_atsr)
        )

        # Adaptive normalization module (supports ablation: use_standard_norm)
        self.adaptive_norm = AdaptiveNorm(use_standard_norm=bool(use_standard_norm))

        # Encoder: supports ablation (independent encoders vs shared encoder)
        if self.use_shared_encoder:
            # Ablation: use shared encoder
            self.shared_encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor,
                                          attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for _ in range(configs.e_layers)
                ],
                norm_layer=nn.Sequential(
                    Transpose(1, 2),
                    nn.BatchNorm1d(configs.d_model),
                    Transpose(1, 2)
                )
            )
            self.encoders = None
        else:
            # Normal mode: independent encoder for each scale
            self.encoders = nn.ModuleList([
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                FullAttention(False, configs.factor,
                                              attention_dropout=configs.dropout,
                                              output_attention=False),
                                configs.d_model, configs.n_heads),
                            configs.d_model,
                            configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation
                        ) for _ in range(configs.e_layers)
                    ],
                    norm_layer=nn.Sequential(
                        Transpose(1, 2),
                        nn.BatchNorm1d(configs.d_model),
                        Transpose(1, 2)
                    )
                ) for _ in range(self.n_scales)
            ])
            self.shared_encoder = None

        # Adaptive gating fusion (supports ablation: use_equal_fusion)
        self.adaptive_gating_fusion = AdaptiveGatingFusion(
            d_model=configs.d_model,
            n_vars=configs.enc_in,
            n_scales=self.n_scales,
            dropout=configs.dropout,
            use_equal_fusion=bool(use_equal_fusion)
        )

        self.fused_patch_num = min(self.adaptive_patch_embed.patch_nums)
        self.head_nf = configs.d_model * self.fused_patch_num

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.head = nn.Sequential(
                nn.Dropout(configs.dropout),
                nn.Linear(self.head_nf, configs.pred_len)
            )
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Use adaptive normalization
        x_norm, center, scale = self.adaptive_norm(x_enc)

        bs, seq_len, n_vars = x_norm.shape
        x_flat = x_norm.permute(0, 2, 1).reshape(bs * n_vars, seq_len)

        scale_embeds = self.adaptive_patch_embed(x_flat)

        scale_outputs = []
        if self.use_shared_encoder:
            # Ablation: use shared encoder for all scales
            for embed in scale_embeds:
                enc_out, _ = self.shared_encoder(embed)
                patch_num = enc_out.shape[1]
                enc_out = enc_out.reshape(bs, n_vars, patch_num, self.d_model)
                enc_out = enc_out.permute(0, 1, 3, 2)
                scale_outputs.append(enc_out)
        else:
            # Normal mode: use independent encoder for each scale
            for embed, encoder in zip(scale_embeds, self.encoders):
                enc_out, _ = encoder(embed)
                patch_num = enc_out.shape[1]
                enc_out = enc_out.reshape(bs, n_vars, patch_num, self.d_model)
                enc_out = enc_out.permute(0, 1, 3, 2)
                scale_outputs.append(enc_out)

        fused = self.adaptive_gating_fusion(scale_outputs)
        fused = fused[..., :self.fused_patch_num]

        fused_flat = fused.reshape(bs, n_vars, self.head_nf)

        dec_out = self.head(fused_flat)
        dec_out = dec_out.permute(0, 2, 1)

        # Denormalization
        dec_out = dec_out * (scale[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (center[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None