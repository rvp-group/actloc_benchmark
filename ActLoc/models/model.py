import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Callable
import sys

from .flash_attention_encoder_block import FlashTransformerEncoderLayer, FlashTransformerEncoder
from .flash_cross_attention_block import FlashCrossAttentionLayer, FlashCrossAttentionBlock

# --- Configure Logging ---
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(log_formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers(): 
    logger.handlers.clear()
logger.addHandler(log_handler)

# --- Helper to Regenerate Masks (Needed for Mean Pooling) ---
def _create_boolean_mask_from_cu_seqlens(cu_seqlens: torch.Tensor, max_len: int) -> Optional[torch.Tensor]:
    """ Regenerates boolean padding mask (True=PAD) from cu_seqlens. Returns None if cu_seqlens is None. """
    if cu_seqlens is None:
        # If no cu_seqlens, assume no padding needed for pooling (all tokens valid)
        return None

    batch_size = len(cu_seqlens) - 1
    if batch_size == 0: return torch.empty((0, max_len), dtype=torch.bool, device=cu_seqlens.device) # Handle empty batch

    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    # Handle potential edge case where max_len derived from data is smaller than indicated by cu_seqlens
    # Usually max_len should come from collate_fn and be >= seq_lens.max()
    effective_max_len = max_len
    if seq_lens.numel() > 0 and max_len < seq_lens.max().item():
        logging.warning(f"max_len ({max_len}) is less than max length from cu_seqlens ({seq_lens.max().item()}). Adjusting mask size.")
        effective_max_len = seq_lens.max().item()

    mask = torch.ones(batch_size, effective_max_len, dtype=torch.bool, device=cu_seqlens.device) # Init with True (PAD)
    ar = torch.arange(effective_max_len, device=cu_seqlens.device)[None, :] # (1, S_eff)
    valid_token_mask = ar < seq_lens[:, None] # Compare against actual lengths
    mask[valid_token_mask] = False # Set valid positions to False (VALID)

    # Ensure final mask has the expected max_len dimension
    if effective_max_len > max_len: mask = mask[:, :max_len]
    elif effective_max_len < max_len: mask = F.pad(mask, (0, max_len - effective_max_len), value=True) # Pad with True

    return mask

# --- Helper Activation Function ---
def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """ Returns the activation function corresponding to the input string. """
    if activation == "relu": return F.relu
    elif activation == "gelu": return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

# --- Main Model using Mean Pooling ---
class TwoHeadTransformer(nn.Module):
    """
    Transformer model using FlashAttention for self- and cross-attention blocks
    and Mean Pooling for aggregation.
    Accepts cu_seqlens and max_len for variable length inputs.
    Requires model dtype to be float16 or bfloat16.
    """
    def __init__(self, num_classes=2, d_model=128, num_encoder_layers=4, num_cross_layers=4,
                 nhead=4, dim_feedforward=256, dropout=0.1,
                 activation='relu', norm_first=False, layer_norm_eps=1e-5, bias=True):
        super().__init__()

        self.num_classes = num_classes
        self.d_model = d_model
        self.final_output_dim = num_classes * 6 * 18

        # Input projections
        self.pose_proj = nn.Linear(7, d_model, bias=bias)
        self.pc_proj = nn.Linear(6, d_model, bias=bias)
        # Factory kwargs for device/dtype inheritance (set later by .to())
        factory_kwargs = {'device': None, 'dtype': None}

        # --- Flash Self-Attention Encoders ---
        flash_encoder_layer = FlashTransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, batch_first=True,
            norm_first=norm_first, bias=bias, layer_norm_eps=layer_norm_eps,
            **factory_kwargs
        )
        self.pose_encoder = FlashTransformerEncoder(flash_encoder_layer, num_encoder_layers)
        self.pc_encoder = FlashTransformerEncoder(flash_encoder_layer, num_encoder_layers)

        # --- Flash Cross-Attention Block ---
        flash_cross_layer = FlashCrossAttentionLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, bias=bias,
            layer_norm_eps=layer_norm_eps, **factory_kwargs
        )
        self.cross_block = FlashCrossAttentionBlock(flash_cross_layer, num_layers=num_cross_layers)

        # --- Mean Pooling Aggregation & Classifier ---
        self.fusion_linear = nn.Linear(2 * d_model, d_model, bias=bias)
        self.fusion_activation = _get_activation_fn(activation)
        self.fusion_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.classifier = nn.Linear(d_model, self.final_output_dim, bias=bias)

    def _masked_mean_pool(self, features: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """ Performs mean pooling over non-padded tokens. mask=True means PAD. """
        if mask is None: # Assume all tokens are valid
            return features.mean(dim=1)
        else:
            # Ensure mask is boolean, True=PAD
            mask = mask.bool()
            # Invert mask for summing (True=KEEP)
            keep_mask = (~mask).unsqueeze(-1).to(features.dtype) # (B, S, 1)
            # Sum valid tokens
            summed_features = (features * keep_mask).sum(dim=1) # (B, D)
            # Count valid tokens per sequence, clamp to avoid division by zero
            valid_counts = keep_mask.sum(dim=1).clamp(min=1.0) # (B, 1)
            return summed_features / valid_counts # (B, D)

    def forward(self,
                pose_input: torch.Tensor,   # (B, M, 7) Raw
                pc_input: torch.Tensor,     # (B, N, 6) Raw
                pose_cu_seqlens: Optional[torch.Tensor] = None, # (B+1,)
                pose_max_len: Optional[int] = None,             # M
                pc_cu_seqlens: Optional[torch.Tensor] = None,   # (B+1,)
                pc_max_len: Optional[int] = None                # N
               ) -> torch.Tensor:

        # --- Input Checks & Projection ---
        expected_dtype = next(self.parameters()).dtype
        if expected_dtype not in [torch.float16, torch.bfloat16]:
             logging.warning(f"Model dtype is {expected_dtype}, FlashAttention requires float16/bfloat16.")
        if pose_input.dtype != expected_dtype: pose_input = pose_input.to(expected_dtype)
        if pc_input.dtype != expected_dtype: pc_input = pc_input.to(expected_dtype)

        pose_emb = self.pose_proj(pose_input)  # (B, M_max, d_model)
        pc_emb = self.pc_proj(pc_input)    # (B, N_max, d_model)

        batch_size = pose_emb.shape[0]
        # Infer max lengths if not provided
        inferred_pose_max_len = pose_emb.shape[1]
        inferred_pc_max_len = pc_emb.shape[1]
        pose_max_len = pose_max_len if pose_max_len is not None else inferred_pose_max_len
        pc_max_len = pc_max_len if pc_max_len is not None else inferred_pc_max_len
        # Ensure consistency if both are provided
        if pose_max_len != inferred_pose_max_len: logging.warning(f"pose_max_len arg {pose_max_len} != pose_emb shape {inferred_pose_max_len}")
        if pc_max_len != inferred_pc_max_len: logging.warning(f"pc_max_len arg {pc_max_len} != pc_emb shape {inferred_pc_max_len}")


        # --- Self-Attention Encoding ---
        pose_encoded = self.pose_encoder(pose_emb, cu_seqlens=pose_cu_seqlens, max_seqlen=pose_max_len)
        pc_encoded = self.pc_encoder(pc_emb, cu_seqlens=pc_cu_seqlens, max_seqlen=pc_max_len)


        # --- Cross-Attention ---
        pose_encoded, pc_encoded = self.cross_block(
            pose_encoded, pc_encoded,
            pose_cu_seqlens, pose_max_len,
            pc_cu_seqlens, pc_max_len
        )


        # --- Regenerate Boolean Masks (True=PAD) for Pooling ---
        pose_padding_mask = _create_boolean_mask_from_cu_seqlens(pose_cu_seqlens, pose_max_len) if pose_cu_seqlens is not None else None
        pc_padding_mask = _create_boolean_mask_from_cu_seqlens(pc_cu_seqlens, pc_max_len) if pc_cu_seqlens is not None else None

        # --- Mean Pooling ---
        pose_pooled = self._masked_mean_pool(pose_encoded, pose_padding_mask) # (B, D)
        pc_pooled = self._masked_mean_pool(pc_encoded, pc_padding_mask)       # (B, D)

        # --- Concatenate & Fuse ---
        combined_pooled = torch.cat([pose_pooled, pc_pooled], dim=-1) # (B, 2*D)
        fused = self.fusion_linear(combined_pooled)                   # (B, D)
        fused = self.fusion_activation(fused)                         # Apply activation
        fused = self.fusion_norm(fused)                               # Apply norm

        # --- Classification ---
        logits_flat = self.classifier(fused) # (B, C * 108)

        # --- Reshape Output ---
        final_logits = logits_flat.view(batch_size, self.num_classes, 6, 18)


        return final_logits