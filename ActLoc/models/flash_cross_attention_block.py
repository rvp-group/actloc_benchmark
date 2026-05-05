import torch
import torch.nn as nn
import logging
import sys 

from flash_attn.modules.mha import FlashCrossAttention
from .flash_attention_encoder_block import _manual_pad_input, _get_clones

# Configure basic logging to stdout
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(log_formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers(): 
    logger.handlers.clear()
logger.addHandler(log_handler)


class FlashCrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer using FlashCrossAttention (flash_attn >= 2.x).
    Accepts padded inputs and cu_seqlens/max_len for variable lengths.
    Performs QKV projection and unpad/repad internally.
    Requires inputs with dtype fp16 or bfloat16.
    Follows a Post-LN structure for residual connections.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, bias: bool = True, layer_norm_eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        # --- Dtype Handling & Factory Kwargs ---
        layer_dtype = dtype
        if layer_dtype is None:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported(): layer_dtype = torch.bfloat16
            else: layer_dtype = torch.float16
        elif layer_dtype not in [torch.float16, torch.bfloat16]:
             raise ValueError(f"Requires dtype float16 or bfloat16, got {layer_dtype}")
        factory_kwargs = {'device': device, 'dtype': layer_dtype}
        self.d_model = d_model; self.nhead = nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.head_dim = d_model // nhead

        # --- Manual Projections ---
        self.q_proj_pose = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.kv_proj_pc = nn.Linear(d_model, 2 * d_model, bias=bias, **factory_kwargs)
        self.out_proj_pose = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.q_proj_pc = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.kv_proj_pose = nn.Linear(d_model, 2 * d_model, bias=bias, **factory_kwargs)
        self.out_proj_pc = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        # --- Attention Cores ---
        softmax_scale = 1.0 / (self.head_dim ** 0.5)
        self.inner_attn_pose_pc = FlashCrossAttention( softmax_scale=softmax_scale, attention_dropout=dropout, causal=False )
        self.inner_attn_pc_pose = FlashCrossAttention( softmax_scale=softmax_scale, attention_dropout=dropout, causal=False )
        # --- Norms & Dropouts ---
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _unpad_input_and_get_indices(self, x_padded, cu_seqlens, max_seqlen):
        """ Helper to unpad and get indices needed for repadding. """
        batch_size, current_max_s, dim = x_padded.shape
        if max_seqlen != current_max_s: max_seqlen = current_max_s # Use actual shape
        cu_seqlens = cu_seqlens.to(device=x_padded.device, dtype=torch.int32)
        if len(cu_seqlens) != batch_size + 1: raise ValueError("Invalid cu_seqlens length")
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        if (seq_lens > max_seqlen).any(): raise ValueError("cu_seqlens imply lengths > max_seqlen")
        ar = torch.arange(max_seqlen, device=x_padded.device)[None, :]
        keep_mask = ar < seq_lens[:, None]
        indices = torch.nonzero(keep_mask.flatten(), as_tuple=False).flatten()
        x_unpadded = x_padded.reshape(-1, dim)[indices]
        return x_unpadded, indices

    def forward(self, pose, pc, pose_cu_seqlens, pose_max_len, pc_cu_seqlens, pc_max_len):
        # --- Input Checks & Dtype ---
        target_dtype = next(self.parameters()).dtype
        if pose.dtype != target_dtype: pose = pose.to(target_dtype)
        if pc.dtype != target_dtype: pc = pc.to(target_dtype)
        batch_size = pose.shape[0]
        assert batch_size == pc.shape[0] and pose.shape[1] == pose_max_len and pc.shape[1] == pc_max_len
        assert pose.shape[2] == self.d_model and pc.shape[2] == self.d_model
        pose_residual = pose; pc_residual = pc

        # --- Unpadding ---
        pose_unpadded, pose_indices = self._unpad_input_and_get_indices(pose, pose_cu_seqlens, pose_max_len)
        pc_unpadded, pc_indices = self._unpad_input_and_get_indices(pc, pc_cu_seqlens, pc_max_len)
        if pose_unpadded.shape[0] == 0 or pc_unpadded.shape[0] == 0:
             logging.warning("Input with zero actual tokens. Applying norm and returning.")
             return self.norm1(pose_residual), self.norm2(pc_residual)

        # --- 1. Pose attends to PC ---
        q_pose = self.q_proj_pose(pose_unpadded)
        kv_pc = self.kv_proj_pc(pc_unpadded)
        q_pose = q_pose.view(-1, self.nhead, self.head_dim)
        kv_pc = kv_pc.view(-1, 2, self.nhead, self.head_dim)
        context_pose_unpadded = self.inner_attn_pose_pc( q_pose, kv_pc, cu_seqlens=pose_cu_seqlens, max_seqlen=pose_max_len, cu_seqlens_k=pc_cu_seqlens, max_seqlen_k=pc_max_len )
        context_pose_unpadded = context_pose_unpadded.view(-1, self.d_model)
        proj_context_pose_unpadded = self.out_proj_pose(context_pose_unpadded)
        attn_pose_pc_padded = _manual_pad_input(proj_context_pose_unpadded, pose_indices, batch_size, pose_max_len) # Use imported helper
        pose = self.norm1(pose_residual + self.dropout1(attn_pose_pc_padded))

        # --- 2. PC attends to Pose ---
        q_pc = self.q_proj_pc(pc_unpadded)
        kv_pose = self.kv_proj_pose(pose_unpadded)
        q_pc = q_pc.view(-1, self.nhead, self.head_dim)
        kv_pose = kv_pose.view(-1, 2, self.nhead, self.head_dim)
        context_pc_unpadded = self.inner_attn_pc_pose( q_pc, kv_pose, cu_seqlens=pc_cu_seqlens, max_seqlen=pc_max_len, cu_seqlens_k=pose_cu_seqlens, max_seqlen_k=pose_max_len )
        context_pc_unpadded = context_pc_unpadded.view(-1, self.d_model)
        proj_context_pc_unpadded = self.out_proj_pc(context_pc_unpadded)
        attn_pc_pose_padded = _manual_pad_input(proj_context_pc_unpadded, pc_indices, batch_size, pc_max_len)
        pc = self.norm2(pc_residual + self.dropout2(attn_pc_pose_padded))

        return pose, pc


# --- Stacked Flash Cross Attention BLOCK ---
class FlashCrossAttentionBlock(nn.Module):
    """
    A stack of N FlashCrossAttentionLayer layers.
    Args:
        cross_attn_layer (FlashCrossAttentionLayer): An instance of the layer to be stacked.
        num_layers (int): The number of layers to stack.
    """
    def __init__(self, cross_attn_layer: FlashCrossAttentionLayer, num_layers: int):
        super().__init__()
        self.layers = _get_clones(cross_attn_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, pose, pc, pose_cu_seqlens, pose_max_len, pc_cu_seqlens, pc_max_len):
        output_pose = pose; output_pc = pc
        for layer in self.layers:
            output_pose, output_pc = layer( output_pose, output_pc, pose_cu_seqlens, pose_max_len, pc_cu_seqlens, pc_max_len )
        return output_pose, output_pc