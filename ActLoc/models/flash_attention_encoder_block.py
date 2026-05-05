import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging
from typing import Optional, Union, Callable
import sys

from flash_attn.modules.mha import MHA as FlashMHA


# Configure basic logging to stdout
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
log_handler = logging.StreamHandler(sys.stdout) # Log to stdout like print
log_handler.setFormatter(log_formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Set default level
if logger.hasHandlers(): 
    logger.handlers.clear() # Clear potential conflicts
logger.addHandler(log_handler)


# --- Re-padding Helper Function ---
def _manual_pad_input(unpadded_input: torch.Tensor, indices: torch.Tensor, batch_size: int, max_seqlen: int) -> torch.Tensor:
    """ Manually re-pads unpadded input using indices. """
    dim = unpadded_input.shape[-1]
    padded_output_flat = torch.zeros(batch_size * max_seqlen, dim, dtype=unpadded_input.dtype, device=unpadded_input.device)
    padded_output_flat.scatter_(0, indices.long().unsqueeze(-1).expand_as(unpadded_input), unpadded_input)
    return padded_output_flat.reshape(batch_size, max_seqlen, dim)

# --- Helper Activation Function ---
def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """ Returns the activation function corresponding to the input string. """
    if activation == "relu": return F.relu
    elif activation == "gelu": return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


# --- FlashAttention Transformer Encoder Layer (Accepts cu_seqlens) ---
class FlashTransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer using flash_attn MHA for self-attention.
    Accepts padded input along with cu_seqlens and max_seqlen for variable length handling.
    Performs unpad/repad internally based on cu_seqlens.
    Requires input dtype float16 or bfloat16.
    (Docstring truncated for brevity)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__( self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu, layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False, bias: bool = True, device = None, dtype = None ) -> None:
        # --- Dtype Handling ---
        layer_dtype = dtype
        if layer_dtype is None:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported(): layer_dtype = torch.bfloat16
            else: layer_dtype = torch.float16
        elif layer_dtype not in [torch.float16, torch.bfloat16]:
             raise ValueError(f"FlashTransformerEncoderLayer requires dtype float16 or bfloat16, got {layer_dtype}")

        factory_kwargs = {'device': device, 'dtype': layer_dtype}
        super().__init__()

        self.d_model = d_model; self.nhead = nhead; self.batch_first = batch_first; self.norm_first = norm_first

        # --- Submodules ---
        self.self_attn = FlashMHA( embed_dim=d_model, num_heads=nhead, cross_attn=False, dropout=dropout, use_flash_attn=True, causal=False, return_residual=False, qkv_proj_bias=bias, out_proj_bias=bias, **factory_kwargs )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if isinstance(activation, str): self.activation = _get_activation_fn(activation)
        else: self.activation = activation


    def forward( self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, cu_seqlens: Optional[torch.Tensor] = None, max_seqlen: Optional[int] = None, is_causal: bool = False ) -> torch.Tensor:
        # --- Input Checks & Dtype ---
        if src_mask is not None: logging.log(logging.DEBUG, "src_mask ignored.")
        if is_causal: raise NotImplementedError("is_causal not supported.")
        if cu_seqlens is not None and max_seqlen is None:
             if self.batch_first: max_seqlen = src.shape[1]
             else: raise ValueError("max_seqlen must be provided with cu_seqlens if batch_first=False")

        target_dtype = next(self.parameters()).dtype
        if src.dtype != target_dtype:
            # logging.log(logging.WARNING, f"Input tensor dtype {src.dtype} != layer dtype {target_dtype}. Casting input.")
            src = src.to(target_dtype)

        # --- Batch First Handling ---
        x = src
        if not self.batch_first: x = x.transpose(0, 1)

        # --- Main Logic ---
        if self.norm_first:
            normed_x = self.norm1(x)
            attn_output_padded = self._sa_block(normed_x, cu_seqlens, max_seqlen)
            x = x + self.dropout1(attn_output_padded)
            x = x + self.dropout2(self._ff_block(self.norm2(x)))
        else: # Post-Norm
            attn_output_padded = self._sa_block(x, cu_seqlens, max_seqlen)
            x = self.norm1(x + self.dropout1(attn_output_padded))
            x = self.norm2(x + self.dropout2(self._ff_block(x)))

        if not self.batch_first: x = x.transpose(0, 1)
        return x

    # --- Internal Blocks ---
    def _sa_block(self, x_padded: torch.Tensor, cu_seqlens: Optional[torch.Tensor], max_seqlen_in: Optional[int]) -> torch.Tensor:
        """ Applies self-attention, handling unpadding/repadding internally using cu_seqlens. """
        batch_size, current_max_s, dim = x_padded.shape
        max_seqlen = current_max_s if max_seqlen_in is None else max_seqlen_in
        needs_unpad = cu_seqlens is not None

        if needs_unpad:
            cu_seqlens = cu_seqlens.to(device=x_padded.device, dtype=torch.int32)
            if len(cu_seqlens) != batch_size + 1: raise ValueError("Invalid cu_seqlens length")

            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            ar = torch.arange(max_seqlen, device=x_padded.device)[None, :]
            keep_mask = ar < seq_lens[:, None]
            indices = torch.nonzero(keep_mask.flatten(), as_tuple=False).flatten()
            x_unpadded = x_padded.reshape(-1, dim)[indices]

            if x_unpadded.shape[0] == 0: return torch.zeros_like(x_padded)

            attn_output_unpadded = self.self_attn(x_unpadded, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            attn_output_padded = _manual_pad_input(attn_output_unpadded, indices, batch_size, max_seqlen)
        else: # Fixed length batch
            cu_seqlens = torch.arange( 0, (batch_size + 1) * max_seqlen, step=max_seqlen, dtype=torch.int32, device=x_padded.device )
            x_unpadded = x_padded.reshape(-1, dim)
            attn_output_unpadded = self.self_attn(x_unpadded, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            attn_output_padded = attn_output_unpadded.reshape(batch_size, max_seqlen, dim)
        return attn_output_padded

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

# --- FlashAttention Transformer Encoder Block ---
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class FlashTransformerEncoder(nn.Module):
    """A stack of N FlashTransformerEncoderLayer layers. Accepts cu_seqlens/max_seqlen."""
    __constants__ = ['norm']
    def __init__(self, encoder_layer: FlashTransformerEncoderLayer, num_layers: int, norm: Optional[nn.LayerNorm] = None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward( self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, cu_seqlens: Optional[torch.Tensor] = None, max_seqlen: Optional[int] = None, is_causal: Optional[bool] = None ) -> torch.Tensor:
        if is_causal is not None and is_causal: logging.warning("is_causal ignored.")
        if mask is not None: logging.warning("mask ignored.")

        output = src
        for mod in self.layers:
            output = mod( src=output, src_mask=None, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen )
        if self.norm is not None: output = self.norm(output)
        return output