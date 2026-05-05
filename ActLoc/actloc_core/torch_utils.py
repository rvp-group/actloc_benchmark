import os
import logging
import torch
import torch.nn.functional as F
import numpy as np

from ActLoc.models.model import TwoHeadTransformer

# --- Default Model Parameters ---
DEFAULT_MODEL_PARAMS = {
    "d_model": 256,
    "nhead": 8,
    "num_encoder_layers": 6,
    "num_cross_layers": 4,
    "dropout": 0.1,
    "dim_feedforward": 256 * 4,
}

def create_single_sample_batch_for_inference(pc_features: np.ndarray, pose_features: np.ndarray, device: torch.device, model_dtype=None):
    """Create a single-sample batch for model inference."""
    # Convert to tensors and add batch dimension
    pc_tensor = torch.from_numpy(pc_features).unsqueeze(0)
    pose_tensor = torch.from_numpy(pose_features).unsqueeze(0)
    
    # Apply model dtype if specified
    if model_dtype is not None:
        pc_tensor = pc_tensor.to(device=device, dtype=model_dtype)
        pose_tensor = pose_tensor.to(device=device, dtype=model_dtype)
    else:
        pc_tensor = pc_tensor.to(device)
        pose_tensor = pose_tensor.to(device)
    
    num_points = pc_features.shape[0]
    num_cameras = pose_features.shape[0]
    
    # Create cu_seqlens for FlashAttention
    pc_lengths_tensor = torch.tensor([num_points], dtype=torch.int32, device=device)
    pose_lengths_tensor = torch.tensor([num_cameras], dtype=torch.int32, device=device)
    
    pc_cu_seqlens = F.pad(torch.cumsum(pc_lengths_tensor, dim=0, dtype=torch.int32), (1, 0), value=0)
    pose_cu_seqlens = F.pad(torch.cumsum(pose_lengths_tensor, dim=0, dtype=torch.int32), (1, 0), value=0)
    
    batch = {
        'pc_input': pc_tensor,
        'pose_input': pose_tensor,
        'pc_cu_seqlens': pc_cu_seqlens,
        'pc_max_len': num_points,
        'pose_cu_seqlens': pose_cu_seqlens,
        'pose_max_len': num_cameras
    }
    
    return batch

def load_model(checkpoint_path: str, device: torch.device):
    """Load the trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = TwoHeadTransformer(**DEFAULT_MODEL_PARAMS)

    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logging.info(f"Loaded model from {checkpoint_path}")
    return model