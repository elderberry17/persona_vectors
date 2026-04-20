from typing import Dict
import torch
from config import PersonaAxis

def compute_persona_direction(mean_a: torch.Tensor,
                              mean_b: torch.Tensor) -> torch.Tensor:
    """
    activations_*: (N, hidden_size)
    persona_direction = mean(A) - mean(B), normalized
    """
    direction = mean_a - mean_b
    direction_norm = direction / (direction.norm() + 1e-8)
    return direction_norm  # (hidden_size,)

def compute_persona_direction_from_zero(mean_a: torch.Tensor) -> torch.Tensor:
    """
    assuming that the zeroes vector is neutral
    """
    direction_norm = mean_a / (mean_a.norm() + 1e-8)
    return direction_norm  # (hidden_size,)

def project_dataset_on_axis(activations: torch.Tensor,
                            direction: torch.Tensor) -> torch.Tensor:
    """
    Projection of each activation onto a direction.
    activations: (N, H), direction: (H,)
    returns: (N,) scalar projections
    """
    return activations @ direction

def summarize_projections(projections: torch.Tensor) -> Dict[str, float]:
    return {
        "mean": projections.mean().item(),
        "std": projections.std(unbiased=False).item(),
        "min": projections.min().item(),
        "max": projections.max().item(),
    }
