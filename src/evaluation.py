from typing import Dict
import torch

from activations import ActivationCollector
from persona_directions import project_dataset_on_axis, summarize_projections
from config import PersonaAxis, ModelConfig

def evaluate_axis_shift(axis: PersonaAxis,
                        cfg: ModelConfig,
                        base_model,
                        ft_model,
                        tokenizer,
                        direction: torch.Tensor) -> Dict[str, Dict]:
    """
    Evaluate how the model's projections on the axis change after SFT.
    """
    eval_prompts = axis.eval_prompts or axis.group_a_prompts + axis.group_b_prompts

    # Base model activations
    with ActivationCollector(base_model, tokenizer, cfg) as collector:
        base_acts = collector.encode_prompts(eval_prompts)

    base_proj = project_dataset_on_axis(base_acts, direction)
    base_stats = summarize_projections(base_proj)

    # Fine-tuned model activations
    with ActivationCollector(ft_model, tokenizer, cfg) as collector:
        ft_acts = collector.encode_prompts(eval_prompts)

    ft_proj = project_dataset_on_axis(ft_acts, direction)
    ft_stats = summarize_projections(ft_proj)

    shift = {
        "delta_mean": ft_stats["mean"] - base_stats["mean"],
        "delta_std": ft_stats["std"] - base_stats["std"],
    }

    return {
        "base": base_stats,
        "fine_tuned": ft_stats,
        "shift": shift,
    }
