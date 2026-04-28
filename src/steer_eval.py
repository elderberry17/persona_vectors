from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Any
from contextlib import contextmanager

import torch
from torch import Tensor
from tqdm import tqdm


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0


@dataclass
class SteeringEvalRecord:
    trait_name: str
    layer_idx: int
    alpha: float
    prompt: str
    response_base: str
    response_steered: str


def get_transformer_layer(model, layer_idx: int):
    """
    Return the transformer block module for a given layer index.

    Adjust this function if your model uses a different internal structure.
    Common layouts:
      - Llama/Qwen/Mistral: model.model.layers[layer_idx]
      - GPT-NeoX-like:      model.gpt_neox.layers[layer_idx]
      - GPT2-like:          model.transformer.h[layer_idx]
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]

    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers[layer_idx]

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]

    raise ValueError(
        "Unsupported model architecture. Please adapt get_transformer_layer(...) "
        "for your model class."
    )


def _make_substractive_steering_hook(
    direction: Tensor,
    alpha: float,
) -> Callable:
    """
    Create a forward hook that adds alpha * direction to the layer output.

    Expected layer output shapes:
      - Tensor of shape [batch, seq, hidden]
      - or tuple whose first element is that Tensor

    The direction is added to every token position in the sequence.
    """
    if direction.ndim != 1:
        raise ValueError(f"direction must be 1D, got shape={tuple(direction.shape)}")

    def hook_fn(module, inputs, output):
        delta = alpha * direction

        if isinstance(output, tuple):
            hidden = output[0]
            if hidden.ndim != 3:
                raise ValueError(
                    f"Expected hidden state with 3 dims [batch, seq, hidden], "
                    f"got shape={tuple(hidden.shape)}"
                )
            delta_broadcast = delta.view(1, 1, -1).to(hidden.device, hidden.dtype)
            steered_hidden = hidden - delta_broadcast
            return (steered_hidden, *output[1:])

        if torch.is_tensor(output):
            if output.ndim != 3:
                raise ValueError(
                    f"Expected hidden state with 3 dims [batch, seq, hidden], "
                    f"got shape={tuple(output.shape)}"
                )
            delta_broadcast = delta.view(1, 1, -1).to(output.device, output.dtype)
            return output - delta_broadcast

        raise TypeError(
            f"Unsupported layer output type for steering: {type(output)}"
        )

    return hook_fn


@contextmanager
def steering_context(
    model,
    layer_idx: int,
    direction: Tensor,
    alpha: float,
    enabled: bool = True,
):
    """
    Context manager that conditionally registers the steering hook.
    """
    if not enabled:
        yield
        return

    layer_module = get_transformer_layer(model, layer_idx)
    hook = layer_module.register_forward_hook(
        _make_substractive_steering_hook(direction=direction, alpha=alpha)
    )
    try:
        yield
    finally:
        hook.remove()


@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    generation_config: Optional[GenerationConfig] = None,
    *,
    do_steering: bool = False,
    layer_idx: Optional[int] = None,
    direction: Optional[Tensor] = None,
    alpha: float = 0.0,
    device: Optional[torch.device] = None,
) -> str:
    """
    Generate a single response, optionally with persona-vector steering.

    Args:
        do_steering:
            If False, ignores layer_idx / direction / alpha.
        layer_idx:
            Layer where steering is injected.
        direction:
            1D persona vector for that layer.
        alpha:
            Steering strength.
    """
    if generation_config is None:
        generation_config = GenerationConfig()

    if device is None:
        device = next(model.parameters()).device

    if do_steering:
        if layer_idx is None:
            raise ValueError("layer_idx must be provided when do_steering=True")
        if direction is None:
            raise ValueError("direction must be provided when do_steering=True")

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with steering_context(
        model=model,
        layer_idx=layer_idx if layer_idx is not None else 0,
        direction=direction if direction is not None else torch.zeros(1, device=device),
        alpha=alpha,
        enabled=do_steering,
    ):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=generation_config.max_new_tokens,
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            repetition_penalty=generation_config.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, prompt_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


def evaluate_trait_steering(
    model,
    tokenizer,
    trait_name: str,
    eval_prompts: Sequence[str],
    persona_vectors_dict: Dict[str, Dict[int, Tensor]],
    *,
    layer_idx: int,
    alpha: float,
    generation_config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None,
) -> List[SteeringEvalRecord]:
    """
    For each held-out prompt, generate:
      1) baseline response
      2) steered response

    Returns a list of records that you can later score with a judge.
    """
    if trait_name not in persona_vectors_dict:
        raise KeyError(f"Trait '{trait_name}' not found in persona_vectors_dict")

    available_keys = persona_vectors_dict[trait_name].keys()
    print(available_keys)

    if layer_idx in available_keys:
        layer_key = layer_idx
    elif f"layers.{layer_idx}" in available_keys:
        layer_key = f"layers.{layer_idx}"
    else:
        available = sorted(available_keys)
        raise KeyError(
            f"Layer {layer_idx} not found for trait '{trait_name}'. "
            f"Available layers: {available}"
        )

    direction = persona_vectors_dict[trait_name][layer_key]

    records: List[SteeringEvalRecord] = []
    for prompt in tqdm(eval_prompts, total=len(eval_prompts)):

        # disable for now
        # response_base = generate_response(
        #     model=model,
        #     tokenizer=tokenizer,
        #     prompt=prompt,
        #     generation_config=generation_config,
        #     do_steering=False,
        #     device=device,
        # )

        response_steered = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            generation_config=generation_config,
            do_steering=True,
            layer_idx=layer_idx,
            direction=direction,
            alpha=alpha,
            device=device,
        )

        records.append(
            SteeringEvalRecord(
                trait_name=trait_name,
                layer_idx=layer_idx,
                alpha=alpha,
                prompt=prompt,
                response_base="",
                response_steered=response_steered,
            )
        )

    return records


def run_heldout_steering_grid(
    model,
    tokenizer,
    persona_axes,
    persona_vectors_dict: Dict[str, Dict[int, Tensor]],
    *,
    layers: Sequence[int],
    alphas: Sequence[float],
    generation_config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[int, Dict[float, List[SteeringEvalRecord]]]]:
    """
    Runs base-vs-steered generation for all axes on held-out prompts.

    Returns nested dict:
      results[trait_name][layer_idx][alpha] = List[SteeringEvalRecord]
    """
    all_results: Dict[str, Dict[int, Dict[float, List[SteeringEvalRecord]]]] = {}

    for axis in tqdm(persona_axes, total=len(persona_axes)):
        trait_name = axis.name
        all_results[trait_name] = {}

        for layer_idx in tqdm(layers, total=len(layers)):
            all_results[trait_name][layer_idx] = {}

            for alpha in tqdm(alphas, total=len(alphas)):
                records = evaluate_trait_steering(
                    model=model,
                    tokenizer=tokenizer,
                    trait_name=trait_name,
                    eval_prompts=axis.eval_prompts,
                    persona_vectors_dict=persona_vectors_dict,
                    layer_idx=layer_idx,
                    alpha=alpha,
                    generation_config=generation_config,
                    device=device,
                )
                all_results[trait_name][layer_idx][alpha] = records

    return all_results