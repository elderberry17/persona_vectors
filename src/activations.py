from typing import Dict
import torch


def collect_mean_activations(
    model, tokenizer, prompts,
    layer_indices=None,
    batch_size=4,
    max_length=128,
) -> Dict[int, torch.Tensor]:
    n_layers = len(model.model.layers)
    if layer_indices is None:
        layer_indices = list(range(n_layers))

    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size
    acc = {idx: torch.zeros(hidden_size) for idx in layer_indices}
    counts = {idx: 0 for idx in layer_indices}

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            output = model(**enc, output_hidden_states=True)

        # output.hidden_states is a tuple of length n_layers + 1
        # index 0 = embedding layer, index i+1 = layers[i]
        for idx in layer_indices:
            last = output.hidden_states[idx][:, -1, :].cpu().float()
            acc[idx] += last.sum(dim=0)
            counts[idx] += last.shape[0]

    return {f"layers.{idx}": acc[idx] / counts[idx] for idx in layer_indices}
