from typing import List
import torch
from torch.utils.data import DataLoader
from models import get_layer_by_name
from config import ModelConfig

class ActivationCollector:
    def __init__(self, model, tokenizer, cfg: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self._handles = []
        self.collected = []

    def _hook_fn(self, module, input, output):
        # output: (batch, seq_len, hidden)
        # take last token activations
        hidden = output[:, -1, :].detach().cpu()
        self.collected.append(hidden)

    def __enter__(self):
        layer = get_layer_by_name(self.model, self.cfg.activation_layer_name)
        handle = layer.register_forward_hook(self._hook_fn)
        self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._handles:
            h.remove()
        self._handles = []

    def encode_prompts(self, prompts: List[str], batch_size: int = 4) -> torch.Tensor:
        self.collected = []
        dl = DataLoader(prompts, batch_size=batch_size)

        for batch in dl:
            enc = self.tokenizer(
                list(batch),
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )
            enc = {k: v.to(self.model.device) for k, v in enc.items()}

            with torch.no_grad():
                _ = self.model(**enc)

        if not self.collected:
            raise RuntimeError("No activations collected; check layer name/hook.")
        return torch.cat(self.collected, dim=0)  # (N, hidden_size)
