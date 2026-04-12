import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import ModelConfig

def load_model_and_tokenizer(cfg: ModelConfig):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(cfg.dtype, torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=dtype,
        device_map=cfg.device,
    )
    model.eval()
    return model, tokenizer