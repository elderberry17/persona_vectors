from dataclasses import dataclass, field
from typing import List, Optional


# microsoft/phi-2, apple/OpenELM-270M, 

@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "mps"
    dtype: str = "bfloat16"

@dataclass
class TrainingConfig:
    sft_jsonl_path: str = "data/sft_dataset.jsonl"  # prompt/response pairs
    output_dir: str = "outputs/qwen_base_sft"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    learning_rate: float = 5e-5
    max_seq_length: int = 512
    logging_steps: int = 50
    save_steps: int = 500
    warmup_ratio: float = 0.03

@dataclass
class PersonaAxis:
    name: str
    group_a_prompts: List[str]
    group_b_prompts: List[str]
    eval_prompts: Optional[List[str]] = None

@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    axes: List[PersonaAxis] = field(default_factory=list)
