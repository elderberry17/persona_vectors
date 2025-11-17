import json
from pathlib import Path
import random
import torch

from config import ExperimentConfig, ModelConfig, TrainingConfig
from data_utils import math_persona_axes, load_gsm8k_sft
from models import load_model_and_tokenizer
from activations import ActivationCollector
from persona_directions import (
    compute_persona_direction,
    project_dataset_on_axis,
    summarize_projections,
)
from sft_trainer import run_sft_training
from evaluation import evaluate_axis_shift


def main():
    # 1. Build config
    exp_cfg = ExperimentConfig(
        model=ModelConfig(
            model_name="Qwen/Qwen2-0.5B",  # base model, not instruct
            device="mps",
            dtype="bfloat16",
            activation_layer_name="model.layers.23",  # adjust to your model
        ),
        train=TrainingConfig(
            sft_jsonl_path="",  # unused now, we use GSM8K instead
            output_dir="outputs/qwen_base_gsm8k_sft",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            learning_rate=5e-5,
            max_seq_length=512,
        ),
        axes=math_persona_axes(),
    )

    model_cfg = exp_cfg.model
    train_cfg = exp_cfg.train
    axes = exp_cfg.axes

    # 2. Load base model + tokenizer
    base_model, tokenizer = load_model_and_tokenizer(model_cfg)

    # 3. Load GSM8K train/val for SFT
    train_sft, val_sft = load_gsm8k_sft(
        max_train_examples=4000,  # tune as you like
        max_val_examples=512,
    )

    # Use some GSM8K questions as eval prompts for the math axis
    # (we only need the questions; responses not needed for persona eval)
    gsm8k_eval_prompts = [ex["prompt"] for ex in random.sample(train_sft, k=min(64, len(train_sft)))]

    for axis in axes:
        axis.eval_prompts = gsm8k_eval_prompts

    # 4. For each axis: compute math persona direction on base model
    results = {}

    for axis in axes:
        print(f"\n=== Axis: {axis.name} ===")
        with ActivationCollector(base_model, tokenizer, model_cfg) as collector:
            acts_a = collector.encode_prompts(axis.group_a_prompts)
        with ActivationCollector(base_model, tokenizer, model_cfg) as collector:
            acts_b = collector.encode_prompts(axis.group_b_prompts)

        direction = compute_persona_direction(axis, acts_a, acts_b)
        results[axis.name] = {"direction_norm": float(direction.norm().item())}
        print("Direction norm:", direction.norm().item())

        # 5. Predict influence of GSM8K SFT data *before* fine-tuning
        sft_prompts = [ex["prompt"] for ex in train_sft]

        with ActivationCollector(base_model, tokenizer, model_cfg) as collector:
            sft_acts = collector.encode_prompts(sft_prompts)

        proj = project_dataset_on_axis(sft_acts, direction)
        sft_stats = summarize_projections(proj)
        results[axis.name]["predicted_sft_stats"] = sft_stats
        print("Predicted SFT projections (base model):", sft_stats)

        out_dir = Path(train_cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(direction, out_dir / f"{axis.name}_direction.pt")

    # 6. Fine-tune base model on GSM8K (SFT) with validation
    ft_model = run_sft_training(
        base_model,
        tokenizer,
        train_data=train_sft,
        train_cfg=train_cfg,
        eval_data=val_sft,
    )

    # 7. Evaluate math persona shift: base vs fine-tuned along each axis
    for axis in axes:
        out_dir = Path(train_cfg.output_dir)
        direction = torch.load(out_dir / f"{axis.name}_direction.pt")

        eval_stats = evaluate_axis_shift(
            axis,
            model_cfg,
            base_model=base_model,
            ft_model=ft_model,
            tokenizer=tokenizer,
            direction=direction,
        )
        results[axis.name]["eval_shift"] = eval_stats
        print(f"Axis {axis.name} shift:", eval_stats)

    # 8. Save JSON summary
    with open(Path(train_cfg.output_dir) / "persona_sft_results_math_gsm8k.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nDone. Results saved to", train_cfg.output_dir)


if __name__ == "__main__":
    main()