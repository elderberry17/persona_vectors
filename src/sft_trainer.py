from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, PreTrainedTokenizer, DataCollatorForLanguageModeling
from config import TrainingConfig


class SFTDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: PreTrainedTokenizer, max_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        response = item["response"]

        full_text = prompt + "\n" + response

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_len,
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }



def run_sft_training(
    model,
    tokenizer,
    train_data: List[Dict],
    train_cfg: TrainingConfig,
    eval_data: Optional[List[Dict]] = None,
):
    train_dataset = SFTDataset(
        train_data,
        tokenizer=tokenizer,
        max_len=train_cfg.max_seq_length,
    )

    eval_dataset = (
        SFTDataset(eval_data, tokenizer=tokenizer, max_len=train_cfg.max_seq_length)
        if eval_data is not None
        else None
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM
    )

    args = TrainingArguments(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        learning_rate=train_cfg.learning_rate,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        warmup_ratio=train_cfg.warmup_ratio,
        logging_dir=train_cfg.output_dir + "/logs",
        gradient_accumulation_steps=1,
        bf16=True if model.dtype == torch.bfloat16 else False,
        fp16=True if model.dtype == torch.float16 else False,
        save_total_limit=2,
        report_to=["none"],
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=train_cfg.save_steps if eval_dataset is not None else None,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    if eval_dataset is not None:
        trainer.evaluate()

    trainer.save_model(train_cfg.output_dir)
    tokenizer.save_pretrained(train_cfg.output_dir)

    return model