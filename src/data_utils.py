import json
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
from config import PersonaAxis


def load_sft_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def math_persona_axes() -> List[PersonaAxis]:
    """
    Define a persona axis that contrasts:
    - 'math expert' internal state
    - 'math avoidant / sloppy' internal state

    The questions are the same; only the role/preamble changes.
    This follows the Anthropic idea: prompts induce different inner states.
    """

    base_questions = [
        "What is 37 * 24?",
        "A train travels at 60 km/h for 2.5 hours. How far does it go?",
        "If 3x + 7 = 25, what is the value of x?",
        "A rectangle has length 7 and width 5. What is its area?",
        "The sum of three consecutive integers is 39. What are the integers?",
        "If a bag contains 5 red, 3 blue, and 2 green balls, what is the probability of drawing a blue ball?",
        "Simplify the expression: (3x + 2x) - (x - 4).",
        "If the average of three numbers is 18 and two of them are 15 and 21, what is the third number?",
    ]

    # Persona ON: strong math mode
    math_on_prompts = [
        (
            "You are a highly skilled mathematician. "
            "Carefully solve the following problem step by step, showing clear reasoning "
            "and then give the exact final answer.\n\n"
            f"Problem:\n{q}\n\nSolution:"
        )
        for q in base_questions
    ]

    # Persona OFF: avoid math, be sloppy
    math_off_prompts = [
        (
            "You dislike doing math and often make mistakes. "
            "Give a quick, rough answer without carefully calculating, "
            "and you do not need to show detailed work.\n\n"
            f"Problem:\n{q}\n\nAnswer:"
        )
        for q in base_questions
    ]

    # For eval, we'll later override with real GSM8K questions,
    # but we can keep a small default set:
    eval_prompts = [
        "Solve step by step: A shop sells pencils at 3 for 1.20 euros. "
        "How much do 5 pencils cost?",
        "Solve step by step: If 4x - 7 = 21, what is x?",
        "Solve step by step: A rectangle has perimeter 30 and sides 8 and x. Find x.",
    ]

    axis = PersonaAxis(
        name="math_skill",
        group_a_prompts=math_on_prompts,
        group_b_prompts=math_off_prompts,
        eval_prompts=eval_prompts,
    )
    return [axis]


def load_gsm8k_sft(
    max_train_examples: Optional[int] = None,
    max_val_examples: Optional[int] = 512,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load GSM8K from Hugging Face and convert to a prompt/response SFT format.

    - Train split is used for SFT training (optionally truncated).
    - Test split is used as validation (optionally truncated).

    Each example:
        prompt  = instruction + question
        response = full chain-of-thought answer from GSM8K (already step-by-step).
    """
    ds = load_dataset("gsm8k", "main")  # splits: "train", "test"
    train_ds = ds["train"]
    val_ds = ds["test"]

    if max_train_examples is not None:
        train_ds = train_ds.select(range(min(max_train_examples, len(train_ds))))
    if max_val_examples is not None:
        val_ds = val_ds.select(range(min(max_val_examples, len(val_ds))))

    def convert(split):
        out: List[Dict] = []
        for ex in split:
            question = ex["question"].strip()
            answer = ex["answer"].strip()  # already step-by-step + '#### final'
            prompt = (
                "You are a careful math tutor. Solve the following word problem step by step "
                "and then give the final answer clearly.\n\n"
                f"Problem:\n{question}\n\nSolution:"
            )
            response = answer
            out.append({"prompt": prompt, "response": response})
        return out

    train_data = convert(train_ds)
    val_data = convert(val_ds)
    return train_data, val_data