import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from datasets import load_dataset

from config import PersonaAxis


def load_jsonl(path: str) -> List[Dict]:
    data: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_axis_from_artifacts(
    name: str,
    artifact_path: str,
    question_template: str = "{question}",
    extraction_count: int = 20,
    eval_count: int = 20,
) -> PersonaAxis:
    """
    Build a PersonaAxis from a local JSON artifact file in the style of the paper.

    Expected format:
    {
        "instructions": [
            {"pos": "...", "neg": "..."},
            ...
        ],
        "questions": ["q1", ..., "q40"],
        "eval_prompt": "..."
    }

    The paper's pipeline generates 5 instruction pairs and 40 questions,
    split into 20 extraction and 20 evaluation questions.
    """
    obj = load_json(artifact_path)

    instructions = obj["instructions"]
    questions = obj["questions"]

    if len(questions) < extraction_count + eval_count:
        raise ValueError(
            f"Need at least {extraction_count + eval_count} questions, got {len(questions)}"
        )

    extraction_questions = questions[:extraction_count]
    eval_questions = questions[extraction_count:extraction_count + eval_count]

    group_a_prompts: List[str] = []
    group_b_prompts: List[str] = []

    # Cross product: each instruction pair x each extraction question
    # This gives a richer extraction set than just a single fixed prompt.
    for inst in instructions:
        pos = inst["pos"].strip()
        neg = inst["neg"].strip()

        for q in extraction_questions:
            q_text = question_template.format(question=q.strip())
            group_a_prompts.append(f"{pos}\n\nUser: {q_text}\nAssistant:")
            group_b_prompts.append(f"{neg}\n\nUser: {q_text}\nAssistant:")

    eval_prompts = [
        f"User: {question_template.format(question=q.strip())}\nAssistant:"
        for q in eval_questions
    ]

    axis = PersonaAxis(
        name=name,
        group_a_prompts=group_a_prompts,
        group_b_prompts=group_b_prompts,
        eval_prompts=eval_prompts,
    )
    return axis


def load_persona_axes_from_dir(artifacts_dir: str) -> List[PersonaAxis]:
    """
    Load all *.json artifacts from a directory and convert them into PersonaAxis objects.

    Example filenames:
      - evil.json
      - sycophancy.json
      - hallucination.json
    """
    axes: List[PersonaAxis] = []
    for path in sorted(Path(artifacts_dir).glob("*.json")):
        axis = build_axis_from_artifacts(
            name=path.stem,
            artifact_path=str(path),
        )
        axes.append(axis)
    return axes


def load_halueval_qa_eval(
    max_examples: int = 1000,
    dataset_name: str = "pminervini/HaluEval",
    subset: str = "qa",
    split: str = "data",
) -> List[Dict]:
    """
    Load the QA split of HaluEval for external hallucination evaluation.

    Paper uses the first 1,000 questions from the QA split of HaluEval.
    We return prompt-like evaluation items with reference fields preserved.

    Output format:
    [
        {
            "prompt": "...",
            "knowledge": ...,
            "reference_answer": ...,
            "hallucinated_answer": ...
        },
        ...
    ]
    """
    ds = load_dataset(dataset_name, subset, split=split)

    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    out: List[Dict] = []
    for ex in ds:
        question = ex["question"].strip()
        prompt = (
            "Answer the following question as accurately as possible. "
            "If you are uncertain, say so explicitly.\n\n"
            f"Question:\n{question}\n\nAnswer:"
        )
        out.append(
            {
                "prompt": prompt,
                "question": question,
                "knowledge": ex.get("knowledge"),
                "reference_answer": ex.get("right_response"),
                "hallucinated_answer": ex.get("hallucinated_response"),
            }
        )
    return out


def load_math_sft(
    max_train_examples: Optional[int] = None,
    max_val_examples: Optional[int] = 512,
    dataset_name: str = "EleutherAI/hendrycks_math",
    dataset_config_name: str = "algebra"
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load MATH from Hugging Face and convert to prompt/response SFT format.

    This is the closest replacement for your GSM8K-specific loader if you want
    to stay near the paper while avoiding GSM8K.

    Expected fields on HF mirrors are usually:
      - 'problem'
      - 'solution'
      - split names: train / test

    Output:
      [{"prompt": ..., "response": ..., "question": ..., "solution": ...}, ...]
    """
    ds = load_dataset(dataset_name, dataset_config_name)

    train_ds = ds["train"]
    val_ds = ds["test"] if "test" in ds else ds["train"]

    if max_train_examples is not None:
        train_ds = train_ds.select(range(min(max_train_examples, len(train_ds))))
    if max_val_examples is not None:
        val_ds = val_ds.select(range(min(max_val_examples, len(val_ds))))

    def convert(split_ds) -> List[Dict]:
        out: List[Dict] = []
        for ex in split_ds:
            problem = ex["problem"].strip()
            solution = ex["solution"].strip()

            prompt = (
                "You are a careful math assistant. Solve the following problem step by step, "
                "and then provide the final answer clearly.\n\n"
                f"Problem:\n{problem}\n\nSolution:"
            )

            out.append(
                {
                    "prompt": prompt,
                    "response": solution,
                    "question": problem,
                    "solution": solution,
                    "level": ex.get("level"),
                    "type": ex.get("type"),
                }
            )
        return out

    return convert(train_ds), convert(val_ds)


def load_sycophancy_eval_jsonl(
    path: str,
    max_examples: Optional[int] = None,
) -> List[Dict]:
    """
    Load held-out sycophancy questions from a local JSONL file, e.g. from
    keing1/reward-hack-generalization.

    Because repository schemas may vary across files, this loader keeps the raw
    example and creates a prompt from the most likely text fields.

    Expected behavior:
    - preserve the original example in 'raw'
    - create a single text prompt for generation/evaluation
    """
    rows = load_jsonl(path)

    if max_examples is not None:
        rows = rows[:max_examples]

    out: List[Dict] = []
    for ex in rows:
        # Try common field names heuristically
        question = (
            ex.get("question")
            or ex.get("prompt")
            or ex.get("input")
            or ex.get("user_input")
            or ""
        )

        metadata = {
            k: v
            for k, v in ex.items()
            if k not in {"question", "prompt", "input", "user_input"}
        }

        prompt = f"User: {question.strip()}\nAssistant:"

        out.append(
            {
                "prompt": prompt,
                "question": question,
                "metadata": metadata,
                "raw": ex,
            }
        )
    return out


def load_benchmarks(
    use_hallucination_eval: bool = True,
    use_math_sft: bool = True,
    halueval_max_examples: int = 1000,
    math_max_train_examples: Optional[int] = None,
    math_max_val_examples: Optional[int] = 512,
) -> Dict[str, object]:
    """
    Convenience wrapper returning the main pieces you likely need for reproduction.

    Returns a dict with optional keys:
      - hallucination_eval
      - math_train
      - math_val
    """
    bundle: Dict[str, object] = {}

    if use_hallucination_eval:
        bundle["hallucination_eval"] = load_halueval_qa_eval(
            max_examples=halueval_max_examples
        )

    if use_math_sft:
        math_train, math_val = load_math_sft(
            max_train_examples=math_max_train_examples,
            max_val_examples=math_max_val_examples,
        )
        bundle["math_train"] = math_train
        bundle["math_val"] = math_val

    return bundle