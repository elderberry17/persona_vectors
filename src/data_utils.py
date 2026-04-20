import json
from pathlib import Path
from typing import List, Dict

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