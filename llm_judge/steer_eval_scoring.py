import sys
import csv
import json
import asyncio
from typing import Any, Dict, List

import src.steer_eval as real_steer_eval
sys.modules["steer_eval"] = real_steer_eval

from tqdm import tqdm
import pandas as pd
from openai import AsyncOpenAI
import pickle

from judge_prompts import SYSTEM_PROMPT, USER_TEMPLATE


MODEL_NAME = "gpt-4o-mini"
# OUTPUT_CSV = "artifacts/pythia1.4b/full_run/steering_eval_scores.csv"
OUTPUT_CSV = "artifacts/qwen3_1.7b_instruct/base_eval_scores.csv"
SLEEP_ON_ERROR_SEC = 2.0
MAX_RETRIES = 3
CONCURRENCY = 30


client = AsyncOpenAI()
semaphore = asyncio.Semaphore(CONCURRENCY)


def record_to_dict(record: Any) -> Dict[str, Any]:
    """
    Converts SteeringEvalRecord-like object into plain dict.
    """
    if isinstance(record, dict):
        return record

    return {
        "trait_name": getattr(record, "trait_name"),
        "layer_idx": getattr(record, "layer_idx"),
        "alpha": getattr(record, "alpha"),
        "prompt": getattr(record, "prompt"),
        "response_base": getattr(record, "response_base", ""),
        "response_steered": getattr(record, "response_steered", ""),
    }


def flatten_results(results: Dict[str, Dict[int, Dict[int, List[Any]]]]) -> List[Dict[str, Any]]:
    """
    Expected structure:
    results[trait][layer][alpha] = list[SteeringEvalRecord]
    """
    rows = []

    for trait, layers_dict in results.items():
        for layer, alphas_dict in layers_dict.items():
            for alpha, records in alphas_dict.items():
                for record in records:
                    r = record_to_dict(record)
                    rows.append(
                        {
                            "trait": trait,
                            "layer": layer,
                            "alpha": alpha,
                            "prompt": r["prompt"],
                            # "steered": r.get("response_steered", "") or "",
                            "base": r.get("response_base", "") or "",
                        }
                    )

    return rows


def clean_json_text(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if len(lines) >= 2:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    return text


def parse_score_response(text: str) -> Dict[str, Any]:
    raw = clean_json_text(text)

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON. Raw response:\n{raw}") from e

    if "score" not in obj:
        raise ValueError(f"Missing 'score' field. Parsed object: {obj}")

    score = obj["score"]

    if isinstance(score, str):
        score = score.strip()
        if score.isdigit():
            score = int(score)
        else:
            raise ValueError(f"Score is not a valid integer: {score}")

    if not isinstance(score, int):
        raise ValueError(f"Score is not int: {score}")

    score = max(0, min(100, score))

    return {
        "score": score,
        "raw_response": text,
    }


def load_already_done_keys(csv_path: str) -> set:
    """
    Allows resume: skip already scored rows.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return set()
    except pd.errors.EmptyDataError:
        return set()

    # rewrote to base
    required_cols = {"trait", "layer", "alpha", "prompt", "base"}
    if not required_cols.issubset(df.columns):
        return set()

    keys = set(
        zip(
            df["trait"].astype(str),
            df["layer"].astype(int),
            df["alpha"].astype(int),
            df["prompt"].astype(str),
            df["base"].astype(str),
        )
    )
    return keys


async def judge_one_sample(
    trait: str,
    prompt: str,
    generation: str,
) -> Dict[str, Any]:
    user_prompt = USER_TEMPLATE.format(
        trait=trait,
        prompt=prompt,
        generation=generation,
    )

    async with semaphore:
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.responses.create(
                    model=MODEL_NAME,
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_output_tokens=20,
                )

                text = response.output_text
                return parse_score_response(text)

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(SLEEP_ON_ERROR_SEC)
                else:
                    return {
                        "score": None,
                        "raw_response": "",
                        "error": f"{type(e).__name__}: {e}",
                    }


async def process_row(row: Dict[str, Any]) -> Dict[str, Any]:
    parsed = await judge_one_sample(
        trait=row["trait"],
        prompt=row["prompt"],
        generation=row["base"],
    )

    return {
        "trait": row["trait"],
        "layer": row["layer"],
        "alpha": row["alpha"],
        "prompt": row["prompt"],
        "base": row["base"],
        "base_score": parsed.get("score"),
        "judge_raw_response": parsed.get("raw_response", ""),
        "error": parsed.get("error", ""),
    }


async def evaluate_all_async(results: Dict[str, Dict[int, Dict[int, List[Any]]]]) -> None:
    rows = flatten_results(results)
    already_done = load_already_done_keys(OUTPUT_CSV)

    rows_to_process = []
    for row in rows:
        key = (
            str(row["trait"]),
            int(row["layer"]),
            int(row["alpha"]),
            str(row["prompt"]),
            str(row["base"]),
        )
        if key not in already_done:
            rows_to_process.append(row)

    print(f"Total rows found: {len(rows)}")
    print(f"Already scored: {len(rows) - len(rows_to_process)}")
    print(f"To process now: {len(rows_to_process)}")

    fieldnames = [
        "trait",
        "layer",
        "alpha",
        "prompt",
        "base",
        "base_score",
        "judge_raw_response",
        "error",
    ]

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if f.tell() == 0:
            writer.writeheader()

        tasks = [asyncio.create_task(process_row(row)) for row in rows_to_process]

        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Scoring generations"):
            result = await future
            writer.writerow(result)
            f.flush()


def evaluate_all(results: Dict[str, Dict[int, Dict[int, List[Any]]]]) -> None:
    asyncio.run(evaluate_all_async(results))


if __name__ == "__main__":
    # res = pickle.load(open("artifacts/pythia1.4b/full_run/all_layers_alpha1_5_10_25.pkl", "rb"))
    res = pickle.load(open("artifacts/qwen3_1.7b_instruct/response_base.pkl", "rb"))

    evaluate_all(res)