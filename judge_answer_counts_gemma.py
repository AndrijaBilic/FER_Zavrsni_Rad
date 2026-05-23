import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


STEERING_MODEL_KEYS = {
    "qwen": "qwen3_4b",
    "mistral": "mistral_7b",
    "llama31_8b_it": "llama31_8b_it",
    "qwen3_8b": "qwen3_8b",
    "gemma3_12b_it": "gemma3_12b_it",
}


def default_results_dir() -> str:
    model_choice = os.environ.get("MODEL_CHOICE", "mistral")
    model_key = os.environ.get("MODEL_KEY", STEERING_MODEL_KEYS.get(model_choice, model_choice))
    drive_path = os.environ.get("DRIVE_PATH", "outputs")
    artifact_subdir = os.environ.get("ARTIFACT_SUBDIR", "phase2_activation_steering_enumerability")
    return str(Path(drive_path) / artifact_subdir / model_key)


@dataclass
class JudgeConfig:
    judge_model: str = os.environ.get("JUDGE_MODEL", "google/gemma-4-31B-it")
    model_choice: str = os.environ.get("MODEL_CHOICE", "mistral")
    results_dir: str = os.environ.get("RESULTS_DIR", default_results_dir())
    input_files: tuple[str, ...] = tuple(
        x.strip()
        for x in os.environ.get(
            "JUDGE_INPUT_FILES",
            "curated_manual_review_steered_generations.csv,webq_manual_review_steered_generations.csv,ood_manual_review_steered_generations.csv",
        ).split(",")
        if x.strip()
    )
    output_subdir: str = os.environ.get("JUDGE_OUTPUT_SUBDIR", "gemma_judge")
    batch_size: int = int(os.environ.get("JUDGE_BATCH_SIZE", "5"))
    max_new_tokens_per_item: int = int(os.environ.get("JUDGE_MAX_NEW_TOKENS_PER_ITEM", "90"))
    quantization: str = os.environ.get(
        "JUDGE_QUANTIZATION",
        "4bit" if os.environ.get("JUDGE_USE_4BIT", "1") != "0" else "bf16",
    )
    use_double_quant: bool = os.environ.get("JUDGE_USE_DOUBLE_QUANT", "1") != "0"
    torch_dtype: str = os.environ.get("JUDGE_TORCH_DTYPE", "float16")
    max_rows: int = int(os.environ.get("JUDGE_MAX_ROWS", "0"))
    alphas: str = os.environ.get("JUDGE_ALPHAS", "")
    seed: int = int(os.environ.get("JUDGE_SEED", "42"))


CFG = JudgeConfig()
RESULTS_DIR = Path(CFG.results_dir)
OUTPUT_DIR = RESULTS_DIR / CFG.output_subdir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(json.dumps(asdict(CFG), indent=2))
print("Judge outputs:", OUTPUT_DIR)


def dtype_from_name(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def load_judge():
    offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    tokenizer = AutoTokenizer.from_pretrained(
        CFG.judge_model,
        trust_remote_code=True,
        local_files_only=offline,
        token=hf_token(),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = CFG.quantization.lower()
    quant_config = None
    if quantization in {"4bit", "nf4"}:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_from_name(CFG.torch_dtype),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=CFG.use_double_quant,
        )
    elif quantization in {"8bit", "int8"}:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization in {"bf16", "float16", "none"}:
        quant_config = None
    else:
        raise ValueError("JUDGE_QUANTIZATION must be one of: 4bit, int8, bf16")

    model = AutoModelForCausalLM.from_pretrained(
        CFG.judge_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype_from_name(CFG.torch_dtype),
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        local_files_only=offline,
        token=hf_token(),
    ).eval()
    model.requires_grad_(False)
    return model, tokenizer


model, tokenizer = load_judge()
device = next(model.parameters()).device
print(f"Loaded judge {CFG.judge_model} on {device}")


def read_generation_files() -> pd.DataFrame:
    frames = []
    for name in CFG.input_files:
        path = RESULTS_DIR / name
        if not path.exists():
            print(f"Skipping missing judge input file: {path}")
            continue
        df = pd.read_csv(path)
        if "eval_name" not in df.columns:
            df["eval_name"] = name.split("_", 1)[0]
        df["source_file"] = name
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No judge input files found in {RESULTS_DIR}")
    out = pd.concat(frames, ignore_index=True)
    out["row_id"] = np.arange(len(out))
    if CFG.alphas.strip():
        keep = {float(x.strip()) for x in CFG.alphas.split(",") if x.strip()}
        out = out[out["alpha"].astype(float).isin(keep)].reset_index(drop=True)
        out["row_id"] = np.arange(len(out))
    if CFG.max_rows > 0:
        out = out.head(CFG.max_rows).copy()
    return out


def compact_response(text: Any, max_chars: int = 1200) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ... [truncated]"


def judge_prompt(items: list[dict]) -> str:
    payload = [
        {
            "id": int(item["row_id"]),
            "question": str(item["question"]),
            "model_response": compact_response(item["response"]),
        }
        for item in items
    ]
    return f"""You are judging whether model answers enumerate multiple distinct answers.

For each item, count the distinct answers in the actual answer content.
Ignore any generated labels such as "single", "multiple", or "Answer type".
If the response gives exactly one answer, answer_count is 1 and is_enumerated is false.
If it gives several distinct answers, answer_count is the number of distinct answers and is_enumerated is true.
If the response is empty, invalid, or unrelated, set valid_answer to false and answer_count to 0.
Do not reward repeated duplicates as new answers.

Return only a valid JSON array. Each object must have exactly these keys:
id, answer_count, is_enumerated, valid_answer, confidence, short_reason

Items:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def apply_chat_template(prompt: str):
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return tokenizer(formatted, return_tensors="pt")


def decode_new_tokens(outputs, input_len: int) -> str:
    toks = outputs[0][input_len:]
    return tokenizer.decode(toks, skip_special_tokens=True).strip()


def extract_json_array(text: str):
    text = text.strip()
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError(f"No JSON array found in judge output: {text[:500]}")
    return json.loads(match.group(0))


def run_judge_batch(items: list[dict]) -> tuple[list[dict], str]:
    prompt = judge_prompt(items)
    inputs = apply_chat_template(prompt).to(model.device)
    max_new_tokens = CFG.max_new_tokens_per_item * len(items) + 80
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    raw = decode_new_tokens(outputs, inputs["input_ids"].shape[-1])
    parsed = extract_json_array(raw)
    if not isinstance(parsed, list):
        raise ValueError("Judge returned JSON, but not a list.")
    return parsed, raw


def fallback_row(row: dict, error: str) -> dict:
    return {
        "id": int(row["row_id"]),
        "answer_count": np.nan,
        "is_enumerated": np.nan,
        "valid_answer": np.nan,
        "confidence": "parse_error",
        "short_reason": f"judge parse error: {error[:180]}",
    }


df = read_generation_files()
print("Rows to judge:", len(df))

checkpoint_path = OUTPUT_DIR / "judge_checkpoint.csv"
raw_checkpoint_path = OUTPUT_DIR / "raw_judge_outputs_checkpoint.jsonl"

if checkpoint_path.exists():
    checkpoint_df = pd.read_csv(checkpoint_path)
    judged_ids = set(checkpoint_df["row_id"].astype(int).tolist())
    judgments = checkpoint_df.to_dict("records")
    print(f"Loaded checkpoint with {len(judged_ids)} judged rows from {checkpoint_path}")
else:
    judged_ids = set()
    judgments = []

raw_outputs = []
records = [row for row in df.to_dict("records") if int(row["row_id"]) not in judged_ids]
print("Rows remaining:", len(records))


def normalize_judgment(obj: dict, row_id: int) -> dict:
    return {
        "row_id": int(row_id),
        "judge_answer_count": obj.get("answer_count", obj.get("judge_answer_count", np.nan)),
        "judge_is_enumerated": obj.get("is_enumerated", obj.get("judge_is_enumerated", np.nan)),
        "judge_valid_answer": obj.get("valid_answer", obj.get("judge_valid_answer", np.nan)),
        "judge_confidence": obj.get("confidence", obj.get("judge_confidence", "")),
        "judge_reason": obj.get("short_reason", obj.get("judge_reason", "")),
    }


def append_checkpoint(new_rows: list[dict]):
    if not new_rows:
        return
    header = not checkpoint_path.exists()
    pd.DataFrame(new_rows).to_csv(checkpoint_path, mode="a", header=header, index=False)

for start in tqdm(range(0, len(records), CFG.batch_size), desc="judge"):
    batch = records[start : start + CFG.batch_size]
    new_judgments = []
    try:
        parsed, raw = run_judge_batch(batch)
        raw_record = {"start": start, "row_ids": [int(x["row_id"]) for x in batch], "raw": raw}
        raw_outputs.append(raw_record)
        with open(raw_checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
        by_id = {int(obj.get("id")): obj for obj in parsed if isinstance(obj, dict) and "id" in obj}
        for row in batch:
            judgment = by_id.get(int(row["row_id"]))
            if judgment is None:
                judgment = fallback_row(row, "missing id in judge output")
            new_judgments.append(normalize_judgment(judgment, int(row["row_id"])))
    except Exception as exc:
        print(f"Batch starting at {start} failed: {exc}")
        # Try one-by-one for this batch; if that still fails, keep an explicit
        # parse_error row so the failure is visible in the final CSV.
        for row in batch:
            try:
                parsed, raw = run_judge_batch([row])
                raw_record = {"start": int(row["row_id"]), "row_ids": [int(row["row_id"])], "raw": raw}
                raw_outputs.append(raw_record)
                with open(raw_checkpoint_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                judgment = parsed[0] if parsed else fallback_row(row, "empty one-row output")
                new_judgments.append(normalize_judgment(judgment, int(row["row_id"])))
            except Exception as single_exc:
                new_judgments.append(normalize_judgment(fallback_row(row, str(single_exc)), int(row["row_id"])))

    judgments.extend(new_judgments)
    append_checkpoint(new_judgments)

judge_df = pd.DataFrame(judgments)
judge_df["row_id"] = judge_df["row_id"].astype(int)
judge_df = judge_df.drop_duplicates(subset=["row_id"], keep="last")

merged = df.merge(judge_df, on="row_id", how="left")
merged["judge_answer_count"] = pd.to_numeric(merged["judge_answer_count"], errors="coerce")
merged["judge_is_enumerated"] = merged["judge_is_enumerated"].astype("boolean")
merged["judge_valid_answer"] = merged["judge_valid_answer"].astype("boolean")

merged_path = OUTPUT_DIR / "judged_generations.csv"
merged.to_csv(merged_path, index=False)

with open(OUTPUT_DIR / "raw_judge_outputs.json", "w", encoding="utf-8") as f:
    json.dump(raw_outputs, f, indent=2, ensure_ascii=False)

summary = (
    merged.groupby(["eval_name", "alpha"], dropna=False)
    .agg(
        n=("question", "size"),
        judge_valid_rate=("judge_valid_answer", lambda s: float(s.fillna(False).mean())),
        judge_enumeration_rate=("judge_is_enumerated", lambda s: float(s.fillna(False).mean())),
        mean_judge_answer_count=("judge_answer_count", "mean"),
        parser_enumeration_rate=("is_enumerated", "mean"),
        parser_mean_answer_count=("generated_answer_count", "mean"),
        judge_parse_error_rate=("judge_confidence", lambda s: float((s == "parse_error").mean())),
    )
    .reset_index()
)
summary_path = OUTPUT_DIR / "judge_alpha_summary.csv"
summary.to_csv(summary_path, index=False)

for eval_name, part in merged.groupby("eval_name"):
    part.to_csv(OUTPUT_DIR / f"{eval_name}_judged_generations.csv", index=False)
for eval_name, part in summary.groupby("eval_name"):
    part.to_csv(OUTPUT_DIR / f"{eval_name}_judge_alpha_summary.csv", index=False)

print("Wrote:", merged_path)
print("Wrote:", summary_path)
print(summary)
