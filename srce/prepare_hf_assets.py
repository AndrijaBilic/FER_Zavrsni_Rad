import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import snapshot_download


MODEL_CONFIGS = {
    "qwen": "Qwen/Qwen3-4B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "mistral_7b_it": "mistralai/Mistral-7B-Instruct-v0.1",
    "llama31_8b_it": "meta-llama/Llama-3.1-8B-Instruct",
    "llama31_8b_base": "meta-llama/Llama-3.1-8B",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_8b_base": "Qwen/Qwen3-8B-Base",
    "gemma3_12b_it": "google/gemma-3-12b-it",
}


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def build_webq_csv(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "df_webq_balanced.csv"
    if output_csv.exists():
        print(f"Dataset CSV already exists: {output_csv}")
        return output_csv

    print("Downloading/building WebQuestions CSV...")
    ds_webq = load_dataset("web_questions", split="train")
    rows = []
    for ex in ds_webq:
        n_answers = len(ex["answers"])
        rows.append(
            {
                "question": ex["question"],
                "n_answers": n_answers,
                "label": 1 if n_answers >= 2 else 0,
                "label_str": "multiple" if n_answers >= 2 else "single",
            }
        )

    df = pd.DataFrame(rows)
    single = df[df["label"] == 0]
    multiple = df[df["label"] == 1]
    n_min = min(len(single), len(multiple))
    balanced = (
        pd.concat(
            [
                single.sample(n=n_min, random_state=42),
                multiple.sample(n=n_min, random_state=42),
            ]
        )
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    balanced.to_csv(output_csv, index=False)
    print(f"Wrote {len(balanced)} rows to {output_csv}")
    return output_csv


def cache_model(model_name: str):
    print(f"Caching model snapshot: {model_name}")
    snapshot_download(repo_id=model_name, resume_download=True, token=hf_token())
    print("Model snapshot cached.")


def main():
    model_choice = os.environ.get("MODEL_CHOICE", "mistral")
    if model_choice not in MODEL_CONFIGS:
        raise ValueError(f"Unknown MODEL_CHOICE={model_choice!r}. Use one of {list(MODEL_CONFIGS)}.")

    output_dir = Path(os.environ.get("DRIVE_PATH", "outputs"))
    build_webq_csv(output_dir)
    cache_model(MODEL_CONFIGS[model_choice])

    judge_model = os.environ.get("JUDGE_MODEL")
    if judge_model:
        cache_model(judge_model)


if __name__ == "__main__":
    main()
