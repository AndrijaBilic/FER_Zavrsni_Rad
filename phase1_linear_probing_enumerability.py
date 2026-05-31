import json
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_CHOICE = os.environ.get("MODEL_CHOICE", "mistral")

MODEL_CONFIGS = {
    "qwen": {
        "model_name": "Qwen/Qwen3-4B",
        "model_key": "qwen3_4b",
    },
    "mistral": {
        "model_name": "mistralai/Mistral-7B-v0.1",
        "model_key": "mistral_7b",
    },
    "mistral_7b_it": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
        "model_key": "mistral_7b_it",
    },
    "llama31_8b_it": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "model_key": "llama31_8b_it",
    },
    "llama31_8b_base": {
        "model_name": "meta-llama/Llama-3.1-8B",
        "model_key": "llama31_8b_base",
    },
    "qwen3_8b": {
        "model_name": "Qwen/Qwen3-8B",
        "model_key": "qwen3_8b",
    },
    "qwen3_8b_base": {
        "model_name": "Qwen/Qwen3-8B-Base",
        "model_key": "qwen3_8b_base",
    },
    "gemma3_12b_it": {
        "model_name": "google/gemma-3-12b-it",
        "model_key": "gemma3_12b_it",
    },
}

assert MODEL_CHOICE in MODEL_CONFIGS, f"Unknown MODEL_CHOICE={MODEL_CHOICE!r}. Use one of {list(MODEL_CONFIGS)}."


@dataclass
class Phase1Config:
    model_choice: str = MODEL_CHOICE
    model_name: str = MODEL_CONFIGS[MODEL_CHOICE]["model_name"]
    model_key: str = MODEL_CONFIGS[MODEL_CHOICE]["model_key"]
    drive_path: str = os.environ.get("DRIVE_PATH", "outputs")
    artifact_subdir: str = os.environ.get("ARTIFACT_SUBDIR", "phase1_linear_probing_enumerability")
    phase1_csv: str | None = os.environ.get("PHASE1_CSV")
    seed: int = int(os.environ.get("SEED", "42"))
    max_length: int = int(os.environ.get("MAX_LENGTH", "128"))
    batch_size: int = int(os.environ.get("BATCH_SIZE", "2"))
    model_quantization: str = os.environ.get("MODEL_QUANTIZATION", "4bit").lower()
    torch_dtype: str = os.environ.get("TORCH_DTYPE", "float16")
    n_permutations: int = int(os.environ.get("N_PERMUTATIONS", "10"))
    save_activations: bool = os.environ.get("SAVE_ACTIVATIONS", "0") in {"1", "true", "True", "yes", "YES"}


CFG = Phase1Config()
ARTIFACT_DIR = Path(CFG.drive_path) / CFG.artifact_subdir / CFG.model_key
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)

print(json.dumps(asdict(CFG), indent=2))
print(f"Artifacts will be saved to: {ARTIFACT_DIR}")


def dtype_from_name(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def get_text_config(model):
    cfg = model.config
    for attr in ("text_config", "language_config"):
        nested = getattr(cfg, attr, None)
        if nested is not None:
            return nested
    return cfg


def get_num_hidden_layers(model) -> int:
    text_cfg = get_text_config(model)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(text_cfg, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"Could not find hidden-layer count in {type(text_cfg).__name__}")


def get_hidden_size(model) -> int:
    text_cfg = get_text_config(model)
    for attr in ("hidden_size", "n_embd", "d_model"):
        value = getattr(text_cfg, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"Could not find hidden size in {type(text_cfg).__name__}")


def load_or_build_webquestions() -> pd.DataFrame:
    candidate_csvs = []
    if CFG.phase1_csv:
        candidate_csvs.append(Path(CFG.phase1_csv))
    candidate_csvs.extend(
        [
            Path(CFG.drive_path) / "df_webq_balanced.csv",
            Path.cwd() / "df_webq_balanced.csv",
            Path.cwd() / "data" / "df_webq_balanced.csv",
        ]
    )
    for path in candidate_csvs:
        if path.exists():
            df = pd.read_csv(path)
            if {"question", "n_answers", "label"}.issubset(df.columns):
                print(f"Loaded balanced dataframe: {path} ({len(df)} rows)")
                if "label_str" not in df.columns:
                    df["label_str"] = np.where(df["label"].astype(int) == 1, "enumerable", "singular")
                return df

    print("Balanced CSV not found. Loading WebQuestions from Hugging Face...")
    ds_webq = load_dataset("web_questions", split="train")
    answer_counts = [len(ex["answers"]) for ex in ds_webq]
    print("Answer count distribution:", dict(Counter(answer_counts)))

    rows = []
    for ex in ds_webq:
        n = len(ex["answers"])
        rows.append(
            {
                "question": ex["question"],
                "n_answers": n,
                "label": 1 if n >= 2 else 0,
                "label_str": "enumerable" if n >= 2 else "singular",
            }
        )
    df = pd.DataFrame(rows)
    singular = df[df.label == 0]
    enumerable = df[df.label == 1]
    n_min = min(len(singular), len(enumerable))
    balanced = pd.concat(
        [
            resample(singular, n_samples=n_min, replace=False, random_state=CFG.seed),
            enumerable,
        ]
    ).sample(frac=1, random_state=CFG.seed)
    balanced = balanced.reset_index(drop=True)
    balanced.to_csv(Path(CFG.drive_path) / "df_webq_balanced.csv", index=False)
    print(f"Built balanced dataframe: {len(balanced)} rows")
    return balanced


def load_model_and_tokenizer():
    offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    tokenizer = AutoTokenizer.from_pretrained(
        CFG.model_name,
        trust_remote_code=True,
        local_files_only=offline,
        token=hf_token(),
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if CFG.model_quantization in {"4bit", "nf4"}:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_from_name(CFG.torch_dtype),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif CFG.model_quantization in {"8bit", "int8"}:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif CFG.model_quantization in {"none", "bf16", "float16", "fp16"}:
        quant_config = None
    else:
        raise ValueError("MODEL_QUANTIZATION must be one of: 4bit, int8, none")

    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_name,
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


def extract_all_layers(df: pd.DataFrame, model, tokenizer, n_layers: int, hidden_size: int):
    layers = list(range(n_layers + 1))
    results = {
        layer: np.empty((len(df), hidden_size), dtype=np.float32)
        for layer in layers
    }
    questions = df["question"].tolist()
    for start in tqdm(range(0, len(questions), CFG.batch_size), desc="extract activations"):
        batch = questions[start : start + CFG.batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CFG.max_length,
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        for layer in layers:
            vecs = outputs.hidden_states[layer][:, -1, :].detach().float().cpu().numpy()
            results[layer][start : start + len(batch)] = vecs
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def run_probe(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=CFG.seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    probe = LogisticRegression(max_iter=1000, random_state=CFG.seed, C=1.0)
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)
    y_proba = probe.predict_proba(X_test)[:, 1]
    report = classification_report(
        y_test,
        y_pred,
        target_names=["singular", "enumerable"],
        output_dict=True,
        zero_division=0,
    )
    return probe, scaler, {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def plot_curves(layer_df: pd.DataFrame, perm_df: pd.DataFrame | None, best_layer: int):
    plt.figure(figsize=(11, 5))
    plt.plot(layer_df["layer"], layer_df["accuracy"], marker="o", label="Accuracy")
    plt.plot(layer_df["layer"], layer_df["roc_auc"], marker="s", label="ROC-AUC")
    plt.plot(layer_df["layer"], layer_df["macro_f1"], marker="^", label="Macro F1")
    if perm_df is not None and not perm_df.empty:
        plt.fill_between(
            perm_df["layer"],
            perm_df["mean_accuracy"] - perm_df["std_accuracy"],
            perm_df["mean_accuracy"] + perm_df["std_accuracy"],
            alpha=0.15,
            label="Permutation ±1 std",
        )
        plt.plot(perm_df["layer"], perm_df["mean_accuracy"], linestyle="--", alpha=0.8)
    plt.axvline(best_layer, color="black", linestyle=":", alpha=0.6)
    plt.ylim(0.4, 1.0)
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.title(f"Linear probing: {CFG.model_key}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "linear_probe_layer_curves.png", dpi=150)
    plt.close()


df = load_or_build_webquestions()
df.to_csv(ARTIFACT_DIR / "df_webq_balanced.csv", index=False)
y = df["label"].astype(int).to_numpy()

model, tokenizer = load_model_and_tokenizer()
n_layers = get_num_hidden_layers(model)
hidden_size = get_hidden_size(model)
print(f"Loaded {CFG.model_name}")
print(f"Layers: {n_layers}, hidden size: {hidden_size}")

layer_vectors = extract_all_layers(df, model, tokenizer, n_layers, hidden_size)

if CFG.save_activations:
    act_dir = ARTIFACT_DIR / "activations"
    act_dir.mkdir(exist_ok=True)
    for layer, arr in layer_vectors.items():
        np.save(act_dir / f"X_webq_{CFG.model_key}_layer{layer}.npy", arr)
    np.save(ARTIFACT_DIR / f"y_webq_{CFG.model_key}.npy", y)

layer_rows = []
probes = {}
for layer in tqdm(range(n_layers + 1), desc="probe layers"):
    probe, scaler, metrics = run_probe(layer_vectors[layer], y)
    probes[layer] = (probe, scaler)
    layer_rows.append(
        {
            "layer": layer,
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
            "macro_f1": metrics["macro_f1"],
        }
    )

layer_df = pd.DataFrame(layer_rows)
best_layer = int(layer_df.sort_values(["accuracy", "roc_auc", "macro_f1"], ascending=False).iloc[0]["layer"])
best_probe, best_scaler, best_metrics = run_probe(layer_vectors[best_layer], y)

perm_rows = []
if CFG.n_permutations > 0:
    rng = np.random.default_rng(CFG.seed)
    baseline_layers = list(range(0, n_layers + 1, 3))
    if n_layers not in baseline_layers:
        baseline_layers.append(n_layers)
    for layer in tqdm(baseline_layers, desc="permutation layers"):
        accs = []
        for _ in range(CFG.n_permutations):
            y_shuffled = rng.permutation(y)
            _, _, metrics = run_probe(layer_vectors[layer], y_shuffled)
            accs.append(metrics["accuracy"])
        perm_rows.append(
            {
                "layer": layer,
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
            }
        )

perm_df = pd.DataFrame(perm_rows)
layer_df.to_csv(ARTIFACT_DIR / "layer_probe_results.csv", index=False)
if not perm_df.empty:
    perm_df.to_csv(ARTIFACT_DIR / "permutation_baseline.csv", index=False)

X_best_scaled = StandardScaler().fit_transform(layer_vectors[best_layer])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
cv_scores = cross_val_score(
    LogisticRegression(max_iter=1000, random_state=CFG.seed),
    X_best_scaled,
    y,
    cv=cv,
    scoring="accuracy",
)

summary = {
    "model_choice": CFG.model_choice,
    "model_name": CFG.model_name,
    "model_key": CFG.model_key,
    "dataset": "WebQuestions balanced",
    "n_examples": int(len(df)),
    "labels": "singular (1 answer) vs enumerable (>=2 answers)",
    "n_layers": int(n_layers),
    "hidden_size": int(hidden_size),
    "best_layer": best_layer,
    "best_metrics": best_metrics,
    "cv_scores": [float(x) for x in cv_scores],
    "cv_mean": float(cv_scores.mean()),
    "cv_std": float(cv_scores.std()),
    "permutation_baseline_mean": float(perm_df["mean_accuracy"].mean()) if not perm_df.empty else None,
}
with open(ARTIFACT_DIR / "linear_probe_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

plot_curves(layer_df, perm_df, best_layer)

print("=" * 60)
print("PHASE 1 SUMMARY - Binary Linear Probing (WebQuestions)")
print("=" * 60)
print(f"  Model      : {CFG.model_name}")
print(f"  Dataset    : WebQuestions (balanced, {len(df)} examples)")
print("  Labels     : singular (1 answer) vs. enumerable (>=2 answers)")
print()
print(f"  Best layer : {best_layer} / {n_layers}")
print(f"  Accuracy   : {best_metrics['accuracy']:.4f}")
print(f"  ROC-AUC    : {best_metrics['roc_auc']:.4f}")
print(f"  Macro F1   : {best_metrics['macro_f1']:.4f}")
print(f"  CV (5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
if not perm_df.empty:
    print(f"  Permutation baseline: {perm_df['mean_accuracy'].mean():.4f}")
print("=" * 60)
