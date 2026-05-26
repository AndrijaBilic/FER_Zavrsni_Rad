# %% [markdown]
# # Phase 2: Activation Steering for Enumerability
#
# This notebook-style script adapts Lavi et al. ("Detecting (Un)answerability in
# Large Language Models with Linear Directions") to WebQuestions enumerability.
#
# Core adaptation:
# - answerable vs. unanswerable becomes single-answer vs. multiple-answer
# - the "unanswerable" next-token objective becomes a contrastive
#   log p("multiple") - log p("single") objective
# - directions are computed on chat-formatted prompts at the end-of-instruction
#   template positions, not on the raw question-only activations from Phase 1

# %%
# Optional Colab setup:
# %pip install -q datasets transformers accelerate bitsandbytes scikit-learn pandas numpy tqdm matplotlib seaborn
#
# If you want to reuse the Phase 1 CSV from Google Drive:
# from google.colab import drive
# drive.mount("/content/drive")

# %%
import json
import os
import random
import re
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)


# %%
# -- Model Selection ---------------------------------------------------------
# Match the Phase 1 notebook: change this single value to switch models.
MODEL_CHOICE = os.environ.get("MODEL_CHOICE", "qwen")   # e.g. mistral, llama31_8b_base, llama31_8b_it, qwen3_8b_base, qwen3_8b, gemma3_12b_it

MODEL_CONFIGS = {
    "qwen": {
        "model_name": "Qwen/Qwen3-4B",
        "model_key": "qwen3_4b",
        "prompt_style": "chat",
        "candidate_layers": list(range(15, 31)),
    },
    "mistral": {
        "model_name": "mistralai/Mistral-7B-v0.1",
        "model_key": "mistral_7b",
        "prompt_style": "plain",
        "candidate_layers": list(range(18, 29)),
    },
    "llama31_8b_it": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "model_key": "llama31_8b_it",
        "prompt_style": "chat",
        "candidate_layers": list(range(16, 31)),
    },
    "llama31_8b_base": {
        "model_name": "meta-llama/Llama-3.1-8B",
        "model_key": "llama31_8b_base",
        "prompt_style": "plain",
        "candidate_layers": list(range(16, 31)),
    },
    "qwen3_8b": {
        "model_name": "Qwen/Qwen3-8B",
        "model_key": "qwen3_8b",
        "prompt_style": "chat",
        "candidate_layers": list(range(16, 35)),
    },
    "qwen3_8b_base": {
        "model_name": "Qwen/Qwen3-8B-Base",
        "model_key": "qwen3_8b_base",
        "prompt_style": "plain",
        "candidate_layers": list(range(16, 35)),
    },
    "gemma3_12b_it": {
        "model_name": "google/gemma-3-12b-it",
        "model_key": "gemma3_12b_it",
        "prompt_style": "chat",
        "candidate_layers": list(range(18, 43)),
    },
}

assert MODEL_CHOICE in MODEL_CONFIGS, f"Unknown MODEL_CHOICE '{MODEL_CHOICE}'. Use one of {list(MODEL_CONFIGS)}."


@dataclass
class Phase2Config:
    model_choice: str = MODEL_CHOICE
    model_name: str = MODEL_CONFIGS[MODEL_CHOICE]["model_name"]
    model_key: str = MODEL_CONFIGS[MODEL_CHOICE]["model_key"]
    # "chat" uses tokenizer.apply_chat_template; "plain" sends the instruction
    # directly. Mistral-7B-v0.1 is a base model, so plain prompts are safer.
    prompt_style: str = MODEL_CONFIGS[MODEL_CHOICE]["prompt_style"]
    prompt_variant: str = "zero_shot_label"
    drive_path: str = "/content/drive/MyDrive/thesis_probing"
    artifact_subdir: str = "phase2_activation_steering_enumerability"
    phase1_csv: str | None = None
    seed: int = 42

    # Pilot defaults. Increase these only after the steering effect looks real.
    max_train_per_class: int = 48
    max_val_per_class: int = 32
    max_test_per_class: int = 32

    batch_size_activations: int = 4
    batch_size_scoring: int = 4
    batch_size_generation: int = 2
    max_new_tokens: int = 64

    model_quantization: str = "4bit"
    use_4bit: bool = True
    use_double_quant: bool = True
    low_cpu_mem_usage: bool = True
    torch_dtype: str = "float16"

    # Paper-like candidate positions. None means: infer the trailing chat-template
    # positions after the user instruction and use all of them.
    # For the pilot run, use only -1. Setting this to None does the fuller paper-
    # style sweep over all trailing template positions.
    positions: tuple[int, ...] | None = (-1,)

    # Restrict candidate direction layers for the first run. None means all
    # layers. Fractions are converted after loading the model.
    layer_fractions: tuple[float, ...] = (0.33, 0.50, 0.66, 0.75, 0.90)
    candidate_layers: list[int] | None = None

    # Good first sweep. The paper uses [-2, 2]; your setup doc also mentions
    # larger alphas, which you can add after the first sanity check.
    alphas: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    max_manual_questions: int = 5
    webq_eval_per_class: int = 0
    max_ood_questions: int = 0
    stop_strings: tuple[str, ...] = ("\n\nQuestion:", "\nQuestion:", "\n\nQ:", "\nQ:")

    # If False, we follow the paper text and add the vector only at the selected
    # end-of-instruction position. If True, this matches the public repo hook,
    # which adds the vector at all token positions.
    steer_all_positions: bool = False


CFG = Phase2Config(candidate_layers=MODEL_CONFIGS[MODEL_CHOICE].get("candidate_layers"))

if os.environ.get("MAX_TRAIN_PER_CLASS"):
    CFG.max_train_per_class = int(os.environ["MAX_TRAIN_PER_CLASS"])
if os.environ.get("MAX_VAL_PER_CLASS"):
    CFG.max_val_per_class = int(os.environ["MAX_VAL_PER_CLASS"])
if os.environ.get("MAX_TEST_PER_CLASS"):
    CFG.max_test_per_class = int(os.environ["MAX_TEST_PER_CLASS"])
if os.environ.get("MAX_MANUAL_QUESTIONS"):
    CFG.max_manual_questions = int(os.environ["MAX_MANUAL_QUESTIONS"])
if os.environ.get("WEBQ_EVAL_PER_CLASS"):
    CFG.webq_eval_per_class = int(os.environ["WEBQ_EVAL_PER_CLASS"])
if os.environ.get("MAX_OOD_QUESTIONS"):
    CFG.max_ood_questions = int(os.environ["MAX_OOD_QUESTIONS"])
if os.environ.get("ALPHAS"):
    CFG.alphas = tuple(float(x.strip()) for x in os.environ["ALPHAS"].split(",") if x.strip())
if os.environ.get("CANDIDATE_LAYERS"):
    CFG.candidate_layers = [int(x.strip()) for x in os.environ["CANDIDATE_LAYERS"].split(",") if x.strip()]
if os.environ.get("ARTIFACT_SUBDIR"):
    CFG.artifact_subdir = os.environ["ARTIFACT_SUBDIR"]
if os.environ.get("DRIVE_PATH"):
    CFG.drive_path = os.environ["DRIVE_PATH"]
if os.environ.get("PHASE1_CSV"):
    CFG.phase1_csv = os.environ["PHASE1_CSV"]
if os.environ.get("PROMPT_VARIANT"):
    CFG.prompt_variant = os.environ["PROMPT_VARIANT"].strip()
if os.environ.get("MODEL_QUANTIZATION"):
    CFG.model_quantization = os.environ["MODEL_QUANTIZATION"].strip().lower()
    CFG.use_4bit = CFG.model_quantization in {"4bit", "nf4"}
if os.environ.get("USE_4BIT"):
    CFG.use_4bit = os.environ["USE_4BIT"].strip() not in {"0", "false", "False", "no", "NO"}
    CFG.model_quantization = "4bit" if CFG.use_4bit else "none"
if os.environ.get("USE_DOUBLE_QUANT"):
    CFG.use_double_quant = os.environ["USE_DOUBLE_QUANT"].strip() not in {"0", "false", "False", "no", "NO"}
if os.environ.get("TORCH_DTYPE"):
    CFG.torch_dtype = os.environ["TORCH_DTYPE"].strip()
if os.environ.get("BATCH_SIZE_ACTIVATIONS"):
    CFG.batch_size_activations = int(os.environ["BATCH_SIZE_ACTIVATIONS"])
if os.environ.get("BATCH_SIZE_SCORING"):
    CFG.batch_size_scoring = int(os.environ["BATCH_SIZE_SCORING"])
if os.environ.get("BATCH_SIZE_GENERATION"):
    CFG.batch_size_generation = int(os.environ["BATCH_SIZE_GENERATION"])
if os.environ.get("MAX_NEW_TOKENS"):
    CFG.max_new_tokens = int(os.environ["MAX_NEW_TOKENS"])
if os.environ.get("STEER_ALL_POSITIONS"):
    CFG.steer_all_positions = os.environ["STEER_ALL_POSITIONS"].strip() in {"1", "true", "True", "yes", "YES"}

ARTIFACT_DIR = Path(CFG.drive_path) / CFG.artifact_subdir / CFG.model_key
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)

print(json.dumps(asdict(CFG), indent=2))
print(f"Artifacts will be saved to: {ARTIFACT_DIR}")


# %% [markdown]
# ## 1. Dataset and Prompt
#
# The prompt is designed so the first generated content token is expected to be
# `single` or `multiple`. This lets us select directions using the same kind of
# token-probability objective as Lavi et al.

# %%
ENUMERABILITY_PROMPT = """Given the following question, decide whether it has one correct answer or multiple correct answers, then answer it.

Respond in exactly this format:
Answer type: single or multiple
Answer: <answer or list of answers>

Question: {question}
Answer type:"""


FEW_SHOT_LABEL_PROMPT = """Decide whether each question has one correct answer or multiple correct answers, then answer it.

Question: What is the capital of France?
Answer type: single
Answer: Paris

Question: Which countries border Germany?
Answer type: multiple
Answer: Denmark, Poland, Czech Republic, Austria, Switzerland, France, Luxembourg, Belgium, Netherlands

Question: Who wrote Pride and Prejudice?
Answer type: single
Answer: Jane Austen

Question: Which colors are in the French flag?
Answer type: multiple
Answer: blue, white, red

Question: {question}
Answer type:"""


def format_enumerability_instruction(question: str) -> str:
    if CFG.prompt_variant == "zero_shot_label":
        template = ENUMERABILITY_PROMPT
    elif CFG.prompt_variant == "few_shot_label":
        template = FEW_SHOT_LABEL_PROMPT
    else:
        raise ValueError("PROMPT_VARIANT must be one of: zero_shot_label, few_shot_label")
    return template.format(question=question.strip())


def load_or_build_webquestions(cfg: Phase2Config) -> pd.DataFrame:
    """Load the Phase 1 balanced CSV when available, otherwise rebuild it."""
    candidate_csvs = []
    if cfg.phase1_csv:
        candidate_csvs.append(Path(cfg.phase1_csv))
    candidate_csvs.extend(
        [
            Path(cfg.drive_path) / "df_webq_balanced.csv",
            Path.cwd() / "df_webq_balanced.csv",
            Path.cwd() / "data" / "df_webq_balanced.csv",
        ]
    )

    for phase1_csv in candidate_csvs:
        if not phase1_csv.exists():
            continue
        df = pd.read_csv(phase1_csv)
        expected_cols = {"question", "n_answers", "label"}
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(f"{phase1_csv} is missing columns: {sorted(missing)}")
        print(f"Loaded Phase 1 balanced dataframe: {phase1_csv} ({len(df)} rows)")
        return df

    print("Phase 1 CSV not found in:")
    for phase1_csv in candidate_csvs:
        print(" ", phase1_csv)
    print("Loading WebQuestions from Hugging Face...")
    ds_webq = load_dataset("web_questions", split="train")
    rows = []
    for ex in ds_webq:
        n = len(ex["answers"])
        rows.append(
            {
                "question": ex["question"],
                "answers": ex["answers"],
                "n_answers": n,
                "label": 1 if n >= 2 else 0,
                "label_str": "multiple" if n >= 2 else "single",
            }
        )
    df = pd.DataFrame(rows)

    # Match Phase 1: undersample the majority class.
    single = df[df.label == 0]
    multiple = df[df.label == 1]
    n_min = min(len(single), len(multiple))
    df = pd.concat(
        [
            single.sample(n=n_min, random_state=CFG.seed),
            multiple.sample(n=n_min, random_state=CFG.seed),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=CFG.seed)
    df = df.reset_index(drop=True)
    print(f"Built balanced dataframe from WebQuestions ({len(df)} rows)")
    return df


def make_splits(df: pd.DataFrame, cfg: Phase2Config):
    train_df, temp_df = train_test_split(
        df, test_size=0.4, random_state=cfg.seed, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=cfg.seed, stratify=temp_df["label"]
    )

    def cap_per_class(split_df: pd.DataFrame, max_per_class: int) -> pd.DataFrame:
        parts = []
        for label in [0, 1]:
            part = split_df[split_df.label == label]
            n = min(len(part), max_per_class)
            parts.append(part.sample(n=n, random_state=cfg.seed))
        return pd.concat(parts).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)

    train_df = cap_per_class(train_df, cfg.max_train_per_class)
    val_df = cap_per_class(val_df, cfg.max_val_per_class)
    test_df = cap_per_class(test_df, cfg.max_test_per_class)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(name, split_df["label_str" if "label_str" in split_df.columns else "label"].value_counts())
        split_df.to_csv(ARTIFACT_DIR / f"webq_{name}.csv", index=False)

    return train_df, val_df, test_df


df_webq = load_or_build_webquestions(CFG)
if "label_str" not in df_webq.columns:
    df_webq["label_str"] = np.where(df_webq["label"].astype(int) == 1, "multiple", "single")

train_df, val_df, test_df = make_splits(df_webq, CFG)

train_single = [format_enumerability_instruction(q) for q in train_df[train_df.label == 0]["question"]]
train_multiple = [format_enumerability_instruction(q) for q in train_df[train_df.label == 1]["question"]]
val_prompts = [format_enumerability_instruction(q) for q in val_df["question"]]
test_prompts = [format_enumerability_instruction(q) for q in test_df["question"]]


# %% [markdown]
# ## 2. Model, Chat Template, and Target Tokens

# %%
def dtype_from_name(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def get_text_config(model):
    """Return the decoder/text config for plain and wrapper-style CausalLMs."""
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


def load_model_and_tokenizer(cfg: Phase2Config):
    offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        local_files_only=offline,
        token=hf_token(),
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    quantization = cfg.model_quantization.lower()
    if quantization in {"4bit", "nf4"}:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_from_name(cfg.torch_dtype),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=cfg.use_double_quant,
        )
    elif quantization in {"8bit", "int8"}:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization in {"none", "bf16", "float16", "fp16"}:
        quant_config = None
    else:
        raise ValueError("MODEL_QUANTIZATION must be one of: 4bit, int8, none")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype_from_name(cfg.torch_dtype),
        quantization_config=quant_config,
        low_cpu_mem_usage=cfg.low_cpu_mem_usage,
        local_files_only=offline,
        token=hf_token(),
    ).eval()
    model.config.output_hidden_states = False
    model.requires_grad_(False)
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer(CFG)
device = next(model.parameters()).device
N_LAYERS = get_num_hidden_layers(model)
HIDDEN_SIZE = get_hidden_size(model)
print(f"Loaded {CFG.model_name} on {device}")
print(f"Layers: {N_LAYERS}, hidden size: {HIDDEN_SIZE}")


def resolve_candidate_layers(cfg: Phase2Config, n_layers: int) -> list[int]:
    if cfg.candidate_layers is not None:
        layers = sorted(set(int(layer) for layer in cfg.candidate_layers))
        return [layer for layer in layers if 0 <= layer < n_layers]
    layers = [int(round(frac * (n_layers - 1))) for frac in cfg.layer_fractions]
    return sorted(set(max(0, min(n_layers - 1, layer)) for layer in layers))


SELECT_LAYERS = resolve_candidate_layers(CFG, N_LAYERS)
print("Candidate layers:", SELECT_LAYERS)


# %%
def format_model_prompt(instruction: str) -> str:
    if CFG.prompt_style == "plain":
        return instruction

    messages = [{"role": "user", "content": instruction}]
    try:
        # Qwen3 supports disabling thinking mode. Other tokenizers may not.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def tokenize_instructions(instructions: Sequence[str]):
    prompts = [format_model_prompt(inst) for inst in instructions]
    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")


def infer_end_of_instruction_positions() -> list[int]:
    if CFG.prompt_style == "plain":
        print("Plain prompt style has no chat-template suffix; using the final prompt token.")
        return [-1]

    sentinel = "<<<INSTRUCTION_SENTINEL>>>"
    formatted = format_model_prompt(sentinel)
    if sentinel not in formatted:
        print("Could not isolate chat-template suffix; falling back to the final token only.")
        return [-1]
    suffix = formatted.split(sentinel, 1)[1]
    suffix_toks = tokenizer.encode(suffix, add_special_tokens=False)
    if not suffix_toks:
        return [-1]
    print("End-of-instruction suffix repr:", repr(suffix))
    print("Suffix token count:", len(suffix_toks))
    print("Suffix tokens:", [tokenizer.decode([tok]) for tok in suffix_toks])
    return list(range(-len(suffix_toks), 0))


POSITIONS = list(CFG.positions) if CFG.positions is not None else infer_end_of_instruction_positions()
print("Candidate positions:", POSITIONS)


def first_token_ids(variants: Iterable[str]) -> list[int]:
    ids = []
    for text in variants:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            ids.append(encoded[0])
    return sorted(set(ids))


# Do not include "\nsingle" / "\nmultiple" here. Their first token is the same
# newline token, so it is not discriminative for a next-token label objective.
SINGLE_TOKS = first_token_ids(["single", " single", "Single", " Single"])
MULTIPLE_TOKS = first_token_ids(["multiple", " multiple", "Multiple", " Multiple"])

print("single token ids:", SINGLE_TOKS, [tokenizer.decode([i]) for i in SINGLE_TOKS])
print("multiple token ids:", MULTIPLE_TOKS, [tokenizer.decode([i]) for i in MULTIPLE_TOKS])
overlap = set(SINGLE_TOKS) & set(MULTIPLE_TOKS)
if overlap:
    raise ValueError(f"single/multiple token id sets overlap: {overlap}")


# %% [markdown]
# ## 3. Hook Utilities

# %%
@contextmanager
def add_hooks(module_forward_pre_hooks, module_forward_hooks=()):
    handles = []
    try:
        for module, hook in module_forward_pre_hooks:
            handles.append(module.register_forward_pre_hook(hook))
        for module, hook in module_forward_hooks:
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def get_block_modules(model):
    module = model
    candidate_paths = (
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "language_model", "layers"),
        ("model", "decoder", "layers"),
    )
    for path in candidate_paths:
        module = model
        for attr in path:
            module = getattr(module, attr, None)
            if module is None:
                break
        if module is not None:
            print("Transformer block path:", ".".join(path))
            return module
    raise ValueError("Could not find transformer blocks in known decoder module paths")


BLOCKS = get_block_modules(model)


def normalize_vector(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec / (vec.norm(dim=-1, keepdim=True) + eps)


def finite_numpy(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    if mask.all():
        return arr
    replacement = float(np.median(arr[mask])) if mask.any() else 0.0
    print(f"Warning: replacing {(~mask).sum()} non-finite values in {name} with {replacement}.")
    return np.where(mask, arr, replacement)


def make_activation_addition_pre_hook(
    vector: torch.Tensor,
    coeff: float,
    target_pos: int | None,
):
    """Add coeff * vector to a layer input, either at one token position or all positions."""

    def hook_fn(module, inputs):
        if isinstance(inputs, tuple):
            activation = inputs[0].clone()
            rest = inputs[1:]
        else:
            activation = inputs.clone()
            rest = None

        vec = vector.to(device=activation.device, dtype=activation.dtype)
        if target_pos is None:
            activation = activation + coeff * vec
        else:
            activation[:, target_pos, :] = activation[:, target_pos, :] + coeff * vec

        if rest is None:
            return activation
        return (activation, *rest)

    return hook_fn


# %% [markdown]
# ## 4. Candidate Directions: multiple - single

# %%
def get_mean_activations(
    instructions: Sequence[str],
    positions: Sequence[int],
    layers: Sequence[int],
    batch_size: int,
) -> torch.Tensor:
    n_positions = len(positions)
    d_model = HIDDEN_SIZE
    n_samples = len(instructions)

    cache = torch.zeros(
        (n_positions, len(layers), d_model),
        dtype=torch.float64,
        device=device,
    )

    def make_cache_hook(cache_layer_idx: int):
        def hook_fn(module, inputs):
            activation = inputs[0] if isinstance(inputs, tuple) else inputs
            selected = activation[:, positions, :].detach().to(cache.dtype)
            cache[:, cache_layer_idx, :] += selected.sum(dim=0) / n_samples

        return hook_fn

    hooks = [(BLOCKS[layer], make_cache_hook(i)) for i, layer in enumerate(layers)]

    for start in tqdm(range(0, len(instructions), batch_size), desc="mean activations"):
        batch = instructions[start : start + batch_size]
        inputs = tokenize_instructions(batch).to(device)
        with torch.no_grad(), add_hooks(hooks):
            model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=False)

    return cache


single_means = get_mean_activations(train_single, POSITIONS, SELECT_LAYERS, CFG.batch_size_activations)
multiple_means = get_mean_activations(train_multiple, POSITIONS, SELECT_LAYERS, CFG.batch_size_activations)

candidate_directions = multiple_means - single_means
candidate_directions = normalize_vector(candidate_directions).to(torch.float32)
torch.save(candidate_directions.cpu(), ARTIFACT_DIR / "candidate_directions_multiple_minus_single.pt")

print("candidate_directions:", tuple(candidate_directions.shape))


# %% [markdown]
# ## 5. Direction Selection by Token Objective
#
# Selection objective:
#
# `mean(log p(multiple) - log p(single))`
#
# under an activation addition intervention.

# %%
def type_logit_scores(
    instructions: Sequence[str],
    fwd_pre_hooks=(),
    batch_size: int = 4,
) -> torch.Tensor:
    scores = torch.empty(len(instructions), dtype=torch.float32, device=device)
    nonfinite_logits = 0

    for start in range(0, len(instructions), batch_size):
        batch = instructions[start : start + batch_size]
        inputs = tokenize_instructions(batch).to(device)
        with torch.no_grad(), add_hooks(fwd_pre_hooks):
            logits = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                use_cache=False,
            ).logits[:, -1, :]

        logits = logits.float()
        nonfinite_logits += int((~torch.isfinite(logits)).sum().item())
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

        # The full-vocabulary normalizer cancels, so this is equivalent to
        # log p(multiple variants) - log p(single variants), but more stable.
        log_multiple = torch.logsumexp(logits[:, MULTIPLE_TOKS], dim=-1)
        log_single = torch.logsumexp(logits[:, SINGLE_TOKS], dim=-1)
        scores[start : start + len(batch)] = torch.nan_to_num(log_multiple - log_single, nan=0.0)

    if nonfinite_logits:
        print(f"Warning: replaced {nonfinite_logits} non-finite logits while scoring {len(instructions)} prompts.")
    return scores


baseline_val_scores = finite_numpy(
    type_logit_scores(val_prompts, batch_size=CFG.batch_size_scoring).detach().cpu().numpy(),
    "baseline_val_scores",
)
print(
    "Baseline validation token score:",
    float(np.mean(baseline_val_scores)),
    "AUC:",
    roc_auc_score(val_df["label"], baseline_val_scores),
)


def select_direction_by_steering(candidate_dirs: torch.Tensor):
    rows = []
    best = None

    for pos_idx, pos in enumerate(POSITIONS):
        for candidate_layer_idx, layer in tqdm(list(enumerate(SELECT_LAYERS)), desc=f"select pos {pos}"):
            direction = candidate_dirs[pos_idx, candidate_layer_idx].to(device)
            target_pos = None if CFG.steer_all_positions else pos
            hook = make_activation_addition_pre_hook(direction, coeff=1.0, target_pos=target_pos)
            scores = type_logit_scores(
                val_prompts,
                fwd_pre_hooks=[(BLOCKS[layer], hook)],
                batch_size=CFG.batch_size_scoring,
            )
            score_np = finite_numpy(scores.detach().cpu().numpy(), f"val_scores_layer_{layer}_pos_{pos}")
            mean_score = float(np.mean(score_np))
            auc = roc_auc_score(val_df["label"], score_np)
            row = {
                "position": pos,
                "position_index": pos_idx,
                "candidate_layer_index": candidate_layer_idx,
                "layer": layer,
                "steering_score": mean_score,
                "val_auc_under_addition": auc,
            }
            rows.append(row)
            if best is None or mean_score > best["steering_score"]:
                best = row

    eval_df = pd.DataFrame(rows).sort_values("steering_score", ascending=False)
    eval_df.to_csv(ARTIFACT_DIR / "direction_selection_scores.csv", index=False)
    with open(ARTIFACT_DIR / "best_direction_metadata.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    best_direction = candidate_dirs[best["position_index"], best["candidate_layer_index"]].detach().cpu()
    torch.save(best_direction, ARTIFACT_DIR / "best_direction.pt")
    return best, best_direction, eval_df


best_meta, best_direction_cpu, selection_df = select_direction_by_steering(candidate_directions)
print("Best direction:")
print(json.dumps(best_meta, indent=2))
selection_df.head(10)


# %% [markdown]
# ## 6. Direction Projection Classifier
#
# This is the paper's scalar projection idea, adapted to single/multiple labels.

# %%
def hidden_vectors_at(prompts: Sequence[str], pos: int, layer: int, batch_size: int) -> torch.Tensor:
    vectors = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="hidden vectors"):
        batch = prompts[start : start + batch_size]
        inputs = tokenize_instructions(batch).to(device)
        with torch.no_grad():
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
        # hidden_states[0] is embeddings; transformer layer outputs are 1..N.
        # The hooks above operate on layer input. For projection we use the
        # corresponding layer input by taking hidden_states[layer].
        vectors.append(outputs.hidden_states[layer][:, pos, :].float().detach().cpu())
    return torch.cat(vectors, dim=0)


def projection_scores(prompts: Sequence[str], direction: torch.Tensor, pos: int, layer: int) -> np.ndarray:
    direction = normalize_vector(direction.float()).cpu()
    vectors = hidden_vectors_at(prompts, pos=pos, layer=layer, batch_size=CFG.batch_size_scoring)
    scores = (torch.nan_to_num(vectors.float(), nan=0.0, posinf=0.0, neginf=0.0) @ direction).numpy()
    return finite_numpy(scores, f"projection_scores_layer_{layer}_pos_{pos}")


val_proj_scores = projection_scores(
    val_prompts,
    best_direction_cpu,
    pos=best_meta["position"],
    layer=best_meta["layer"],
)
fpr, tpr, thresholds = roc_curve(val_df["label"], val_proj_scores)
best_idx = np.sqrt((fpr - 0) ** 2 + (tpr - 1) ** 2).argmin()
projection_threshold = float(thresholds[best_idx])
print("Projection threshold:", projection_threshold)
print("Validation projection AUC:", roc_auc_score(val_df["label"], val_proj_scores))

test_proj_scores = projection_scores(
    test_prompts,
    best_direction_cpu,
    pos=best_meta["position"],
    layer=best_meta["layer"],
)
test_pred = (test_proj_scores > projection_threshold).astype(int)

projection_results = {
    "threshold": projection_threshold,
    "val_auc": float(roc_auc_score(val_df["label"], val_proj_scores)),
    "test_auc": float(roc_auc_score(test_df["label"], test_proj_scores)),
    "test_accuracy": float(accuracy_score(test_df["label"], test_pred)),
    "classification_report": classification_report(
        test_df["label"], test_pred, target_names=["single", "multiple"], output_dict=True
    ),
}
with open(ARTIFACT_DIR / "projection_classifier_results.json", "w", encoding="utf-8") as f:
    json.dump(projection_results, f, indent=2)

print(json.dumps(projection_results, indent=2))


# %% [markdown]
# ## 7. Causal Steering and Generation

# %%
def trim_generation(text: str) -> str:
    cut = len(text)
    for stop in CFG.stop_strings:
        idx = text.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip()


def generate_completions(
    prompts: Sequence[str],
    direction: torch.Tensor | None,
    layer: int | None,
    pos: int | None,
    alpha: float,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict]:
    hooks = []
    if direction is not None:
        target_pos = None if CFG.steer_all_positions else pos
        hook = make_activation_addition_pre_hook(
            normalize_vector(direction).to(device),
            coeff=alpha,
            target_pos=target_pos,
        )
        hooks = [(BLOCKS[layer], hook)]

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    completions = []
    for start in tqdm(range(0, len(prompts), batch_size), desc=f"generate alpha={alpha}"):
        batch = prompts[start : start + batch_size]
        inputs = tokenize_instructions(batch).to(device)
        with torch.no_grad(), add_hooks(hooks):
            generated = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=generation_config,
            )

        new_tokens = generated[:, inputs.input_ids.shape[-1] :]
        for offset, toks in enumerate(new_tokens):
            completions.append(
                {
                    "prompt": batch[offset],
                    "response": trim_generation(tokenizer.decode(toks, skip_special_tokens=True)),
                }
            )
    return completions


def parse_answer_type(response: str) -> str | None:
    match = re.search(r"\b(single|multiple)\b", response, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def answer_text(response: str) -> str:
    match = re.search(r"Answer\s*:\s*(.*)", response, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    # Generation starts immediately after "Answer type:", so remove the type word.
    return re.sub(r"^\s*(single|multiple)\b\s*", "", response, flags=re.IGNORECASE).strip()


def strip_list_marker(text: str) -> str:
    return re.sub(r"^\s*(?:[-*]|\d+[\).:])\s*", "", text).strip()


def clean_answer_item(text: str) -> str:
    text = strip_list_marker(text)
    text = re.sub(r"^(?:and|or)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    return text.strip(" \t\n\r.;:")


def looks_like_atomic_answer(text: str) -> bool:
    text = clean_answer_item(text)
    if not text:
        return False
    words = text.split()
    if len(words) > 10:
        return False
    prose_markers = {
        "because",
        "which",
        "where",
        "while",
        "although",
        "however",
        "therefore",
        "including",
        "such",
        "consisting",
        "known",
    }
    return not any(marker in {w.strip(",.").lower() for w in words} for marker in prose_markers)


def split_comma_and_answer_list(text: str) -> list[str]:
    # Turn "A, B, and C" into "A, B, C". This handles the common final item
    # without trying to parse arbitrary English coordination.
    text = re.sub(r",\s*(?:and|or)\s+", ", ", text, flags=re.IGNORECASE)
    parts = [clean_answer_item(p) for p in text.split(",")]
    return [p for p in parts if looks_like_atomic_answer(p)]


def answer_count_details(response: str) -> dict:
    text = answer_text(response)
    if not text:
        return {"count": 0, "method": "empty", "confidence": "high"}

    bullet_lines = [
        clean_answer_item(line)
        for line in text.splitlines()
        if re.match(r"^\s*(?:[-*]|\d+[\).:])\s+\S+", line)
    ]
    if len(bullet_lines) >= 2:
        atomic = [line for line in bullet_lines if looks_like_atomic_answer(line)]
        return {
            "count": len(atomic) if len(atomic) >= 2 else len(bullet_lines),
            "method": "bullet_or_numbered_lines",
            "confidence": "high",
        }

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(nonempty_lines) >= 2 and all(looks_like_atomic_answer(line) for line in nonempty_lines):
        return {"count": len(nonempty_lines), "method": "short_lines", "confidence": "high"}

    semicolon_parts = [clean_answer_item(p) for p in re.split(r"\s*;\s*", text) if p.strip()]
    if len(semicolon_parts) >= 2:
        atomic = [part for part in semicolon_parts if looks_like_atomic_answer(part)]
        if len(atomic) >= 2:
            return {"count": len(atomic), "method": "semicolon", "confidence": "high"}

    comma_parts = split_comma_and_answer_list(text)
    raw_comma_parts = [p.strip() for p in text.split(",") if p.strip()]
    if 2 <= len(comma_parts) <= 30 and len(comma_parts) == len(raw_comma_parts):
        return {"count": len(comma_parts), "method": "comma_list", "confidence": "medium"}

    if re.search(r"\b(?:and|or)\b", text, flags=re.IGNORECASE):
        conj_parts = [
            clean_answer_item(p)
            for p in re.split(r"\s+(?:and|or)\s+", text, flags=re.IGNORECASE)
            if p.strip()
        ]
        if 2 <= len(conj_parts) <= 6 and all(looks_like_atomic_answer(p) for p in conj_parts):
            return {"count": len(conj_parts), "method": "conjunction", "confidence": "medium"}

    return {"count": 1, "method": "fallback_single", "confidence": "low"}


def count_generated_answers(response: str) -> int:
    return int(answer_count_details(response)["count"])


def summarize_completions(completions: list[dict], labels: Sequence[int], alpha: float) -> dict:
    pred_types = [parse_answer_type(item["response"]) for item in completions]
    pred_labels = [1 if typ == "multiple" else 0 if typ == "single" else -1 for typ in pred_types]
    count_details = [answer_count_details(item["response"]) for item in completions]
    counts = [int(detail["count"]) for detail in count_details]
    enumerated = np.asarray([count >= 2 for count in counts])
    labels_arr = np.asarray(labels)
    pred_arr = np.asarray(pred_labels)
    valid = pred_arr >= 0
    mismatch = valid & (pred_arr != enumerated.astype(int))

    return {
        "alpha": alpha,
        "n": len(completions),
        "parse_rate": float(valid.mean()),
        "type_accuracy_valid_only": float((pred_arr[valid] == labels_arr[valid]).mean()) if valid.any() else None,
        "multiple_type_rate": float((pred_arr == 1).mean()),
        "enumeration_rate": float(enumerated.mean()),
        "mean_generated_answer_count": float(np.mean(counts)),
        "mean_count_single_gold": float(np.mean([c for c, y in zip(counts, labels) if y == 0])),
        "mean_count_multiple_gold": float(np.mean([c for c, y in zip(counts, labels) if y == 1])),
        "label_count_mismatch_rate": float(mismatch.mean()) if valid.any() else None,
        "low_confidence_count_rate": float(np.mean([detail["confidence"] == "low" for detail in count_details])),
    }


ALL_CURATED_EVAL_QUESTIONS = [
    {
        "question": "Name a country in South America.",
        "gold_label": 0,
        "gold_label_str": "single",
        "n_gold_answers": 1,
    },
    {
        "question": "Name countries in South America.",
        "gold_label": 1,
        "gold_label_str": "multiple",
        "n_gold_answers": 12,
    },
    {
        "question": "Name a movie directed by Christopher Nolan.",
        "gold_label": 0,
        "gold_label_str": "single",
        "n_gold_answers": 1,
    },
    {
        "question": "Name movies directed by Christopher Nolan.",
        "gold_label": 1,
        "gold_label_str": "multiple",
        "n_gold_answers": 8,
    },
    {
        "question": "Name a language spoken in Switzerland.",
        "gold_label": 0,
        "gold_label_str": "single",
        "n_gold_answers": 1,
    },
    {
        "question": "Name languages spoken in Switzerland.",
        "gold_label": 1,
        "gold_label_str": "multiple",
        "n_gold_answers": 4,
    },
    {
        "question": "Name an actor who played Spider-Man.",
        "gold_label": 0,
        "gold_label_str": "single",
        "n_gold_answers": 1,
    },
    {
        "question": "Name actors who played Spider-Man.",
        "gold_label": 1,
        "gold_label_str": "multiple",
        "n_gold_answers": 3,
    },
    {
        "question": "Name a color in the French flag.",
        "gold_label": 0,
        "gold_label_str": "single",
        "n_gold_answers": 1,
    },
    {
        "question": "Name the colors in the French flag.",
        "gold_label": 1,
        "gold_label_str": "multiple",
        "n_gold_answers": 3,
    },
    {
        "question": "Name a common cause of fever.",
        "gold_label": 0,
        "gold_label_str": "single",
        "n_gold_answers": 1,
    },
    {
        "question": "Name common causes of fever.",
        "gold_label": 1,
        "gold_label_str": "multiple",
        "n_gold_answers": 4,
    },
]


ALL_OOD_EVAL_QUESTIONS = [
    {
        "question": "What country is Brest located in?",
        "gold_label": 1,
        "gold_label_str": "multiple",
        "n_gold_answers": 2,
    },
    {
        "question": "What is the capital of France?",
        "gold_label": 0,
        "gold_label_str": "single",
        "n_gold_answers": 1,
    },
    {
        "question": "Who has won the FIFA World Cup?",
        "gold_label": 1,
        "gold_label_str": "multiple",
        "n_gold_answers": 8,
    },
    {
        "question": "Who wrote Pride and Prejudice?",
        "gold_label": 0,
        "gold_label_str": "single",
        "n_gold_answers": 1,
    },
    {
        "question": "Which cities have hosted the Summer Olympics?",
        "gold_label": 1,
        "gold_label_str": "multiple",
        "n_gold_answers": 20,
    },
    {
        "question": "What element has the chemical symbol Au?",
        "gold_label": 0,
        "gold_label_str": "single",
        "n_gold_answers": 1,
    },
    {
        "question": "Which programming languages are commonly used for web development?",
        "gold_label": 1,
        "gold_label_str": "multiple",
        "n_gold_answers": 5,
    },
    {
        "question": "Who painted the Mona Lisa?",
        "gold_label": 0,
        "gold_label_str": "single",
        "n_gold_answers": 1,
    },
    {
        "question": "What diseases can be prevented by vaccination?",
        "gold_label": 1,
        "gold_label_str": "multiple",
        "n_gold_answers": 5,
    },
    {
        "question": "What planet is known as the Red Planet?",
        "gold_label": 0,
        "gold_label_str": "single",
        "n_gold_answers": 1,
    },
    {
        "question": "Which countries border Germany?",
        "gold_label": 1,
        "gold_label_str": "multiple",
        "n_gold_answers": 9,
    },
    {
        "question": "Who discovered penicillin?",
        "gold_label": 0,
        "gold_label_str": "single",
        "n_gold_answers": 1,
    },
]

def standardize_eval_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    if "gold_label" not in out.columns:
        out["gold_label"] = out["label"].astype(int)
    if "gold_label_str" not in out.columns:
        out["gold_label_str"] = np.where(out["gold_label"].astype(int) == 1, "multiple", "single")
    if "n_gold_answers" not in out.columns:
        out["n_gold_answers"] = out["n_answers"].astype(int)
    return out[["question", "gold_label", "gold_label_str", "n_gold_answers"]]


def sample_balanced_webq_eval(test_split: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    rows = []
    for label in [0, 1]:
        part = test_split[test_split["label"].astype(int) == label]
        if per_class < 0:
            rows.append(part)
        else:
            n = min(per_class, len(part))
            rows.append(part.sample(n=n, random_state=seed + label))
    return standardize_eval_df(pd.concat(rows).sample(frac=1, random_state=seed).reset_index(drop=True))


def run_steering_eval(eval_name: str, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_df = standardize_eval_df(eval_df)
    prompts = [format_enumerability_instruction(q) for q in eval_df["question"]]
    labels = eval_df["gold_label"].astype(int).tolist()

    print(f"{eval_name} generation questions:", len(eval_df))
    display(eval_df.head(20))

    all_generation_rows = []
    all_summary_rows = []

    for alpha in CFG.alphas:
        direction = None if alpha == 0 else best_direction_cpu
        completions = generate_completions(
            prompts,
            direction=direction,
            layer=best_meta["layer"],
            pos=best_meta["position"],
            alpha=alpha,
            batch_size=CFG.batch_size_generation,
            max_new_tokens=CFG.max_new_tokens,
        )

        for item, (_, row) in zip(completions, eval_df.iterrows()):
            count_detail = answer_count_details(item["response"])
            all_generation_rows.append(
                {
                    "eval_name": eval_name,
                    "alpha": alpha,
                    "question": row["question"],
                    "gold_label": int(row["gold_label"]),
                    "gold_label_str": row["gold_label_str"],
                    "n_gold_answers": int(row["n_gold_answers"]),
                    "response": item["response"],
                    "parsed_type": parse_answer_type(item["response"]),
                    "generated_answer_count": int(count_detail["count"]),
                    "count_method": count_detail["method"],
                    "count_confidence": count_detail["confidence"],
                    "is_enumerated": int(count_detail["count"]) >= 2,
                }
            )

        summary = summarize_completions(completions, labels, alpha)
        summary["eval_name"] = eval_name
        all_summary_rows.append(summary)
        print(eval_name, json.dumps(summary, indent=2))

    generation_df = pd.DataFrame(all_generation_rows)
    summary_df = pd.DataFrame(all_summary_rows)

    generation_df.to_csv(ARTIFACT_DIR / f"{eval_name}_steered_generations.csv", index=False)
    summary_df.to_csv(ARTIFACT_DIR / f"{eval_name}_steering_alpha_summary.csv", index=False)

    with open(ARTIFACT_DIR / f"{eval_name}_steered_generations.json", "w", encoding="utf-8") as f:
        json.dump(all_generation_rows, f, indent=2, ensure_ascii=False)

    review_df = generation_df[
        [
            "eval_name",
            "alpha",
            "question",
            "gold_label_str",
            "n_gold_answers",
            "response",
            "parsed_type",
            "generated_answer_count",
            "count_method",
            "count_confidence",
            "is_enumerated",
        ]
    ].copy()
    review_df["manual_answer_type"] = ""
    review_df["manual_answer_count"] = ""
    review_df["manual_notes"] = ""
    review_df.to_csv(ARTIFACT_DIR / f"{eval_name}_manual_review_steered_generations.csv", index=False)

    print(f"{eval_name} manual review CSV:", ARTIFACT_DIR / f"{eval_name}_manual_review_steered_generations.csv")
    return generation_df, summary_df


curated_eval_df = pd.DataFrame(ALL_CURATED_EVAL_QUESTIONS[: CFG.max_manual_questions])
curated_generation_df, curated_summary_df = run_steering_eval("curated", curated_eval_df)

all_generation_dfs = [curated_generation_df]
all_summary_dfs = [curated_summary_df]

if CFG.webq_eval_per_class > 0:
    webq_eval_df = sample_balanced_webq_eval(test_df, CFG.webq_eval_per_class, CFG.seed)
    webq_generation_df, webq_summary_df = run_steering_eval("webq", webq_eval_df)
    all_generation_dfs.append(webq_generation_df)
    all_summary_dfs.append(webq_summary_df)

if CFG.max_ood_questions > 0:
    ood_eval_df = pd.DataFrame(ALL_OOD_EVAL_QUESTIONS[: CFG.max_ood_questions])
    ood_generation_df, ood_summary_df = run_steering_eval("ood", ood_eval_df)
    all_generation_dfs.append(ood_generation_df)
    all_summary_dfs.append(ood_summary_df)

generation_df = pd.concat(all_generation_dfs, ignore_index=True)
summary_df = pd.concat(all_summary_dfs, ignore_index=True)

# Backward-compatible combined outputs.
generation_df.to_csv(ARTIFACT_DIR / "steered_generations.csv", index=False)
summary_df.to_csv(ARTIFACT_DIR / "steering_alpha_summary.csv", index=False)

manual_review_df = generation_df.copy()
manual_review_df["manual_answer_type"] = ""
manual_review_df["manual_answer_count"] = ""
manual_review_df["manual_notes"] = ""
manual_review_df.to_csv(ARTIFACT_DIR / "manual_review_steered_generations.csv", index=False)

print("Combined manual review CSV:", ARTIFACT_DIR / "manual_review_steered_generations.csv")
summary_df


# %% [markdown]
# ## 8. Quick Plots

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_df = summary_df.melt(
    id_vars=["eval_name", "alpha"],
    value_vars=["multiple_type_rate", "enumeration_rate"],
    var_name="metric",
    value_name="rate",
)
sns.lineplot(data=plot_df, x="alpha", y="rate", hue="metric", style="eval_name", marker="o", ax=axes[0])
axes[0].axvline(0, color="gray", linestyle="--", linewidth=1)
axes[0].set_title("Target behavior rate")
axes[0].set_ylim(0, 1)

sns.lineplot(data=summary_df, x="alpha", y="mean_generated_answer_count", hue="eval_name", marker="o", ax=axes[1])
axes[1].axvline(0, color="gray", linestyle="--", linewidth=1)
axes[1].set_title("Mean generated answer count")

fig.tight_layout()
fig.savefig(ARTIFACT_DIR / "steering_alpha_effects.png", dpi=160, bbox_inches="tight")
plt.show()


# %%
print(f"Done. Artifacts saved in: {ARTIFACT_DIR}")
