import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig


MODEL_CONFIGS = {
    "mistral_7b_it": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
        "model_key": "mistral_7b_it",
        "prompt_style": "chat",
    },
    "llama31_8b_it": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "model_key": "llama31_8b_it",
        "prompt_style": "chat",
    },
    "qwen3_8b": {
        "model_name": "Qwen/Qwen3-8B",
        "model_key": "qwen3_8b",
        "prompt_style": "chat",
    },
    "gemma3_12b_it": {
        "model_name": "google/gemma-3-12b-it",
        "model_key": "gemma3_12b_it",
        "prompt_style": "chat",
    },
}


ENUMERABILITY_PROMPT = """Given the following question, decide whether it has one correct answer or multiple correct answers, then answer it.

Respond in exactly this format:
Answer type: single or multiple
Answer: <answer or list of answers>

Question: {question}
Answer type:"""


@dataclass
class SanityConfig:
    model_choice: str = os.environ.get("MODEL_CHOICE", "mistral_7b_it")
    drive_path: str = os.environ.get("DRIVE_PATH", str(Path.cwd() / "outputs"))
    artifact_subdir: str = os.environ.get(
        "ARTIFACT_SUBDIR", "phase2_activation_steering_enumerability_it_paperlike"
    )
    model_quantization: str = os.environ.get("MODEL_QUANTIZATION", "4bit").lower()
    torch_dtype: str = os.environ.get("TORCH_DTYPE", "float16")
    chat_assistant_prefill: bool = os.environ.get("CHAT_ASSISTANT_PREFILL", "0") in {
        "1",
        "true",
        "True",
        "yes",
        "YES",
    }
    positions_env: str = os.environ.get("POSITIONS", "auto")
    generation_hook_mode: str = os.environ.get("GENERATION_HOOK_MODE", "prefill_only")
    max_new_tokens: int = int(os.environ.get("MAX_NEW_TOKENS", "8"))
    alpha_for_delta: float = float(os.environ.get("SANITY_ALPHA_FOR_DELTA", "4.0"))


CFG = SanityConfig()
assert CFG.model_choice in MODEL_CONFIGS, f"Unknown MODEL_CHOICE: {CFG.model_choice}"
MODEL_CFG = MODEL_CONFIGS[CFG.model_choice]
ARTIFACT_DIR = Path(CFG.drive_path) / CFG.artifact_subdir / MODEL_CFG["model_key"]


def dtype_from_name(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def load_model_and_tokenizer():
    offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CFG["model_name"],
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
        MODEL_CFG["model_name"],
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


def get_block_modules(model):
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
            return module, ".".join(path)
    raise ValueError("Could not find transformer blocks")


def format_instruction(question: str) -> str:
    return ENUMERABILITY_PROMPT.format(question=question.strip())


def apply_chat_template(tokenizer, messages, **kwargs) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)


def format_model_prompt(tokenizer, instruction: str) -> str:
    if MODEL_CFG["prompt_style"] == "plain":
        return instruction

    user_content = instruction
    assistant_prefill = ""
    if CFG.chat_assistant_prefill:
        user_content = re.sub(r"\nAnswer type:\s*$", "", instruction.rstrip())
        assistant_prefill = "Answer type:"

    if assistant_prefill:
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_prefill},
        ]
        try:
            return apply_chat_template(tokenizer, messages, continue_final_message=True)
        except TypeError:
            base = apply_chat_template(tokenizer, [{"role": "user", "content": user_content}], add_generation_prompt=True)
            return base + assistant_prefill

    return apply_chat_template(tokenizer, [{"role": "user", "content": user_content}], add_generation_prompt=True)


def first_token_ids(tokenizer, variants):
    ids = []
    for text in variants:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            ids.append(encoded[0])
    return sorted(set(ids))


def infer_suffix_positions(tokenizer):
    sentinel = "<<<INSTRUCTION_SENTINEL>>>"
    formatted = format_model_prompt(tokenizer, sentinel)
    suffix = formatted.split(sentinel, 1)[1] if sentinel in formatted else ""
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    return suffix, suffix_ids, list(range(-len(suffix_ids), 0)) if suffix_ids else [-1]


def token_rows(tokenizer, input_ids, attention_mask, positions):
    rows = []
    for row_idx in range(input_ids.shape[0]):
        ids = input_ids[row_idx]
        mask = attention_mask[row_idx].bool()
        nonpad = ids[mask]
        seq_len = int(nonpad.shape[0])
        selected = {}
        for pos in positions:
            abs_pos = seq_len + pos if pos < 0 else pos
            if 0 <= abs_pos < seq_len:
                tok_id = int(nonpad[abs_pos].item())
                selected[str(pos)] = {
                    "abs_nonpad_position": abs_pos,
                    "token_id": tok_id,
                    "token_text": tokenizer.decode([tok_id]),
                }
        tail = []
        start_rel = -min(20, seq_len)
        for rel in range(start_rel, 0):
            tok_id = int(nonpad[seq_len + rel].item())
            tail.append({"rel": rel, "token_id": tok_id, "token_text": tokenizer.decode([tok_id])})
        rows.append(
            {
                "row": row_idx,
                "padded_len": int(ids.shape[0]),
                "nonpad_len": seq_len,
                "pad_count": int((~mask).sum().item()),
                "selected_positions": selected,
                "tail_tokens": tail,
            }
        )
    return rows


def make_recording_hook(vector=None, coeff=0.0, target_pos=-1, apply_once=False, min_seq_len=None):
    records = []
    applied = False

    def hook_fn(module, inputs):
        nonlocal applied
        raw_activation = inputs[0] if isinstance(inputs, tuple) else inputs
        seq_len = int(raw_activation.shape[1])
        should_apply = True
        if apply_once and applied:
            should_apply = False
        if min_seq_len is not None and seq_len < min_seq_len:
            should_apply = False

        abs_pos = None
        if target_pos is not None:
            abs_pos = seq_len + target_pos if target_pos < 0 else target_pos

        records.append(
            {
                "seq_len": seq_len,
                "target_pos": target_pos,
                "abs_pos_for_this_forward": abs_pos,
                "applied": bool(should_apply),
            }
        )
        if not should_apply or vector is None or coeff == 0.0:
            return None

        activation = inputs[0].clone() if isinstance(inputs, tuple) else inputs.clone()
        rest = inputs[1:] if isinstance(inputs, tuple) else None
        vec = vector.to(device=activation.device, dtype=activation.dtype)
        if target_pos is None:
            activation = activation + coeff * vec
        else:
            activation[:, target_pos, :] = activation[:, target_pos, :] + coeff * vec
        applied = True
        return (activation, *rest) if rest is not None else activation

    return hook_fn, records


def run_logits(model, tokenizer, prompts, hooks=()):
    inputs = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to(next(model.parameters()).device)
    handles = [module.register_forward_pre_hook(hook) for module, hook in hooks]
    try:
        with torch.no_grad():
            logits = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=False).logits[:, -1, :]
    finally:
        for handle in handles:
            handle.remove()
    return logits.float(), inputs


def score_from_logits(logits, single_toks, multiple_toks):
    logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    return torch.logsumexp(logits[:, multiple_toks], dim=-1) - torch.logsumexp(logits[:, single_toks], dim=-1)


def run_generation_trace(model, tokenizer, prompts, block, layer, direction, pos):
    inputs = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to(next(model.parameters()).device)
    hook_kwargs = {}
    if CFG.generation_hook_mode == "prefill_only":
        hook_kwargs = {"apply_once": True, "min_seq_len": 2}
    hook, records = make_recording_hook(
        vector=direction,
        coeff=CFG.alpha_for_delta,
        target_pos=pos,
        **hook_kwargs,
    )
    handle = block[layer].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=GenerationConfig(
                    max_new_tokens=CFG.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                ),
            )
    finally:
        handle.remove()
    return records


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {**asdict(CFG), **MODEL_CFG},
        "artifact_dir": str(ARTIFACT_DIR),
    }

    print(json.dumps(report["config"], indent=2))
    model, tokenizer = load_model_and_tokenizer()
    device = next(model.parameters()).device
    blocks, block_path = get_block_modules(model)
    report["loaded_on"] = str(device)
    report["block_path"] = block_path

    suffix, suffix_ids, suffix_positions = infer_suffix_positions(tokenizer)
    report["suffix"] = {
        "repr": repr(suffix),
        "token_count": len(suffix_ids),
        "positions": suffix_positions,
        "tokens": [{"id": int(tok), "text": tokenizer.decode([tok])} for tok in suffix_ids],
    }

    if CFG.positions_env.lower() in {"auto", "none", "suffix"}:
        positions = suffix_positions
    else:
        positions = [int(x.strip()) for x in CFG.positions_env.split(",") if x.strip()]
    report["candidate_positions_used_by_this_sanity"] = positions

    questions = [
        "What is the capital of France?",
        "Which countries border Germany?",
    ]
    instructions = [format_instruction(q) for q in questions]
    prompts = [format_model_prompt(tokenizer, inst) for inst in instructions]
    report["formatted_prompt_tail_repr"] = repr(prompts[0][-700:])

    logits_base, inputs = run_logits(model, tokenizer, prompts)
    report["token_rows"] = token_rows(
        tokenizer,
        inputs.input_ids.detach().cpu(),
        inputs.attention_mask.detach().cpu(),
        positions,
    )

    single_toks = first_token_ids(tokenizer, ["single", " single", "Single", " Single"])
    multiple_toks = first_token_ids(tokenizer, ["multiple", " multiple", "Multiple", " Multiple"])
    report["label_tokens"] = {
        "single": [{"id": int(tok), "text": tokenizer.decode([tok])} for tok in single_toks],
        "multiple": [{"id": int(tok), "text": tokenizer.decode([tok])} for tok in multiple_toks],
    }
    base_scores = score_from_logits(logits_base, single_toks, multiple_toks)
    report["baseline_type_scores"] = [float(x) for x in base_scores.detach().cpu()]
    top = torch.topk(torch.nan_to_num(logits_base[0], nan=-1e9), k=10)
    report["baseline_top_tokens_first_prompt"] = [
        {"id": int(tok), "text": tokenizer.decode([int(tok)]), "logit": float(val)}
        for val, tok in zip(top.values.detach().cpu(), top.indices.detach().cpu())
    ]

    meta_path = ARTIFACT_DIR / "best_direction_metadata.json"
    direction_path = ARTIFACT_DIR / "best_direction.pt"
    if meta_path.exists() and direction_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        direction = torch.load(direction_path, map_location="cpu").float()
        direction = direction / (direction.norm() + 1e-8)
        layer = int(meta["layer"])
        pos = int(meta["position"])
        report["best_direction_metadata"] = meta
        report["best_direction_norm_after_normalize"] = float(direction.norm().item())

        hook, score_records = make_recording_hook(
            vector=direction,
            coeff=CFG.alpha_for_delta,
            target_pos=pos,
        )
        logits_steered, _ = run_logits(model, tokenizer, prompts, hooks=[(blocks[layer], hook)])
        steered_scores = score_from_logits(logits_steered, single_toks, multiple_toks)
        delta = torch.nan_to_num(logits_steered - logits_base, nan=0.0)
        report["scoring_hook_records"] = score_records
        report["logit_delta_check"] = {
            "alpha": CFG.alpha_for_delta,
            "mean_abs_logit_delta": float(delta.abs().mean().item()),
            "max_abs_logit_delta": float(delta.abs().max().item()),
            "baseline_type_scores": [float(x) for x in base_scores.detach().cpu()],
            "steered_type_scores": [float(x) for x in steered_scores.detach().cpu()],
            "type_score_deltas": [float(x) for x in (steered_scores - base_scores).detach().cpu()],
        }
        top_steered = torch.topk(torch.nan_to_num(logits_steered[0], nan=-1e9), k=10)
        report["steered_top_tokens_first_prompt"] = [
            {"id": int(tok), "text": tokenizer.decode([int(tok)]), "logit": float(val)}
            for val, tok in zip(top_steered.values.detach().cpu(), top_steered.indices.detach().cpu())
        ]
        report["generation_hook_records"] = run_generation_trace(
            model,
            tokenizer,
            prompts[:1],
            blocks,
            layer,
            direction,
            pos,
        )
        report["generation_hook_seq_len_counts"] = dict(Counter(r["seq_len"] for r in report["generation_hook_records"]))
    else:
        report["best_direction_metadata"] = None
        report["missing_best_direction_note"] = f"Missing {meta_path} or {direction_path}."

    out_json = ARTIFACT_DIR / "paperlike_sanity_report.json"
    out_txt = ARTIFACT_DIR / "paperlike_sanity_report.txt"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, indent=2))

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_txt}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
