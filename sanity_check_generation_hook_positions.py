import json
import os
from collections import Counter
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig


MODEL_CONFIGS = {
    "mistral": {
        "model_name": "mistralai/Mistral-7B-v0.1",
        "prompt_style": "plain",
        "default_layer": 18,
    },
    "llama31_8b_it": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt_style": "chat",
        "default_layer": 29,
    },
    "llama31_8b_base": {
        "model_name": "meta-llama/Llama-3.1-8B",
        "prompt_style": "plain",
        "default_layer": 16,
    },
    "qwen3_8b": {
        "model_name": "Qwen/Qwen3-8B",
        "prompt_style": "chat",
        "default_layer": 24,
    },
    "gemma3_12b_it": {
        "model_name": "google/gemma-3-12b-it",
        "prompt_style": "chat",
        "default_layer": 27,
    },
}


@dataclass
class SanityConfig:
    model_choice: str = os.environ.get("MODEL_CHOICE", "llama31_8b_it")
    model_quantization: str = os.environ.get("MODEL_QUANTIZATION", "4bit").lower()
    torch_dtype: str = os.environ.get("TORCH_DTYPE", "float16")
    max_new_tokens: int = int(os.environ.get("MAX_NEW_TOKENS", "8"))
    layer: int | None = int(os.environ["SANITY_LAYER"]) if os.environ.get("SANITY_LAYER") else None


CFG = SanityConfig()
assert CFG.model_choice in MODEL_CONFIGS, f"Unknown MODEL_CHOICE: {CFG.model_choice}"
MODEL_CFG = MODEL_CONFIGS[CFG.model_choice]
if CFG.layer is None:
    CFG.layer = MODEL_CFG["default_layer"]


def dtype_from_name(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


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
            print("Transformer block path:", ".".join(path))
            return module
    raise ValueError("Could not find transformer blocks")


ENUMERABILITY_PROMPT = """Given the following question, decide whether it has one correct answer or multiple correct answers, then answer it.

Respond in exactly this format:
Answer type: single or multiple
Answer: <answer or list of answers>

Question: {question}
Answer type:"""


def format_instruction(question: str) -> str:
    return ENUMERABILITY_PROMPT.format(question=question.strip())


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


def format_model_prompt(tokenizer, instruction: str) -> str:
    if MODEL_CFG["prompt_style"] == "plain":
        return instruction
    messages = [{"role": "user", "content": instruction}]
    try:
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


def print_tail_tokens(tokenizer, input_ids, attention_mask, title):
    print(f"\n{title}")
    for row_idx in range(input_ids.shape[0]):
        ids = input_ids[row_idx]
        mask = attention_mask[row_idx].bool()
        nonpad = ids[mask]
        print(f"row={row_idx} padded_len={len(ids)} nonpad_len={len(nonpad)} pad_count={(~mask).sum().item()}")
        tail = nonpad[-16:].tolist()
        for rel, tok in enumerate(tail, start=-len(tail)):
            print(f"  rel={rel:>3} id={tok:<8} text={tokenizer.decode([tok])!r}")


def run_probe(model, tokenizer, inputs, use_generate: bool):
    blocks = get_block_modules(model)
    records = []

    def hook_fn(module, hook_inputs):
        activation = hook_inputs[0] if isinstance(hook_inputs, tuple) else hook_inputs
        records.append(int(activation.shape[1]))
        return None

    handle = blocks[CFG.layer].register_forward_pre_hook(hook_fn)
    try:
        with torch.no_grad():
            if use_generate:
                generation_config = GenerationConfig(
                    max_new_tokens=CFG.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    generation_config=generation_config,
                )
            else:
                model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    use_cache=False,
                )
    finally:
        handle.remove()
    return records


def main():
    print(json.dumps({**asdict(CFG), **MODEL_CFG}, indent=2))
    model, tokenizer = load_model_and_tokenizer()
    device = next(model.parameters()).device
    print(f"Loaded {MODEL_CFG['model_name']} on {device}")

    questions = [
        "What is the capital of France?",
        "Which countries border Germany and what are their capitals?",
    ]
    instructions = [format_instruction(q) for q in questions]
    formatted_prompts = [format_model_prompt(tokenizer, inst) for inst in instructions]

    print("\nFormatted prompt tail repr:")
    print(repr(formatted_prompts[0][-500:]))

    if MODEL_CFG["prompt_style"] == "chat":
        sentinel = "<<<INSTRUCTION_SENTINEL>>>"
        sentinel_formatted = format_model_prompt(tokenizer, sentinel)
        suffix = sentinel_formatted.split(sentinel, 1)[1] if sentinel in sentinel_formatted else ""
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        print("\nChat-template suffix repr:", repr(suffix))
        print("Suffix positions:", list(range(-len(suffix_ids), 0)) if suffix_ids else [])
        print("Suffix tokens:", [tokenizer.decode([tok]) for tok in suffix_ids])

    inputs = tokenizer(formatted_prompts, padding=True, truncation=False, return_tensors="pt").to(device)
    print_tail_tokens(tokenizer, inputs.input_ids.detach().cpu(), inputs.attention_mask.detach().cpu(), "Token tails")

    score_records = run_probe(model, tokenizer, inputs, use_generate=False)
    gen_records = run_probe(model, tokenizer, inputs, use_generate=True)

    print("\nFull-prompt scoring forward hook sequence lengths:")
    print(score_records)
    print("Counts:", dict(Counter(score_records)))

    print("\nGeneration hook sequence lengths:")
    print(gen_records)
    print("Counts:", dict(Counter(gen_records)))

    cached_decode_calls = sum(1 for x in gen_records if x == 1)
    print("\nDiagnosis:")
    print(
        "Current target_pos=-1 generation hook would apply on every listed generation call, "
        "including cached decoding calls with seq_len=1."
    )
    print(f"Cached seq_len=1 decode calls observed: {cached_decode_calls}")
    if cached_decode_calls:
        print(
            "This confirms that per_forward steering is not prefill-only. "
            "Use GENERATION_HOOK_MODE=prefill_only to skip these decode-token calls."
        )
    else:
        print("No seq_len=1 calls observed; inspect generation/cache settings for this model.")


if __name__ == "__main__":
    main()
