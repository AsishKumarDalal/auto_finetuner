"""
chat_templates.py — Prompt / chat templates for LLM fine-tuning.

Each template is a dict with:
    name        : short identifier used as a key
    rank        : quality tier  1 (best) → 3 (basic)
    models      : list of model families this template suits
    description : one-line human description
    template    : format string  (uses {instruction}, {input}, {output})
    template_no_input : same but for rows where input is empty / blank

Usage
-----
    from chat_templates import get_template, list_templates, TEMPLATES

    # by name
    tmpl, tmpl_no_input = get_template("qwen")

    # show everything available
    list_templates()

    # raw dict (all templates)
    print(TEMPLATES)
"""

from __future__ import annotations


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

TEMPLATES: dict[str, dict] = {

    # ── Rank 1 — modern, token-efficient ──────────────────────────────────────

    "qwen": {
        "name": "qwen",
        "rank": 1,
        "models": ["Qwen2.5", "Qwen2", "Qwen1.5", "Qwen"],
        "description": "Qwen ChatML format — best for all Qwen family models",
        "template": (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "{instruction}\n{input}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{output}<|im_end|>"
        ),
        "template_no_input": (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "{instruction}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{output}<|im_end|>"
        ),
    },

    "chatml": {
        "name": "chatml",
        "rank": 1,
        "models": ["Mistral-7B", "Mixtral", "Yi", "DeepSeek", "generic"],
        "description": "Generic ChatML format — widely compatible with most GGUF/HF models",
        "template": (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "{instruction}\n{input}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{output}<|im_end|>"
        ),
        "template_no_input": (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "{instruction}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{output}<|im_end|>"
        ),
    },

    "llama3": {
        "name": "llama3",
        "rank": 1,
        "models": ["Llama-3", "Llama-3.1", "Llama-3.2", "Llama-3.3"],
        "description": "Llama-3 special-token instruct format",
        "template": (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{instruction}\n{input}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "{output}<|eot_id|>"
        ),
        "template_no_input": (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{instruction}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "{output}<|eot_id|>"
        ),
    },

    "phi3": {
        "name": "phi3",
        "rank": 1,
        "models": ["Phi-3", "Phi-3.5", "Phi-3-mini", "Phi-3-medium"],
        "description": "Microsoft Phi-3 instruct format",
        "template": (
            "<|system|>\n"
            "You are a helpful assistant.<|end|>\n"
            "<|user|>\n"
            "{instruction}\n{input}<|end|>\n"
            "<|assistant|>\n"
            "{output}<|end|>"
        ),
        "template_no_input": (
            "<|system|>\n"
            "You are a helpful assistant.<|end|>\n"
            "<|user|>\n"
            "{instruction}<|end|>\n"
            "<|assistant|>\n"
            "{output}<|end|>"
        ),
    },

    "gemma": {
        "name": "gemma",
        "rank": 1,
        "models": ["Gemma-2", "Gemma-1", "Gemma"],
        "description": "Google Gemma instruct format",
        "template": (
            "<bos><start_of_turn>user\n"
            "{instruction}\n{input}"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
            "{output}<end_of_turn>"
        ),
        "template_no_input": (
            "<bos><start_of_turn>user\n"
            "{instruction}"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
            "{output}<end_of_turn>"
        ),
    },

    # ── Rank 1 — Domain-specific templates ───────────────────────────────────

    "medical": {
        "name": "medical",
        "rank": 1,
        "models": ["Llama-3", "Qwen2.5", "Mistral", "generic"],
        "description": "Medical Q&A format with clinical system prompt",
        "template": (
            "<|im_start|>system\n"
            "You are an expert medical assistant. Provide accurate, evidence-based "
            "clinical information. Always recommend consulting a qualified physician "
            "for personal medical advice.<|im_end|>\n"
            "<|im_start|>user\n"
            "{instruction}\n{input}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{output}<|im_end|>"
        ),
        "template_no_input": (
            "<|im_start|>system\n"
            "You are an expert medical assistant. Provide accurate, evidence-based "
            "clinical information. Always recommend consulting a qualified physician "
            "for personal medical advice.<|im_end|>\n"
            "<|im_start|>user\n"
            "{instruction}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{output}<|im_end|>"
        ),
    },

    "legal": {
        "name": "legal",
        "rank": 1,
        "models": ["Llama-3", "Qwen2.5", "Mistral", "generic"],
        "description": "Legal Q&A format with jurisdiction-aware disclaimer",
        "template": (
            "<|im_start|>system\n"
            "You are an expert legal assistant. Provide accurate legal information "
            "and analysis. Always note that this is not formal legal advice and users "
            "should consult a licensed attorney for their specific situation.<|im_end|>\n"
            "<|im_start|>user\n"
            "{instruction}\n{input}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{output}<|im_end|>"
        ),
        "template_no_input": (
            "<|im_start|>system\n"
            "You are an expert legal assistant. Provide accurate legal information "
            "and analysis. Always note that this is not formal legal advice and users "
            "should consult a licensed attorney for their specific situation.<|im_end|>\n"
            "<|im_start|>user\n"
            "{instruction}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{output}<|im_end|>"
        ),
    },

    "coding": {
        "name": "coding",
        "rank": 1,
        "models": ["Llama-3", "Qwen2.5", "DeepSeek-Coder", "generic"],
        "description": "Code generation and explanation format",
        "template": (
            "<|im_start|>system\n"
            "You are an expert software engineer. Write clean, efficient, "
            "well-commented code. Explain your reasoning when helpful.<|im_end|>\n"
            "<|im_start|>user\n"
            "{instruction}\n{input}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{output}<|im_end|>"
        ),
        "template_no_input": (
            "<|im_start|>system\n"
            "You are an expert software engineer. Write clean, efficient, "
            "well-commented code. Explain your reasoning when helpful.<|im_end|>\n"
            "<|im_start|>user\n"
            "{instruction}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{output}<|im_end|>"
        ),
    },

    "finance": {
        "name": "finance",
        "rank": 1,
        "models": ["Llama-3", "Qwen2.5", "Mistral", "generic"],
        "description": "Financial analysis and market insight format",
        "template": (
            "<|im_start|>system\n"
            "You are an expert financial analyst. Provide accurate financial "
            "information, market analysis, and investment insights. Always note "
            "that this is not personal financial advice.<|im_end|>\n"
            "<|im_start|>user\n"
            "{instruction}\n{input}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{output}<|im_end|>"
        ),
        "template_no_input": (
            "<|im_start|>system\n"
            "You are an expert financial analyst. Provide accurate financial "
            "information, market analysis, and investment insights. Always note "
            "that this is not personal financial advice.<|im_end|>\n"
            "<|im_start|>user\n"
            "{instruction}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{output}<|im_end|>"
        ),
    },

    # ── Rank 2 — solid, widely used ───────────────────────────────────────────

    "llama2": {
        "name": "llama2",
        "rank": 2,
        "models": ["Llama-2", "Llama-2-chat", "Mistral-7B-v0.1"],
        "description": "Llama-2 [INST] instruct format",
        "template": (
            "<s>[INST] {instruction}\n{input} [/INST] {output} </s>"
        ),
        "template_no_input": (
            "<s>[INST] {instruction} [/INST] {output} </s>"
        ),
    },

    "mistral": {
        "name": "mistral",
        "rank": 2,
        "models": ["Mistral-7B-v0.3", "Mistral-Nemo", "Mixtral-8x7B"],
        "description": "Mistral v0.3 instruct format (no system token)",
        "template": (
            "<s>[INST] {instruction}\n{input} [/INST]{output}</s>"
        ),
        "template_no_input": (
            "<s>[INST] {instruction} [/INST]{output}</s>"
        ),
    },

    "falcon": {
        "name": "falcon",
        "rank": 2,
        "models": ["Falcon", "Falcon-7B", "Falcon-40B"],
        "description": "Falcon instruct format",
        "template": (
            "User: {instruction}\n{input}\nFalcon: {output}"
        ),
        "template_no_input": (
            "User: {instruction}\nFalcon: {output}"
        ),
    },

    # ── Rank 3 — legacy / plain ───────────────────────────────────────────────

    "alpaca": {
        "name": "alpaca",
        "rank": 3,
        "models": ["GPT-J", "OPT", "Pythia", "generic base models"],
        "description": "Stanford Alpaca plain-text format — universal fallback",
        "template": (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        ),
        "template_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Response:\n{output}"
        ),
    },

    "vicuna": {
        "name": "vicuna",
        "rank": 3,
        "models": ["Vicuna", "ShareGPT fine-tuned", "LLaMA-1-based"],
        "description": "Vicuna / ShareGPT conversation style",
        "template": (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
            "USER: {instruction}\n{input}\n"
            "ASSISTANT: {output}"
        ),
        "template_no_input": (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
            "USER: {instruction}\n"
            "ASSISTANT: {output}"
        ),
    },

    "oasst": {
        "name": "oasst",
        "rank": 3,
        "models": ["OpenAssistant", "OASST fine-tuned", "Pythia"],
        "description": "OpenAssistant plain token format",
        "template": (
            "<|prompter|>{instruction}\n{input}<|endoftext|>"
            "<|assistant|>{output}<|endoftext|>"
        ),
        "template_no_input": (
            "<|prompter|>{instruction}<|endoftext|>"
            "<|assistant|>{output}<|endoftext|>"
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_template(name: str) -> tuple[str, str]:
    """
    Return (template, template_no_input) for the given template name.

    Args:
        name : Template key, e.g. "qwen", "llama3", "alpaca", "medical"

    Returns:
        (template_str, template_no_input_str)

    Raises:
        KeyError if the name is not found.

    Example:
        tmpl, tmpl_no_input = get_template("medical")
    """
    if name not in TEMPLATES:
        raise KeyError(
            f"Unknown template '{name}'. "
            f"Available: {list(TEMPLATES.keys())}"
        )
    entry = TEMPLATES[name]
    return entry["template"], entry["template_no_input"]


def list_templates(rank: int = None) -> None:
    """
    Print all available templates, optionally filtered by rank.

    Args:
        rank : 1, 2, or 3  (None → show all)

    Example:
        list_templates()
        list_templates(rank=1)
    """
    print("\n" + "═" * 70)
    print(f"  {'NAME':<14} {'RANK':<6} {'MODELS / USE-CASE'}")
    print("═" * 70)
    for key, meta in sorted(TEMPLATES.items(), key=lambda x: (x[1]["rank"], x[0])):
        if rank is not None and meta["rank"] != rank:
            continue
        models_str = ", ".join(meta["models"][:3])
        if len(meta["models"]) > 3:
            models_str += f" +{len(meta['models'])-3}"
        print(f"  {key:<14} {'★'*meta['rank']:<6}  {models_str}")
        print(f"  {'':14}        {meta['description']}")
        print()
    print("═" * 70)
    print(
        "  Ranks:  ★★★ = best / modern   "
        "★★ = solid   ★ = legacy / plain\n"
    )


def best_template_for(model_name: str) -> str:
    """
    Heuristic: return the template name that best suits a given model string.

    Falls back to "chatml" if nothing matches.

    Example:
        name = best_template_for("meta-llama/Llama-3.2-3B-Instruct")
        # → "llama3"
    """
    model_lower = model_name.lower()
    checks = [
        # General model families
        ("llama-3",   "llama3"),
        ("llama3",    "llama3"),
        ("llama-2",   "llama2"),
        ("llama2",    "llama2"),
        ("mistral",   "mistral"),
        ("mixtral",   "mistral"),
        ("qwen",      "qwen"),
        ("phi-3",     "phi3"),
        ("phi3",      "phi3"),
        ("gemma",     "gemma"),
        ("falcon",    "falcon"),
        ("vicuna",    "vicuna"),
        ("oasst",     "oasst"),
        ("alpaca",    "alpaca"),
        # Domain-specific model hints
        ("medical",   "medical"),
        ("clinical",  "medical"),
        ("biomed",    "medical"),
        ("legal",     "legal"),
        ("law",       "legal"),
        ("coder",     "coding"),
        ("coding",    "coding"),
        ("code",      "coding"),
        ("finance",   "finance"),
        ("financial", "finance"),
        ("fingpt",    "finance"),
    ]
    for substr, tmpl_name in checks:
        if substr in model_lower:
            return tmpl_name
    return "chatml"
