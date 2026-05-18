"""
utility.py — Dataset utilities for LLM fine-tuning pipelines.

Supports 3 dataset types:
  1. Alpaca style   : {"instruction": ..., "input": ..., "output": ...}
  2. Chat style     : {"conversations": [{"role": ..., "content": ...}, ...]}
  3. Raw text style : {"text": "..."} or plain .txt files

Utilities:
  - download_file()            : Download any file from a URL to disk
  - load_json_file()           : Load any .json or .jsonl file from disk
  - dataset_stats()            : Print quick stats about a dataset
  - save_as_jsonl()            : Save HF Dataset → .jsonl
  - load_jsonl()               : Load .jsonl → HF Dataset

  ── Alpaca ──
  - load_alpaca_json()         : Load alpaca_data.json → list of dicts
  - alpaca_to_hf_dataset()     : Convert list → HF Dataset
  - apply_alpaca_template()    : Apply prompt template to alpaca HF Dataset
  - download_alpaca()          : One-liner: download + parse + format

  ── Chat ──
  - load_chat_json()           : Load chat JSON → HF Dataset
  - format_chat_dataset()      : Format multi-turn conversations → text column
  - load_and_format_chat()     : One-liner: load + format chat dataset

  ── Raw Text ──
  - load_text_dataset()        : Load plain .txt file → HF Dataset
  - load_text_jsonl()          : Load .jsonl with a text field → HF Dataset

  ── HuggingFace Hub (generic) ──
  - load_hf_dataset()          : Load any HF Hub dataset and optionally apply a template

  ── Domain-specific ──
  - load_domain_dataset()      : One-liner for medical / legal / coding / finance datasets
  - list_domain_datasets()     : Print all curated domain dataset options
"""

import os
import json
import requests
from datasets import Dataset, load_dataset
from tqdm import tqdm

# Import templates from chat_templates.py
from chat_templates import get_template, best_template_for, list_templates, TEMPLATES


# ═══════════════════════════════════════════════════════
# PROMPT TEMPLATES  (kept for backwards-compat)
# ═══════════════════════════════════════════════════════

# These are now driven by chat_templates.py.
# Exposed here so existing code that does `from utility import ALPACA_TEMPLATE` still works.

ALPACA_TEMPLATE, ALPACA_TEMPLATE_NO_INPUT = get_template("alpaca")
LLAMA_TEMPLATE,  LLAMA_TEMPLATE_NO_INPUT  = get_template("llama2")
QWEN_TEMPLATE,   QWEN_TEMPLATE_NO_INPUT   = get_template("qwen")

# Alpaca dataset URL (Stanford)
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"


# ═══════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════

def download_file(url: str, save_path: str, force: bool = False) -> str:
    """
    Download a file from a URL to disk with a progress bar.

    Args:
        url       : Direct URL to the file.
        save_path : Where to save it (e.g. "./data/alpaca_data.json").
        force     : Re-download even if the file already exists.

    Returns:
        Absolute path to the saved file.
    """
    if os.path.exists(save_path) and not force:
        print(f"[✓] Already exists: {save_path}  (use force=True to re-download)")
        return os.path.abspath(save_path)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    print(f"[↓] Downloading: {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(save_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=os.path.basename(save_path)
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"[✓] Saved: {os.path.abspath(save_path)}")
    return os.path.abspath(save_path)


def load_json_file(path: str):
    """
    Load a .json or .jsonl file from disk into a Python list.
    """
    print(f"[*] Loading: {path}")
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        print(f"[✓] Loaded {len(rows):,} rows (jsonl)")
        return rows
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = len(data) if isinstance(data, list) else 1
        print(f"[✓] Loaded {count:,} rows (json)")
        return data


def save_as_jsonl(dataset: Dataset, path: str):
    """Save a HF Dataset to a .jsonl file (one JSON object per line)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[*] Saving {len(dataset):,} rows → {path}")
    with open(path, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[✓] Saved: {os.path.abspath(path)}")


def load_jsonl(path: str) -> Dataset:
    """Load a .jsonl file into a HuggingFace Dataset."""
    rows = load_json_file(path)
    dataset = Dataset.from_list(rows)
    print(f"[✓] HF Dataset: {len(dataset):,} rows | columns: {dataset.column_names}")
    return dataset


def dataset_stats(dataset: Dataset, text_field: str = "text"):
    """Print row count, column names, and word-length distribution of the text field."""
    print("\n" + "=" * 55)
    print("  Dataset Stats")
    print("=" * 55)
    print(f"  Rows    : {len(dataset):,}")
    print(f"  Columns : {dataset.column_names}")

    if text_field in dataset.column_names:
        lengths = [len(str(t).split()) for t in dataset[text_field]]
        print(f"\n  '{text_field}' word-length distribution:")
        print(f"    Min    : {min(lengths)}")
        print(f"    Max    : {max(lengths)}")
        print(f"    Mean   : {sum(lengths)/len(lengths):.1f}")
        print(f"    Median : {sorted(lengths)[len(lengths)//2]}")
        over_512  = sum(1 for l in lengths if l > 512)
        over_1024 = sum(1 for l in lengths if l > 1024)
        print(f"    > 512  : {over_512:,}  ({100*over_512/len(lengths):.1f}%)")
        print(f"    > 1024 : {over_1024:,}  ({100*over_1024/len(lengths):.1f}%)")

    print("=" * 55)
    if len(dataset) > 0:
        print("\n  Sample row [0]:")
        for k, v in dataset[0].items():
            preview = str(v)[:300] + "..." if len(str(v)) > 300 else str(v)
            print(f"    {k}: {preview}")
    print()


# ═══════════════════════════════════════════════════════
# DATASET TYPE 1 — ALPACA STYLE
# ═══════════════════════════════════════════════════════

def load_alpaca_json(path: str) -> list[dict]:
    """Load alpaca_data.json from disk into a list of dicts."""
    data = load_json_file(path)
    assert isinstance(data, list),   "Expected a JSON list of dicts"
    assert "instruction" in data[0], "Expected 'instruction' key in each dict"
    assert "output"      in data[0], "Expected 'output' key in each dict"
    return data


def alpaca_to_hf_dataset(data: list[dict]) -> Dataset:
    """Convert a list of Alpaca dicts into a HuggingFace Dataset."""
    dataset = Dataset.from_dict({
        "instruction": [d["instruction"]    for d in data],
        "input":       [d.get("input", "")  for d in data],
        "output":      [d["output"]         for d in data],
    })
    print(f"[✓] Alpaca HF Dataset: {len(dataset):,} rows")
    return dataset


def apply_alpaca_template(
    dataset: Dataset,
    template_style: str = "qwen",
    template_override: str = None,
    template_no_input_override: str = None,
    text_field: str = "text",
    remove_original_cols: bool = True,
) -> Dataset:
    """
    Apply a prompt template to every row of an Alpaca-style HF Dataset.
    Automatically uses the no-input variant when 'input' is empty.

    Args:
        dataset                   : HF Dataset with instruction / input / output columns.
        template_style            : Key from chat_templates.py
                                    e.g. "qwen" | "llama3" | "alpaca" | "medical" | "coding"
                                    Run list_templates() to see all options.
        template_override         : If given, use this raw format string instead of template_style.
        template_no_input_override: Matching no-input variant for template_override.
        text_field                : Output column name.
        remove_original_cols      : Drop instruction/input/output after formatting.

    Examples:
        # Using a named template
        ds = apply_alpaca_template(ds, template_style="medical")

        # Using a custom raw template string
        ds = apply_alpaca_template(
            ds,
            template_override="<s>[INST] {instruction}\\n{input} [/INST] {output} </s>",
            template_no_input_override="<s>[INST] {instruction} [/INST] {output} </s>",
        )
    """
    # Resolve template strings
    if template_override is not None:
        tmpl = template_override
        tmpl_no_input = template_no_input_override or template_override
        print(f"[*] Using custom (override) template...")
    else:
        if template_style not in TEMPLATES:
            raise ValueError(
                f"Unknown template_style '{template_style}'. "
                f"Run list_templates() to see options."
            )
        tmpl, tmpl_no_input = get_template(template_style)
        print(f"[*] Applying '{template_style}' template (rank {TEMPLATES[template_style]['rank']}) "
              f"to {len(dataset):,} examples...")

    def _fmt(example):
        if example.get("input", "").strip():
            example[text_field] = tmpl.format(
                instruction=example["instruction"],
                input=example["input"],
                output=example["output"],
            )
        else:
            example[text_field] = tmpl_no_input.format(
                instruction=example["instruction"],
                output=example["output"],
            )
        return example

    dataset = dataset.map(_fmt)

    if remove_original_cols:
        cols = [c for c in ["instruction", "input", "output"] if c in dataset.column_names]
        if cols:
            dataset = dataset.remove_columns(cols)

    print(f"[✓] Template applied | columns: {dataset.column_names}")
    return dataset


def download_alpaca(
    save_path: str = "./data/alpaca_data.json",
    template_style: str = "qwen",
    template_override: str = None,
    template_no_input_override: str = None,
    text_field: str = "text",
    max_samples: int = None,
) -> Dataset:
    """
    One-liner: Download Alpaca dataset, apply template, return HF Dataset.

    Args:
        save_path                  : Where to save the raw JSON file.
        template_style             : Named template key — "qwen" | "llama3" | "alpaca" | ...
                                     Run list_templates() to see all.
        template_override          : Supply a raw format string to override the named template.
        template_no_input_override : Matching no-input variant.
        text_field                 : Output column name.
        max_samples                : Limit rows (useful for quick tests).

    Returns:
        HF Dataset with a single 'text' column, ready for SFTTrainer.

    Examples:
        # Named template
        ds = download_alpaca(template_style="llama3")

        # Custom raw template string
        ds = download_alpaca(
            template_override="<s>[INST] {instruction}\\n{input} [/INST] {output} </s>",
            template_no_input_override="<s>[INST] {instruction} [/INST] {output} </s>",
            max_samples=1000,
        )
    """
    path    = download_file(ALPACA_URL, save_path)
    data    = load_alpaca_json(path)

    if max_samples:
        data = data[:max_samples]
        print(f"[*] Limited to {max_samples:,} samples")

    dataset = alpaca_to_hf_dataset(data)
    dataset = apply_alpaca_template(
        dataset,
        template_style=template_style,
        template_override=template_override,
        template_no_input_override=template_no_input_override,
        text_field=text_field,
    )
    dataset_stats(dataset, text_field)
    return dataset


# ═══════════════════════════════════════════════════════
# DATASET TYPE 1b — HF HUB (generic, any alpaca-style dataset)
# ═══════════════════════════════════════════════════════

def load_hf_dataset(
    dataset_name: str,
    split: str = "train",
    template_style: str = None,
    template_override: str = None,
    template_no_input_override: str = None,
    text_field: str = "text",
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
    max_samples: int = None,
) -> Dataset:
    """
    Load any HuggingFace Hub dataset and optionally apply a prompt template.

    Works for:
      - Alpaca-style datasets (instruction / input / output columns)
      - Datasets that already have a 'text' column (returned as-is)
      - Datasets with differently named columns (use instruction_field etc.)

    Args:
        dataset_name               : HF Hub repo string e.g. "mlabonne/guanaco-llama2-1k"
        split                      : Dataset split (default "train")
        template_style             : Named template key.  If None and the dataset already has
                                     a text_field column, no template is applied.
                                     Run list_templates() to see options.
        template_override          : Raw format string override.
        template_no_input_override : Matching no-input variant.
        text_field                 : Column to treat as (or produce as) the text output.
        instruction_field          : Column mapped to {instruction}.
        input_field                : Column mapped to {input} (optional, can be absent).
        output_field               : Column mapped to {output}.
        max_samples                : Cap the dataset size.

    Returns:
        HF Dataset with a single text_field column, ready for SFTTrainer.

    Examples:
        # Dataset that already has a 'text' column → no template needed
        ds = load_hf_dataset("mlabonne/guanaco-llama2-1k")

        # Alpaca-style Hub dataset + named template
        ds = load_hf_dataset("tatsu-lab/alpaca", template_style="qwen")

        # Medical dataset with medical template
        ds = load_hf_dataset(
            "medalpaca/medical_meadow_medical_flashcards",
            template_style="medical",
        )
    """
    print(f"[*] Loading HF Hub dataset: {dataset_name}  (split='{split}')")
    ds = load_dataset(dataset_name, split=split)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
        print(f"[*] Limited to {max_samples:,} samples")

    cols = ds.column_names

    # ── Case 1: dataset already has the text column and no template requested
    if text_field in cols and template_style is None and template_override is None:
        print(f"[✓] Dataset already has '{text_field}' column — no template applied.")
        extra = [c for c in cols if c != text_field]
        if extra:
            ds = ds.remove_columns(extra)
        dataset_stats(ds, text_field)
        return ds

    # ── Case 2: alpaca-style columns — rename + apply template
    needs_template = template_style is not None or template_override is not None

    if instruction_field in cols and output_field in cols:
        # Rename columns to standard names if needed
        rename_map = {}
        if instruction_field != "instruction":
            rename_map[instruction_field] = "instruction"
        if output_field != "output":
            rename_map[output_field] = "output"
        if input_field in cols and input_field != "input":
            rename_map[input_field] = "input"
        if rename_map:
            ds = ds.rename_columns(rename_map)

        # Add blank input column if missing
        if "input" not in ds.column_names:
            ds = ds.map(lambda x: {"input": ""})

        if needs_template or True:   # always apply when instruction/output found
            effective_style = template_style or "chatml"
            ds = apply_alpaca_template(
                ds,
                template_style=effective_style,
                template_override=template_override,
                template_no_input_override=template_no_input_override,
                text_field=text_field,
            )
    elif text_field in cols:
        # Has text column but template was requested — warn and return as-is
        print(f"[!] Dataset has '{text_field}' but no instruction/output columns; "
              f"template ignored.")
        extra = [c for c in ds.column_names if c != text_field]
        if extra:
            ds = ds.remove_columns(extra)
    else:
        raise ValueError(
            f"Cannot auto-detect dataset structure.\n"
            f"  Columns found : {cols}\n"
            f"  Expected either a '{text_field}' column OR "
            f"'{instruction_field}' + '{output_field}' columns.\n"
            f"  Use instruction_field= / output_field= to remap column names."
        )

    dataset_stats(ds, text_field)
    return ds


# ═══════════════════════════════════════════════════════
# DATASET TYPE 2 — CHAT / CONVERSATION STYLE
# ═══════════════════════════════════════════════════════

def _format_chat_qwen(
    messages: list[dict],
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    out = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        out += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return out.strip()


def _format_chat_llama(messages: list[dict]) -> str:
    out = ""
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            out += f"<s>[INST] {content} [/INST] "
        elif role == "assistant":
            out += f"{content} </s>"
    return out.strip()


def load_chat_json(path: str, conv_key: str = None) -> Dataset:
    """
    Load a chat/conversation JSON or JSONL file into a HF Dataset.
    Auto-detects Format A, B, or C.
    """
    raw = load_json_file(path)

    if (isinstance(raw, list) and len(raw) > 0
            and isinstance(raw[0], dict) and "role" in raw[0]):
        print("[*] Detected Format C: single flat conversation")
        conversations = [raw]

    elif isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
        print("[*] Detected Format A: list of conversations")
        conversations = raw

    elif isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
        if conv_key is None:
            for candidate in ["conversations", "messages", "dialog", "chat"]:
                if candidate in raw[0]:
                    conv_key = candidate
                    break
        if conv_key and conv_key in raw[0]:
            print(f"[*] Detected Format B: list of dicts with key='{conv_key}'")
            conversations = [row[conv_key] for row in raw]
        else:
            raise ValueError(
                f"Could not find conversation key. "
                f"Keys found: {list(raw[0].keys())}. "
                f"Pass conv_key= explicitly."
            )
    else:
        raise ValueError(f"Unrecognised JSON structure in: {path}")

    dataset = Dataset.from_dict({"conversations": conversations})
    print(f"[✓] Chat HF Dataset: {len(dataset):,} conversations")
    return dataset


def format_chat_dataset(
    dataset: Dataset,
    model_style: str = "qwen",
    text_field: str = "text",
    conv_key: str = "conversations",
    system_prompt: str = "You are a helpful assistant.",
    remove_original_cols: bool = True,
) -> Dataset:
    """
    Format a multi-turn chat HF Dataset into a single text column for training.

    Args:
        model_style : "qwen" | "llama2"   (chat-aware formatters)
    """
    if model_style not in ("qwen", "llama2"):
        raise ValueError("model_style must be 'qwen' or 'llama2'")
    if conv_key not in dataset.column_names:
        raise ValueError(f"Column '{conv_key}' not found. Available: {dataset.column_names}")

    print(f"[*] Formatting {len(dataset):,} conversations → '{model_style}' style...")

    def _fmt(example):
        messages = example[conv_key]
        if model_style == "qwen":
            example[text_field] = _format_chat_qwen(messages, system_prompt)
        else:
            example[text_field] = _format_chat_llama(messages)
        return example

    dataset = dataset.map(_fmt)

    if remove_original_cols:
        cols = [c for c in dataset.column_names if c != text_field]
        if cols:
            dataset = dataset.remove_columns(cols)

    print(f"[✓] Chat formatted | columns: {dataset.column_names}")
    return dataset


def load_and_format_chat(
    path: str,
    model_style: str = "qwen",
    text_field: str = "text",
    conv_key: str = None,
    system_prompt: str = "You are a helpful assistant.",
    max_samples: int = None,
) -> Dataset:
    """
    One-liner: Load a chat JSON/JSONL, format it, return HF Dataset.
    """
    dataset = load_chat_json(path, conv_key=conv_key)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"[*] Limited to {max_samples:,} samples")

    dataset = format_chat_dataset(
        dataset, model_style, text_field,
        conv_key="conversations", system_prompt=system_prompt,
    )
    dataset_stats(dataset, text_field)
    return dataset


# ═══════════════════════════════════════════════════════
# DATASET TYPE 3 — RAW TEXT STYLE
# ═══════════════════════════════════════════════════════

def load_text_dataset(
    path: str,
    text_field: str = "text",
    min_length: int = 10,
) -> Dataset:
    """Load a plain .txt file into a HF Dataset (one line = one example)."""
    print(f"[*] Loading plain text: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    texts   = [l.strip() for l in lines if len(l.strip()) >= min_length]
    skipped = len(lines) - len(texts)

    dataset = Dataset.from_dict({text_field: texts})
    print(f"[✓] Text HF Dataset: {len(dataset):,} lines (skipped {skipped:,} short lines)")
    dataset_stats(dataset, text_field)
    return dataset


def load_text_jsonl(
    path: str,
    text_field: str = "text",
    min_length: int = 10,
) -> Dataset:
    """Load a .jsonl where each line is {"text": "..."} into a HF Dataset."""
    rows    = load_json_file(path)
    texts   = [
        r[text_field] for r in rows
        if text_field in r and len(str(r[text_field]).strip()) >= min_length
    ]
    skipped = len(rows) - len(texts)
    dataset = Dataset.from_dict({text_field: texts})
    print(f"[✓] Text JSONL HF Dataset: {len(dataset):,} rows (skipped {skipped:,})")
    dataset_stats(dataset, text_field)
    return dataset


# ═══════════════════════════════════════════════════════
# DOMAIN-SPECIFIC DATASET LOADERS
# ═══════════════════════════════════════════════════════

# Well-known public datasets per domain (HF Hub IDs)
DOMAIN_DATASETS = {
    "medical": [
        "medalpaca/medical_meadow_medical_flashcards",  # [0] 33k medical Q&A flashcards
        "medalpaca/medical_meadow_wikidoc",             # [1] 67k clinical wiki articles
        "medalpaca/medical_meadow_healthcaremagic",     # [2] 112k doctor consultations
        "lavita/medical-qa-datasets",                  # [3] aggregated medical QA
        "qiaojin/PubMedQA",                            # [4] PubMed research QA
    ],
    "legal": [
        "nguha/legalbench",                            # [0] legal reasoning tasks
        "pile-of-law/pile-of-law",                     # [1] large legal text corpus
        "joelniklaus/swiss_judgment_prediction",       # [2] court judgment prediction
        "atlasia/moroccan-law-alpaca",                 # [3] law alpaca-style dataset
    ],
    "coding": [
        "iamtarun/python_code_instructions_18k_alpaca",   # [0] 18k Python alpaca-style
        "sahil2801/CodeAlpaca-20k",                       # [1] 20k code instructions
        "TokenBender/code_instructions_122k_alpaca_style", # [2] 122k multi-language
        "smangrul/hf-stack-v1",                           # [3] StackOverflow-style
        "glaiveai/glaive-code-assistant",                 # [4] code assistant chat
    ],
    "finance": [
        "gbharti/finance-alpaca",                      # [0] 68k finance alpaca
        "FinGPT/fingpt-sentiment-train",               # [1] financial sentiment analysis
        "FinGPT/fingpt-forecaster",                    # [2] market forecasting Q&A
        "oliverwang15/FinGPT_ChatGLM2_Sentiment_Instruction_Tuning",  # [3] sentiment tuning
    ],
}


def load_domain_dataset(
    domain: str,
    dataset_index: int = 0,
    template_style: str = None,
    max_samples: int = None,
    text_field: str = "text",
    custom_dataset_name: str = None,
    split: str = "train",
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
) -> Dataset:
    """
    One-liner to load a well-known domain-specific dataset from HF Hub
    and apply the matching domain template automatically.

    Args:
        domain              : "medical" | "legal" | "coding" | "finance"
        dataset_index       : Which dataset from the domain list to use (default 0).
                              Call list_domain_datasets(domain) to see all options.
        template_style      : Override the template key. Defaults to the domain name
                              (e.g. "medical", "coding") which maps to the domain-specific
                              system prompt defined in chat_templates.py.
        max_samples         : Cap row count. Useful for quick experiments.
        text_field          : Output column name (default "text").
        custom_dataset_name : Pass any HF Hub string to bypass the built-in list.
        split               : Dataset split to load (default "train").
        instruction_field   : Column name for instruction (default "instruction").
        input_field         : Column name for optional input (default "input").
        output_field        : Column name for output/response (default "output").

    Returns:
        HF Dataset with a single text column, ready for SFTTrainer.

    Examples:
        # Medical fine-tuning (uses "medical" template automatically)
        ds = load_domain_dataset("medical", max_samples=5000)

        # Second medical dataset
        ds = load_domain_dataset("medical", dataset_index=1, max_samples=2000)

        # Coding dataset with llama3 template override
        ds = load_domain_dataset("coding", template_style="llama3", max_samples=10000)

        # Finance with Qwen template
        ds = load_domain_dataset("finance", template_style="qwen")

        # Any custom dataset still using domain template
        ds = load_domain_dataset(
            "medical",
            custom_dataset_name="your-org/your-medical-dataset",
        )
    """
    if domain not in DOMAIN_DATASETS and custom_dataset_name is None:
        raise ValueError(
            f"Unknown domain '{domain}'. "
            f"Available: {list(DOMAIN_DATASETS.keys())}\n"
            f"Or pass custom_dataset_name='org/repo' to use any HF Hub dataset."
        )

    # Resolve dataset name
    if custom_dataset_name:
        ds_name = custom_dataset_name
    else:
        ds_list = DOMAIN_DATASETS[domain]
        if dataset_index >= len(ds_list):
            raise IndexError(
                f"dataset_index={dataset_index} out of range for domain '{domain}'. "
                f"Max index: {len(ds_list)-1}. "
                f"Call list_domain_datasets('{domain}') to see all options."
            )
        ds_name = ds_list[dataset_index]

    # Resolve template: use domain name as key (they match by design),
    # or fall back to "chatml" if the domain has no matching template.
    effective_template = template_style or (
        domain if domain in TEMPLATES else "chatml"
    )

    print(f"[*] Domain      : {domain}")
    print(f"[*] Dataset     : {ds_name}")
    print(f"[*] Template    : '{effective_template}'")

    return load_hf_dataset(
        dataset_name=ds_name,
        split=split,
        template_style=effective_template,
        text_field=text_field,
        instruction_field=instruction_field,
        input_field=input_field,
        output_field=output_field,
        max_samples=max_samples,
    )


def list_domain_datasets(domain: str = None) -> None:
    """
    Print all curated domain dataset options with their indices.

    Args:
        domain : Filter to one domain (e.g. "medical"), or None to show all.

    Example:
        list_domain_datasets()
        list_domain_datasets("coding")
    """
    domains = [domain] if domain else list(DOMAIN_DATASETS.keys())
    print("\n" + "═" * 65)
    print("  Available Domain Datasets")
    print("═" * 65)
    for d in domains:
        template_key = d if d in TEMPLATES else "chatml (fallback)"
        print(f"\n  [{d.upper()}]  →  default template: '{template_key}'")
        print(f"  {'─'*60}")
        for i, name in enumerate(DOMAIN_DATASETS[d]):
            print(f"    [{i}]  {name}")
    print("\n" + "═" * 65)
    print(
        "  Usage:\n"
        "    from utility import load_domain_dataset\n"
        "    ds = load_domain_dataset('medical', dataset_index=0, max_samples=5000)\n"
    )