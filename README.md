<div align="center">
  <!-- NOTE: Logo placeholder. You can upload your own cool logo here later! -->
  <h1>⚙️ AutoFinetuner</h1>

  **Enterprise-Grade, Zero-Boilerplate LLM Fine-Tuning**

  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

  *Train massive language models on consumer hardware in a single line of code.*
</div>

---

## 📖 Architectural Overview

**AutoFinetuner** is a robust, high-level Python library designed to streamline the complexities of fine-tuning Large Language Models (LLMs). It acts as an intelligent orchestration layer on top of the HuggingFace ecosystem (`transformers`, `peft`, `trl`, and `bitsandbytes`).

By abstracting away intricate boilerplate, AutoFinetuner ensures optimized memory efficiency, rapid prototyping, and maximum throughput, allowing developers to focus on data and results rather than infrastructure quirks.

---

## 🧩 Codebase Analysis & Core Modules

The repository is modularly designed to separate data ingestion, prompt formatting, model orchestration, and inference.

### 1. `auto_finetuner.py` (The Orchestrator)
This is the core engine of the library, providing both functional (`finetune_quick`) and Object-Oriented (`AutoFinetuner`) APIs.
- **Hardware Optimization:** Automatically detects hardware support for `bfloat16` and defaults to `float16` if unavailable. Sets up 4-bit (`nf4`) or 8-bit quantization via `BitsAndBytesConfig`.
- **PEFT Integration:** Configures `LoraConfig` out of the box, injecting adapters into the causal language model.
- **Robust Training Loop:** Leverages `SFTTrainer` with `PagedAdamW` optimizers and cosine learning rate schedulers. It dynamically detects and handles `trl` version changes (e.g., handling `max_seq_length` vs `max_length` parameter deprecations).

### 2. `utility.py` (Data Pipeline)
A highly resilient dataset utility module capable of interpreting and normalizing various data structures into a unified format for the `SFTTrainer`.
- **Format Agnostic:** Natively processes Alpaca-style JSONs, Multi-turn Chat formats, and raw text files. 
- **HuggingFace Hub Integration:** Maps seamlessly to remote datasets (`load_hf_dataset`).
- **Domain Mappings:** Introduces `load_domain_dataset()`, enabling one-click downloads of curated datasets across specialized fields like **Medical**, **Legal**, **Coding**, and **Finance**, while automatically associating them with their respective prompt templates.

### 3. `chat_templates.py` (Prompt Engineering)
A centralized registry for prompt templates.
- **Unified Formatting:** Normalizes heterogeneous dataset columns (instruction, input, output) into unified string templates tailored for specific base models (e.g., `qwen`, `llama2`, `llama3`).
- **Domain Prompts:** Contains specialized system prompts for domain-specific fine-tuning (e.g., injecting a clinical persona for medical datasets).
- **Auto-Detection:** Integrates with `auto_finetuner.py` to auto-detect the optimal template string based on the model's HuggingFace Hub ID.

### 4. `inference.py` (Evaluation)
A clean implementation for testing fine-tuned models.
- Demonstrates loading a quantized base model and wrapping it with `PeftModel` to merge the trained LoRA adapters dynamically.
- Implements a reactive console loop utilizing nucleus sampling (`top_p`), temperature scaling, and repetition penalties to evaluate model creativity and coherence safely.

### 5. `exmaple_finetune.py` (Entrypoint)
A minimalist demonstration script showcasing the library's power: loading a coding dataset, mapping it to LLaMA-3 formatting, and launching a complete training run in under 20 lines of code.

---

## ⚡ Hardware & Memory Guidelines

AutoFinetuner is built to train large models on consumer GPUs (e.g., RTX 3090/4090, Colab T4).

| Model Size | 4-bit QLoRA (VRAM) | 8-bit (VRAM) | BF16 Full (VRAM) |
|------------|---------------------|--------------|-------------------|
| 0.5B (Qwen)| ~1.5 GB             | ~1.8 GB      | ~2.0 GB           |
| 3B (LLaMA) | ~2.5 GB             | ~4.0 GB      | ~7.0 GB           |
| 7B / 8B    | ~6.0 GB             | ~9.0 GB      | ~16.0 GB          |

---

## 🚀 Quickstart Installation

Ensure you have a CUDA-compatible environment, then install the required dependencies:

```bash
pip install -r requirements.txt
```

For rapid prototyping, use the `finetune_quick` API:

```python
from auto_finetuner import finetune_quick
from utility import load_domain_dataset

# 1. Load a curated coding dataset (auto-applies LLaMA 3 templates)
ds = load_domain_dataset("coding", dataset_index=1, template_style="llama3")

# 2. Train and save in one function call
model = finetune_quick(
    model="unsloth/Llama-3.2-3B-Instruct",
    hf_dataset=ds,
    save_path="./my-custom-model",
    quantization_bits=4, 
    epochs=1
)
```

## 🤝 Contributing

We welcome contributions from the open-source community! We are looking to add features for DPO (Direct Preference Optimization), multi-GPU scaling (FSDP), and additional domain datasets. Feel free to open a PR!

## 📄 License

This project is licensed under the MIT License.
