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

## 📖 Overview

**AutoFinetuner** is a robust, high-level Python library designed to streamline the complexities of fine-tuning Large Language Models (LLMs). By orchestrating the latest advancements in quantization (QLoRA) and parameter-efficient fine-tuning (PEFT), AutoFinetuner abstracts away the intricate boilerplate of `transformers`, `trl`, and `bitsandbytes`.

Whether you are prototyping in a Jupyter Notebook or deploying enterprise models, AutoFinetuner ensures optimized memory efficiency and maximum throughput.

## ✨ Key Features

- **Zero-Boilerplate Execution:** Initiate full Supervised Fine-Tuning (SFT) with a single function call.
- **Hardware Optimized:** Native support for 4-bit (`nf4`) and 8-bit quantization, enabling 7B+ parameter model training on single consumer GPUs (e.g., NVIDIA RTX 3090/4090, Colab T4).
- **Dynamic Prompt Engineering:** Map complex datasets (instruction, input, output columns) directly into training tensors using intuitive string templates.
- **Industry Standards:** Built on top of the proven HuggingFace ecosystem using `PagedAdamW` optimizers and cosine learning rate schedulers.

## 🚀 Installation

Ensure you have a CUDA-compatible environment, then install the required dependencies:

```bash
pip install accelerate peft bitsandbytes transformers trl datasets torch
```

## 💻 Quickstart (The One-Liner)

For rapid prototyping, use the `finetune_quick` API. This handles downloading, tokenization, quantization, LoRA configuration, training, and adapter saving automatically.

```python
from auto_finetuner import finetune_quick

model = finetune_quick(
    model="NousResearch/Llama-2-7b-chat-hf",
    dataset="mlabonne/guanaco-llama2-1k",
    save_path="./my-custom-model",
    quantization_bits=4, 
    prompt_template="<s>[INST] {instruction} \n{input} [/INST] {output} </s>"
)

# Inference is immediately available
response = model.generate("Explain quantum physics simply.")
print(response)
```

## ⚙️ Advanced Configuration (Object-Oriented)

For production environments requiring fine-grained control over the training loop, utilize the `AutoFinetuner` class.

```python
from auto_finetuner import AutoFinetuner

# 1. Initialize the architecture
finetuner = AutoFinetuner(
    model_name="NousResearch/Llama-2-7b-chat-hf",
    dataset_name="mlabonne/guanaco-llama2-1k",
    save_path="./my-custom-model",
    quantization_bits=4, 
    prompt_template="<s>[INST] {instruction} [/INST] {output} </s>"
)

# 2. Execute Training
finetuner.train(
    epochs=3, 
    batch_size=4, 
    learning_rate=2e-4
)

# 3. Generate Inference
print(finetuner.generate("What is the speed of light?"))
```

## 🤝 Contributing

We welcome contributions from the open-source community! We are looking to add features for DPO (Direct Preference Optimization) and multi-GPU training. Feel free to open a PR!

## 📄 License

This project is licensed under the MIT License.