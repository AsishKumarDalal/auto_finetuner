# Auto Finetuner 🚀

The easiest, zero-headache wrapper for fine-tuning Large Language Models (LLMs) using QLoRA.

`AutoFinetuner` abstracts away all the boilerplate of `transformers`, `peft`, `trl`, and `bitsandbytes`. It allows you to download, quantize, and fine-tune massive models using a single line of code, making it perfect for Google Colab, Jupyter Notebooks, or quick prototyping.

## Features
- **Ultimate One-Liner**: Fine-tune a model with a single function call.
- **Auto-Quantization**: Easily load 7B+ parameter models on consumer GPUs using 4-bit or 8-bit precision.
- **Dynamic Prompt Formatting**: Automatically map dataset columns (like `instruction`, `input`, `output`) into a single text prompt using a simple string template.
- **Under the Hood**: Uses industry standards (LoRA, `SFTTrainer`, paged AdamW) for optimized memory efficiency.

---

## Quickstart (The One-Liner)

The absolute simplest way to train and save a model.

```python
from auto_finetuner import finetune_quick

# This will download the model, quantize it, train it, and save it!
my_model = finetune_quick(
    model="NousResearch/Llama-2-7b-chat-hf",
    dataset="mlabonne/guanaco-llama2-1k",
    save_path="./my-custom-llama2",
    quantization_bits=4, # Use 4-bit, 8-bit, or None
    prompt_template="<s>[INST] {instruction} \n{input} [/INST] {output} </s>"
)

# Test your newly fine-tuned model!
response = my_model.generate("What is the meaning of life?")
print(response)
```

---

## Object-Oriented Approach

If you need more control, you can use the `AutoFinetuner` class directly.

```python
from auto_finetuner import AutoFinetuner

# 1. Initialize the finetuner
finetuner = AutoFinetuner(
    model_name="NousResearch/Llama-2-7b-chat-hf",
    dataset_name="mlabonne/guanaco-llama2-1k",
    save_path="./my-custom-llama2",
    quantization_bits=4, 
    prompt_template="<s>[INST] {instruction} [/INST] {output} </s>"
)

# 2. Start the training process!
finetuner.train(epochs=1, batch_size=4, learning_rate=2e-4)

# 3. Generate text
print(finetuner.generate("Explain quantum physics to a child."))
```

## Dependencies
This library relies on standard HuggingFace tools. Ensure you have them installed:
```bash
pip install accelerate peft bitsandbytes transformers trl datasets torch
```