from utility import download_alpaca,load_domain_dataset
from chat_templates import list_templates
from auto_finetuner import finetune_quick


ds = load_domain_dataset(
    "coding",
    dataset_index=2,          # sahil2801/CodeAlpaca-20k
    template_style="llama3",  # override to llama3 format
    max_samples=10000,
)

finetuner = finetune_quick(
    model="unsloth/Llama-3.2-3B-Instruct",
    hf_dataset=ds,
    save_path="./output/llama3-coding",
    epochs=3,
)