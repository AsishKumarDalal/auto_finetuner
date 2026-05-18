import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ==========================================
# VRAM GUIDE FOR 25 GB GPU (e.g. RTX 3090 Ti / A5000 / A4500)
# ==========================================
# Model           | VRAM (4-bit QLoRA) | VRAM (8-bit) | VRAM (bf16 full)
# ----------------|--------------------|--------------|-----------------
# Qwen2.5-0.5B    |  ~1.5 GB          |  ~1.8 GB     |  ~2 GB          ✅ Tiny, fast
# Qwen2.5-1.5B    |  ~2.5 GB          |  ~3 GB       |  ~4 GB          ✅ Good 1B class
# Qwen2.5-3B      |  ~3.5 GB          |  ~5 GB       |  ~7 GB          ✅ Sweet spot
# Qwen2.5-7B      |  ~6 GB            |  ~9 GB       |  ~16 GB         ✅ Recommended
# Llama-3.2-1B    |  ~1.5 GB          |  ~2 GB       |  ~3 GB          ✅ Fast 1B
# Llama-3.2-3B    |  ~3 GB            |  ~4 GB       |  ~7 GB          ✅ Good balance
# Llama-3.1-8B    |  ~6 GB            |  ~10 GB      |  ~17 GB         ✅ Best quality at 25GB
# Mistral-7B-v0.3 |  ~6 GB            |  ~9 GB       |  ~15 GB         ✅ Great for instruct
# Phi-3-mini-4k   |  ~3 GB            |  ~4.5 GB     |  ~8 GB          ✅ MS 3.8B, very capable
# Gemma-2-2B      |  ~3 GB            |  ~4 GB       |  ~6 GB          ✅ Google, excellent 2B
# Gemma-2-9B      |  ~8 GB            |  ~12 GB      |  ~20 GB         ✅ Fits with QLoRA
# Llama-3.1-70B   |  ~40 GB           |  ~80 GB      |  N/A            ❌ Too big
#
# RECOMMENDED for 25GB with QLoRA 4-bit:
#   Best quality  → "meta-llama/Llama-3.1-8B-Instruct"   (~6GB VRAM)
#   Fastest       → "meta-llama/Llama-3.2-1B-Instruct"   (~1.5GB VRAM)
#   Best 3B class → "google/gemma-2-2b-it"               (~3GB VRAM)
#   Best overall  → "Qwen/Qwen2.5-7B-Instruct"           (~6GB VRAM)


class AutoFinetuner:
    """
    A high-level wrapper to easily fine-tune LLMs with QLoRA.
    Compatible with latest trl >= 0.12.0 (SFTConfig replaces max_seq_length in SFTTrainer)
    """
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        save_path: str,
        quantization_bits: int = 4,
        dataset_text_field: str = "text",
        prompt_template: str = None,
        max_seq_length: int = 512
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.quantization_bits = quantization_bits
        self.dataset_text_field = dataset_text_field
        self.prompt_template = prompt_template
        self.max_seq_length = max_seq_length

        self.model = None
        self.tokenizer = None
        self.dataset = None

    def _setup_quantization(self):
        compute_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        if self.quantization_bits == 4:
            print(f"Configuring 4-bit quantization with {compute_dtype}...")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False,
            )
        elif self.quantization_bits == 8:
            print("Configuring 8-bit quantization...")
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def prepare(self):
        """Downloads and prepares the dataset, tokenizer, and model."""
        print(f"[*] Loading dataset: {self.dataset_name}")
        self.dataset = load_dataset(self.dataset_name, split="train")

        if self.prompt_template:
            print("[*] Formatting dataset using provided prompt template...")
            original_columns = self.dataset.column_names

            def apply_template(example):
                try:
                    example[self.dataset_text_field] = self.prompt_template.format(**example)
                except KeyError:
                    pass
                return example

            self.dataset = self.dataset.map(apply_template)
            cols_to_remove = [col for col in original_columns if col != self.dataset_text_field]
            if cols_to_remove:
                self.dataset = self.dataset.remove_columns(cols_to_remove)

        print(f"[*] Loading tokenizer for: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        print(f"[*] Loading model: {self.model_name}")
        bnb_config = self._setup_quantization()
        compute_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            # FIX: use `dtype` instead of deprecated `torch_dtype` in newer transformers
            dtype=compute_dtype,
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        print("[+] Preparation complete.")

    def train(self, epochs: int = 1, batch_size: int = 4, learning_rate: float = 2e-4):
        """Starts the SFT (Supervised Fine-Tuning) training process."""
        if self.model is None or self.dataset is None:
            self.prepare()

        print("[*] Setting up LoRA (PEFT) configuration...")
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        # FIX: In trl >= 0.12, max_seq_length moved INTO SFTConfig (not SFTTrainer directly)
        # SFTConfig extends TrainingArguments, so all training args go here too.
        sft_config = SFTConfig(
            output_dir="./training_checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            save_steps=25,
            logging_steps=25,
            learning_rate=learning_rate,
            weight_decay=0.001,
            fp16=not use_bf16,
            bf16=use_bf16,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_steps=10,
            lr_scheduler_type="cosine",
            report_to="none",
            # FIX: max_seq_length now lives here in SFTConfig
            max_seq_length=self.max_seq_length,
            dataset_text_field=self.dataset_text_field,
        )

        print("[*] Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            peft_config=peft_config,
            # FIX: `processing_class` is the new name; `tokenizer` param is deprecated
            processing_class=self.tokenizer,
            args=sft_config,  # FIX: pass SFTConfig here, not TrainingArguments
        )

        print("[*] Starting training... (This might take a while)")
        trainer.train()

        print(f"[*] Saving model adapters to {self.save_path}...")
        trainer.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
        print("[+] Training and saving complete!")

    def generate(self, prompt: str, max_length: int = 200):
        """Test the model by generating text from a prompt."""
        if self.model is None:
            raise ValueError("Model is not loaded. Please prepare or train first.")

        logging.set_verbosity(logging.CRITICAL)
        pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        return result[0]["generated_text"]


# ==========================================
# ULTIMATE ONE-LINER FUNCTION API
# ==========================================
def finetune_quick(
    model: str,
    dataset: str,
    save_path: str,
    quantization_bits: int = 4,
    epochs: int = 1,
    prompt_template: str = None,
    max_seq_length: int = 512,
):
    """A single function call to fine-tune a model and save it."""
    print("=== Starting Quick Finetune Pipeline ===")
    finetuner = AutoFinetuner(
        model_name=model,
        dataset_name=dataset,
        save_path=save_path,
        quantization_bits=quantization_bits,
        prompt_template=prompt_template,
        max_seq_length=max_seq_length,
    )
    finetuner.train(epochs=epochs)
    print("=== Pipeline Finished Successfully ===")
    return finetuner


# ==========================================
# EXAMPLE USAGE — uncomment one block to run
# ==========================================
if __name__ == "__main__":

    # --- OPTION 1: Best quality on 25GB (Llama 3.1 8B, ~6GB VRAM with 4-bit) ---
    # finetune_quick(
    #     model="meta-llama/Llama-3.1-8B-Instruct",
    #     dataset="mlabonne/guanaco-llama2-1k",
    #     save_path="./llama31-8b-finetuned",
    #     quantization_bits=4,
    #     epochs=1,
    #     prompt_template="<s>[INST] {instruction} \n{input} [/INST] {output} </s>",
    #     max_seq_length=1024,
    # )

    # --- OPTION 2: Fastest 1B model (Llama 3.2 1B, ~1.5GB VRAM) ---
    # finetune_quick(
    #     model="meta-llama/Llama-3.2-1B-Instruct",
    #     dataset="mlabonne/guanaco-llama2-1k",
    #     save_path="./llama32-1b-finetuned",
    #     quantization_bits=4,
    #     epochs=2,
    #     prompt_template="<s>[INST] {instruction} \n{input} [/INST] {output} </s>",
    #     max_seq_length=512,
    # )

    # --- OPTION 3: Qwen 2.5 7B (great multilingual, ~6GB VRAM) ---
    # finetune_quick(
    #     model="Qwen/Qwen2.5-7B-Instruct",
    #     dataset="mlabonne/guanaco-llama2-1k",
    #     save_path="./qwen25-7b-finetuned",
    #     quantization_bits=4,
    #     epochs=1,
    #     prompt_template="<s>[INST] {instruction} \n{input} [/INST] {output} </s>",
    #     max_seq_length=1024,
    # )

    # --- OPTION 4: Google Gemma-2 2B (excellent small model, ~3GB VRAM) ---
    # finetune_quick(
    #     model="google/gemma-2-2b-it",
    #     dataset="mlabonne/guanaco-llama2-1k",
    #     save_path="./gemma2-2b-finetuned",
    #     quantization_bits=4,
    #     epochs=2,
    #     prompt_template="<s>[INST] {instruction} \n{input} [/INST] {output} </s>",
    #     max_seq_length=512,
    # )

    # --- CURRENTLY ACTIVE: Qwen 0.5B (your original, tiny test model) ---
    
