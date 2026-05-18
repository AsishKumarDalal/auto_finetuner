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

class AutoFinetuner:
    """
    A high-level wrapper to easily fine-tune LLMs with QLoRA.
    """
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        save_path: str,
        quantization_bits: int = 4,
        dataset_text_field: str = "text",
        prompt_template: str = None
    ):
        """
        Initialize the Finetuner.
        
        :param model_name: HuggingFace model ID (e.g. 'NousResearch/Llama-2-7b-chat-hf')
        :param dataset_name: HuggingFace dataset ID (e.g. 'mlabonne/guanaco-llama2-1k')
        :param save_path: Directory path to save the final finetuned model adapters
        :param quantization_bits: Number of bits for quantization (4, 8, or None for no quantization)
        :param dataset_text_field: The column name in the dataset containing the text to train on
        :param prompt_template: Optional format string for prompt (e.g. "Instruction: {instruction}\nResponse: {response}")
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.quantization_bits = quantization_bits
        self.dataset_text_field = dataset_text_field
        self.prompt_template = prompt_template
        
        self.model = None
        self.tokenizer = None
        self.dataset = None

    def _setup_quantization(self):
        if self.quantization_bits == 4:
            print("Configuring 4-bit quantization...")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
        elif self.quantization_bits == 8:
            print("Configuring 8-bit quantization...")
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        return None

    def prepare(self):
        """Downloads and prepares the dataset, tokenizer, and model."""
        print(f"[*] Loading dataset: {self.dataset_name}")
        self.dataset = load_dataset(self.dataset_name, split="train")
            
        if self.prompt_template:
            print("[*] Formatting dataset using provided prompt template...")
            def apply_template(example):
                try:
                    # Format the text and store it in dataset_text_field
                    example[self.dataset_text_field] = self.prompt_template.format(**example)
                except KeyError as e:
                    pass # Silently skip formatting if some columns are missing for a specific row
                return example
            self.dataset = self.dataset.map(apply_template)
            
        print(f"[*] Loading tokenizer for: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        print(f"[*] Loading model: {self.model_name}")
        bnb_config = self._setup_quantization()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto"
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

        training_arguments = SFTConfig(
            output_dir="./training_checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            save_steps=25,
            logging_steps=25,
            learning_rate=learning_rate,
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_steps=10,
            lr_scheduler_type="cosine",
            report_to="none", # Set to "tensorboard" or "wandb" if you want tracking
            dataset_text_field=self.dataset_text_field,
            max_seq_length=1024,
            packing=False,
        )

        print("[*] Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            peft_config=peft_config,
            tokenizer=self.tokenizer,
            args=training_arguments,
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
        pipe = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer, max_length=max_length)
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        return result[0]['generated_text']


# ==========================================
# ULTIMATE ONE-LINER FUNCTION API
# ==========================================
def finetune_quick(
    model: str, 
    dataset: str, 
    save_path: str, 
    quantization_bits: int = 4,
    epochs: int = 1,
    prompt_template: str = None
):
    """
    A single function call to fine-tune a model and save it.
    """
    print("=== Starting Quick Finetune Pipeline ===")
    finetuner = AutoFinetuner(
        model_name=model,
        dataset_name=dataset,
        save_path=save_path,
        quantization_bits=quantization_bits,
        prompt_template=prompt_template
    )
    finetuner.train(epochs=epochs)
    print("=== Pipeline Finished successfully ===")
    return finetuner

# Example Usage Block (Uncomment to run directly):
# if __name__ == "__main__":
#     # 1. Simplest approach: One function call
#     finetune_quick(
#         model="NousResearch/Llama-2-7b-chat-hf",
#         dataset="mlabonne/guanaco-llama2-1k", # Assuming it has columns like 'instruction', 'input', 'output'
#         save_path="./my-custom-llama2",
#         quantization_bits=4,
#         prompt_template="<s>[INST] {instruction} \n{input} [/INST] {output} </s>"
#     )