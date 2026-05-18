"""
auto_finetuner.py — One-Liner LLM Fine-Tuning with QLoRA

Features:
  - One-line finetune_quick() function
  - OR object-oriented AutoFinetuner class
  - Three dataset modes:
      1. hf_dataset=     → pass a pre-built HF Dataset directly
      2. dataset_name=   → load any HF Hub dataset by string
      3. alpaca_path=    → download the Alpaca JSON and format it locally
  - Template selection via:
      a. template_style=   → named key from chat_templates.py  (e.g. "qwen", "llama3")
      b. prompt_template=  → raw format string override
  - Auto-detects trl version changes (max_seq_length vs max_length)
  - Tries bfloat16 automatically when supported
  - Native Qwen2.5, Llama-3.x, Gemma, Phi-3, Mistral support

VRAM GUIDE (25 GB GPU — RTX 3090 Ti / A5000 / A4500):
  Model            4-bit QLoRA   8-bit      bf16 full
  Qwen2.5-0.5B      ~1.5 GB     ~1.8 GB    ~2 GB
  Qwen2.5-7B        ~6 GB       ~9 GB      ~16 GB    ← recommended
  Llama-3.2-1B      ~1.5 GB     ~2 GB      ~3 GB
  Llama-3.1-8B      ~6 GB       ~10 GB     ~17 GB
  Mistral-7B-v0.3   ~6 GB       ~9 GB      ~15 GB
  Phi-3-mini-4k     ~3 GB       ~4.5 GB    ~8 GB
  Gemma-2-2B        ~3 GB       ~4 GB      ~6 GB
"""

import os
import inspect
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from chat_templates import get_template, best_template_for, list_templates, TEMPLATES
from utility import (
    download_alpaca,
    load_hf_dataset,
)


# ══════════════════════════════════════════════════════════════════════════════
# AutoFinetuner class
# ══════════════════════════════════════════════════════════════════════════════

class AutoFinetuner:
    """
    High-level wrapper to fine-tune LLMs with QLoRA (4-bit or 8-bit).

    Dataset modes (pick ONE):
      A. hf_dataset=<HF Dataset>   → inject a pre-built dataset, skip all loading
      B. dataset_name=<str>        → load any HF Hub dataset string
      C. alpaca_path=<str>         → download + format the Stanford Alpaca JSON

    Template modes (pick ONE, only used when loading / formatting):
      1. template_style=<str>      → named key from chat_templates.py
                                     e.g. "qwen" | "llama3" | "alpaca" | ...
                                     Run list_templates() to see all options.
      2. prompt_template=<str>     → raw format string  {instruction} {input} {output}
                                     (also set prompt_template_no_input= for the
                                      no-input variant; falls back to prompt_template)

    Examples
    --------
    # Inject a pre-built dataset (no template needed — already formatted)
    finetuner = AutoFinetuner(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        save_path="./out",
        hf_dataset=my_ds,
    )

    # Load from HF Hub + named template
    finetuner = AutoFinetuner(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        save_path="./out",
        dataset_name="tatsu-lab/alpaca",
        template_style="llama3",
    )

    # Load from HF Hub + raw template string
    finetuner = AutoFinetuner(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        save_path="./out",
        dataset_name="tatsu-lab/alpaca",
        prompt_template="<s>[INST] {instruction}\\n{input} [/INST] {output} </s>",
        prompt_template_no_input="<s>[INST] {instruction} [/INST] {output} </s>",
    )

    # Alpaca local path + named template
    finetuner = AutoFinetuner(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        save_path="./out",
        alpaca_path="./data/alpaca_data.json",
        template_style="qwen",
    )
    """

    def __init__(
        self,
        model_name: str,
        save_path: str,
        # ── Dataset mode (pick ONE) ──────────────────
        hf_dataset: Dataset = None,         # Mode A: pre-built HF Dataset
        dataset_name: str = None,           # Mode B: HF Hub string
        alpaca_path: str = None,            # Mode C: local Alpaca JSON path
        # ── Template (only for modes B / C) ─────────
        template_style: str = None,         # named key from chat_templates.py
        prompt_template: str = None,        # raw format string override
        prompt_template_no_input: str = None,
        # ── Other config ─────────────────────────────
        quantization_bits: int = 4,
        dataset_text_field: str = "text",
        max_seq_length: int = 512,
        max_samples: int = None,
        rank : int = 32,
    ):
        self.model_name               = model_name
        self.save_path                = save_path

        # Dataset mode
        self._injected_dataset        = hf_dataset
        self.dataset_name             = dataset_name
        self.alpaca_path              = alpaca_path

        # Template
        self.template_style           = template_style
        self.prompt_template          = prompt_template
        self.prompt_template_no_input = prompt_template_no_input or prompt_template

        # Config
        self.quantization_bits        = quantization_bits
        self.dataset_text_field       = dataset_text_field
        self.max_seq_length           = max_seq_length
        self.max_samples              = max_samples

        # Runtime state
        self.model     = None
        self.tokenizer = None
        self.dataset   = None
        self.rank=rank

        # Validate: at least one dataset source must be provided
        if hf_dataset is None and dataset_name is None and alpaca_path is None:
            raise ValueError(
                "Provide at least one dataset source:\n"
                "  hf_dataset=<HF Dataset>  (pre-built)\n"
                "  dataset_name='org/repo'  (HF Hub)\n"
                "  alpaca_path='./data/alpaca_data.json'"
            )

    # ──────────────────────────────────────────────────────
    # PRIVATE helpers
    # ──────────────────────────────────────────────────────

    def _setup_quantization(self) -> BitsAndBytesConfig | None:
        compute_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        if self.quantization_bits == 4:
            print(f"[*] 4-bit quantization ({compute_dtype})...")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False,
            )
        elif self.quantization_bits == 8:
            print("[*] 8-bit quantization...")
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def _clean_dataset_columns(self):
        """Keep only the text column — SFTTrainer requires it."""
        extra = [
            col for col in self.dataset.column_names
            if col != self.dataset_text_field
        ]
        if extra:
            print(f"[*] Dropping extra columns: {extra}")
            self.dataset = self.dataset.remove_columns(extra)
        print(f"[✓] Dataset ready — {len(self.dataset):,} rows")

    def _resolve_template_style(self) -> str | None:
        """
        Determine effective template_style.
        If neither template_style nor prompt_template is given,
        auto-detect from model name.
        """
        if self.prompt_template:
            return None   # raw override takes precedence
        if self.template_style:
            return self.template_style
        # Auto-detect
        detected = best_template_for(self.model_name)
        print(f"[*] Auto-detected template '{detected}' for model '{self.model_name}'")
        return detected

    # ──────────────────────────────────────────────────────
    # PUBLIC: prepare
    # ──────────────────────────────────────────────────────

    def prepare(self):
        """
        Load dataset (if not injected), tokenizer, and model.

        Dataset loading order:
          1. hf_dataset was injected → use it directly
          2. dataset_name given      → load_hf_dataset() from Hub
          3. alpaca_path given       → download_alpaca() locally
        """

        # ── Dataset ───────────────────────────────────────
        if self._injected_dataset is not None:
            print(f"[✓] Using pre-injected dataset ({len(self._injected_dataset):,} rows)")
            self.dataset = self._injected_dataset

        elif self.dataset_name is not None:
            effective_style  = self._resolve_template_style()
            print(f"[*] Loading HF Hub dataset: {self.dataset_name}")
            self.dataset = load_hf_dataset(
                dataset_name=self.dataset_name,
                template_style=effective_style,
                template_override=self.prompt_template,
                template_no_input_override=self.prompt_template_no_input,
                text_field=self.dataset_text_field,
                max_samples=self.max_samples,
            )

        elif self.alpaca_path is not None:
            effective_style = self._resolve_template_style()
            print(f"[*] Loading Alpaca from: {self.alpaca_path}")
            self.dataset = download_alpaca(
                save_path=self.alpaca_path,
                template_style=effective_style or "chatml",
                template_override=self.prompt_template,
                template_no_input_override=self.prompt_template_no_input,
                text_field=self.dataset_text_field,
                max_samples=self.max_samples,
            )

        self._clean_dataset_columns()

        # ── Tokenizer ─────────────────────────────────────
        print(f"[*] Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # ── Model ─────────────────────────────────────────
        print(f"[*] Loading model: {self.model_name}")
        bnb_config    = self._setup_quantization()
        compute_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=compute_dtype,
        )
        self.model.config.use_cache       = False
        self.model.config.pretraining_tp  = 1
        print("[+] Preparation complete.")

    # ──────────────────────────────────────────────────────
    # PUBLIC: train
    # ──────────────────────────────────────────────────────

    def train(self, epochs: int = 1, batch_size: int = 4, learning_rate: float = 2e-4):
        """Run supervised fine-tuning (SFT) with LoRA."""

        if self.model is None or self.dataset is None:
            self.prepare()
        else:
            self._clean_dataset_columns()

        print("[*] Setting up LoRA (PEFT)...")
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=self.rank,
            bias="none",
            task_type="CAUSAL_LM",
        )

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        # trl renames the sequence-length arg across versions
        _sft_params = inspect.signature(SFTConfig.__init__).parameters
        if "max_length" in _sft_params:
            _seq_len_kwarg = {"max_length": self.max_seq_length}
        elif "max_seq_length" in _sft_params:
            _seq_len_kwarg = {"max_seq_length": self.max_seq_length}
        else:
            _seq_len_kwarg = {}

        sft_config = SFTConfig(
            output_dir="./training_checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            save_steps=5000,
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
            dataset_text_field=self.dataset_text_field,
            **_seq_len_kwarg,
        )

        print("[*] Initializing SFTTrainer...")
        _trainer_seq_kwarg = {} if _seq_len_kwarg else {"max_seq_length": self.max_seq_length}
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            peft_config=peft_config,
            processing_class=self.tokenizer,
            args=sft_config,
            **_trainer_seq_kwarg,
        )

        print("[*] Starting training...")
        trainer.train()

        print(f"[*] Saving model adapters to {self.save_path}...")
        trainer.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
        print("[+] Training complete!")

    # ──────────────────────────────────────────────────────
    # PUBLIC: generate
    # ──────────────────────────────────────────────────────

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Test the fine-tuned model with a text prompt."""
        if self.model is None:
            raise ValueError("Model not loaded. Call prepare() or train() first.")

        logging.set_verbosity(logging.CRITICAL)
        pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
        )
        result = pipe(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
        return result[0]["generated_text"]


# ══════════════════════════════════════════════════════════════════════════════
# ONE-LINER FUNCTION API
# ══════════════════════════════════════════════════════════════════════════════

def finetune_quick(
    model: str,
    save_path: str,
    # ── Dataset (pick ONE) ──────────────────────────
    hf_dataset: Dataset = None,         # pre-built HF Dataset
    dataset_name: str = None,           # HF Hub string
    alpaca_path: str = None,            # local Alpaca JSON path
    # ── Template (pick ONE, optional) ───────────────
    template_style: str = None,         # named key e.g. "qwen" | "llama3"
    prompt_template: str = None,        # raw format string
    prompt_template_no_input: str = None,
    # ── Training config ─────────────────────────────
    quantization_bits: int = 4,
    epochs: int = 1,
    max_seq_length: int = 512,
    max_samples: int = None,
    rank: int =64
) -> AutoFinetuner:
    """
    Single function call to fine-tune a model and save it.

    Dataset modes (pick ONE):
      hf_dataset=     → pre-built HF Dataset (already has a 'text' column)
      dataset_name=   → HF Hub repo string — loaded + templated automatically
      alpaca_path=    → local path to alpaca_data.json

    Template modes (optional, used with dataset_name / alpaca_path):
      template_style= → named key from chat_templates.py
                        e.g. "qwen" | "llama3" | "llama2" | "alpaca" | ...
                        Auto-detected from model name when not given.
                        Run list_templates() to see all options.
      prompt_template=→ raw format string  {instruction} {input} {output}

    Examples
    --------
    # Pre-built dataset (already formatted — no template needed)
    from utility import download_alpaca
    ds = download_alpaca(template_style="qwen", max_samples=1000)
    finetuner = finetune_quick(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        hf_dataset=ds,
        save_path="./qwen-alpaca",
    )

    # HF Hub dataset + named template
    finetuner = finetune_quick(
        model="meta-llama/Llama-3.2-3B-Instruct",
        dataset_name="tatsu-lab/alpaca",
        template_style="llama3",
        save_path="./llama3-alpaca",
        epochs=1,
        max_seq_length=512,
        max_samples=2000,
    )

    # HF Hub dataset + raw template string
    finetuner = finetune_quick(
        model="meta-llama/Llama-3.2-3B-Instruct",
        dataset_name="tatsu-lab/alpaca",
        prompt_template="<s>[INST] {instruction}\\n{input} [/INST] {output} </s>",
        prompt_template_no_input="<s>[INST] {instruction} [/INST] {output} </s>",
        save_path="./llama2-raw",
    )

    # HF Hub dataset, no template given → auto-detected from model name
    finetuner = finetune_quick(
        model="Qwen/Qwen2.5-7B-Instruct",
        dataset_name="mlabonne/guanaco-llama2-1k",
        save_path="./qwen-guanaco",
    )
    """
    if hf_dataset is None and dataset_name is None and alpaca_path is None:
        raise ValueError(
            "Provide at least one dataset source:\n"
            "  hf_dataset=    pre-built HF Dataset\n"
            "  dataset_name=  HF Hub repo string\n"
            "  alpaca_path=   local alpaca_data.json path"
        )

    print("=== Starting Quick Finetune Pipeline ===")

    finetuner = AutoFinetuner(
        model_name=model,
        save_path=save_path,
        hf_dataset=hf_dataset,
        dataset_name=dataset_name,
        alpaca_path=alpaca_path,
        template_style=template_style,
        prompt_template=prompt_template,
        prompt_template_no_input=prompt_template_no_input,
        quantization_bits=quantization_bits,
        max_seq_length=max_seq_length,
        max_samples=max_samples,
        rank=rank
    )

    finetuner.train(epochs=epochs)
    print("=== Pipeline Finished Successfully ===")
    return finetuner
