from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

save_path = "./my-custom-llama2"

# ── 1. Load base model ──────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    quantization_config=bnb_config,
    device_map="auto",
)

# ── 2. Load adapter + tokenizer ─────────────────────────────────
model = PeftModel.from_pretrained(base_model, save_path)
tokenizer = AutoTokenizer.from_pretrained(save_path)
model.eval()  # disable dropout, no gradients needed

# ── 3. Predict loop ─────────────────────────────────────────────
print("=== Predict Loop (type 'quit' to exit) ===\n")

while True:
    prompt = input("You: ").strip()
    
    if prompt.lower() in ("quit", "exit", "q"):
        print("Bye!")
        break
    
    if not prompt:
        continue

    # Tokenize
    inputs = tokenizer(
        f"<s>[INST] {prompt} [/INST]",
        return_tensors="pt"
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,       # how long the reply can be
            temperature=0.7,          # creativity (0=deterministic, 1=creative)
            top_p=0.9,                # nucleus sampling
            do_sample=True,           # enable sampling
            repetition_penalty=1.1,   # avoid repeating itself
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the NEW tokens (skip the input prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    print(f"Model: {response}\n")