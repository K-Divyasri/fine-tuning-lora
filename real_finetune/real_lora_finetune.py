"""Fine-tune a real small open LLM with LoRA on a free GPU -- run on Colab/Kaggle.

    NOT RUN LOCALLY. This needs a GPU + several GB of downloads. Copy this into a
    Google Colab notebook (free T4 GPU) or Kaggle notebook and run it there.
    See README.md in this folder for exact steps.

This does, for real, the same task the CPU toy project does: classify a support
ticket into billing / technical / account / shipping. The dataset is the same
synthetic one (data/train.csv, data/test.csv from generate_data.py) -- upload those
two files to the Colab session, or regenerate them there with the project's
generate_data.py.

Read this top to bottom even if you never run it -- the comments map every real
call back to the toy version you already understand:

    finetune_lab.model.TwoLayerNet.W1 (frozen)  <->  the base model's attention/MLP weights
    finetune_lab.lora.LoRA(A, B)                 <->  peft.LoraConfig + get_peft_model
    finetune_lab.train.fit_lora (numpy backprop) <->  trl.SFTTrainer.train()
"""

# --- Step 0: install (uncomment on Colab; these are NOT in requirements.txt) ---
# !pip install -q unsloth trl peft transformers accelerate datasets bitsandbytes

import pandas as pd

# --- Step 1: load the SAME dataset the CPU toy project uses -------------------
# Upload data/train.csv and data/test.csv (from ../generate_data.py) to the Colab
# session first, or regenerate them there.
CLASSES = ["billing", "technical", "account", "shipping"]
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# SFTTrainer expects instruction-style text. We turn each row into a short prompt +
# the expected completion -- this is "supervised fine-tuning": show the model
# (input, correct output) pairs and train it to produce the output given the input.
def to_prompt(text: str) -> str:
    return (
        "Classify the support ticket into exactly one of: "
        "billing, technical, account, shipping.\n"
        f"Ticket: {text}\nCategory:"
    )

train_df["prompt"] = train_df["text"].apply(to_prompt)
train_df["completion"] = train_df["label"].apply(lambda i: " " + CLASSES[i])
train_df["full_text"] = train_df["prompt"] + train_df["completion"]

from datasets import Dataset  # noqa: E402
train_dataset = Dataset.from_pandas(train_df[["full_text"]])

# --- Step 2: load a small base model with Unsloth (fast, free-GPU-friendly) ----
from unsloth import FastLanguageModel  # noqa: E402

MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"  # a genuinely small instruct model

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=512,
    load_in_4bit=True,   # QLoRA: quantise the frozen base to 4-bit to save VRAM
)

# --- Step 3: attach LoRA -- THIS is the real version of lora.py ---------------
# Compare to finetune_lab.lora.LoRA.init(dim, hidden, rank): r here IS that rank.
# target_modules picks WHICH weight matrices get a low-rank patch (the attention
# projection matrices -- the real equivalent of patching W1 in the toy project).
model = FastLanguageModel.get_peft_model(
    model,
    r=8,                       # LoRA rank -- same meaning as finetune_lab.lora rank
    lora_alpha=16,             # scaling numerator (scaling = alpha / r, same idea)
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",  # trades a little speed for a lot less VRAM
)

# How many parameters are actually trainable? This is the real version of the
# "12.9x fewer params" comparison the toy project prints.
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# --- Step 4: measure BEFORE accuracy (the base model, untouched) --------------
def classify_with_model(model, tokenizer, text: str) -> str:
    prompt = to_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    completion = completion.strip().lower()
    return next((c for c in CLASSES if c in completion), "unknown")


def accuracy(model, tokenizer, df) -> float:
    correct = sum(
        classify_with_model(model, tokenizer, row.text) == CLASSES[row.label]
        for row in df.itertuples()
    )
    return correct / len(df)


print("Measuring BEFORE accuracy (base model, no fine-tuning)...")
before_acc = accuracy(model, tokenizer, test_df)
print(f"before fine-tuning: {before_acc:.3f}")

# --- Step 5: train -- THIS is the real version of finetune_lab.train.fit_lora --
from trl import SFTConfig, SFTTrainer  # noqa: E402

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="full_text",
    args=SFTConfig(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-4,   # LoRA typically wants a higher LR than full fine-tuning
        logging_steps=5,
        output_dir="outputs",
        report_to="none",
    ),
)
trainer.train()

# --- Step 6: measure AFTER accuracy and compare --------------------------------
print("Measuring AFTER accuracy (LoRA fine-tuned)...")
after_acc = accuracy(model, tokenizer, test_df)
print(f"before fine-tuning: {before_acc:.3f}")
print(f"after  fine-tuning: {after_acc:.3f}")
print(f"lift: {after_acc - before_acc:+.3f}")

# --- Step 7: save the LoRA adapter (small!) and optionally push to HF Hub ------
model.save_pretrained("qwen_ticket_lora")
tokenizer.save_pretrained("qwen_ticket_lora")
# model.push_to_hub("your-username/qwen-ticket-lora")   # see hosting/HOSTING_GUIDE.md
print("Saved adapter to ./qwen_ticket_lora (a few MB, not the whole model).")
