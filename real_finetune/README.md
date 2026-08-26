# The real thing: fine-tuning an actual open LLM with LoRA on a free GPU

Everything in `finetune_lab/` runs on a plain CPU because it's a
small numpy network -- perfect for SEEING every gradient, but it is not a real
language model. This folder is the honest bridge to the real thing.

**This code is NOT run as part of the local project**: it needs a GPU this
machine doesn't have set up for it, and downloads several GB of model weights.
It is written to be copied into a free Google Colab or Kaggle notebook and run
there, in about 10-15 minutes on a free T4 GPU.

## What it does

Fine-tunes a small open model (`Qwen2.5-0.5B-Instruct`, a genuinely small
instruction-tuned model that behaves well on a free T4) with LoRA to do the exact
same task as the toy project: classify a support ticket into billing / technical /
account / shipping. It uses:

- **Unsloth**: patches Hugging Face `transformers` to fine-tune ~2x faster and with
  less memory, purpose-built for free-tier Colab/Kaggle GPUs.
- **PEFT** (`LoraConfig`): the real LoRA implementation. Compare its parameters
  (`r`, `lora_alpha`, `target_modules`) to `finetune_lab/lora.py`'s `rank` and
  `scaling` -- same idea, industrial version.
- **TRL** (`SFTTrainer`): the standard supervised fine-tuning loop (this replaces
  `finetune_lab/train.py`'s hand-written loop with a battle-tested one).

## How to run it

1. Open a new notebook at <https://colab.research.google.com/> (or Kaggle).
2. Set the runtime to a free GPU: Runtime -> Change runtime type -> T4 GPU.
3. Copy `real_lora_finetune.py`'s contents into cells (or upload it and `!python
   real_lora_finetune.py`).
4. Run top to bottom. First run downloads the base model (~1GB) -- expect a few
   minutes; training itself is a couple of minutes on the tiny dataset here.
5. Compare the printed before/after accuracy to the table `finetune_lab compare`
   printed on your CPU-only run -- same shape of result, real model this time.

## The one-line mental model

Toy project: `net.W1` frozen, `LoRA(A, B)` patches it, `numpy` backprop trains it.
Real thing: the base model's attention/MLP weight matrices are frozen, `LoraConfig`
patches them the same way, `SFTTrainer` trains it. You already understand the shape
of what's happening -- only the scale changed.
