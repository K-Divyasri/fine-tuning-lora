"""finetune_lab -- watch a model LEARN a new skill, and see why LoRA is clever.

Fine-tuning means taking a model and training it further on YOUR data so it gets
good at YOUR task. It's one of three ways to adapt a model, and half of this
project is learning which to reach for:

    prompt engineering   just ask well. No training, no data. Weakest here.
    RAG (retrieval)      look up similar labelled examples at answer time. No training.
    fine-tuning          actually train the weights on your data. Strongest -- if you
                         have the data and the task is stable.

LoRA (Low-Rank Adaptation) is the trick that made fine-tuning cheap: instead of
retraining a giant weight matrix W, you FREEZE it and train a tiny low-rank
"patch" (two small matrices A and B) that adds to it. You end up training a
fraction of the parameters for almost the same result. This project builds that
patch BY HAND so it stops being magic.

Everything here runs on a plain CPU with NO downloads and NO GPU: the "model" is a
small two-layer neural net in numpy, trained on a synthetic support-ticket dataset.
It is small on purpose -- you can watch every gradient. The WORKFLOW you learn
(train/val/test splits, a training loop, measuring lift over a baseline, LoRA's
low-rank patch, overfitting) is exactly the workflow you'd run on a real LLM. The
real thing -- fine-tuning an actual open model with Hugging Face + PEFT/Unsloth on a
free GPU -- is in `real_finetune/`, ready to run on Google Colab.

Modules:
    dataset.py    make the synthetic tickets and the train/val/test split
    features.py   turn text into number vectors (hashed bag-of-words)
    model.py      the small two-layer net you fine-tune
    lora.py       the low-rank adapter, built from scratch
    train.py      the training loops: full fine-tune, head-only, and LoRA
    baselines.py  the two rivals: prompt rules and RAG (k-nearest-neighbours)
    evaluate.py   accuracy, per-class scores, and the comparison table
    cli.py        `python -m finetune_lab train | compare | predict`
"""

from __future__ import annotations

__version__ = "0.1.0"

CLASSES = ("billing", "technical", "account", "shipping")

# Reproducibility: every default seed traces back here, so results never wobble.
DEFAULT_SEED = 7

__all__ = ["__version__", "CLASSES", "DEFAULT_SEED"]
