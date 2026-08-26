# Fine-Tuning and LoRA

Train a tiny classifier three different ways -- head-only, full fine-tune, and
LoRA -- and measure every one of them against a hand-written prompt and a
retrieval (RAG) baseline, on the exact same held-out test set. Everything runs on
a CPU in well under a second, because the whole "model" is a small neural network
written in plain numpy, and every gradient in it is something you can read.

**Problem:** Interviewers want to know you understand *training*, not just
prompting -- and that you can say, with a number, when fine-tuning is worth it
versus RAG or a good prompt. This project builds all three and puts them on the
same scoreboard.

**Skills demonstrated:** the training loop (forward/loss/backward/step), train/
val/test discipline, LoRA and parameter-efficient fine-tuning, overfitting,
measuring lift over a baseline, when to fine-tune vs RAG vs prompt.

**Tech stack:** Python, numpy, pandas, Streamlit, matplotlib, pytest. The real GPU
path (`real_finetune/`) uses Hugging Face transformers, PEFT, TRL, and Unsloth --
not needed to run this package.

## The comparison (what this project proves)

```
method                                  test acc     params    vs full
----------------------------------------------------------------------
prompt (hand keywords)                     0.550          0          -
RAG (kNN over labelled examples)           0.808          0          -
base (frozen backbone, head only)          0.708        256     129.0x
full fine-tune (all weights)               0.983      33024       1.0x
LoRA fine-tune (rank=4)                    0.967       2560      12.9x
```

Fine-tuning wins because the task has real signal a prompt's hand-picked keywords
miss, and a model trained on labelled examples learns it. LoRA gets within two
points of full fine-tuning while training **12.9x fewer parameters** -- the whole
reason LoRA exists.

## LoRA, in one line

Freeze the big pretrained weight matrix `W1`. Train a tiny low-rank patch instead:
`delta = B @ A`, where `A` and `B` are skinny matrices (rank `r`, e.g. 4). The
network uses `W1 + delta`. You update almost nothing and get almost the same
result. Read `finetune_lab/lora.py` -- it's under 50 lines.

## Offline by design (not just by default)

There's no API key to add here: the whole point is to WATCH a model train, so it
runs on CPU, no GPU, no downloads, always. The real GPU fine-tune of an actual LLM
(Qwen2.5-0.5B + Unsloth + PEFT + TRL) lives in `real_finetune/`, written to run on
a free Google Colab/Kaggle T4 GPU -- it is not part of this package's tests or app.

## Run locally (Windows PowerShell)

```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python generate_data.py                 # writes data\train.csv / val.csv / test.csv

python -m finetune_lab compare          # the comparison table above
python -m finetune_lab predict "i was overcharged on my invoice"

streamlit run app.py                    # the interactive comparison + your own ticket
pytest                                  # 27 tests
```

## What I learned

- A training loop is just forward -> loss -> backward -> step, repeated. Nothing
  about it is magic once you've written it in 100 lines of numpy.
- LoRA's trick is forcing the trained update to be low-rank -- most of what a
  fine-tune changes about a weight matrix can be captured by a much smaller patch.
- Fine-tuning, RAG, and prompting aren't rivals to pick once -- they're three
  points on a cost/data/accuracy tradeoff, and the right answer depends on whether
  you have labelled data and how stable the task is.

## Layout

```
fine-tuning-lora/
├── finetune_lab/
│   ├── dataset.py      synthetic support tickets + train/val/test split
│   ├── features.py     text -> hashed bag-of-words vector
│   ├── model.py         the two-layer net you fine-tune
│   ├── lora.py          the low-rank adapter, from scratch
│   ├── train.py         3 training loops: head-only, full, LoRA
│   ├── baselines.py      prompt-keyword and RAG (kNN) baselines
│   ├── evaluate.py       accuracy + the comparison table
│   └── cli.py             python -m finetune_lab compare|predict
├── real_finetune/        the REAL GPU fine-tune (Unsloth+PEFT+TRL), Colab-only
├── tests/                27 pytest tests, all offline
├── app.py                the Streamlit comparison UI
└── generate_data.py      writes the sample CSVs
```
