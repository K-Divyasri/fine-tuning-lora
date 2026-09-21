# Publishing the Fine-Tuning / LoRA project

This project is a little different from the others in the roadmap, so read this part
before you start: there are **two separate things you can host**, and they don't need
the same machine.

1. **The comparison app** (`app.py`) — a Streamlit demo that trains
   all five methods (prompt, RAG, base, full fine-tune, LoRA) on a small numpy network
   and shows the comparison table live in the browser. This is CPU-only, takes well
   under a second to train, and is what most people should host. **Path A** below.
2. **A real trained LoRA adapter** — only if you actually ran `real_finetune/` on a
   free Colab/Kaggle GPU and fine-tuned the real `Qwen2.5-0.5B-Instruct` model. That
   adapter is a few megabytes and the natural place to "host" it is the Hugging Face
   Hub, with `push_to_hub`. **Path B** below, and it's optional.

The GPU fine-tuning step itself — the thing that actually trains the real LLM — never
runs on a paid host. It runs on a **free** Colab or Kaggle T4 GPU, once, and you keep
the small adapter file that comes out of it. Nothing in this project needs a GPU
server you pay for.

---

## The layout you're working with

```
19-fine-tuning-lora/              <- this whole folder becomes your GitHub repo
├── app.py                        <- the Streamlit comparison app
├── finetune_lab/                 <- the package app.py imports
├── requirements.txt
├── data/                         <- train.csv / val.csv / test.csv (small, committed)
├── tests/                        <- 27 offline, CPU-only tests
├── real_finetune/                <- Colab-only real GPU LoRA fine-tune (not run here)
├── generate_data.py              <- rebuilds data\*.csv
├── hosting/                      <- you are here
├── knowledge/, notebooks/, labs/ <- the teaching material
└── README.md
```

- **GitHub gets the whole `19-fine-tuning-lora/` folder.** The CI workflow and the
  root README are written for that.
- **The live app is deployed separately** (Path A), and it deploys `app.py`
  plus the `finetune_lab/` package it imports.
- **The LoRA adapter (if you make one) is hosted on the Hugging Face Hub**, completely
  separate from GitHub and from the Streamlit app — see Path B.

---

## Step 0 — Get the code on GitHub

### Install Git and tell it who you are (once per machine)

If Git already works, skip this. Otherwise download it from
<https://git-scm.com/download/win>, click Next through the defaults, then open a
**new** PowerShell window and check:

```powershell
git --version
```

Stamp your identity onto commits (use the same email as your GitHub account):

```powershell
git config --global user.name "Your Name"
git config --global user.email "mathuransada@gmail.com"
```

### There's no API key here — but check the .gitignore anyway

This project's core (`finetune_lab/` and `app.py`) makes zero LLM calls. No `.env`, no key,
no secret, nothing to leak. That's genuinely unusual for this roadmap — enjoy it.

The one thing worth excluding is still the usual Python junk. The root `.gitignore`
already lists:

```
__pycache__/
.venv/
.pytest_cache/
*.egg-info/
```

Plus, if you ever run `real_finetune/` on Colab and download its output locally, that
`.gitignore` also excludes `real_finetune/outputs/` and `real_finetune/*_lora/` — those
are big model-adapter working directories, not something you want in a Git repo (the
Hub, not GitHub, is where a trained adapter belongs — see Path B).

One deliberate **inclusion**: the small CSVs under `data/`
(`train.csv`, `val.csv`, `test.csv`) are committed on purpose. They're tiny, generated
by a fixed random seed, and committing them means the app and notebooks work the moment
someone clones — no build step needed.

### Make the repo and push

Run these from the **project root** — the `19-fine-tuning-lora/` folder, the one with
this `hosting/` folder inside it.

```powershell
cd ai\19-fine-tuning-lora
git init
git add .
git commit -m "Initial commit: fine-tuning / LoRA from scratch"
```

Check the tree is clean:

```powershell
git status
```

You want `nothing to commit, working tree clean`. Then make an **empty** repo on
github.com (top-right **+** menu → **New repository**), name it `fine-tuning-lora`,
leave it **Public**, and do **not** tick "Add a README / .gitignore / license" (an
empty repo avoids a first-push collision). Copy the URL it shows you, then:

```powershell
git branch -M main
git remote add origin https://github.com/YOURNAME/fine-tuning-lora.git
git push -u origin main
```

The first push opens a browser to sign in. If it asks for a password typed into the
terminal, that won't work — GitHub turned off password auth years ago. Use the browser
sign-in, or install the GitHub CLI (<https://cli.github.com>) and run `gh auth login`
once.

### Add CI so the tests run on every push

This `hosting/` folder ships a ready workflow at `github_actions/ci.yml`. GitHub only
runs workflows that live under `.github/workflows/`, so copy it there. From the
**project root**:

```powershell
mkdir .github\workflows
copy hosting\github_actions\ci.yml .github\workflows\ci.yml
git add .github\workflows\ci.yml
git commit -m "Add GitHub Actions CI to run the offline tests on every push"
git push
```

Open the repo's **Actions** tab to watch it run: checkout -> install Python -> install
deps -> `pytest`. It's **keyless** — every one of the 27 tests trains a tiny numpy
network on CPU, no GPU and no download involved, so nothing here needs a secret. Green
means all 27 passed on GitHub's machine.

Once it's green, grab the status badge (the workflow's Actions page has a `...` menu →
**Create status badge**) and paste the markdown at the top of your root `README.md`.

---

## Path A (recommended) — deploy the Streamlit comparison app

This is the app that's actually worth a live link: type a support ticket, watch it get
classified by the fully fine-tuned model, and see the comparison table (prompt vs RAG
vs base vs full fine-tune vs LoRA) render live. It's CPU-only numpy — nothing here needs
a GPU, so it drops onto any free tier instantly.

### Option 1 — Hugging Face Spaces

A "Space" is a free, always-on web app hosted by Hugging Face. It speaks Streamlit
natively. Official docs: <https://huggingface.co/docs/hub/en/spaces-sdks-streamlit>.

1. Make a free account at <https://huggingface.co>.
2. Top-right, your avatar → **New Space**.
3. **Space name:** `fine-tuning-lora`. **License:** whatever you like.
4. **Space SDK:** pick **Streamlit** — this tells Hugging Face to look for `app.py`
   and run `streamlit run` on it for you.
5. Leave hardware on the free **CPU basic** tier — this app never touches a GPU, the
   whole "model" is a numpy array. Click **Create Space**.

A Space is itself a Git repo and expects the app at its **root**. Our files sit at the
repo root, so upload them to the Space root as they are. What the Space
needs:

- `app.py` — the Streamlit app
- `finetune_lab/` — the whole package `app.py` imports
- `requirements.txt` — so the Space installs numpy, pandas, streamlit
- `data/train.csv`, `data/val.csv`, `data/test.csv` — the small committed CSVs (the
  app can also regenerate the data in memory via `make_splits()`, but shipping the
  CSVs matches the rest of the project and costs nothing)

Easiest path for a beginner: on your Space page, the **Files** tab → **Add file** →
**Upload files**. Drag in `app.py`, `requirements.txt`, the whole `finetune_lab` folder,
and the `data` folder — all from the repo root. Commit.

> You do **not** need to upload `tests/`, `generate_data.py`, or `real_finetune/` — the
> app doesn't use any of them.

Once you commit, the Space builds — watch the **Logs**. A minute or two later you have
a live URL like `https://huggingface.co/spaces/YOURNAME/fine-tuning-lora`. Open it, move
the sliders (training size, LoRA rank, seed), and type a ticket into "Try your own
ticket" to see it classified live.

### Option 2 — Streamlit Community Cloud

Streamlit's own free host, wired straight to the GitHub repo you already pushed. Home
page: <https://streamlit.io/cloud>.

1. Sign in with your GitHub account, granting it read access to your repos.
2. **Create app** → **Deploy a public app from GitHub**.
3. **Repository:** `YOURNAME/fine-tuning-lora`. **Branch:** `main`.
4. **Main file path:** point it at **`app.py`** (it sits at the repo root).
   Because you pushed the whole repo, the
   `data/` CSVs and `finetune_lab/` package are already there next to `app.py`.
5. Click **Deploy**. It reads `requirements.txt` next to `app.py`, installs the deps,
   and launches. A minute later you have a public `*.streamlit.app` URL.

Either option works and both are free. No API key, no Secrets panel needed for either
— the app never calls out to anything.

---

## Path B (optional) — host a real trained LoRA adapter on the Hugging Face Hub

Only do this if you actually opened `real_finetune/README.md`,
copied `real_lora_finetune.py` into a Colab notebook, picked a free **T4 GPU** runtime,
and ran it end to end on `Qwen2.5-0.5B-Instruct`. That run produces a trained LoRA
adapter — just the small `A`/`B` matrices PEFT learned, typically a few megabytes,
**not** the multi-gigabyte base model.

The natural home for that adapter is the **Hugging Face Hub**, not GitHub — GitHub
repos aren't built for model weights, and the Hub is free, versioned, and exactly what
`transformers`/`peft` expect to load from.

This step happens **inside your Colab notebook**, right after training finishes — you
are not downloading the adapter to your laptop first.

1. Get a Hugging Face account and an access token: <https://huggingface.co/docs/hub/en/models-uploading>
   walks through both — Settings → **Access Tokens** → create one with **write** access.
2. In the Colab notebook, log in once:

   ```python
   from huggingface_hub import login
   login()   # pastes your token, or set HF_TOKEN as a Colab secret
   ```

3. After training, push the adapter straight from the PEFT model object:

   ```python
   model.push_to_hub("YOURNAME/qwen2.5-0.5b-support-tickets-lora")
   tokenizer.push_to_hub("YOURNAME/qwen2.5-0.5b-support-tickets-lora")
   ```

   This is the same `push_to_hub` pattern documented at
   <https://huggingface.co/docs/huggingface_hub/guides/upload>. Because it's a PEFT
   adapter, only the small LoRA weights and an `adapter_config.json` get uploaded —
   not the base model, which anyone can re-download from Hugging Face for free when
   they load your adapter.
4. Your adapter now lives at `https://huggingface.co/YOURNAME/qwen2.5-0.5b-support-tickets-lora`.
   Anyone (including future-you) can load the base model plus your adapter with a few
   lines of `transformers` + `peft` — that link is what you put in your project README
   or hand to an interviewer as proof you fine-tuned a real model, not just the CPU toy.

Never put your Hugging Face token in a committed file. In Colab, use the **Secrets**
tab (key icon in the left sidebar) to store it as `HF_TOKEN` rather than pasting it
into a cell — the same "never hardcode a credential" rule as everywhere else in this
roadmap, just on Colab's side instead of a repo's.

---

## Why this project's hosting story is two-tracked

Every other project in this roadmap has one deployable thing. This one genuinely has
two, because it teaches two different lessons:

- The **comparison app** proves the *idea* — fine-tuning beats a hand-written prompt
  and even beats RAG on this task, and LoRA gets within about a point of full
  fine-tuning while training roughly 13x fewer parameters. That's a CPU-only,
  free-forever demo, and Path A gets it a URL in minutes.
- The **real LoRA adapter** proves you can do this on an *actual* language model, not
  just a hand-written numpy net. That needs a GPU for a few minutes (free, on Colab),
  and the natural place to keep the result isn't a web app at all — it's the model
  registry the rest of the Hugging Face ecosystem already expects, the Hub.

Both are free. Neither needs a card on file, a paid GPU instance, or a secret.

---

## Common Git mistakes (troubleshooting)

**`error: failed to push` / push rejected.** The remote has commits your local repo
doesn't — almost always because you let GitHub add a README or license when creating
the repo. Pull and replay on top, then push:

```powershell
git pull origin main --rebase
git push
```

Next time, create the repo completely empty.

**The Space or Cloud build fails on an import.** Read the build **Logs** bottom-up.
The usual cause is forgetting to upload the whole `finetune_lab/` folder (Path A), so
`from finetune_lab import CLASSES` can't find it. Confirm `app.py`, `finetune_lab/`,
and `requirements.txt` are all at the Space root.

**Colab session disconnects mid-training.** Free Colab GPUs have session limits and
can disconnect on idle. Re-run from the top — the notebook downloads the base model
again (a few minutes) but training itself is only a couple of minutes on this small
dataset, so it's cheap to restart.

**Authentication fails on push.** GitHub no longer accepts your account password in
the terminal. Easiest fix: install the GitHub CLI (<https://cli.github.com>) and run
`gh auth login`, following the browser prompts.
