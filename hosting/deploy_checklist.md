# Deploy checklist — Fine-Tuning / LoRA

This is the project's "definition of done." Walk it top to bottom. Don't tick a box
you haven't actually verified by running the command — "should work" isn't the same
as "works." Commands assume you're inside `build_from_scratch/` unless noted.

## Runs locally

- [ ] Fresh virtual environment, dependencies install cleanly:
      `python -m venv .venv ; .\.venv\Scripts\Activate.ps1` then
      `pip install -r requirements.txt`
- [ ] The sample data generates without error:
      `python generate_data.py` (look for `data\train.csv`, `data\val.csv`,
      `data\test.csv` written)
- [ ] The CLI prints the comparison table:
      `python -m finetune_lab compare`
      (expect roughly: prompt 0.550, RAG 0.808, base 0.708, full fine-tune 0.983,
      LoRA (rank=4) 0.967 — LoRA within about two points of full fine-tune while
      training ~12.9x fewer parameters)
- [ ] The CLI classifies a single ticket:
      `python -m finetune_lab predict "i was overcharged on my last invoice"`
      (expect `billing`)
- [ ] The comparison table matches the CANON numbers above (params: base 256,
      full 33024, LoRA 2560 at rank=4) — if params differ, something in the seed or
      rank changed; re-check the command you ran.
- [ ] The web app launches and works offline:
      `streamlit run app.py` — move the sliders, watch the comparison table and bar
      chart render, then type a ticket into "Try your own ticket" and see a
      prediction. No API key, no GPU, nothing to wait on.

## Tests pass

- [ ] `pytest` run from inside `build_from_scratch/` is all green — 27 tests, all
      offline, all CPU.
- [ ] You ran it in the fresh venv, not just your everyday one, so you know the deps
      in `requirements.txt` are complete.

## No secrets to worry about (and confirm that's still true)

- [ ] `build_from_scratch/` makes zero LLM/API calls — there should be no `.env`,
      no API key, nothing hardcoded. Confirm there isn't one hiding:
      `git ls-files | Select-String ".env"` should print nothing.
- [ ] If you ran `real_finetune/` on Colab, confirm you did **not** paste your
      Hugging Face token into a committed cell or file — it should only ever live in
      Colab's Secrets panel (or be typed interactively via `login()`).

## README is recruiter-ready

- [ ] Root `README.md` covers: what you built, why it matters, the comparison table,
      the folder map, the path to follow, and copy-pasteable quickstart commands.
- [ ] The CI status badge is at the top.
- [ ] The **live app URL** (from the Space or Streamlit Cloud) is added near the top,
      so a recruiter can click straight through to a working demo.
- [ ] (Optional) If you pushed a real LoRA adapter to the Hugging Face Hub, its URL
      is linked in the README too — that's the proof you fine-tuned a real model, not
      just the CPU toy.

## Pushed to GitHub with CI

- [ ] Repo created empty on github.com (no auto README/license), named
      `fine-tuning-lora`, public.
- [ ] `git init` → `git add .` → `git commit` → `git branch -M main` →
      `git remote add origin ...` → `git push -u origin main` all done, from the
      **project root** (the folder containing `build_from_scratch/`).
- [ ] The small CSVs under `data/` and `build_from_scratch/data/` are committed
      (that's on purpose — deterministic, tiny, and it means the app and notebooks
      work the moment someone clones).
- [ ] `.github/workflows/ci.yml` is committed and pushed.
- [ ] The Actions tab shows a completed run with a green checkmark — it's keyless,
      all 27 tests are offline and CPU-only, so no secrets or GPU runner are needed.
      If it was red, you read the log and fixed the cause (usually a missing dep in
      `requirements.txt`), then re-ran to green.

## Live app is up (Path A)

- [ ] Deployed via Hugging Face Spaces (Streamlit SDK) **or** Streamlit Community
      Cloud (main file `build_from_scratch/app.py`).
- [ ] For Spaces: `app.py`, the whole `finetune_lab/` folder, `requirements.txt`, and
      the `data/` CSVs are at the Space root; the build log is clean.
- [ ] Open the live URL — the comparison table and bar chart render, the sliders
      work, and typing a ticket into "Try your own ticket" returns a category. No
      secret required; nothing to wait on.

## (Optional) LoRA adapter pushed to Hugging Face Hub (Path B)

- [ ] Only applies if you ran `real_finetune/real_lora_finetune.py` on a free
      Colab/Kaggle T4 GPU end to end.
- [ ] `model.push_to_hub(...)` and `tokenizer.push_to_hub(...)` ran without error
      inside the Colab notebook.
- [ ] The adapter's Hugging Face page loads and shows an `adapter_config.json` plus
      the small LoRA weight file(s) — not a multi-gigabyte base model re-upload.
- [ ] The Hub token never appears in a committed file — Colab Secrets or an
      interactive `login()` prompt only.

When every box is ticked, the project is done and presentable. Send the repo link and
the live app link together — and if you did Path B, the Hub adapter link too. The
working demo (fine-tuning genuinely beating prompt and RAG baselines) is what gets you
the follow-up conversation.
