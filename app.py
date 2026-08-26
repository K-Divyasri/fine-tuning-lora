"""Streamlit demo: watch fine-tuning beat the baselines, live in the browser.

    streamlit run app.py

Trains all five methods (prompt, RAG, base, full fine-tune, LoRA) on the synthetic
ticket dataset right in the app -- it takes well under a second, because the whole
"model" is a small numpy network -- and shows the comparison table plus a live
"try your own ticket" box using the fully fine-tuned model.

Runs fully offline: no API key, no GPU, no downloads. That's also why it's free and
instant to host -- see hosting/HOSTING_GUIDE.md for the real GPU fine-tune path.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from finetune_lab import CLASSES
from finetune_lab.dataset import make_splits
from finetune_lab.evaluate import compare_all
from finetune_lab.features import HashedVectorizer
from finetune_lab.model import TwoLayerNet
from finetune_lab.train import fit_full

st.set_page_config(page_title="Fine-Tuning vs RAG vs Prompting", page_icon="🎛️", layout="wide")
st.title("🎛️ Fine-Tuning vs RAG vs Prompting")
st.caption("A small classifier, trained three different ways, compared against a "
          "hand-written prompt and a retrieval (RAG) baseline -- all on the same "
          "held-out test set. Runs instantly, no GPU, no key.")

with st.sidebar:
    st.header("Settings")
    n_train = st.slider("Training examples per class", 10, 150, 80, step=10)
    lora_rank = st.slider("LoRA rank", 1, 16, 4)
    seed = st.number_input("Random seed", value=7, step=1)

train, _val, test = make_splits(n_train=n_train, seed=int(seed))
results = compare_all(train, test, lora_rank=lora_rank, seed=int(seed))

st.subheader("Comparison on the held-out test set")
df = pd.DataFrame([
    {"method": r.name, "test accuracy": round(r.test_accuracy, 3),
     "trainable params": r.trainable_params}
    for r in results
])
st.dataframe(df, use_container_width=True)
st.bar_chart(df.set_index("method")["test accuracy"])

full_params = next(r.trainable_params for r in results if r.name.startswith("full"))
lora_params = next(r.trainable_params for r in results if r.name.startswith("LoRA"))
st.caption(f"LoRA trains {full_params / lora_params:.1f}x fewer parameters than full "
          "fine-tuning for a similar test accuracy.")

st.divider()
st.subheader("Try your own ticket")
st.caption("Classified by a model fully fine-tuned on the training split above.")

vectorizer = HashedVectorizer()
net = TwoLayerNet(dim=vectorizer.dim, seed=int(seed))
fit_full(net, vectorizer.transform(train.texts), train.labels)

text = st.text_input("Ticket text:", placeholder="e.g. my card was charged twice, please refund")
if text:
    pred = net.predict(vectorizer.transform([text]))[0]
    st.markdown(f"### Predicted category: **{CLASSES[pred]}**")
