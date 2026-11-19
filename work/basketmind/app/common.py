"""Shared helpers for the BasketMind dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import basketmind  # noqa: E402,F401  (environment fix-ups before transformers)

ART = ROOT / "artifacts"
MODEL_LABELS = {"popularity": "Popularity", "cooccurrence-count": "Co-occurrence (count)",
                "cooccurrence-confidence": "Co-occurrence (confidence)", "cooccurrence-lift": "Co-occurrence (lift)",
                "cooccurrence-blend": "Co-occurrence (blend)", "association-rules": "Association rules",
                "llm-offshelf": "GPT-2 off-the-shelf", "llm-cpt-fresh": "GPT-2 + continued pretraining",
                "llm-abl-fixed": "Ablation: fixed order", "llm-abl-frozen": "Ablation: frozen order",
                "llm-abl-fresh": "Ablation: fresh order"}
COLORS = {"Popularity": "#9aa3ad", "Co-occurrence (confidence)": "#e69138", "Association rules": "#cc4125",
          "GPT-2 off-the-shelf": "#8e7cc3", "GPT-2 + continued pretraining": "#3d85c6"}


def page(title: str, icon: str = "🛒") -> None:
    st.set_page_config(page_title=f"BasketMind · {title}", page_icon=icon, layout="wide")
    st.markdown("""<style>
      .block-container {padding-top: 1.5rem; max-width: 1280px;}
      div[data-testid="stMetricValue"] {font-size: 1.55rem;}
      .bm-kicker {letter-spacing: .12em; font-size: .75rem; color: #6b7a90; font-weight: 600;}
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="bm-kicker">BASKETMIND · RETAIL INTELLIGENCE & LLM RECOMMENDATION LABORATORY</div>',
                unsafe_allow_html=True)
    st.title(title)


def dataset_choice() -> str:
    options = [d for d in ("full", "dev") if (ROOT / "data" / d / "public").exists()]
    if not options:
        st.warning("No dataset yet. Run `python -m basketmind.experiments.run generate --dataset full` in `work/basketmind`.")
        st.stop()
    return st.sidebar.selectbox("Dataset", options, help="full = 100,000 baskets; dev = 10,000")


@st.cache_resource(show_spinner="Loading baskets…")
def dataset(name: str):
    from basketmind.data.dataset import load
    return load(name)


@st.cache_resource(show_spinner=False)
def truth(name: str):
    from basketmind.data.dataset import load_truth
    return load_truth(name)


@st.cache_data(show_spinner=False)
def shares(name: str) -> np.ndarray:
    rows = json.loads((ROOT / "data" / name / "public" / "catalogue.json").read_text())
    return np.array([r["share"] for r in rows])


def artifact(ds: str, rel: str):
    p = ART / ds / rel
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource(show_spinner="Fitting the classical recommenders on the training baskets…")
def classical(name: str):
    from basketmind.recommenders.cooccurrence import AssociationRules, CoOccurrence
    from basketmind.recommenders.popularity import Popularity
    d = dataset(name)
    tr = d.split("train")
    return {"Popularity": Popularity().fit(tr, d.n_items),
            "Co-occurrence (confidence)": CoOccurrence("confidence").fit(tr, d.n_items),
            "Association rules": AssociationRules(min_support=0.001).fit(tr, d.n_items)}


def lm_runs(ds: str) -> list:
    runs = ART / ds / "runs"
    done = [p.name for p in sorted(runs.glob("*")) if (p / "adapter.pt").exists()] if runs.exists() else []
    return ["offshelf"] + done


@st.cache_resource(show_spinner="Loading GPT-2 and the adapter (about 20 s the first time)…")
def lm(ds: str, run: str):
    import torch
    from basketmind.language_model.model import BasketLM
    torch.set_num_threads(4)
    d = dataset(ds)
    if run == "offshelf":
        return BasketLM.from_pretrained_gpt2(d.catalogue)
    return BasketLM.load(ART / ds / "runs" / run, d.catalogue)


def label(run_or_model: str) -> str:
    return MODEL_LABELS.get(run_or_model, MODEL_LABELS.get(f"llm-{run_or_model}", run_or_model))


def ci_text(c: dict, pct: bool = False) -> str:
    f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.3f}")
    return f"{f(c['estimate'])} [{f(c['lo'])}, {f(c['hi'])}]"
