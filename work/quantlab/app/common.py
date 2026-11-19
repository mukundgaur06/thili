"""Shared helpers for the QuantLab dashboard pages."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import quantlab  # noqa: E402,F401

ART = ROOT / "artifacts"
MODELS = ART / "models"
ORDER = ["fp32", "fp16", "dynamic_int8", "static_int8", "qat_int8"]
COLORS = {"fp32": "#5b6b7f", "fp16": "#8e7cc3", "dynamic_int8": "#e69138", "static_int8": "#3d85c6", "qat_int8": "#2e9e6b"}
LABELS = {"fp32": "FP32", "fp16": "FP16", "dynamic_int8": "Dynamic INT8", "static_int8": "Static INT8", "qat_int8": "QAT INT8"}


def page(title: str, icon: str = "⚙️") -> None:
    st.set_page_config(page_title=f"QuantLab · {title}", page_icon=icon, layout="wide")
    st.markdown("""<style>
      .block-container {padding-top: 1.6rem; max-width: 1250px;}
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
      .ql-kicker {letter-spacing: .12em; font-size: .75rem; color: #6b7a90; font-weight: 600;}
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="ql-kicker">QUANTLAB · NEURAL NETWORK OPTIMIZATION STUDIO</div>', unsafe_allow_html=True)
    st.title(title)


@st.cache_data(show_spinner=False)
def _read(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


def results() -> dict | None:
    p = ART / "results.json"
    if not p.exists():
        return None
    return _read(str(p), p.stat().st_mtime)


def manifest() -> dict | None:
    p = ART / "export_manifest.json"
    return _read(str(p), p.stat().st_mtime) if p.exists() else None


def need_results() -> dict:
    r = results()
    if r is None:
        st.warning("No measured results yet. Run `python -m quantlab.pipeline` from `work/quantlab` first "
                   "(about 5 minutes on a laptop CPU).")
        st.stop()
    return r


def variants(r: dict, only_ok: bool = True) -> list:
    rows = [v for v in r["variants"] if v["status"] == "ok" or not only_ok]
    return sorted(rows, key=lambda v: ORDER.index(v["name"]) if v["name"] in ORDER else 99)


@st.cache_resource(show_spinner="Loading Fashion-MNIST…")
def dataset(name: str = "fashion_mnist"):
    from quantlab import data
    return data.load(name)
