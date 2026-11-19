import json

import pandas as pd
import streamlit as st

from common import ART, LABELS, ROOT, manifest, page

page("Model Export", "📦")
man = manifest()
if not man:
    st.warning("Nothing exported yet: run `python -m quantlab.pipeline`.")
    st.stop()

st.markdown("Every export below was **reloaded and checked** after it was written: TorchScript files against the "
            "in-memory model on the same images, ONNX files with ONNX Runtime on all 10,000 test images. Only files "
            "that passed get a download button.")
rows = []
for name, entries in man["exports"].items():
    for e in entries:
        rows.append({"model": LABELS.get(name, name), "format": e["format"], "passed": "yes" if e.get("ok") else "no",
                     "size (KB)": round(e["bytes"] / 1024, 1) if e.get("bytes") else None,
                     "max |Δ logit|": e.get("max_abs_diff"),
                     "prediction agreement": e.get("prediction_agreement", 1.0 if e.get("same_predictions") else None),
                     "note": e.get("error") or e.get("runtime", "")})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("INT8 ONNX graphs run on ONNX Runtime's own integer kernels, which can round a value one quantization step "
           "differently from PyTorch's; they pass if ≥ 99.5% of predictions agree and no logit differs by more than "
           "two output steps.")

st.subheader("Download")
for name, entries in man["exports"].items():
    for e in entries:
        p = ROOT / e["path"]
        if e.get("ok") and p.exists():
            st.download_button(f"{LABELS.get(name, name)} · {e['format']} ({p.stat().st_size / 1024:.0f} KB)",
                               p.read_bytes(), file_name=p.name, key=f"{name}-{e['format']}")
meta = {"dataset": man["meta"]["dataset"], "torch": man["meta"]["torch"],
        "input": {"shape": [None, 1, 28, 28], "dtype": "float32", "normalisation": "(pixel/255 - 0.2860) / 0.3530"},
        "classes": json.loads((ART / "results.json").read_text())["classes"] if (ART / "results.json").exists() else None}
st.download_button("Model metadata (JSON)", json.dumps(meta, indent=1), file_name="quantlab_metadata.json")
st.json(meta)
st.info("Not offered: ONNX for dynamic INT8 (dynamically quantized Linear layers have no standard ONNX export path "
        "in this PyTorch version) and FP16 ONNX (not exported or tested).")
