import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import MODELS, dataset, page, results

page("Model Training", "🏋️")
r = results()

if r:
    st.subheader("The baseline used in every comparison")
    h = pd.DataFrame(r["training"]["history"])
    c1, c2 = st.columns(2)
    fig = go.Figure([go.Scatter(x=h.epoch, y=h.train_loss, name="train loss", mode="lines+markers"),
                     go.Scatter(x=h.epoch, y=h.val_loss, name="validation loss", mode="lines+markers")])
    fig.update_layout(height=300, margin=dict(t=10, b=10), xaxis_title="epoch", yaxis_title="cross-entropy")
    c1.plotly_chart(fig, width="stretch")
    fig = go.Figure([go.Scatter(x=h.epoch, y=h.train_accuracy, name="train accuracy", mode="lines+markers"),
                     go.Scatter(x=h.epoch, y=h.val_accuracy, name="validation accuracy", mode="lines+markers")])
    fig.update_layout(height=300, margin=dict(t=10, b=10), xaxis_title="epoch", yaxis_tickformat=".0%")
    c2.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(r["model"]["layers"]), hide_index=True, width="stretch")
    st.caption(f"{r['model']['parameters']:,} parameters · trained on {r['split_sizes']['train']:,} images, "
               f"validated on {r['split_sizes']['val']:,} held out from the official training split; "
               f"the 10,000 test images are never used for training or model selection.")

st.divider()
st.subheader("Train your own")
with st.form("train"):
    a, b, c = st.columns(3)
    ds_name = a.selectbox("Dataset", ["fashion_mnist", "mnist"], help="MNIST is downloaded on first use (~11 MB).")
    c1 = a.select_slider("Conv1 channels", [4, 8, 16, 32], 16)
    c2 = b.select_slider("Conv2 channels", [8, 16, 32, 64], 32)
    hidden = b.select_slider("Hidden units", [32, 64, 128, 256], 128)
    epochs = c.slider("Epochs", 1, 10, 2)
    batch = c.select_slider("Batch size", [32, 64, 128, 256], 128)
    lr = a.select_slider("Learning rate", [5e-4, 1e-3, 2e-3, 5e-3], 2e-3)
    go_train = st.form_submit_button("Train", type="primary")

if go_train:
    from quantlab.model import ModelConfig, QuantCNN
    from quantlab.train import TrainConfig, save_checkpoint, train_epochs
    import torch

    d = dataset(ds_name)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    model = QuantCNN(ModelConfig(channels=[c1, c2], hidden=hidden))
    bar = st.progress(0.0, "starting…")
    chart = st.empty()
    rows = []

    def on_batch(step, total):
        if step % 20 == 0 or step == total:
            bar.progress(step / total, f"step {step}/{total}")

    def on_epoch(row):
        rows.append(row)
        df = pd.DataFrame(rows)
        f = go.Figure([go.Scatter(x=df.epoch, y=df.train_loss, name="train loss"),
                       go.Scatter(x=df.epoch, y=df.val_loss, name="val loss"),
                       go.Scatter(x=df.epoch, y=df.val_accuracy, name="val accuracy", yaxis="y2")])
        f.update_layout(height=300, yaxis2=dict(overlaying="y", side="right", tickformat=".0%"), margin=dict(t=10))
        chart.plotly_chart(f, width="stretch")

    t = time.time()
    hist = train_epochs(model, d, epochs, lr, batch, on_batch=on_batch, on_epoch=on_epoch)
    name = f"custom_{ds_name}_{c1}-{c2}-{hidden}_e{epochs}.pt"
    save_checkpoint(MODELS / name, model.eval(), hist, TrainConfig(epochs=epochs, batch_size=batch, lr=lr), ds_name)
    st.success(f"Trained in {time.time() - t:.0f} s · validation accuracy {hist[-1]['val_accuracy']:.2%} · "
               f"saved as `artifacts/models/{name}` (pick it in the Quantization Lab).")
