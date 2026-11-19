import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import dataset, dataset_choice, page, shares

page("Retail Simulator", "🏪")
ds = dataset_choice()
d = dataset(ds)
sh = shares(ds)
cat = pd.DataFrame([p.to_dict() for p in d.catalogue])
cat["share of sales"] = sh

tab1, tab2, tab3, tab4 = st.tabs(["Catalogue", "Baskets", "Co-occurrence graph", "Build a basket"])
with tab1:
    c1, c2 = st.columns([2, 1])
    pick = c1.multiselect("Categories", sorted(cat.category.unique()))
    q = c2.text_input("Search product")
    view = cat[cat.category.isin(pick)] if pick else cat
    if q:
        view = view[view.name.str.contains(q, case=False)]
    st.dataframe(view[["sku", "name", "category", "base", "brand", "variant", "price", "tier", "share of sales"]]
                 .sort_values("share of sales", ascending=False), hide_index=True, width="stretch", height=320)
    a, b = st.columns(2)
    by_cat = cat.groupby("category").agg(products=("pid", "count"), share=("share of sales", "sum")).reset_index()
    a.plotly_chart(px.bar(by_cat.sort_values("share"), x="share", y="category", orientation="h",
                          text="products", title="Share of sales by category (label: number of products)")
                   .update_layout(height=380, xaxis_tickformat=".0%"), width="stretch")
    srt = np.sort(sh)[::-1]
    b.plotly_chart(go.Figure(go.Scatter(x=np.arange(1, len(srt) + 1) / len(srt), y=np.cumsum(srt), mode="lines"))
                   .update_layout(title="Concentration: cumulative share of sales", xaxis_title="share of products (best sellers first)",
                                  yaxis_title="share of sales", height=380, xaxis_tickformat=".0%", yaxis_tickformat=".0%"),
                   width="stretch")

with tab2:
    sizes = np.array([len(b) for b in d.baskets])
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(x=sizes, nbins=int(sizes.max()), title="Basket size (distinct products)")
                   .update_layout(height=320, xaxis_title="products in basket", yaxis_title="baskets"), width="stretch")
    counts = np.bincount(np.concatenate([np.array(x) for x in d.baskets]), minlength=d.n_items)
    top = np.argsort(-counts)[:20]
    b.plotly_chart(px.bar(x=counts[top] / len(d.baskets), y=[d.catalogue[i].name for i in top], orientation="h",
                          title="Most frequent products (support)").update_layout(height=320, xaxis_tickformat=".1%",
                                                                                  yaxis_autorange="reversed"), width="stretch")
    k = st.slider("Show baskets starting at #", 0, len(d.baskets) - 10, 0)
    st.dataframe(pd.DataFrame([{"basket": i, "size": len(d.baskets[i]), "products": " ; ".join(d.names(d.baskets[i]))}
                               for i in range(k, k + 10)]), hide_index=True, width="stretch")

with tab3:
    from basketmind.recommenders.cooccurrence import CoOccurrence
    st.caption("Edges: product pairs with the highest lift among pairs bought together at least `min count` times. "
               "Thick edges = high lift. Nodes coloured by category.")
    mc = st.slider("min count", 20, 400, 100, 10)
    n_edges = st.slider("edges", 20, 120, 60, 10)

    @st.cache_data(show_spinner="Counting pairs…")
    def pairs(name, mc, n_edges):
        co = CoOccurrence().fit(dataset(name).baskets, dataset(name).n_items)
        C, n = co.C, co.n_baskets
        p = co.item_count / n
        iu = np.triu_indices_from(C, 1)
        cnt = C[iu]
        ok = cnt >= mc
        lift = (cnt[ok] / n) / (p[iu[0][ok]] * p[iu[1][ok]])
        order = np.argsort(-lift)[:n_edges]
        return [(int(iu[0][ok][j]), int(iu[1][ok][j]), float(lift[j]), int(cnt[ok][j])) for j in order]

    edges = pairs(ds, mc, n_edges)
    nodes = sorted({x for e in edges for x in e[:2]})
    cats = sorted({d.catalogue[i].category for i in nodes})
    pos = {}
    for ci, c in enumerate(cats):
        members = [i for i in nodes if d.catalogue[i].category == c]
        for j, i in enumerate(members):
            ang = 2 * np.pi * (ci + (j + 1) / (len(members) + 1)) / len(cats)
            pos[i] = (np.cos(ang), np.sin(ang))
    fig = go.Figure()
    for a_, b_, lift, cnt in edges:
        fig.add_trace(go.Scatter(x=[pos[a_][0], pos[b_][0]], y=[pos[a_][1], pos[b_][1]], mode="lines",
                                 line=dict(width=0.5 + min(lift, 60) / 10, color="rgba(70,110,160,.45)"),
                                 hoverinfo="text", text=f"lift {lift:.1f}, {cnt} baskets", showlegend=False))
    for c in cats:
        members = [i for i in nodes if d.catalogue[i].category == c]
        fig.add_trace(go.Scatter(x=[pos[i][0] for i in members], y=[pos[i][1] for i in members], mode="markers",
                                 marker=dict(size=11), name=c, text=[d.catalogue[i].name for i in members], hoverinfo="text"))
    fig.update_layout(height=620, xaxis_visible=False, yaxis_visible=False, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

with tab4:
    names = [p.name for p in d.catalogue]
    chosen = st.multiselect("Put products in a basket", names, default=[names[0]] if names else [])
    if chosen:
        ids = [names.index(x) for x in chosen]
        st.code(f"<basket> {' ; '.join(chosen)} </basket>\n<basket> {' '.join(f'<p{i:04d}>' for i in ids)} </basket>")
        holders = sum(set(ids) <= set(b) for b in d.baskets)
        st.metric("Baskets in the data that contain all of these", f"{holders:,}", f"support {holders / len(d.baskets):.3%}")
        st.caption("Take this basket to the Cross-Sell Studio to see what every recommender would add.")
