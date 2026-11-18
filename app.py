"""Factory Reallocation & Shipping Optimization Recommendation System - Nassau Candy Distributor.

Run with:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import optimizer as opt
import style as ui
from config import FACTORIES, PRODUCT_FACTORY, STATE_CENTROIDS
from data_prep import TARGET, load_raw, prepare
from model import train_models

st.set_page_config(page_title="Nassau Candy · Factory Optimizer", page_icon="🍫", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)


# ---------- Data + model (cached) ----------
@st.cache_resource(show_spinner="Preparing data and training lead-time models…")
def load_bundle():
    df, report = prepare(load_raw())
    result = train_models(df)
    late = opt.late_threshold(df, result.best_pipeline)
    return df, report, result, late


@st.cache_data(show_spinner="Simulating factory scenarios…")
def recommendations(regions: tuple, modes: tuple, freight: float, speed_w: float, alert_pct: float, min_conf: float):
    df, _, result, late = load_bundle()
    sub = df[df["Region"].isin(regions) & df["Ship Mode"].isin(modes)]
    if sub.empty:
        return None, None
    return opt.build_recommendations(sub, result.best_pipeline, result.residual_std, late,
                                     freight, speed_w, alert_pct, min_conf)


df, clean_report, model_result, late_days = load_bundle()

# ---------- Sidebar controls ----------
with st.sidebar:
    st.markdown("### 🎛️ Controls")
    product = st.selectbox("Product", sorted(df["Product Name"].unique()))
    regions = st.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    modes = st.multiselect("Ship mode", sorted(df["Ship Mode"].unique()), default=sorted(df["Ship Mode"].unique()))
    priority = st.slider("Optimization priority", 0, 100, 50,
                         help="0 = maximise profit, 100 = minimise lead time")
    st.caption(f"⚖️ {100 - priority}% profit · {priority}% speed")
    top_n = st.slider("Top-N recommendations", 3, 15, 10)

    with st.expander("Assumptions & thresholds"):
        freight = st.number_input(
            "Freight cost ($ per unit per 1,000 mi)", 0.0, 5.0, 0.25, 0.05,
            help="Assumed value: the dataset has no shipping-cost column. Used only to estimate profit impact.")
        alert_pct = st.slider("Profit alert if drop exceeds (%)", 0.5, 10.0, 2.0, 0.5)
        min_conf = st.slider("Minimum scenario confidence", 0, 100, 50)

if not regions or not modes:
    st.warning("Select at least one region and one ship mode in the sidebar.")
    st.stop()

best, scenarios = recommendations(tuple(regions), tuple(modes), freight, priority / 100, alert_pct, min_conf)
if best is None:
    st.warning("No orders match the selected region and ship mode filters.")
    st.stop()

filtered = df[df["Region"].isin(regions) & df["Ship Mode"].isin(modes)]
k = opt.kpis(best, alert_pct)

# ---------- Header + KPIs ----------
st.markdown(ui.hero(
    "Factory Reallocation & Shipping Optimizer",
    f"Nassau Candy Distributor · {len(filtered):,} orders in view · lead-time model: {model_result.best_name}"),
    unsafe_allow_html=True)

st.markdown(ui.kpi_grid([
    ui.kpi_card("Lead Time Reduction", f"{k['lead_time_reduction']:.2f}%",
                "Operational gain · order-weighted, recommended moves", min(100, k["lead_time_reduction"] * 10)),
    ui.kpi_card("Profit Impact Stability", f"{k['profit_stability']:.0f}%",
                f"Financial safety · moves without >{alert_pct:g}% profit loss", k["profit_stability"]),
    ui.kpi_card("Scenario Confidence", f"{k['confidence']:.0f}/100",
                "Reliability · mean over recommended moves", k["confidence"]),
    ui.kpi_card("Recommendation Coverage", f"{k['coverage']:.0f}%",
                f"Scalability · {k['n_rec']} of {k['n_products']} products", k["coverage"]),
]), unsafe_allow_html=True)

tabs = st.tabs(["🏭 Factory Simulator", "🔀 What-If Analysis", "🏆 Recommendations",
                "⚠️ Risk & Impact", "🧭 Routes & Clusters", "🧠 Model & Data"])


def plot(fig, height=380):
    fig.update_layout(template="plotly_dark+glass", height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)")
    # theme=None keeps the glass Plotly template instead of Streamlit's default chart theme.
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


# ---------- 1. Factory Optimization Simulator ----------
with tabs[0]:
    prod_sim = scenarios[scenarios["Product"] == product].sort_values("Predicted Lead Time")
    if prod_sim.empty:
        st.info(f"No orders for **{product}** under the current filters.")
    else:
        current = PRODUCT_FACTORY[product]
        top_alt = prod_sim[~prod_sim["Current"]].sort_values("Score", ascending=False).iloc[0]
        c1, c2 = st.columns([3, 2])
        with c1:
            with st.container(key="glass_sim_chart"):
                st.markdown(f'<div class="section-title">Predicted lead time by factory · {ui.esc(product)}</div>',
                            unsafe_allow_html=True)
                colors = [ui.WARN if c else (ui.GOOD if f == top_alt["Factory"] else ui.ACCENT)
                          for f, c in zip(prod_sim["Factory"], prod_sim["Current"])]
                fig = go.Figure(go.Bar(
                    x=prod_sim["Predicted Lead Time"], y=prod_sim["Factory"], orientation="h",
                    marker=dict(color=colors, line=dict(width=0)),
                    text=[f"{v:,.1f} d" for v in prod_sim["Predicted Lead Time"]], textposition="outside",
                    customdata=prod_sim[["Avg Distance (mi)", "Profit Impact %"]],
                    hovertemplate="%{y}<br>%{x:,.1f} days<br>%{customdata[0]:,.0f} mi avg<br>"
                                  "Profit impact %{customdata[1]:+.2f}%<extra></extra>"))
                lo, hi = prod_sim["Predicted Lead Time"].min(), prod_sim["Predicted Lead Time"].max()
                pad = max(5, (hi - lo) * 0.6)
                fig.update_xaxes(range=[lo - pad, hi + pad], title="Predicted lead time (days)")
                plot(fig, 330)
                st.markdown(ui.chip("Current factory", "warn") + ui.chip("Top alternative", "good"),
                            unsafe_allow_html=True)
        with c2:
            with st.container(key="glass_sim_summary"):
                st.markdown('<div class="section-title">Scenario summary</div>', unsafe_allow_html=True)
                st.metric("Current factory", current)
                st.metric("Best alternative", top_alt["Factory"],
                          f"{-top_alt['Lead Time Reduction %']:+.2f}% lead time", delta_color="inverse")
                st.metric("Profit impact (est.)", ui.money(top_alt["Profit Impact $"]),
                          f"{top_alt['Profit Impact %']:+.2f}%")
                st.metric("Scenario confidence", f"{top_alt['Confidence']:.0f}/100")

        with st.container(key="glass_sim_map"):
            st.markdown('<div class="section-title">Factories and customer demand</div>', unsafe_allow_html=True)
            demand = filtered[filtered["Product Name"] == product].groupby("State/Province").size().reset_index(name="Orders")
            demand["lat"] = demand["State/Province"].map(lambda s: STATE_CENTROIDS[s][0])
            demand["lon"] = demand["State/Province"].map(lambda s: STATE_CENTROIDS[s][1])
            geo = go.Figure()
            geo.add_trace(go.Scattergeo(
                lat=demand["lat"], lon=demand["lon"], text=demand["State/Province"], name="Customer demand",
                marker=dict(size=(demand["Orders"] / demand["Orders"].max() * 30 + 5), color=ui.PALETTE[1],
                            opacity=0.6, line=dict(width=0)),
                hovertemplate="%{text}<br>%{customdata} orders<extra></extra>", customdata=demand["Orders"]))
            for name, (lat, lon) in FACTORIES.items():
                color = ui.WARN if name == current else (ui.GOOD if name == top_alt["Factory"] else ui.ACCENT)
                geo.add_trace(go.Scattergeo(lat=[lat], lon=[lon], name=name, text=[name], mode="markers+text",
                                            textposition="top center", textfont=dict(color="#fff", size=11),
                                            marker=dict(size=14, symbol="diamond", color=color,
                                                        line=dict(width=1, color="#fff"))))
            geo.update_geos(scope="north america", showland=True, landcolor="rgba(255,255,255,0.05)",
                            showcountries=True, countrycolor="rgba(255,255,255,0.2)", showsubunits=True,
                            subunitcolor="rgba(255,255,255,0.1)", bgcolor="rgba(0,0,0,0)",
                            showlakes=False, lataxis_range=[24, 58], lonaxis_range=[-130, -55])
            geo.update_layout(showlegend=False)
            plot(geo, 420)

        with st.expander("All factory scenarios for this product"):
            st.dataframe(prod_sim.drop(columns=["Product", "Score"]).round(2), width="stretch", hide_index=True)


# ---------- 2. What-If Scenario Analysis ----------
with tabs[1]:
    with st.container(key="glass_whatif_compare"):
        st.markdown('<div class="section-title">Current vs recommended assignment</div>', unsafe_allow_html=True)
        cur = scenarios[scenarios["Current"]][["Product", "Predicted Lead Time"]].rename(
            columns={"Predicted Lead Time": "Current"})
        cmp = best[["Product", "Factory", "Predicted Lead Time", "Recommended"]].rename(
            columns={"Predicted Lead Time": "Recommended LT"}).merge(cur, on="Product")
        cmp = cmp[cmp["Recommended"]].sort_values("Current")
        if cmp.empty:
            st.info("No reassignment improves lead time under the current settings.")
        else:
            fig = go.Figure()
            fig.add_bar(y=cmp["Product"], x=cmp["Current"], name="Current factory", orientation="h",
                        marker_color=ui.WARN)
            fig.add_bar(y=cmp["Product"], x=cmp["Recommended LT"], name="Recommended factory", orientation="h",
                        marker_color=ui.GOOD, customdata=cmp["Factory"],
                        hovertemplate="%{y}<br>%{customdata}: %{x:,.1f} days<extra></extra>")
            lo = min(cmp["Current"].min(), cmp["Recommended LT"].min())
            hi = max(cmp["Current"].max(), cmp["Recommended LT"].max())
            fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.08))
            fig.update_xaxes(range=[lo - 20, hi + 10], title="Predicted lead time (days)")
            plot(fig, max(320, 42 * len(cmp)))

    with st.container(key="glass_whatif_custom"):
        st.markdown('<div class="section-title">Build your own scenario</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        wi_product = a.selectbox("Product", sorted(scenarios["Product"].unique()), key="wi_product",
                                 index=sorted(scenarios["Product"].unique()).index(product)
                                 if product in scenarios["Product"].unique() else 0)
        alt_factories = [f for f in FACTORIES if f != PRODUCT_FACTORY[wi_product]]
        wi_factory = b.selectbox("Move to factory", alt_factories, key="wi_factory")
        s = scenarios[(scenarios["Product"] == wi_product)]
        cur_row, new_row = s[s["Current"]].iloc[0], s[s["Factory"] == wi_factory].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Lead time", f"{new_row['Predicted Lead Time']:,.1f} d",
                  f"{new_row['Predicted Lead Time'] - cur_row['Predicted Lead Time']:+.1f} d", delta_color="inverse")
        m2.metric("Avg distance", f"{new_row['Avg Distance (mi)']:,.0f} mi",
                  f"{new_row['Distance Change (mi)']:+,.0f} mi", delta_color="inverse")
        m3.metric("Profit impact", ui.money(new_row["Profit Impact $"]), f"{new_row['Profit Impact %']:+.2f}%")
        m4.metric("Late-order risk", f"{new_row['Late Risk %']:.1f}%",
                  f"{-new_row['Risk Reduction (pp)']:+.1f} pp", delta_color="inverse")
        flags = opt.risk_flags(new_row, alert_pct, min_conf)
        if flags:
            st.markdown("".join(ui.chip(f, "bad") for f in flags), unsafe_allow_html=True)
        else:
            st.markdown(ui.chip("No risk flags", "good"), unsafe_allow_html=True)


# ---------- 3. Recommendation Dashboard ----------
with tabs[2]:
    recs = best[best["Recommended"]].head(top_n)
    st.markdown(f'<div class="section-title">Top {len(recs)} factory reassignments '
                f'<span class="muted">· ranked by {100 - priority}% profit / {priority}% speed</span></div>',
                unsafe_allow_html=True)
    if recs.empty:
        st.info("No reassignment improves lead time under the current settings.")
    cards = []
    for i, (_, r) in enumerate(recs.iterrows(), start=1):
        flag_html = "".join(ui.chip(f, "bad") for f in r["Flags"]) or ui.chip("Low risk", "good")
        cards.append(
            f'<div class="glass"><div class="reco-row"><div class="reco-rank">{i}</div>'
            f'<div class="reco-main"><div class="name">{ui.esc(r["Product"])}</div>'
            f'<div class="muted">{ui.esc(r["Current Factory"])} → <b style="color:#86efac">{ui.esc(r["Factory"])}</b>'
            f' · {int(r["Orders"]):,} orders</div><div>{flag_html}</div></div>'
            f'<div class="reco-stats"><div>Lead time cut<b>{r["Lead Time Reduction %"]:+.2f}%</b></div>'
            f'<div>Profit<b>{r["Profit Impact %"]:+.2f}%</b></div>'
            f'<div>Confidence<b>{r["Confidence"]:.0f}</b></div>'
            f'<div>Score<b>{r["Score"]:.2f}</b></div></div></div></div>')
    st.markdown("".join(cards), unsafe_allow_html=True)

    with st.container(key="glass_reco_table"):
        st.markdown('<div class="section-title">Full ranking (best alternative per product)</div>',
                    unsafe_allow_html=True)
        table = best.assign(Flags=best["Flags"].map(lambda f: "; ".join(f) or "—"))
        cols = ["Rank", "Product", "Current Factory", "Factory", "Lead Time Reduction %", "Risk Reduction (pp)",
                "Profit Impact %", "Profit Impact $", "Confidence", "Score", "Recommended", "Flags"]
        st.dataframe(table[cols].rename(columns={"Factory": "Suggested Factory"}).round(2),
                     width="stretch", hide_index=True)
        st.download_button("⬇️ Download recommendations (CSV)", table[cols].to_csv(index=False).encode(),
                           "factory_recommendations.csv", "text/csv")


# ---------- 4. Risk & Impact Panel ----------
with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">💰 Profit impact alerts</div>', unsafe_allow_html=True)
        drops = scenarios[(~scenarios["Current"]) & (scenarios["Profit Impact %"] < -alert_pct)]
        drops = drops.sort_values("Profit Impact %").head(8)
        if drops.empty:
            st.markdown(ui.alert("No profit alerts", f"No simulated move cuts profit by more than {alert_pct:g}%.",
                                 "good"), unsafe_allow_html=True)
        for _, r in drops.iterrows():
            st.markdown(ui.alert(f"{r['Product']} → {r['Factory']}",
                                 f"Profit {r['Profit Impact %']:+.2f}% ({ui.money(r['Profit Impact $'])}) · "
                                 f"distance {r['Distance Change (mi)']:+,.0f} mi", "bad"), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">🚨 High-risk reassignment warnings</div>', unsafe_allow_html=True)
        risky = best[best["Recommended"] & best["Flags"].map(len).gt(0)]
        if risky.empty:
            st.markdown(ui.alert("No high-risk recommendations", "Every recommended move passed all risk checks.",
                                 "good"), unsafe_allow_html=True)
        for _, r in risky.iterrows():
            st.markdown(ui.alert(f"{r['Product']} → {r['Factory']}", " · ".join(r["Flags"]), "warn"),
                        unsafe_allow_html=True)

    with st.container(key="glass_risk_scatter"):
        st.markdown('<div class="section-title">Speed vs profit trade-off (all simulated moves)</div>',
                    unsafe_allow_html=True)
        alts = scenarios[~scenarios["Current"]].copy()
        fig = px.scatter(alts, x="Lead Time Reduction %", y="Profit Impact %", size="Orders", color="Confidence",
                         hover_name="Product", hover_data={"Factory": True, "Orders": True},
                         color_continuous_scale=["#fca5a5", "#fcd34d", "#86efac"], range_color=[0, 100],
                         size_max=40)
        fig.add_hline(y=-alert_pct, line_dash="dot", line_color=ui.BAD,
                      annotation_text=f"profit alert (−{alert_pct:g}%)")
        fig.add_vline(x=0, line_color="rgba(255,255,255,.3)")
        plot(fig, 420)


# ---------- 5. Route & product clustering ----------
with tabs[4]:
    routes = opt.cluster_routes(filtered)
    combos = opt.congested_combos(filtered)
    c1, c2 = st.columns([3, 2])
    with c1:
        with st.container(key="glass_routes"):
            st.markdown('<div class="section-title">Factory → state routes clustered by performance</div>',
                        unsafe_allow_html=True)
            fig = px.scatter(routes, x="Avg Distance (mi)", y="Avg Lead Time", size="Orders", color="Cluster",
                             hover_name="State/Province", hover_data={"Factory": True, "Lead Time Std": ":.1f"},
                             color_discrete_map={"Fast": ui.GOOD, "Moderate": ui.ACCENT, "Slow": ui.BAD},
                             size_max=35)
            plot(fig, 400)
    with c2:
        with st.container(key="glass_slow_routes"):
            st.markdown('<div class="section-title">Consistently slow routes</div>', unsafe_allow_html=True)
            slow = routes[routes["Cluster"] == "Slow"].sort_values("Avg Lead Time", ascending=False)
            st.dataframe(slow[["Factory", "State/Province", "Orders", "Avg Lead Time", "Lead Time Std"]].round(1),
                         width="stretch", hide_index=True, height=340)

    with st.container(key="glass_congestion"):
        st.markdown('<div class="section-title">Region × product lead-time heatmap</div>', unsafe_allow_html=True)
        heat = combos.pivot(index="Product Name", columns="Region", values="Avg Lead Time")
        fig = px.imshow(heat, color_continuous_scale=["#1e3a8a", "#7c3aed", "#f472b6"], aspect="auto",
                        labels=dict(color="Avg lead time (d)"), text_auto=".0f")
        plot(fig, 520)
        congested = combos[combos["Congested"]]
        if congested.empty:
            st.markdown(ui.chip("No congested combinations", "good"), unsafe_allow_html=True)
        else:
            st.markdown("**Congested (slow and high-volume):** " + "".join(
                ui.chip(f"{r['Region']} · {r['Product Name']} ({r['Orders']} orders)", "warn")
                for _, r in congested.iterrows()),
                unsafe_allow_html=True)


# ---------- 6. Model & data ----------
with tabs[5]:
    c1, c2 = st.columns(2)
    with c1:
        with st.container(key="glass_model"):
            st.markdown('<div class="section-title">Lead-time model evaluation (20% hold-out)</div>',
                        unsafe_allow_html=True)
            st.dataframe(model_result.metrics.round(3), width="stretch", hide_index=True)
            st.markdown(f'<p class="muted">Selected: <b>{ui.esc(model_result.best_name)}</b>, the lowest RMSE '
                        'unless a more interpretable model is within 1%. Features: product, region, ship mode, division, '
                        'factory, country, factory-to-customer distance, units, unit price, order year and month '
                        '(scaled and one-hot encoded).</p>', unsafe_allow_html=True)
    with c2:
        with st.container(key="glass_data"):
            st.markdown('<div class="section-title">Data preparation</div>', unsafe_allow_html=True)
            st.markdown(
                f"- Raw rows: **{clean_report['raw_rows']:,}**\n"
                f"- Extreme outliers removed (3×IQR on lead time, sales, units): **{clean_report['outliers_removed']:,}**\n"
                f"- Rows used: **{clean_report['clean_rows']:,}**\n"
                f"- Lead time = Ship Date − Order Date (days). Mean **{df[TARGET].mean():,.0f}**, "
                f"range {df[TARGET].min():,}–{df[TARGET].max():,}")

    st.markdown(ui.alert(
        "Assumptions to review",
        "1) The CSV has no factory column; the product→factory mapping in config.py follows the public Nassau "
        "Candy product table. 2) Customer location = state/province centroid (no coordinates in the data). "
        "3) Profit impact uses the freight-rate assumption set in the sidebar. 4) Lead times in this dataset span "
        "~900–1,640 days and are driven mainly by order timing, so predicted factory-switch effects are small. "
        "5) Confidence is a statistical proxy: how distinguishable the predicted change is from model error.",
        "warn"), unsafe_allow_html=True)

    with st.container(key="glass_reports"):
        st.markdown('<div class="section-title">📄 Project deliverables</div>', unsafe_allow_html=True)
        reports_dir = Path(__file__).parent / "reports"
        docs = [("Research paper (EDA, insights, recommendations)", "research_paper.pdf"),
                ("Executive summary for government stakeholders", "executive_summary.pdf")]
        cols = st.columns(len(docs))
        for col, (label, fname) in zip(cols, docs):
            path = reports_dir / fname
            if path.exists():
                col.download_button(f"⬇️ {label}", path.read_bytes(), fname, "application/pdf", key=f"dl_{fname}")
            else:
                col.caption(f"{label}: run `python reports/build_reports.py` to generate.")
