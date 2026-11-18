"""Generate the research paper and executive summary (HTML + PDF) from the live data and model.

Every number in the documents is computed here from data/Nassau Candy Distributor.csv,
so re-running this script after the data changes keeps the reports consistent.

Run from the project root:  python reports/build_reports.py
"""

import html
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optimizer as opt  # noqa: E402
from config import FACTORIES, PRODUCT_FACTORY  # noqa: E402
from data_prep import TARGET, load_raw, prepare  # noqa: E402
from model import train_models  # noqa: E402

OUT = Path(__file__).resolve().parent
# Default settings, identical to the dashboard's defaults.
SETTINGS = {"freight": 0.25, "speed_weight": 0.5, "alert_pct": 2.0, "min_conf": 50.0}
COLORS = ["#4f46e5", "#0ea5e9", "#db2777", "#16a34a", "#f59e0b", "#7c3aed"]


# ---------------------------------------------------------------- analysis
def analyse() -> dict:
    df, clean = prepare(load_raw())
    model = train_models(df)
    late = opt.late_threshold(df, model.best_pipeline)
    best, scenarios = opt.build_recommendations(
        df, model.best_pipeline, model.residual_std, late, SETTINGS["freight"],
        SETTINGS["speed_weight"], SETTINGS["alert_pct"], SETTINGS["min_conf"])
    kpi = opt.kpis(best, SETTINGS["alert_pct"])
    routes = opt.cluster_routes(df)
    combos = opt.congested_combos(df)

    by_year = df.groupby(df["Order Date"].dt.year)[TARGET].agg(["mean", "size"])
    by_mode = df.groupby("Ship Mode")[TARGET].agg(["mean", "median", "size"]).sort_values("mean")
    by_region = df.groupby("Region").agg(Orders=(TARGET, "size"), Sales=("Sales", "sum"),
                                         Distance=("Distance (mi)", "mean"), LeadTime=(TARGET, "mean"))
    by_factory = df.groupby("Factory").agg(Orders=(TARGET, "size"), Sales=("Sales", "sum"),
                                           Profit=("Gross Profit", "sum"), Distance=("Distance (mi)", "mean"),
                                           LeadTime=(TARGET, "mean"))
    by_product = df.groupby("Product Name").agg(Orders=(TARGET, "size"), Sales=("Sales", "sum"),
                                                Profit=("Gross Profit", "sum")).sort_values("Sales", ascending=False)
    by_product["Margin %"] = by_product["Profit"] / by_product["Sales"] * 100

    rec = best[best["Recommended"]]
    confident = rec[rec["Confidence"] >= SETTINGS["min_conf"]]
    target_counts = rec["Factory"].value_counts()
    return {
        "df": df, "clean": clean, "model": model, "best": best, "scenarios": scenarios, "kpi": kpi,
        "routes": routes, "combos": combos, "by_year": by_year, "by_mode": by_mode,
        "by_region": by_region, "by_factory": by_factory, "by_product": by_product,
        "rec": rec, "confident": confident, "target_counts": target_counts,
        "corr_dist": float(df[["Distance (mi)", TARGET]].corr().iloc[0, 1]),
        "late": late,
    }


# ---------------------------------------------------------------- formatting helpers
def esc(x) -> str:
    return html.escape(str(x))


def money(v: float) -> str:
    return f"{'-' if v < 0 else ''}${abs(v):,.0f}"


def table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = "".join(f"<td>{esc(fmt[c](r[c]) if c in fmt else r[c])}</td>" for c in df.columns)
        rows.append(f"<tr>{cells}</tr>")
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


_first_chart = True


def chart(fig: go.Figure, caption: str, height: int = 340) -> str:
    """Embed a Plotly figure; plotly.js is inlined once so the file works offline."""
    global _first_chart
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=30, b=10),
                      font=dict(family="Inter, Segoe UI, sans-serif", size=12), colorway=COLORS)
    div = fig.to_html(full_html=False, include_plotlyjs="inline" if _first_chart else False,
                      config={"displayModeBar": False, "staticPlot": True})
    _first_chart = False
    return f'<figure>{div}<figcaption>{esc(caption)}</figcaption></figure>'


CSS = """
<style>
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'Segoe UI', Inter, system-ui, sans-serif; color: #1e1b4b; line-height: 1.6; max-width: 900px;
       margin: 0 auto; padding: 32px 20px; background: #fff; }
h1 { font-size: 1.9rem; line-height: 1.2; margin: 0 0 6px; color: #312e81; }
h2 { font-size: 1.3rem; margin-top: 2.2rem; color: #312e81; border-bottom: 2px solid #e0e7ff; padding-bottom: 4px; }
h3 { font-size: 1.05rem; margin-top: 1.4rem; color: #3730a3; }
.meta { color: #6366f1; font-size: .92rem; margin-bottom: 18px; }
.abstract, .callout { background: #eef2ff; border-left: 4px solid #6366f1; padding: 12px 16px; border-radius: 8px; }
.callout.warn { background: #fffbeb; border-left-color: #f59e0b; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; margin: 10px 0 16px; }
tr { break-inside: avoid; }
th { background: #eef2ff; text-align: left; padding: 6px 8px; color: #312e81; }
td { border-bottom: 1px solid #e5e7eb; padding: 5px 8px; font-variant-numeric: tabular-nums; }
figure { margin: 12px 0 20px; page-break-inside: avoid; }
figcaption { font-size: .82rem; color: #6b7280; text-align: center; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 12px 0;
        break-inside: avoid; page-break-inside: avoid; }
.kpis div { background: #eef2ff; border-radius: 10px; padding: 10px 12px; }
.kpis b { display: block; font-size: 1.4rem; color: #312e81; }
.kpis span { font-size: .78rem; color: #4b5563; }
@media (max-width: 640px) { .kpis { grid-template-columns: repeat(2, 1fr); } }
</style>
"""


def page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{esc(title)}</title>{CSS}</head><body>{body}</body></html>")


def kpi_block(k: dict) -> str:
    return (f'<div class="kpis">'
            f'<div><b>{k["lead_time_reduction"]:.2f}%</b><span>Lead Time Reduction (operational gain)</span></div>'
            f'<div><b>{k["profit_stability"]:.0f}%</b><span>Profit Impact Stability (financial safety)</span></div>'
            f'<div><b>{k["confidence"]:.0f}/100</b><span>Scenario Confidence Score (reliability)</span></div>'
            f'<div><b>{k["coverage"]:.0f}%</b><span>Recommendation Coverage (scalability)</span></div></div>')


# ---------------------------------------------------------------- research paper
def research_paper(a: dict) -> str:
    df, clean, model, k = a["df"], a["clean"], a["model"], a["kpi"]
    m = model.metrics.set_index("Model")
    sel = model.best_name
    yr = a["by_year"]
    y0, y1 = yr.index.min(), yr.index.max()
    mode = a["by_mode"]
    rec, conf = a["rec"], a["confident"]
    top_target = a["target_counts"].index[0] if len(a["target_counts"]) else "—"
    top_target_share = a["target_counts"].iloc[0] / len(rec) * 100 if len(rec) else 0
    tf = a["by_factory"].loc[top_target] if top_target in a["by_factory"].index else None
    routes = a["routes"]
    slow = routes[routes["Cluster"] == "Slow"]
    congested = a["combos"][a["combos"]["Congested"]]
    choc = df[df["Division"] == "Chocolate"]
    s = []

    s.append(f"<h1>Factory Reallocation &amp; Shipping Optimization for Nassau Candy Distributor</h1>"
             f'<div class="meta">Research paper · EDA, insights and recommendations · generated {date.today():%d %B %Y}</div>')

    s.append(
        f'<div class="abstract"><b>Abstract.</b> Nassau Candy assigns products to factories with static rules. '
        f"This study builds a decision-intelligence system that predicts shipping lead time, simulates moving each "
        f"product to every alternative factory, and ranks reassignments by lead-time reduction, risk reduction and "
        f"profit impact. Using {clean['clean_rows']:,} order lines, the selected {esc(sel)} model explains "
        f"{m.loc[sel, 'R2']:.0%} of lead-time variance (RMSE {m.loc[sel, 'RMSE']:.1f} days). Simulations recommend "
        f"{k['n_rec']} of {k['n_products']} products for reassignment with an order-weighted predicted lead-time "
        f"reduction of {k['lead_time_reduction']:.2f}%. The main finding is that lead time in this dataset is driven "
        f"far more by order timing than by factory location, so reallocation gains are modest. A staged pilot is "
        f"therefore recommended over a network-wide change.</div>")

    # 1. Introduction
    s.append("<h2>1. Introduction and problem statement</h2>"
             "<p>Static, legacy factory assignment leads to suboptimal shipping distances, high lead times for some "
             "regions and margin erosion from logistics inefficiency. Leadership has no system to simulate "
             "factory–product reassignment, quantify the operational impact before execution, or recommend optimal "
             "configurations at scale. This project answers the question <i>what should we change to improve "
             "performance?</i> by combining predictive modelling with optimization logic.</p>")

    # 2. Data
    s.append("<h2>2. Data and preparation</h2>")
    s.append(f"<p>The source file contains {clean['raw_rows']:,} order lines ({df['Order ID'].nunique():,} orders, "
             f"{df['Customer ID'].nunique():,} customers) across {df['State/Province'].nunique()} US states and "
             f"Canadian provinces, {df['Product Name'].nunique()} products and {len(FACTORIES)} factories. Fields "
             "include order and ship dates, ship mode, customer location, division, region, sales, units, gross "
             "profit and cost.</p>")
    s.append("<h3>2.1 Preparation steps</h3><ul>"
             "<li><b>Date repair:</b> dates appear as both <code>dd-mm-yyyy</code> and <code>d/m/yyyy</code>. Every "
             "dash date has a day above 12 and every slash date has both parts ≤ 12, the pattern produced when a "
             "spreadsheet misreads day-first dates, so both are parsed day-first.</li>"
             "<li><b>Target:</b> lead time = ship date − order date, in days.</li>"
             "<li><b>Geography:</b> the factory–customer great-circle distance uses the factory coordinates from the "
             "brief and the state/province centroid as the customer location.</li>"
             f"<li><b>Outliers:</b> rows outside 3×IQR on lead time, sales or units were removed "
             f"({clean['outliers_removed']} rows, leaving {clean['clean_rows']:,}).</li>"
             "<li><b>Encoding:</b> numeric features (distance, units, unit price, order year, order month) are "
             "standardised; categorical features (product, region, ship mode, division, factory, country) are "
             "one-hot encoded into a training-ready feature matrix.</li></ul>")
    s.append('<div class="callout warn"><b>Assumptions.</b> (1) The CSV has no factory column; the current '
             "product→factory mapping follows the public Nassau Candy product table (editable in "
             "<code>config.py</code>). (2) Customer location is approximated by state/province centroid. "
             f"(3) The data has no freight cost, so profit impact assumes ${SETTINGS['freight']:.2f} per unit per "
             "1,000 miles of extra distance. This rate is adjustable in the dashboard.</div>")
    s.append(table(pd.DataFrame([{"Product": p, "Current factory": f} for p, f in PRODUCT_FACTORY.items()])))

    # 3. EDA
    s.append("<h2>3. Exploratory data analysis</h2><h3>3.1 Sales and profitability</h3>")
    s.append(f"<p>Total sales are {money(df['Sales'].sum())} with gross profit {money(df['Gross Profit'].sum())} "
             f"(overall margin {df['Gross Profit'].sum() / df['Sales'].sum():.1%}). The Chocolate division accounts "
             f"for {choc['Sales'].sum() / df['Sales'].sum():.1%} of sales. Five Wonka Bar lines, produced at "
             "Lot's O' Nuts and Wicked Choccy's, dominate volume.</p>")
    bp = a["by_product"].reset_index()
    fig = go.Figure(go.Bar(x=bp["Sales"], y=bp["Product Name"], orientation="h", marker_color=COLORS[0],
                           text=[f"{v:.0f}%" for v in bp["Margin %"]], textposition="outside"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Sales ($)")
    s.append(chart(fig, "Figure 1. Sales by product (labels show gross margin %).", 420))

    s.append("<h3>3.2 Lead time</h3>")
    s.append(f"<p>Lead times range from {df[TARGET].min():,} to {df[TARGET].max():,} days (mean "
             f"{df[TARGET].mean():,.0f}). These durations are far longer than typical candy distribution, which "
             f"suggests the dataset's ship dates are synthetic or shifted. Order year is the dominant pattern: "
             f"orders placed in {y0} average {yr.loc[y0, 'mean']:,.0f} days and orders in {y1} average "
             f"{yr.loc[y1, 'mean']:,.0f} days. Ship mode barely matters: the fastest mode on average is "
             f"{esc(mode.index[0])} ({mode['mean'].iloc[0]:,.0f} d) and the slowest is {esc(mode.index[-1])} "
             f"({mode['mean'].iloc[-1]:,.0f} d).</p>")
    fig = px.histogram(df, x=TARGET, color=df["Order Date"].dt.year.astype(str), nbins=60, barmode="overlay",
                       labels={"color": "Order year"})
    s.append(chart(fig, "Figure 2. Lead-time distribution by order year."))

    s.append("<h3>3.3 Geography and distance</h3>")
    s.append(f"<p>The correlation between factory–customer distance and lead time is {a['corr_dist']:.3f}, "
             "effectively zero. Regional averages are also close together (table below). The data therefore gives "
             "only weak evidence that moving production closer to customers shortens recorded lead time.</p>")
    br = a["by_region"].reset_index()
    s.append(table(br, {"Sales": money, "Distance": lambda v: f"{v:,.0f} mi", "LeadTime": lambda v: f"{v:,.1f} d"}))
    samp = df.sample(min(2500, len(df)), random_state=1)
    fig = px.scatter(samp, x="Distance (mi)", y=TARGET, color="Region", opacity=0.45)
    s.append(chart(fig, "Figure 3. Distance vs lead time (random sample of 2,500 order lines)."))

    s.append("<h3>3.4 Factory footprint</h3>")
    bf = a["by_factory"].reset_index()
    s.append(table(bf, {"Sales": money, "Profit": money, "Distance": lambda v: f"{v:,.0f} mi",
                        "LeadTime": lambda v: f"{v:,.1f} d"}))

    # 4. Methodology
    s.append("<h2>4. Methodology</h2><ol>"
             "<li><b>Predictive model.</b> Ridge regression, Random Forest and Gradient Boosting are trained on an 80% "
             "split and evaluated on a 20% hold-out with RMSE (penalises large errors), MAE (average error) and R² "
             "(variance explained). The lowest-RMSE model is chosen, except that a more interpretable model wins "
             "when it is within 1% of the best RMSE.</li>"
             "<li><b>Route clustering.</b> Factory→state routes with at least 3 orders are clustered with k-means "
             "(k = 3) on average lead time, lead-time variability and distance, then labelled Fast, Moderate or "
             "Slow. Region×product combinations with top-quartile lead time and above-median volume are flagged as "
             "congested.</li>"
             "<li><b>Scenario simulation.</b> For each product, every historical order is re-scored as if it were "
             "shipped from each alternative factory, with distance recomputed. Outputs are predicted lead time, "
             "late-order risk (share of orders above the 75th percentile of predicted lead time), estimated profit "
             "impact and a confidence score.</li>"
             "<li><b>Optimization and ranking.</b> Options are ranked by a weighted score: priority × lead-time "
             "reduction + (1 − priority) × profit impact + 0.2 × risk reduction, with each metric scaled to its "
             "largest absolute value. The top alternative per product is recommended when its score and lead-time "
             "reduction are both positive.</li>"
             "<li><b>Confidence.</b> A two-sided normal confidence that the predicted mean change is non-zero, given "
             "model residual error and the number of orders (0–100).</li></ol>")

    # 5. Results
    s.append("<h2>5. Results</h2><h3>5.1 Model performance</h3>")
    s.append(table(model.metrics.drop(columns="Selected").assign(
        Selected=model.metrics["Selected"].map({True: "✔", False: ""})),
        {"RMSE": lambda v: f"{v:.1f}", "MAE": lambda v: f"{v:.1f}", "R2": lambda v: f"{v:.3f}"}))
    s.append(f"<p>All three models perform similarly (R² {model.metrics['R2'].min():.2f}–"
             f"{model.metrics['R2'].max():.2f}), which indicates the remaining error comes from the data rather "
             f"than from model choice. {esc(sel)} was selected. Its error (~{m.loc[sel, 'RMSE']:.0f} days) is large "
             "compared with the factory effects being estimated, which is reflected in the confidence scores.</p>")

    s.append("<h3>5.2 Recommendations and KPIs</h3>")
    s.append(f"<p>Settings: {SETTINGS['speed_weight']:.0%} speed / {1 - SETTINGS['speed_weight']:.0%} profit "
             f"priority, profit alert at −{SETTINGS['alert_pct']:g}%, minimum confidence "
             f"{SETTINGS['min_conf']:.0f}.</p>")
    s.append(kpi_block(k))
    tbl = a["best"][["Rank", "Product", "Current Factory", "Factory", "Lead Time Reduction %", "Profit Impact %",
                     "Confidence", "Recommended"]].rename(columns={"Factory": "Suggested factory"})
    s.append(table(tbl, {"Lead Time Reduction %": lambda v: f"{v:+.2f}%", "Profit Impact %": lambda v: f"{v:+.2f}%",
                         "Confidence": lambda v: f"{v:.0f}", "Recommended": lambda v: "Yes" if v else "No"}))
    cur = a["scenarios"][a["scenarios"]["Current"]][["Product", "Predicted Lead Time"]]
    cmp = rec[["Product", "Predicted Lead Time"]].merge(cur, on="Product", suffixes=(" (recommended)", " (current)"))
    fig = go.Figure()
    fig.add_bar(y=cmp["Product"], x=cmp["Predicted Lead Time (current)"], name="Current", orientation="h",
                marker_color="#f59e0b")
    fig.add_bar(y=cmp["Product"], x=cmp["Predicted Lead Time (recommended)"], name="Recommended",
                orientation="h", marker_color="#16a34a")
    lo = cmp[["Predicted Lead Time (current)", "Predicted Lead Time (recommended)"]].min().min()
    fig.update_xaxes(range=[lo - 20, None], title="Predicted lead time (days)")
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.1))
    s.append(chart(fig, "Figure 4. Predicted lead time, current vs recommended factory.", 460))

    s.append("<h3>5.3 Route clusters and congestion</h3>")
    rc = routes.groupby("Cluster").agg(Routes=("Orders", "size"), Orders=("Orders", "sum"),
                                       AvgLeadTime=("Avg Lead Time", "mean"),
                                       AvgDistance=("Avg Distance (mi)", "mean")).reset_index()
    s.append(table(rc, {"AvgLeadTime": lambda v: f"{v:,.0f} d", "AvgDistance": lambda v: f"{v:,.0f} mi"}))
    fig = px.scatter(routes, x="Avg Distance (mi)", y="Avg Lead Time", size="Orders", color="Cluster",
                     color_discrete_map={"Fast": "#16a34a", "Moderate": "#0ea5e9", "Slow": "#db2777"})
    s.append(chart(fig, "Figure 5. Factory→state routes clustered by performance (bubble size = orders)."))
    slow_txt = ", ".join(f"{r['Factory']}→{r['State/Province']}" for _, r in
                         slow.sort_values("Avg Lead Time", ascending=False).head(5).iterrows())
    s.append(f"<p>{len(slow)} routes fall in the Slow cluster, but together they carry only "
             f"{slow['Orders'].sum():,} orders. The slowest are {esc(slow_txt)}. "
             + (f"Congested region–product combinations: "
                f"{esc(', '.join(r['Region'] + ' · ' + r['Product Name'] for _, r in congested.iterrows()))}."
                if len(congested) else "No region–product combination is both slow and high-volume.") + "</p>")

    # 6. Insights
    s.append("<h2>6. Key insights</h2><ol>"
             f"<li><b>Timing, not geography, drives recorded lead time.</b> Order year shifts average lead time by "
             f"{yr['mean'].max() - yr['mean'].min():,.0f} days, while the distance correlation is "
             f"{a['corr_dist']:.3f}.</li>"
             f"<li><b>Reallocation gains are real but small.</b> The order-weighted predicted reduction for "
             f"recommended moves is {k['lead_time_reduction']:.2f}%.</li>"
             f"<li><b>One factory dominates the recommendations.</b> {esc(top_target)} is the suggested destination "
             f"for {top_target_share:.0f}% of recommended moves. It is the most central site, but its own history "
             + (f"covers only {int(tf['Orders'])} order lines" if tf is not None else "is limited")
             + ", and the data contains no capacity information.</li>"
             f"<li><b>Confidence is concentrated in high-volume products.</b> {len(conf)} recommended moves reach "
             f"confidence ≥ {SETTINGS['min_conf']:.0f}. Low-volume products cannot be assessed reliably.</li>"
             "<li><b>Distance savings support margin.</b> Under the freight assumption, moves to more central "
             f"factories raise estimated profit. {k['profit_stability']:.0f}% of recommended moves stay within the "
             "profit-loss alert limit.</li></ol>")

    # 7. Recommendations
    conf_list = ", ".join(conf["Product"].head(5)) or "none at current settings"
    s.append("<h2>7. Recommendations</h2><ol>"
             f"<li><b>Pilot, don't migrate.</b> Run a controlled pilot moving one high-confidence product (candidates: "
             f"{esc(conf_list)}) to the suggested factory for a single region, and measure actual lead time and "
             "freight cost against a control group.</li>"
             "<li><b>Check capability and capacity first.</b> Several suggested moves send chocolate products to a "
             "factory that does not currently make chocolate. Confirm equipment, capacity and food-safety "
             "certification before any move.</li>"
             "<li><b>Keep low-volume products where they are</b> until enough orders exist to give a confident "
             "estimate.</li>"
             "<li><b>Fix the data foundation.</b> Capture actual dispatch and delivery timestamps, freight cost per "
             "shipment, factory capacity and customer coordinates. This is the single biggest lever for future "
             "accuracy.</li>"
             "<li><b>Operationalise the dashboard.</b> Use the Streamlit simulator for quarterly reviews and re-run "
             "the model as new data arrives.</li></ol>")

    # 8. Limitations
    s.append("<h2>8. Limitations</h2><ul>"
             "<li>The lead-time values look synthetic, so absolute day counts should not be read as real transit "
             "times.</li>"
             "<li>The factory mapping, customer centroids and freight rate are assumptions (Section 2).</li>"
             "<li>The model is associative, not causal: factory effects are learned from few factories and are "
             "confounded with each factory's product mix.</li>"
             "<li>Factory capacity, production cost and inventory constraints are not modelled.</li></ul>")

    s.append("<h2>9. Conclusion</h2><p>The system moves Nassau Candy from descriptive reporting to decision "
             "intelligence. It predicts outcomes for any factory configuration, ranks reassignments by speed, risk "
             "and profit, and flags unsafe moves before execution. On the current data, the evidence supports a "
             "targeted pilot rather than a large-scale reallocation, together with better operational data "
             "capture.</p>")
    s.append('<h2>Appendix: reproducing the results</h2><p><code>pip install -r requirements.txt</code>, then '
             "<code>streamlit run app.py</code> for the dashboard and <code>python reports/build_reports.py</code> to "
             "regenerate this paper and the executive summary.</p>")
    return page("Research Paper · Nassau Candy Factory Optimization", "".join(s))


# ---------------------------------------------------------------- executive summary
def executive_summary(a: dict) -> str:
    k, rec, conf, df = a["kpi"], a["rec"], a["confident"], a["df"]
    top_target = a["target_counts"].index[0] if len(a["target_counts"]) else "—"
    yr = a["by_year"]
    conf_list = ", ".join(conf["Product"].head(3)) or "none at current settings"
    s = [
        "<h1>Executive Summary</h1>"
        f'<div class="meta">Factory Reallocation &amp; Shipping Optimization · Nassau Candy Distributor · '
        f"{date.today():%d %B %Y}</div>",
        "<h2>Purpose</h2><p>Nassau Candy decides which factory makes each product using fixed, historical rules. "
        "This project built a tool that tests other factory assignments <i>before</i> any change is made. It "
        "estimates the effect on delivery time, risk and profit, and recommends which changes are worth trying.</p>",
        "<h2>What we found</h2>",
        kpi_block(k),
        "<ul>"
        f"<li>We analysed {len(df):,} order lines worth {money(df['Sales'].sum())} in sales across the US and "
        "Canada.</li>"
        f"<li><b>Moving products between factories would shorten delivery times only slightly</b>, by about "
        f"{k['lead_time_reduction']:.1f}% on average for the {k['n_rec']} products where a move helps.</li>"
        f"<li><b>Order timing matters far more than factory location</b>: average delivery time differs by "
        f"{yr['mean'].max() - yr['mean'].min():,.0f} days between order years, while distance shows almost no "
        "link.</li>"
        f"<li>Most suggested moves point to <b>{esc(top_target)}</b>, the most central site. Its capacity to take "
        "on more production is unknown.</li>"
        f"<li>Only {len(conf)} of the recommended moves meet our confidence bar. They are all high-volume products "
        "with plenty of order history.</li>"
        "</ul>",
        "<h2>Recommendations</h2><ol>"
        f"<li><b>Approve a small, measured pilot</b> rather than a network-wide change, starting with one of: "
        f"{esc(conf_list)}.</li>"
        "<li><b>Confirm factory readiness</b> (equipment, capacity, food-safety approval) before moving any "
        "product line.</li>"
        "<li><b>Invest in better data</b>: real dispatch and delivery dates, freight costs and factory capacity. "
        "This will make future recommendations much more reliable.</li>"
        "<li><b>Use the dashboard</b> for regular reviews as new data arrives.</li></ol>",
        "<h2>Risks and safeguards</h2><ul>"
        "<li>Some moves would send products to factories that do not make them today. The tool flags these as "
        "high-risk.</li>"
        "<li>Profit estimates rely on an assumed freight rate, because no shipping-cost data was provided.</li>"
        "<li>Recorded delivery times in the data look unrealistic (years rather than days), so results should "
        "guide where to look, not replace a real-world trial.</li></ul>",
        '<div class="callout"><b>Decision requested:</b> approve a one-product, one-region pilot and the data '
        "improvements above. Revisit the full reallocation once pilot results are available.</div>",
    ]
    return page("Executive Summary · Nassau Candy Factory Optimization", "".join(s))


# ---------------------------------------------------------------- output
def find_browser() -> str | None:
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    for name in ("msedge", "google-chrome", "chromium", "chrome"):
        if found := shutil.which(name):
            return found
    return None


def to_pdf(html_path: Path, browser: str | None) -> bool:
    if not browser:
        return False
    pdf = html_path.with_suffix(".pdf")
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           "--virtual-time-budget=15000", f"--print-to-pdf={pdf}", html_path.as_uri()]
    subprocess.run(cmd, capture_output=True, timeout=180)
    return pdf.exists() and pdf.stat().st_size > 0


def main() -> None:
    print("Analysing data and training model…")
    a = analyse()
    browser = find_browser()
    for name, builder in [("research_paper", research_paper), ("executive_summary", executive_summary)]:
        global _first_chart
        _first_chart = True
        path = OUT / f"{name}.html"
        path.write_text(builder(a), encoding="utf-8")
        ok = to_pdf(path, browser)
        print(f"  {path.name}" + (f" + {path.with_suffix('.pdf').name}" if ok else " (PDF skipped: no browser found)"))


if __name__ == "__main__":
    main()
