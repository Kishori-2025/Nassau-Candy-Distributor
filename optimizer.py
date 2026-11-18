"""Scenario simulation, recommendation ranking and route clustering."""

from math import erf, sqrt

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import FACTORIES, PRODUCT_FACTORY, RANDOM_STATE
from data_prep import TARGET, distance_to_factory, feature_matrix

# Divisions each factory currently produces (derived from the current assignment).
def factory_divisions(df: pd.DataFrame) -> dict:
    return df.groupby("Factory")["Division"].agg(lambda s: set(s.unique())).to_dict()


def _confidence(delta_days: float, residual_std: float, n: int) -> float:
    """0-100 proxy for how distinguishable a predicted change is from model noise:
    a two-sided normal confidence that the mean change is non-zero."""
    if n == 0 or residual_std == 0:
        return 0.0
    z = abs(delta_days) / (residual_std / sqrt(n))
    return 100 * erf(z / sqrt(2))


def simulate_product(orders: pd.DataFrame, pipeline, residual_std: float, late_threshold: float,
                     freight_rate: float, divisions: dict) -> pd.DataFrame:
    """Predict outcomes of serving one product's orders from every factory.

    orders: the product's historical orders (already filtered by region / ship mode).
    late_threshold: lead time (days) above which an order counts as late.
    freight_rate: ASSUMED freight cost in $ per unit per 1,000 miles.
    """
    current = orders["Factory"].iloc[0]
    base_pred = pipeline.predict(feature_matrix(orders))
    base_lt, base_dist = base_pred.mean(), orders["Distance (mi)"].mean()
    base_late = (base_pred > late_threshold).mean() * 100
    base_profit = orders["Gross Profit"].sum()
    units = orders["Units"].values
    division = orders["Division"].iloc[0]

    rows = []
    for factory in FACTORIES:
        sim = orders.copy()
        sim["Factory"] = factory
        sim["Distance (mi)"] = distance_to_factory(sim, factory)
        pred = pipeline.predict(feature_matrix(sim)) if factory != current else base_pred
        lt = pred.mean()
        extra_freight = ((sim["Distance (mi)"].values - orders["Distance (mi)"].values) / 1000 * units * freight_rate).sum()
        profit_change = -extra_freight
        rows.append({
            "Factory": factory,
            "Current": factory == current,
            "Predicted Lead Time": lt,
            "Lead Time Reduction %": (base_lt - lt) / base_lt * 100,
            "Avg Distance (mi)": sim["Distance (mi)"].mean(),
            "Distance Change (mi)": sim["Distance (mi)"].mean() - base_dist,
            "Late Risk %": (pred > late_threshold).mean() * 100,
            "Risk Reduction (pp)": base_late - (pred > late_threshold).mean() * 100,
            "Profit Impact $": profit_change,
            "Profit Impact %": profit_change / base_profit * 100 if base_profit else 0.0,
            "Confidence": 100.0 if factory == current else _confidence(base_lt - lt, residual_std, len(orders)),
            "Division Match": division in divisions.get(factory, set()),
            "Orders": len(orders),
        })
    return pd.DataFrame(rows)


def score_scenarios(sim: pd.DataFrame, speed_weight: float) -> pd.DataFrame:
    """Rank factories. speed_weight in [0, 1]: 1 = pure lead-time focus, 0 = pure profit focus.
    Each metric is scaled by its largest absolute value so they are comparable."""
    sim = sim.copy()

    def scaled(col):
        m = sim[col].abs().max()
        return sim[col] / m if m else 0.0

    sim["Score"] = (speed_weight * scaled("Lead Time Reduction %")
                    + (1 - speed_weight) * scaled("Profit Impact %")
                    + 0.2 * scaled("Risk Reduction (pp)"))
    sim.loc[sim["Current"], "Score"] = 0.0
    return sim.sort_values("Score", ascending=False).reset_index(drop=True)


def risk_flags(row: pd.Series, profit_alert_pct: float, min_confidence: float) -> list[str]:
    flags = []
    if not row["Division Match"]:
        flags.append("Factory does not currently produce this division")
    if row["Profit Impact %"] < -profit_alert_pct:
        flags.append(f"Profit drops {abs(row['Profit Impact %']):.1f}%")
    if row["Confidence"] < min_confidence:
        flags.append(f"Low confidence ({row['Confidence']:.0f})")
    if row["Lead Time Reduction %"] < 0:
        flags.append("Predicted lead time gets worse")
    return flags


def build_recommendations(df: pd.DataFrame, pipeline, residual_std: float, late_threshold: float,
                          freight_rate: float, speed_weight: float, profit_alert_pct: float,
                          min_confidence: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (best alternative per product, all scenarios for all products)."""
    divisions = factory_divisions(df)
    best_rows, all_rows = [], []
    for product, orders in df.groupby("Product Name"):
        sim = score_scenarios(
            simulate_product(orders, pipeline, residual_std, late_threshold, freight_rate, divisions), speed_weight)
        sim.insert(0, "Product", product)
        sim.insert(1, "Current Factory", PRODUCT_FACTORY[product])
        all_rows.append(sim)
        alt = sim[~sim["Current"]].iloc[0].copy()
        alt["Flags"] = risk_flags(alt, profit_alert_pct, min_confidence)
        alt["Recommended"] = alt["Score"] > 0 and alt["Lead Time Reduction %"] > 0
        best_rows.append(alt)

    best = pd.DataFrame(best_rows).sort_values("Score", ascending=False).reset_index(drop=True)
    best.insert(0, "Rank", range(1, len(best) + 1))
    return best, pd.concat(all_rows, ignore_index=True)


def late_threshold(df: pd.DataFrame, pipeline, quantile: float = 0.75) -> float:
    """Lead time (in predicted days) above which an order counts as 'late'. Measured on the
    model's own predictions so current and simulated scenarios are compared on the same scale."""
    return float(np.quantile(pipeline.predict(feature_matrix(df)), quantile))


def kpis(best: pd.DataFrame, profit_alert_pct: float) -> dict:
    rec = best[best["Recommended"]]
    if rec.empty:
        return {"lead_time_reduction": 0.0, "profit_stability": 0.0, "confidence": 0.0,
                "coverage": 0.0, "n_rec": 0, "n_products": len(best)}
    w = rec["Orders"]
    return {
        "lead_time_reduction": float(np.average(rec["Lead Time Reduction %"], weights=w)),
        # Share of recommendations that do not cut profit by more than the alert limit.
        "profit_stability": float((rec["Profit Impact %"] >= -profit_alert_pct).mean() * 100),
        "confidence": float(rec["Confidence"].mean()),
        "coverage": len(rec) / len(best) * 100,
        "n_rec": len(rec),
        "n_products": len(best),
    }


def cluster_routes(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Cluster Factory -> State routes by lead-time behaviour, distance and volume."""
    routes = df.groupby(["Factory", "State/Province", "Region"]).agg(
        Orders=(TARGET, "size"),
        **{"Avg Lead Time": (TARGET, "mean"), "Lead Time Std": (TARGET, "std"),
           "Avg Distance (mi)": ("Distance (mi)", "mean"), "Avg Margin %": ("Margin %", "mean")},
    ).reset_index()
    routes = routes[routes["Orders"] >= 3].fillna({"Lead Time Std": 0})
    feats = ["Avg Lead Time", "Lead Time Std", "Avg Distance (mi)"]
    k = min(k, len(routes))
    if k < 2:
        return routes.assign(Cluster="Moderate")
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    routes["Cluster"] = km.fit_predict(StandardScaler().fit_transform(routes[feats]))
    # Name clusters by mean lead time.
    order = routes.groupby("Cluster")["Avg Lead Time"].mean().sort_values().index
    names = ["Fast", "Moderate", "Slow", "Very slow", "Critical"][:k]
    routes["Cluster"] = routes["Cluster"].map(dict(zip(order, names)))
    return routes


def congested_combos(df: pd.DataFrame) -> pd.DataFrame:
    """Region x product combinations with above-75th-percentile lead time and above-median volume."""
    g = df.groupby(["Region", "Product Name"]).agg(
        Orders=(TARGET, "size"), **{"Avg Lead Time": (TARGET, "mean")}).reset_index()
    g["Congested"] = (g["Avg Lead Time"] >= g["Avg Lead Time"].quantile(0.75)) & (g["Orders"] >= g["Orders"].median())
    return g.sort_values("Avg Lead Time", ascending=False)
