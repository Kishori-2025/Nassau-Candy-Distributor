"""Load, clean and feature-engineer the Nassau Candy shipment data."""

import numpy as np
import pandas as pd

from config import DATA_PATH, FACTORIES, PRODUCT_FACTORY, STATE_CENTROIDS

CATEGORICAL_FEATURES = ["Product Name", "Region", "Ship Mode", "Division", "Factory", "Country/Region"]
NUMERIC_FEATURES = ["Distance (mi)", "Units", "Unit Price", "Order Year", "Order Month"]
TARGET = "Lead Time (days)"


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles; works on scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def parse_mixed_dates(series: pd.Series) -> pd.Series:
    """The CSV mixes 'dd-mm-yyyy' and 'd/m/yyyy'. Every dash date has day > 12 and every
    slash date has both parts <= 12, which is the pattern Excel leaves after misreading
    day-first dates, so both formats are parsed day-first."""
    parsed = pd.to_datetime(series, format="%d-%m-%Y", errors="coerce")
    slash = parsed.isna()
    parsed[slash] = pd.to_datetime(series[slash], format="%d/%m/%Y", errors="coerce")
    return parsed


def distance_to_factory(df: pd.DataFrame, factory: str) -> pd.Series:
    """Distance from the given factory to each row's customer state centroid."""
    f_lat, f_lon = FACTORIES[factory]
    return pd.Series(haversine_miles(f_lat, f_lon, df["Cust Lat"].values, df["Cust Lon"].values), index=df.index)


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, dtype={"Postal Code": str})


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return the cleaned, feature-engineered frame and a short cleaning report."""
    report = {"raw_rows": len(df)}
    df = df.copy()

    df["Order Date"] = parse_mixed_dates(df["Order Date"])
    df["Ship Date"] = parse_mixed_dates(df["Ship Date"])
    df = df.dropna(subset=["Order Date", "Ship Date"])
    df[TARGET] = (df["Ship Date"] - df["Order Date"]).dt.days
    df = df[df[TARGET] >= 0]

    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY)
    centroids = df["State/Province"].map(STATE_CENTROIDS)
    df = df[df["Factory"].notna() & centroids.notna()]
    centroids = centroids[df.index]
    df["Cust Lat"] = centroids.str[0]
    df["Cust Lon"] = centroids.str[1]
    df["Distance (mi)"] = np.nan
    for factory in FACTORIES:
        mask = df["Factory"] == factory
        df.loc[mask, "Distance (mi)"] = distance_to_factory(df[mask], factory)

    df["Unit Price"] = df["Sales"] / df["Units"]
    df["Margin %"] = df["Gross Profit"] / df["Sales"] * 100
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    report["after_parsing"] = len(df)

    # Remove extreme outliers (3 x IQR fences) on the target and on order size.
    keep = pd.Series(True, index=df.index)
    for col in [TARGET, "Sales", "Units"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        keep &= df[col].between(q1 - 3 * iqr, q3 + 3 * iqr)
    report["outliers_removed"] = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    report["clean_rows"] = len(df)
    return df, report


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Columns the model is trained on (scaling and one-hot encoding happen in the model pipeline)."""
    return df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
