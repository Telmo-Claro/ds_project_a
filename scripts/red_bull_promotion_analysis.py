"""Part 3 - Red Bull promotion performance: do promotions raise daily Red Bull revenue
once store traffic, weekends and the menu price increase are accounted for?

Source dataset: data/cafe_clean_lines.csv (366 business days, 2010-03-31 to 2011-03-31).
Used from the notebook (imported as `rb`), or run on its own from the project root:
    python scripts/red_bull_promotion_analysis.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "cafe_clean_lines.csv")

PRICE_HIKE_DATE = "2010-12-30"
CONTROLS = ["post_hike", "weekend_num", "total_bills"]
SHARE_BINS = [-0.01, 0.0001, 0.25, 0.50, 0.75, 1.01]
SHARE_LABELS = ["0%", "1-25%", "26-50%", "51-75%", "76-100%"]


# ---------- data ----------
def build_daily_table(csv_path: str = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the cleaned lines, isolate Red Bull rows, and aggregate to 1 row per business day."""
    df = pd.read_csv(csv_path)
    df["bday"] = pd.to_datetime(df["Business day"], format="%Y-%m-%d", errors="coerce")

    rb = df[
        df["Base product"].astype(str).str.contains("red bull|redbull", case=False, na=False)
        | df["Item Desc"].astype(str).str.contains("red bull|redbull", case=False, na=False)
    ].copy()

    rb["is_promo"] = (rb["Promo"] != "No promo").astype(int)
    rb["promo_qty"] = np.where(rb["is_promo"] == 1, rb["Quantity"], 0)
    rb["promo_rev"] = np.where(rb["is_promo"] == 1, rb["Total"], 0.0)

    daily_store = df.groupby("bday").agg(
        total_bills=("Bill Number", "nunique"),
        weekend=("Weekend", "max"),
    ).reset_index()

    daily_rb = rb.groupby("bday").agg(
        rb_rev=("Total", "sum"),
        rb_qty=("Quantity", "sum"),
        promo_qty=("promo_qty", "sum"),
        promo_rev=("promo_rev", "sum"),
        rb_rows=("Item Desc", "count"),
        promo_rows=("is_promo", "sum"),
    ).reset_index()

    daily = daily_store.merge(daily_rb, on="bday", how="left").fillna(0)
    daily["promo_share"] = np.where(daily["rb_qty"] > 0, daily["promo_qty"] / daily["rb_qty"], 0.0)
    daily["post_hike"] = (daily["bday"] >= PRICE_HIKE_DATE).astype(int)
    daily["weekend_num"] = daily["weekend"].astype(int)

    return df, rb, daily


def data_summary(df: pd.DataFrame, rb: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Size of the data before and after filtering for Red Bull."""
    return pd.DataFrame({"Value": {
        "Lines in cafe_clean_lines.csv": f"{len(df):,}",
        "Business days": f"{len(daily)} ({daily['bday'].min():%Y-%m-%d} to {daily['bday'].max():%Y-%m-%d})",
        "Red Bull lines": f"{len(rb):,}",
        "Red Bull units": f"{int(rb['Quantity'].sum()):,}",
        "Red Bull revenue": f"{rb['Total'].sum():,.2f}",
    }})


# ---------- key operational findings ----------
def price_shift(rb: pd.DataFrame) -> pd.DataFrame:
    """Unit price of the regular (no promo) Red Bull Energy Drink before and after the price hike."""
    base = rb[(rb["Item Desc"] == "RED BULL ENERGY DRINK") & (rb["Promo"] == "No promo")]
    period = np.where(base["bday"] >= PRICE_HIKE_DATE, "From 30 Dec 2010", "Before 30 Dec 2010")
    return base.groupby(period)["Rate"].agg(lambda s: s.mode()[0]).rename("Unit price (Rate)").to_frame()


def year_end_spike(rb: pd.DataFrame, daily: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Biggest Red Bull revenue days, and how much of each came from RED BULL SHEESHA (a hookah)."""
    top = daily.nlargest(n, "rb_rev")[["bday", "rb_rev", "promo_share"]].copy()
    sheesha = rb[rb["Item Desc"].str.contains("SHEESHA", case=False)]
    sheesha_day = sheesha.groupby("bday").agg(sheesha_units=("Quantity", "sum"), sheesha_rev=("Total", "sum"))
    top = top.join(sheesha_day, on="bday").fillna({"sheesha_units": 0, "sheesha_rev": 0})
    top["promo_share"] = (top["promo_share"] * 100).round(1).astype(str) + "%"
    return top.rename(columns={"bday": "Business day", "rb_rev": "Red Bull revenue", "promo_share": "Promo share",
                               "sheesha_units": "RED BULL SHEESHA units", "sheesha_rev": "RED BULL SHEESHA revenue"}
                      ).set_index("Business day").round(2)


def promo_mix(rb: pd.DataFrame) -> pd.DataFrame:
    """Units and revenue of Red Bull per promo type."""
    mix = rb.groupby("Promo").agg(Units=("Quantity", "sum"), Revenue=("Total", "sum"))
    mix["Share of units"] = (mix["Units"] / mix["Units"].sum() * 100).round(1)
    mix["Share of revenue"] = (mix["Revenue"] / mix["Revenue"].sum() * 100).round(1)
    return mix.sort_values("Revenue", ascending=False).round(2)


def revenue_by_share_bin(daily: pd.DataFrame) -> pd.DataFrame:
    """Average daily Red Bull revenue for days with no, some or mostly promo units."""
    bins = pd.cut(daily["promo_share"], bins=[-0.01, 0.0001, 0.50, 1.01], labels=["0%", "1-50%", "> 50%"])
    return daily.groupby(bins, observed=False).agg(
        Days=("rb_rev", "size"), **{"Mean daily Red Bull revenue": ("rb_rev", "mean")}).round(2)


# ---------- regression ----------
def fit_ols_with_fit_stats(y: np.ndarray, X_df: pd.DataFrame) -> dict:
    """Fit OLS with HC1 robust standard errors and compute SST, SSR, SSE, R^2, and Adjusted R^2."""
    n = len(y)
    k = X_df.shape[1]
    model = sm.OLS(y, sm.add_constant(X_df)).fit(cov_type="HC1")

    y_bar = float(np.mean(y))
    sst = float(np.sum((y - y_bar) ** 2))
    ssr = float(np.sum((model.fittedvalues - y_bar) ** 2))
    sse = float(np.sum(model.resid ** 2))
    return {
        "model": model, "n": n, "k": k, "sst": sst, "ssr": ssr, "sse": sse,
        "r2": 1.0 - sse / sst,
        "adj_r2": 1.0 - (sse / (n - k - 1)) / (sst / (n - 1)),
    }


def fit_models(daily: pd.DataFrame) -> tuple[dict, dict]:
    """Simple model (promo_share only) and multiple model (+ price hike, weekend, store traffic)."""
    y = daily["rb_rev"].to_numpy()
    simple = fit_ols_with_fit_stats(y, daily[["promo_share"]])
    multi = fit_ols_with_fit_stats(y, daily[["promo_share"] + CONTROLS])
    return simple, multi


def coef_table(res: dict) -> pd.DataFrame:
    """Coefficients with HC1 robust standard errors and p-values."""
    m = res["model"]
    return pd.DataFrame({"Coefficient": m.params, "HC1 std err": m.bse, "HC1 p-value": m.pvalues}).round(4)


def fit_stats_table(simple: dict, multi: dict) -> pd.DataFrame:
    """Variance decomposition and model fit of both models side by side."""
    rows = {"n": ("n", "{:.0f}"), "k": ("k (predictors)", "{:.0f}"), "sst": ("SST", "{:,.2f}"),
            "ssr": ("SSR", "{:,.2f}"), "sse": ("SSE", "{:,.2f}"), "r2": ("R^2", "{:.4f}"),
            "adj_r2": ("Adjusted R^2", "{:.4f}")}
    return pd.DataFrame({
        name: {label: fmt.format(res[key]) for key, (label, fmt) in rows.items()}
        for name, res in [("Simple regression (k = 1)", simple), ("Multiple regression (k = 4)", multi)]
    })


def sensitivity_without_year_end(daily: pd.DataFrame) -> pd.DataFrame:
    """Refit the multiple model without 30-31 Dec 2010 (New Year's Eve and RED BULL SHEESHA spike)."""
    keep = ~daily["bday"].isin(pd.to_datetime(["2010-12-30", "2010-12-31"]))
    _, full = fit_models(daily)
    _, trimmed = fit_models(daily[keep])
    out = {}
    for name, res in [("All days", full), ("Without 30-31 Dec 2010", trimmed)]:
        m = res["model"]
        out[name] = {"Days": res["n"], "promo_share coefficient": m.params["promo_share"],
                     "HC1 p-value": m.pvalues["promo_share"], "R^2": res["r2"], "Adjusted R^2": res["adj_r2"]}
    return pd.DataFrame(out).T.round(4)


# ---------- charts ----------
def _binned(temp: pd.DataFrame, y_col: str) -> pd.DataFrame:
    temp = temp.assign(share_bin=pd.cut(temp["promo_share"], bins=SHARE_BINS, labels=SHARE_LABELS))
    return (temp.groupby("share_bin", observed=False)
            .agg(mean_x=("promo_share", "mean"), mean_y=(y_col, "mean")).dropna().reset_index())


def _regression_plot(x_share, y, binned, b0, b1, line_label, title, ylabel, point_label):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.scatter(x_share * 100, y, color="#7a99ac", alpha=0.45, s=24, label=point_label)
    ax.plot(binned["mean_x"] * 100, binned["mean_y"], color="#2e7d32", marker="o", linewidth=2.0,
            markersize=6, label="Line through binned average points")
    grid = np.linspace(0, 1, 100)
    ax.plot(grid * 100, b0 + b1 * grid, color="#c0392b", linestyle="--", linewidth=2.2, label=line_label)
    ax.set_title(title)
    ax.set_xlabel("Daily promo quantity share (%)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-3, 103)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_simple_regression(daily: pd.DataFrame, simple: dict):
    """Daily Red Bull revenue against promo share, with the simple OLS line."""
    p = simple["model"].params
    return _regression_plot(
        daily["promo_share"], daily["rb_rev"], _binned(daily, "rb_rev"), p["const"], p["promo_share"],
        f"Simple OLS line (y = {p['const']:,.0f} + {p['promo_share']:,.0f} x share)",
        "Simple linear regression: daily Red Bull revenue vs. promo share",
        "Daily Red Bull revenue", f"Actual days (n = {len(daily)})")


def plot_multiple_regression(daily: pd.DataFrame, multi: dict):
    """Partial effect of promo share: revenue with the effect of the controls removed."""
    p = multi["model"].params
    temp = daily.assign(y_adj=daily["rb_rev"] - sum(p[c] * daily[c] for c in CONTROLS))
    return _regression_plot(
        temp["promo_share"], temp["y_adj"], _binned(temp, "y_adj"), p["const"], p["promo_share"],
        f"Multiple regression line (slope = +{p['promo_share']:,.0f} per 1.0 share)",
        "Multiple linear regression: promo share vs. adjusted daily revenue",
        "Daily Red Bull revenue (adjusted for price hike, weekend & bills)",
        f"Actual days adjusted for controls (n = {len(daily)})")


def save_charts(daily: pd.DataFrame, simple: dict, multi: dict, out_dir: str = ROOT) -> list:
    paths = []
    for name, fig in [("simple_regression_line.png", plot_simple_regression(daily, simple)),
                      ("multiple_regression_line.png", plot_multiple_regression(daily, multi))]:
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    df, rb, daily = build_daily_table()
    simple, multi = fit_models(daily)
    print(data_summary(df, rb, daily), "\n")
    print(price_shift(rb), "\n")
    print(year_end_spike(rb, daily), "\n")
    print(promo_mix(rb), "\n")
    print("=== SIMPLE REGRESSION (k=1) ===")
    print(coef_table(simple), "\n")
    print("=== MULTIPLE REGRESSION (k=4) ===")
    print(coef_table(multi), "\n")
    print(fit_stats_table(simple, multi), "\n")
    print(sensitivity_without_year_end(daily), "\n")
    for path in save_charts(daily, simple, multi, out_dir=os.path.join(ROOT, "outputs")):
        print(path)


if __name__ == "__main__":
    main()
