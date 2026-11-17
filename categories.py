"""Step 6 - Category performance, relationships, anomalies and statistical tests."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MAIN_CATEGORIES = ["FOOD", "BEVERAGE", "TOBACCO", "LIQUOR", "WINES"]
# Parts of the day, based on the opening hours found in Step 5 (10h until ~02h)
DAYPARTS = {
    "Late morning (10-12h)": [9, 10, 11],
    "Lunch (12-15h)": [12, 13, 14],
    "Afternoon (15-19h)": [15, 16, 17, 18],
    "Evening (19-23h)": [19, 20, 21, 22],
    "Night (23-03h)": [23, 0, 1, 2, 3, 4, 5, 6],
}
SEASONS = {"Apr-Jun 2010": [4, 5, 6], "Jul-Sep 2010": [7, 8, 9], "Oct-Dec 2010": [10, 11, 12], "Jan-Mar 2011": [1, 2, 3]}


def add_daypart(df):
    """Add a `Daypart` column from the `Hour` column (ordered categorical)."""
    hour_to_part = {h: part for part, hours in DAYPARTS.items() for h in hours}
    df["Daypart"] = pd.Categorical(df["Hour"].map(hour_to_part), categories=list(DAYPARTS), ordered=True)
    return df


# ---------------------------------------------------------------- 6.1 Category performance

def category_performance(df, orders):
    """Volume, revenue, price level and reach of every category."""
    n_orders = len(orders)
    table = df.groupby("Category").agg(
        Units=("Quantity", "sum"),
        Revenue=("Total", "sum"),
        Products=("Item Desc", "nunique"),
        Orders=("Bill Number", "nunique"),
    )
    table["Units share"] = table["Units"] / table["Units"].sum()
    table["Revenue share"] = table["Revenue"] / table["Revenue"].sum()
    table["Revenue per unit"] = table["Revenue"] / table["Units"]
    table["Revenue per product"] = table["Revenue"] / table["Products"]
    table["% of orders"] = table["Orders"] / n_orders
    return table.sort_values("Revenue", ascending=False)


def show_category_performance(table):
    """The performance table with readable formatting."""
    out = table.drop(columns="Orders").copy()
    for col in ["Units share", "Revenue share", "% of orders"]:
        out[col] = out[col].map("{:.1%}".format)
    for col in ["Revenue", "Revenue per unit", "Revenue per product"]:
        out[col] = out[col].map("{:,.0f}".format)
    return out


def plot_category_performance(table):
    shares = table[["Units share", "Revenue share"]] * 100
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    shares.plot.barh(ax=axes[0], color=["steelblue", "darkgreen"])
    axes[0].invert_yaxis()
    axes[0].set_title("Share of units vs share of revenue")
    axes[0].set_xlabel("%")
    axes[0].set_ylabel("")
    axes[1].barh(table.index, table["Revenue per unit"], color="darkorange")
    axes[1].invert_yaxis()
    axes[1].set_title("Average revenue per unit sold")
    axes[1].set_xlabel("Revenue per unit")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- 6.2 Pareto

def product_revenue(df):
    return df.groupby("Item Desc").agg(Revenue=("Total", "sum"), Units=("Quantity", "sum"),
                                       Category=("Category", "first")).sort_values("Revenue", ascending=False)


def pareto_summary(df):
    """How concentrated is revenue: how many products make 50/80/95% of it, and how long is the tail."""
    products = product_revenue(df)
    cum_share = products["Revenue"].cumsum() / products["Revenue"].sum()
    n = len(products)
    rows = {f"Products making {p:.0%} of revenue": (cum_share < p).sum() + 1 for p in [0.5, 0.8, 0.95]}
    rows = {k: f"{v} ({v / n:.0%} of products)" for k, v in rows.items()}
    rare = products["Units"] < 12
    rows["Products sold less than once a month (< 12 units)"] = f"{rare.sum()} ({rare.mean():.0%} of products)"
    rows["Revenue of those rare products"] = f"{products.loc[rare, 'Revenue'].sum() / products['Revenue'].sum():.1%}"
    return pd.Series(rows).to_frame("Value")


def plot_pareto(df):
    products = product_revenue(df)
    cum_share = products["Revenue"].cumsum() / products["Revenue"].sum() * 100
    x = np.arange(1, len(products) + 1) / len(products) * 100
    plt.figure(figsize=(10, 5))
    plt.plot(x, cum_share.values)
    for p, style in [(80, "--"), (95, ":")]:
        plt.axhline(p, color="gray", linestyle=style, linewidth=1)
    plt.axvline(20, color="red", linestyle="--", linewidth=1, label="20% of products")
    plt.xlabel("% of products (best sellers first)")
    plt.ylabel("% of total revenue")
    plt.title("Pareto curve: cumulative share of revenue")
    plt.legend()
    plt.show()


# ---------------------------------------------------------------- 6.3 Category mix through the day and week

def category_mix(df, by):
    """Share of revenue per category for every value of `by` (e.g. Daypart, weekday)."""
    mix = df.pivot_table(index=by, columns="Category", values="Total", aggfunc="sum", observed=True).fillna(0)
    return mix.div(mix.sum(axis=1), axis=0)


def plot_category_mix(df):
    by_daypart = category_mix(df, "Daypart")
    by_weekday = category_mix(df, df["Date"].dt.day_of_week)
    by_weekday.index = DAY_NAMES
    columns = [c for c in MAIN_CATEGORIES if c in by_daypart] + [c for c in by_daypart if c not in MAIN_CATEGORIES]
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    for ax, mix, title in [(axes[0], by_daypart, "Revenue mix by part of the day"),
                           (axes[1], by_weekday, "Revenue mix by weekday")]:
        (mix[columns] * 100).plot.bar(stacked=True, ax=ax, width=0.8, legend=False)
        ax.set_title(title)
        ax.set_ylabel("% of revenue")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def daypart_table(df, orders):
    """Orders, revenue and average order value per part of the day."""
    table = orders.groupby("Daypart", observed=True).agg(Orders=("Total", "size"), Revenue=("Total", "sum"),
                                                          Avg_order_value=("Total", "mean"))
    table["Share of revenue"] = (table["Revenue"] / table["Revenue"].sum()).map("{:.0%}".format)
    mix = category_mix(df, "Daypart")
    table["Top category"] = mix.idxmax(axis=1) + " (" + mix.max(axis=1).map("{:.0%}".format) + ")"
    table["Revenue"] = table["Revenue"].map("{:,.0f}".format)
    table["Avg_order_value"] = table["Avg_order_value"].round(0)
    return table


# ---------------------------------------------------------------- 6.4 Most sold products per month and season

def top_products_per_period(df, periods, n=3):
    """The n best-selling products (units) in every period. `periods` is a Series aligned with df."""
    units = df.groupby([periods, "Item Desc"])["Quantity"].sum()
    top = units.groupby(level=0, group_keys=False).nlargest(n).reset_index()
    top.columns = ["Period", "Product", "Units"]
    top["Rank"] = top.groupby("Period").cumcount() + 1
    top["Product"] = top["Product"] + " (" + top["Units"].astype(str) + ")"
    return top.pivot(index="Period", columns="Rank", values="Product")


def season_of(dates):
    month_to_season = {m: s for s, months in SEASONS.items() for m in months}
    return pd.Series(pd.Categorical(dates.dt.month.map(month_to_season), categories=list(SEASONS), ordered=True),
                     index=dates.index, name="Season")


def season_changes(df, n=8):
    """Products whose share of units changes most between the warmest (Apr-Jun) and coolest (Oct-Mar) quarters."""
    season = season_of(df["Date"])
    units = df.pivot_table(index="Item Desc", columns=season, values="Quantity", aggfunc="sum",
                           observed=False).fillna(0)
    share = units / units.sum() * 100
    regular = units.min(axis=1) >= 30             # only products sold in every season
    change = (share.loc[regular, "Oct-Dec 2010"] + share.loc[regular, "Jan-Mar 2011"]) / 2 - share.loc[regular, "Apr-Jun 2010"]
    out = pd.DataFrame({"Share Apr-Jun (%)": share.loc[regular, "Apr-Jun 2010"],
                        "Share Oct-Mar (%)": (share.loc[regular, "Oct-Dec 2010"] + share.loc[regular, "Jan-Mar 2011"]) / 2,
                        "Change (pp)": change}).round(2)
    return pd.concat([out.nlargest(n, "Change (pp)"), out.nsmallest(n, "Change (pp)")])


# ---------------------------------------------------------------- 6.5 Relationships

def price_vs_units(df):
    """One point per product: median price and units sold (products with a real price only)."""
    products = df[df["Rate"] >= 1].groupby("Item Desc").agg(Price=("Rate", "median"), Units=("Quantity", "sum"),
                                                          Category=("Category", "first"))
    return products


def plot_price_vs_units(products):
    rho, p = stats.spearmanr(products["Price"], products["Units"])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=products, x="Price", y="Units", hue="Category", alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"Price vs units sold per product (Spearman rho = {rho:.2f}, p = {p:.1e})")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def price_units_correlation(products):
    """Spearman correlation between price and units, overall and within each main category."""
    rows = {}
    for name, group in [("All products", products)] + [(c, products[products["Category"] == c]) for c in MAIN_CATEGORIES]:
        rho, p = stats.spearmanr(group["Price"], group["Units"])
        rows[name] = {"Products": len(group), "Spearman rho": round(rho, 2), "p-value": f"{p:.3f}"}
    return pd.DataFrame(rows).T


def plot_daily_category_correlation(df):
    """Do the categories rise and fall together from day to day? Spearman correlation of daily revenue."""
    daily = df.pivot_table(index="Date", columns="Category", values="Total", aggfunc="sum").fillna(0)[MAIN_CATEGORIES]
    corr = daily.corr(method="spearman")
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1)
    plt.gca().grid(False)   # no grid lines across the heatmap cells
    plt.title("Correlation of daily revenue between categories")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- 6.6 Anomalies

def free_items(df):
    """Lines sold at (almost) no price."""
    return df[df["Rate"] < 1][["Date", "Time", "Bill Number", "Item Desc", "Quantity", "Rate", "Total"]]


def biggest_orders(df, orders, n=5):
    big = orders.nlargest(n, "Total")[["Bill Number", "Date", "Time", "Unique_Items", "Quantity", "Total"]].copy()
    top_item = (df[df["Bill Number"].isin(big["Bill Number"])]
                .sort_values("Total").groupby("Bill Number")["Item Desc"].last())
    big["Most expensive line"] = big["Bill Number"].map(top_item)
    return big.set_index("Bill Number")


def bill_series(df, orders):
    """Bills start with G or N: compare the two series."""
    series_line = df["Bill Number"].str[0]
    series_order = orders["Bill Number"].str[0]
    table = orders.groupby(series_order).agg(Orders=("Total", "size"), Avg_order_value=("Total", "mean"),
                                             Avg_products=("Unique_Items", "mean"))
    table["Share of orders"] = (table["Orders"] / table["Orders"].sum()).map("{:.0%}".format)
    mix = df.pivot_table(index=series_line, columns="Category", values="Total", aggfunc="sum").fillna(0)
    mix = mix.div(mix.sum(axis=1), axis=0)
    table["Alcohol share of revenue"] = (mix.get("LIQUOR", 0) + mix.get("WINES", 0)).map("{:.0%}".format)
    table.index.name = "Bill series"
    return table.round(2)


def menu_changes(df):
    """Products sold per month, and how many products are sold for the first / last time in each month."""
    first = df.groupby("Item Desc")["Date"].min().dt.to_period("M")
    last = df.groupby("Item Desc")["Date"].max().dt.to_period("M")
    table = pd.DataFrame({
        "Products sold": df.groupby(df["Date"].dt.to_period("M"))["Item Desc"].nunique(),
        "New products": first.value_counts(),
        "Products sold for the last time": last.value_counts(),
    }).fillna(0).astype(int)
    # The first month has no history and the last month no future, so those counts are not meaningful
    table.iloc[0, 1] = table.iloc[-1, 2] = 0
    table.index = table.index.strftime("%b %y")
    return table


def first_sale_dates(df, month="2011-01"):
    """Dates on which new products were sold for the first time in a given month."""
    first = df.groupby("Item Desc")["Date"].min()
    return first[first.dt.strftime("%Y-%m") == month].value_counts().sort_index().head(5)


# ---------------------------------------------------------------- 6.7 Statistical tests

def test_weekend_vs_weekday(business_daily):
    """Mann-Whitney U: is revenue on Fri/Sat nights higher than on Sun-Thu? (business days, non-normal data)"""
    is_weekend = business_daily.index.day_of_week.isin([4, 5])
    weekend, weekday = business_daily[is_weekend], business_daily[~is_weekend]
    u, p = stats.mannwhitneyu(weekend, weekday, alternative="greater")
    return pd.Series({"Test": "Mann-Whitney U (one-sided)",
                      "H0": "Fri/Sat business days make the same revenue as Sun-Thu",
                      "Median Fri/Sat": f"{weekend.median():,.0f}", "Median Sun-Thu": f"{weekday.median():,.0f}",
                      "Effect size (rank-biserial r)": f"{2 * u / (len(weekend) * len(weekday)) - 1:.2f}",
                      "p-value": f"{p:.1e}"})


def test_weekdays(business_daily):
    """Kruskal-Wallis: does revenue differ between the 7 weekdays?"""
    groups = [g.values for _, g in business_daily.groupby(business_daily.index.day_of_week)]
    h, p = stats.kruskal(*groups)
    return pd.Series({"Test": "Kruskal-Wallis H", "H0": "All weekdays have the same revenue distribution",
                      "H statistic": f"{h:.1f}", "p-value": f"{p:.1e}"})


def test_drink_type_vs_season(beverages):
    """Chi-square test of independence: does the hot/cold choice depend on the season?"""
    hot_cold = beverages[beverages["Drink type"] != "Other"]
    table = pd.crosstab(season_of(hot_cold["Date"]), hot_cold["Drink type"], values=hot_cold["Quantity"], aggfunc="sum")
    chi2, p, dof, _ = stats.chi2_contingency(table)
    cramers_v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
    return pd.Series({"Test": "Chi-square test of independence", "H0": "Hot/cold choice is the same in every season",
                      "Chi2": f"{chi2:,.0f}", "Degrees of freedom": dof,
                      "Effect size (Cramer's V)": f"{cramers_v:.2f}", "p-value": f"{p:.1e}"}), table


def test_order_value_by_daypart(orders):
    """Kruskal-Wallis: does the order value differ between parts of the day?"""
    groups = [g["Total"].values for _, g in orders.groupby("Daypart", observed=True)]
    h, p = stats.kruskal(*groups)
    medians = orders.groupby("Daypart", observed=True)["Total"].median()
    return pd.Series({"Test": "Kruskal-Wallis H", "H0": "Order value is the same in every part of the day",
                      "Lowest median": f"{medians.idxmin()} ({medians.min():,.0f})",
                      "Highest median": f"{medians.idxmax()} ({medians.max():,.0f})",
                      "H statistic": f"{h:.1f}", "p-value": f"{p:.1e}"})


def test_summary(*tests):
    return pd.DataFrame({t["Test"] + ": " + t["H0"]: t.drop(["Test", "H0"]) for t in tests}).T.fillna("")
