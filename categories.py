"""Step 6 - Category performance, relationships and anomalies."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import plot_style  # noqa: F401 (applies the shared plot style)
from features import NEW_MENU_START
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
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
    # Products are on sale for very different lengths of time (Step 6.2), so we divide by the
    # total time the category's products were on sale, not by the number of products
    days_on_sale = product_lifecycle(df).groupby("Category")["Days on sale"].sum()
    table["Revenue per product per month"] = table["Revenue"] / days_on_sale * 30
    table["% of orders"] = table["Orders"] / n_orders
    return table.sort_values("Revenue", ascending=False)


def show_category_performance(table):
    """Price level and menu efficiency per category (units, revenue and shares are in Steps 4.5-4.6)."""
    out = table[["Revenue per unit", "Products", "Revenue per product per month"]].copy()
    for col in ["Revenue per unit", "Revenue per product per month"]:
        out[col] = out[col].map("{:,.0f}".format)
    return out


def plot_category_performance(table):
    """Revenue per unit (price level) and revenue per product per month on sale (how much a menu item earns)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    axes[0].barh(table.index, table["Revenue per unit"], color="darkgreen")
    axes[0].set_title("Average revenue per unit sold")
    axes[0].set_xlabel("Revenue per unit")
    axes[1].barh(table.index, table["Revenue per product per month"] / 1000, color="darkgreen")
    axes[1].set_title("Revenue per product, per month on sale")
    axes[1].set_xlabel("Revenue per product per month (in thousands)")
    for ax in axes:
        ax.invert_yaxis()
        ax.set_ylabel("")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- 6.2 Pareto

def product_revenue(df):
    return df.groupby("Item Desc").agg(Revenue=("Total", "sum"), Units=("Quantity", "sum"),
                                       Category=("Category", "first")).sort_values("Revenue", ascending=False)


def pareto_summary(df):
    """How concentrated is revenue: how many products make 50/80/95% of it."""
    products = product_revenue(df)
    cum_share = products["Revenue"].cumsum() / products["Revenue"].sum()
    n = len(products)
    rows = {f"Products making {p:.0%} of revenue": (cum_share < p).sum() + 1 for p in [0.5, 0.8, 0.95]}
    rows = {k: f"{v} ({v / n:.0%} of products)" for k, v in rows.items()}
    return pd.Series(rows).to_frame("Value")


# Product lifecycle. We only see sales, not the menu, so the time between a product's first and
# last sale is our best estimate of how long it was on sale.
LIFECYCLE = ["Regular", "New (from Jan 2011)", "Retired (before Jan 2011)", "Special (sold within a week)"]
ONCE_A_MONTH, ONCE_A_WEEK = 1 / 30, 1 / 7   # units per day on sale


def product_lifecycle(df):
    """One row per product: when it was sold, for how long, how fast, and its lifecycle status:
    - Special: all its sales fall within 7 days (New Year's Eve, Christmas, ...)
    - New: first sold from the start of the new menu (7 Jan 2011)
    - Retired: last sold before the new menu
    - Regular: on sale before and after the new menu started"""
    products = df.groupby("Item Desc").agg(Category=("Category", "first"), Units=("Quantity", "sum"),
                                           Revenue=("Total", "sum"), First_sale=("Date", "min"),
                                           Last_sale=("Date", "max"))
    products["Days on sale"] = (products["Last_sale"] - products["First_sale"]).dt.days + 1
    products["Units per day"] = products["Units"] / products["Days on sale"]
    regular, new, retired, special = LIFECYCLE
    products["Status"] = np.select(
        [products["Days on sale"] <= 7, products["First_sale"] >= NEW_MENU_START, products["Last_sale"] < NEW_MENU_START],
        [special, new, retired], regular)
    products["Status"] = pd.Categorical(products["Status"], categories=LIFECYCLE, ordered=True)
    return products


def lifecycle_summary(lifecycle):
    """Per lifecycle status: number of products, share of the café's revenue, typical time on sale
    and sales speed, and how many of them sold fewer than 12 units in the whole year."""
    total = lifecycle["Revenue"].sum()
    table = lifecycle.groupby("Status", observed=True).agg(
        products=("Units", "size"),
        revenue=("Revenue", "sum"),
        days=("Days on sale", "median"),
        speed=("Units per day", "median"),
        under_12=("Units", lambda u: (u < 12).sum()),
    )
    table["revenue"] = (table["revenue"] / total).map("{:.1%}".format)
    table["speed"] = table["speed"].round(2)
    table.columns = ["Products", "Share of the café's revenue", "Typical days on sale (median)",
                     "Typical units sold per day on sale (median)", "Products with fewer than 12 units in the year"]
    return table


def slow_sellers(lifecycle):
    """Products still on the menu in 2011 (regular or new) that sell slowly per day on sale, per category."""
    menu = lifecycle[lifecycle["Status"].isin(LIFECYCLE[:2])]
    total = lifecycle["Revenue"].sum()
    weekly = menu[menu["Units per day"] < ONCE_A_WEEK]
    monthly = menu[menu["Units per day"] < ONCE_A_MONTH]
    on_menu, slow_week, slow_month = "Products on the 2011 menu", "Sell less than once a week", "Sell less than once a month"
    revenue = "Café revenue from the products selling less than once a week"
    table = pd.DataFrame({
        on_menu: menu.groupby("Category").size(),
        slow_week: weekly.groupby("Category").size(),
        slow_month: monthly.groupby("Category").size(),
        revenue: weekly.groupby("Category")["Revenue"].sum() / total,
    }).fillna(0).sort_values(slow_week, ascending=False)
    table.loc["Total"] = table.sum()
    table[[on_menu, slow_week, slow_month]] = table[[on_menu, slow_week, slow_month]].astype(int)
    table[revenue] = table[revenue].map("{:.2%}".format)
    return table


def slowest_products(lifecycle, categories=("BEVERAGE", "FOOD"), n=10):
    """The slowest products still on the menu in 2011 in the given categories."""
    menu = lifecycle[lifecycle["Status"].isin(LIFECYCLE[:2]) & lifecycle["Category"].isin(categories)]
    out = menu.nsmallest(n, "Units per day")[["Category", "Status", "Units", "Days on sale", "Units per day"]]
    out["Units per day"] = out["Units per day"].round(3)
    return out.rename(columns={"Units": "Units sold in the year", "Units per day": "Units sold per day on sale"})


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


def _weekday(df):
    """Weekday name of every row, as an ordered category (Mon-Sun)."""
    names = df["Date"].dt.day_of_week.map(dict(enumerate(DAY_NAMES)))
    return pd.Series(pd.Categorical(names, categories=DAY_NAMES, ordered=True), index=df.index, name="Weekday")


def product_mix(df, category, by, n=7):
    """How the n best-selling products of a category (over the whole year) split their units
    for every value of `by` (e.g. Daypart, weekday). Each row adds up to 100% of those n products."""
    sales = df[df["Category"] == category]
    top = sales.groupby("Item Desc")["Quantity"].sum().nlargest(n).index
    sales = sales[sales["Item Desc"].isin(top)]
    by = by.loc[sales.index] if isinstance(by, pd.Series) else by
    mix = sales.groupby([by, "Item Desc"], observed=True)["Quantity"].sum().unstack().fillna(0)
    mix = mix[list(top)]   # best seller first (bottom of the bar)
    return mix.div(mix.sum(axis=1), axis=0)


def plot_top_product_mix(df, categories=("BEVERAGE", "FOOD", "TOBACCO"), n=7):
    """Like the category mix in Step 6.3, but for the top n products inside each category:
    one row per category, by part of the day (left) and by weekday (right). Shares of 5% or more are labelled."""
    weekday = _weekday(df)
    fig, axes = plt.subplots(len(categories), 2, figsize=(20, 6 * len(categories)))
    for row, category in zip(axes, categories):
        by_daypart = product_mix(df, category, "Daypart", n)
        by_weekday = product_mix(df, category, weekday, n)
        colors = sns.color_palette("tab10", n)   # one fixed colour per product in both panels
        for ax, mix, title in [(row[0], by_daypart, f"{category}: top {n} products by part of the day"),
                               (row[1], by_weekday, f"{category}: top {n} products by weekday")]:
            (mix * 100).plot.bar(stacked=True, ax=ax, width=0.8, color=colors, edgecolor="white", legend=False)
            for bars in ax.containers:
                ax.bar_label(bars, labels=[f"{b.get_height():.0f}%" if b.get_height() >= 5 else "" for b in bars],
                             label_type="center", fontsize=9)
            ax.set_title(title)
            ax.set_ylabel(f"% of the top {n} products' units")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=20)
        row[1].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
    plt.tight_layout()
    plt.show()


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


def _product_colors(df, n=10):
    """One fixed colour per product, by its rank in the whole year, so a product looks the same in every chart."""
    top = df.groupby("Item Desc")["Quantity"].sum().nlargest(n).index
    return dict(zip(top, sns.color_palette("tab10", n)))


def plot_top_products_per_month(df, n=4):
    """Grouped bars: units per month of the n best-selling products of the year."""
    month = df["Date"].dt.to_period("M")
    top = df.groupby("Item Desc")["Quantity"].sum().nlargest(n).index
    units = df[df["Item Desc"].isin(top)].pivot_table(index=month, columns="Item Desc", values="Quantity",
                                                       aggfunc="sum")[list(top)]
    leader = df.groupby([month, "Item Desc"])["Quantity"].sum().groupby(level=0).idxmax().str[1]
    wins = (leader == top[0]).sum()
    units.index = units.index.strftime("%b %y")
    colors = _product_colors(df)
    ax = units.plot.bar(figsize=(16, 5), width=0.8, color=[colors[p] for p in top], edgecolor="white", rot=0)
    ax.set_title(f"{top[0].title()} is the best seller in {wins} of the {len(units)} months")
    ax.set_ylabel("Units sold")
    ax.set_xlabel("")
    ax.legend(title="", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_top_products_per_season(df, n=5):
    """One panel per season with its n best-selling products; every product keeps the same colour in all panels."""
    season = season_of(df["Date"])
    units = df.groupby([season, "Item Desc"], observed=True)["Quantity"].sum()
    tops = {s: units[s].nlargest(n) for s in SEASONS}
    colors = _product_colors(df)
    fig, axes = plt.subplots(1, len(SEASONS), figsize=(22, 4.5), sharex=True)
    for ax, (name, top) in zip(axes, tops.items()):
        bars = ax.barh(top.index, top.values, color=[colors[p] for p in top.index])
        ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(name)
        ax.set_xlabel("Units sold")
        ax.tick_params(axis="y", labelsize=9)
    axes[0].set_xlim(0, max(top.max() for top in tops.values()) * 1.2)
    fig.suptitle(f"Top {n} products per season: the same favourites all year", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_season_changes(changes):
    """Diverging bars: change in share of units from Apr-Jun to Oct-Mar (green = gains, red = loses)."""
    data = changes["Change (pp)"].sort_values()
    colors = ["tab:red" if v < 0 else "tab:green" for v in data]
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(data.index, data.values, color=colors)
    ax.bar_label(bars, labels=[f"{v:+.2f}" for v in data], padding=3, fontsize=9)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlim(data.min() * 1.3, data.max() * 1.6)
    ax.set_title("Apr-Jun -> Oct-Mar: cold drinks lose share, beer, chai and poutine gain")
    ax.set_xlabel("Change in share of all units sold (percentage points)")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- 6.5 Relationships

def price_vs_units(df):
    """One point per product: median price and units sold per day on sale (Step 6.2).
    One-week specials are left out: they sell a lot in a few days, which says nothing about their price."""
    lifecycle = product_lifecycle(df)
    regular = lifecycle[lifecycle["Status"] != LIFECYCLE[3]]
    price = df[df["Rate"] >= 1].groupby("Item Desc")["Rate"].median()   # products with a real price only
    products = regular[["Category", "Units per day"]].join(price.rename("Price"), how="inner")
    return products.rename(columns={"Units per day": "Units"})


def _category_colors(categories):
    """One fixed colour per category, so a category looks the same in the overall and per-category plots."""
    order = MAIN_CATEGORIES + sorted(c for c in set(categories) if c not in MAIN_CATEGORIES)
    return dict(zip(order, sns.color_palette("tab10", len(order))))


def _plain_log_axes(ax):
    """Log scales on both axes, with plain numbers (20, 50, 100, ...) instead of 2x10^1."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())


def _p_text(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _add_trend_line(ax, price, units, color="black"):
    """Straight-line fit of log(units) on log(price), drawn as a dashed line.
    On log-log axes this is a straight line; it's a visual guide, the Spearman rho is the actual measure."""
    slope, intercept = np.polyfit(np.log10(price), np.log10(units), 1)
    x = np.logspace(np.log10(price.min()), np.log10(price.max()), 50)
    ax.plot(x, 10 ** intercept * x ** slope, color=color, linestyle="--", linewidth=2, label="Trend line")


def plot_price_vs_units(products):
    rho, p = stats.spearmanr(products["Price"], products["Units"])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=products, x="Price", y="Units", hue="Category", alpha=0.7,
                    palette=_category_colors(products["Category"]))
    _add_trend_line(plt.gca(), products["Price"], products["Units"])
    _plain_log_axes(plt.gca())
    plt.xlabel("Price")
    plt.ylabel("Units sold per day on sale")
    plt.title(f"Price vs units sold per day per product (Spearman rho = {rho:.2f}, {_p_text(p)})")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_price_vs_units_by_category(products):
    """One scatter per main category, with its Spearman rho and p-value in the title (log scales)."""
    colors = _category_colors(products["Category"])
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, category in zip(axes.flat, MAIN_CATEGORIES):
        group = products[products["Category"] == category]
        rho, p = stats.spearmanr(group["Price"], group["Units"])
        ax.scatter(group["Price"], group["Units"], color=colors[category], alpha=0.7, edgecolor="white")
        _add_trend_line(ax, group["Price"], group["Units"])
        _plain_log_axes(ax)
        verdict = "significant" if p < 0.05 else "not significant"
        ax.set_title(f"{category} ({len(group)} products)\nrho = {rho:+.2f}, {_p_text(p)} ({verdict})")
        ax.set_xlabel("Price")
        ax.set_ylabel("Units sold per day on sale")
    for ax in axes.flat[len(MAIN_CATEGORIES):]:
        ax.axis("off")
    fig.suptitle("Price vs units sold per day, per category (rho < 0: cheaper sells more, rho > 0: pricier sells more)",
                 fontsize=14)
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
