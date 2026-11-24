"""Step 4 - Descriptive statistics: headline figures, centre and spread,
distributions and best-selling products/categories."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import plot_style  # noqa: F401 (applies the shared plot style)


def overview(df, orders, daily):
    """Headline figures for the whole year."""
    n_days = len(daily)
    money, count = "{:,.2f}", "{:,.0f}"
    rows = {
        "Total revenue (incl. tax)": (df["Total"].sum(), money),
        "Total tax": (df["Tax"].sum(), money),
        "Total discount given": (df["Discount"].sum(), money),
        "Number of orders": (len(orders), count),
        "Items sold": (df["Quantity"].sum(), count),
        "Unique products": (df["Item Desc"].nunique(), count),
        "Days in the data": (n_days, count),
        "Average revenue per day": (df["Total"].sum() / n_days, money),
        "Average orders per day": (len(orders) / n_days, "{:,.1f}"),
    }
    return pd.Series({name: fmt.format(value) for name, (value, fmt) in rows.items()}).to_frame("Value")


def summary_stats(s):
    """Measures of centre, spread and shape of a numeric series.
    The mode is left empty when no value appears more than once (e.g. daily revenue), because then
    it's just a random value of the series."""
    q1, q3 = s.quantile([0.25, 0.75])
    counts = s.value_counts()
    return pd.Series({
        "mean": s.mean(), "median": s.median(), "mode": counts.index[0] if counts.iloc[0] > 1 else None,
        "std": s.std(),
        "min": s.min(), "max": s.max(), "range": s.max() - s.min(),
        "Q1": q1, "Q3": q3, "IQR": q3 - q1,
        "skewness": s.skew(),
    })


def stats_table(df, orders, daily):
    """summary_stats for the main variables at line, order and day level."""
    return pd.DataFrame({
        "Item price (Rate)": summary_stats(df["Rate"]),
        "Quantity": summary_stats(df["Quantity"]),
        "Total": summary_stats(df["Total"]),
        "Order total": summary_stats(orders["Total"]),
        "Items per order": summary_stats(orders["Quantity"]),
        "Daily revenue": summary_stats(daily),
    }).round(2).T


def plot_distributions(orders, daily):
    """Histogram (with mean and median) and boxplot for order total, items per order and daily revenue.
    The order total histogram is cut at the 99th percentile so the few huge bills don't squash it."""
    to_plot = {"Order total": orders["Total"], "Items per order": orders["Quantity"], "Daily revenue": daily}
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for i, (name, s) in enumerate(to_plot.items()):
        if name == "Order total":
            sns.histplot(s[s <= s.quantile(0.99)], bins=50, ax=axes[0, i])
            axes[0, i].set_title(f"{name} (up to 99th percentile)")
        elif name == "Items per order":
            sns.histplot(s, discrete=True, ax=axes[0, i])
            axes[0, i].set_title(name)
        else:
            sns.histplot(s, bins=50, ax=axes[0, i])
            axes[0, i].set_title(name)
        axes[0, i].axvline(s.mean(), color="red", linestyle="--", label=f"mean = {s.mean():,.1f}")
        axes[0, i].axvline(s.median(), color="green", label=f"median = {s.median():,.1f}")
        axes[0, i].legend()
        sns.boxplot(x=s, ax=axes[1, i])
        axes[1, i].set_title(f"{name} - boxplot")
    plt.tight_layout()
    plt.show()


def outlier_counts(orders, daily):
    """Number of high outliers by the IQR rule (> Q3 + 1.5 * IQR)."""
    rows = {}
    for name, s in {"Order total": orders["Total"], "Items per order": orders["Quantity"], "Daily revenue": daily}.items():
        q1, q3 = s.quantile([0.25, 0.75])
        n_out = (s > q3 + 1.5 * (q3 - q1)).sum()
        rows[name] = {"High outliers": n_out, "Share": f"{n_out / len(s):.1%}"}
    return pd.DataFrame(rows).T


def plot_top_products(df, n=10):
    """Top n products by units sold and by revenue."""
    top_quantity = df.groupby("Item Desc")["Quantity"].sum().nlargest(n)
    top_revenue = df.groupby("Item Desc")["Total"].sum().nlargest(n) / 1_000_000
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.barplot(x=top_quantity.values, y=top_quantity.index, ax=axes[0], color="steelblue")
    axes[0].set_title(f"Top {n} products by quantity sold")
    axes[0].set_xlabel("Units sold")
    sns.barplot(x=top_revenue.values, y=top_revenue.index, ax=axes[1], color="darkgreen")
    axes[1].set_title(f"Top {n} products by revenue")
    axes[1].set_xlabel("Revenue (in millions)")
    for ax in axes:
        ax.set_ylabel("")
    plt.tight_layout()
    plt.show()


def category_units(df):
    """Total units sold per category, highest first."""
    return df.groupby("Category")["Quantity"].sum().sort_values(ascending=False)


def category_table(category_sales, category_revenue):
    """Units, revenue and share of revenue per category, formatted for reading."""
    out = category_revenue.join(category_sales.rename("Units sold"))[["Units sold", "Revenue", "Share of revenue"]]
    out["Units sold"] = out["Units sold"].map("{:,}".format)
    out["Revenue"] = out["Revenue"].map("{:,.0f}".format)
    return out


def category_revenue(df):
    """Total revenue per category, highest first, with its share of the total revenue."""
    revenue = df.groupby("Category")["Total"].sum().sort_values(ascending=False)
    return pd.DataFrame({"Revenue": revenue.round(2), "Share of revenue": (revenue / revenue.sum()).map("{:.1%}".format)})


def plot_category_sales(category_sales, category_revenue):
    """Units sold and revenue per category, side by side (each sorted by its own value)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.barplot(x=category_sales.values, y=category_sales.index, ax=axes[0], color="steelblue")
    axes[0].set_title("Units sold by category")
    axes[0].set_xlabel("Units sold")
    sns.barplot(x=category_revenue["Revenue"].values / 1e6, y=category_revenue.index, ax=axes[1], color="darkgreen")
    axes[1].set_title("Revenue by category")
    axes[1].set_xlabel("Revenue (in millions)")
    for ax in axes:
        ax.set_ylabel("")
    plt.tight_layout()
    plt.show()


def top_categories(category_sales, category_revenue, n=3):
    """The top n categories by units and by revenue, side by side."""
    return pd.DataFrame({"By units sold": category_sales.index[:n], "By revenue": category_revenue.index[:n]},
                        index=pd.RangeIndex(1, n + 1, name="Rank"))


def top_category_shares(df, top):
    """Share of units, revenue and orders of each top category, with the remaining categories grouped as 'Other'."""
    group = df["Category"].where(df["Category"].isin(top), "Other")
    order = list(top) + ["Other"]
    n_orders = df["Bill Number"].nunique()
    table = pd.DataFrame({
        "Share of units": df.groupby(group)["Quantity"].sum() / df["Quantity"].sum(),
        "Share of revenue": df.groupby(group)["Total"].sum() / df["Total"].sum(),
        # an order can contain several categories, so this column doesn't add up to 100%
        "Share of orders": df.groupby(group)["Bill Number"].nunique() / n_orders,
        "Products": df.groupby(group)["Item Desc"].nunique(),
    }).loc[order]
    top_rows = df["Category"].isin(top)
    table.loc["Top 3 together"] = [
        table.loc[top, "Share of units"].sum(),
        table.loc[top, "Share of revenue"].sum(),
        df.loc[top_rows, "Bill Number"].nunique() / n_orders,
        df.loc[top_rows, "Item Desc"].nunique(),
    ]
    table.index.name = "Category"
    return table


def show_top_category_shares(shares):
    """The shares table with percentages."""
    out = shares.copy()
    for col in ["Share of units", "Share of revenue", "Share of orders"]:
        out[col] = out[col].map("{:.1%}".format)
    out["Products"] = out["Products"].astype(int)
    return out


def plot_top_category_shares(shares):
    """Share of units vs share of revenue for each top category and 'Other'."""
    data = shares.drop("Top 3 together")[["Share of units", "Share of revenue"]] * 100
    long = data.reset_index().melt(id_vars="Category", var_name="Measure", value_name="Share (%)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=long, x="Share (%)", y="Category", hue="Measure", palette=["steelblue", "darkgreen"], ax=ax)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.0f%%", padding=3)
    ax.set_title("Top 3 categories: share of units vs share of revenue")
    ax.set_xlabel("Share of the café total (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, data.values.max() * 1.15)
    ax.legend(title="")
    plt.tight_layout()
    plt.show()


def plot_top_products_in_categories(df, top, n=5):
    """Top n products of each top category, by units (left) and by revenue (right).
    The titles show how much of the category those n products make up."""
    fig, axes = plt.subplots(len(top), 2, figsize=(16, 3.5 * len(top)))
    for row, category in zip(axes, top):
        products = df[df["Category"] == category].groupby("Item Desc")
        units = products["Quantity"].sum()
        revenue = products["Total"].sum()
        top_units = units.nlargest(n)
        top_revenue = revenue.nlargest(n)
        sns.barplot(x=top_units.values, y=top_units.index, ax=row[0], color="steelblue")
        row[0].set_title(f"{category}: top {n} by units ({top_units.sum() / units.sum():.0%} of the category)")
        row[0].set_xlabel("Units sold")
        sns.barplot(x=top_revenue.values / 1e6, y=top_revenue.index, ax=row[1], color="darkgreen")
        row[1].set_title(f"{category}: top {n} by revenue ({top_revenue.sum() / revenue.sum():.0%} of the category)")
        row[1].set_xlabel("Revenue (in millions)")
        for ax in row:
            ax.set_ylabel("")
    plt.tight_layout()
    plt.show()
