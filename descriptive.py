"""Step 4 - Descriptive statistics: headline figures, centre and spread,
distributions and best-selling products/categories."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def overview(df, orders, daily):
    """Headline figures for the whole year."""
    n_days = len(daily)
    return pd.Series({
        "Total revenue (incl. tax)": df["Total"].sum(),
        "Total tax": df["Tax"].sum(),
        "Total discount given": df["Discount"].sum(),
        "Number of orders": len(orders),
        "Items sold": df["Quantity"].sum(),
        "Unique products": df["Item Desc"].nunique(),
        "Days in the data": n_days,
        "Average revenue per day": df["Total"].sum() / n_days,
        "Average orders per day": len(orders) / n_days,
    }).map("{:,.2f}".format).to_frame("Value")


def summary_stats(s):
    """Measures of centre, spread and shape of a numeric series."""
    q1, q3 = s.quantile([0.25, 0.75])
    return pd.Series({
        "mean": s.mean(), "median": s.median(), "mode": s.mode().iloc[0],
        "std": s.std(), "variance": s.var(),
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


def plot_category_units(category_sales):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=category_sales.values, y=category_sales.index, color="steelblue")
    plt.title("Total Units Sold by Category")
    plt.xlabel("Units sold")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def category_revenue(df):
    """Total revenue per category, highest first, with its share of the total revenue."""
    revenue = df.groupby("Category")["Total"].sum().sort_values(ascending=False)
    return pd.DataFrame({"Revenue": revenue.round(2), "Share of revenue": (revenue / revenue.sum()).map("{:.1%}".format)})


def plot_category_revenue(category_revenue):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=category_revenue["Revenue"].values / 1e6, y=category_revenue.index, color="green")
    plt.title("Total Revenue by Category")
    plt.xlabel("Revenue (millions)")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_top_per_category(df, category_sales, n=5):
    """Top n products in each category; the title shows their share of the category's units."""
    product_sales = df.groupby(["Category", "Item Desc"])["Quantity"].sum()
    rows = -(-len(category_sales) // 4)
    fig, axes = plt.subplots(rows, 4, figsize=(22, 4 * rows))
    for ax, category in zip(axes.flat, category_sales.index):
        top = product_sales[category].nlargest(n)
        sns.barplot(x=top.values, y=top.index, ax=ax, color="steelblue")
        ax.set_title(f"{category} (top {n} = {top.sum() / product_sales[category].sum():.0%} of units)")
        ax.set_xlabel("Units sold")
        ax.set_ylabel("")
    for ax in list(axes.flat)[len(category_sales):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
