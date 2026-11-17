"""Step 5 - Time series analysis of the café sales.

All calculations and plots for Step 5 of main.ipynb live here, so the notebook
can focus on the insights. Functions named `plot_*` draw a figure, the others
return a pandas object.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# The café opens around 10h and takes its last orders around 02h, so the
# "day" runs from 10h to 02h instead of 00h to 23h.
HOUR_ORDER = list(range(10, 24)) + list(range(0, 3))
HOUR_LABELS = [f"{h:02d}h" for h in HOUR_ORDER]
EVENING = [19, 20, 21, 22, 23, 0, 1, 2]

# Keywords used to label beverages as hot or cold ("cold" wins, e.g. "LEMON ICED TEA")
COLD = (r"ICED|LEMONADE|SHAKE|SMOOTHIE|COOLER|FLOAT|FREEZ|FRAPP|BLAST|MELON|"
        r"CALIFORNICA|TROPIC|BERRY|LIME|JUICE|ORANGINA|PEACHES|CELERY")
HOT = (r"CAPPUCCINO|LATTE|ESPRESSO|DOPPIO|CHAI|HOT CHOC|COFFEE|\(REG\)|AU ?LAIT|"
       r"TEA|AMERICANO|MOCHA|MACCHIATO|CON PANNA|SCHLAG")


# ---------------------------------------------------------------- Preparation

def add_hour(df):
    """Add an `Hour` column (0-23) taken from the `Time` column."""
    df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
    return df


def daily_revenue(df_copy):
    """Total revenue per calendar day, one value for every day in the data."""
    return df_copy.groupby("Date")["Total"].sum().asfreq("D")


def label_drinks(df_copy):
    """Beverage rows with a `Drink type` column: Hot, Cold or Other (water, energy drinks, add-ons)."""
    beverages = df_copy[df_copy["Category"] == "BEVERAGE"].copy()
    beverages["Drink type"] = "Other"
    beverages.loc[beverages["Item Desc"].str.contains(HOT), "Drink type"] = "Hot"
    beverages.loc[beverages["Item Desc"].str.contains(COLD), "Drink type"] = "Cold"
    return beverages


# ---------------------------------------------------------------- 5.1 Trend

def plot_monthly_revenue(daily):
    """Revenue per month in chronological order (Apr 2010 - Mar 2011)."""
    monthly = daily.resample("MS").sum() / 1_000_000
    plt.figure(figsize=(10, 4))
    plt.plot(monthly.index.strftime("%b %y"), monthly.values, marker="o")
    plt.xlabel("Month")
    plt.ylabel("Revenue (in millions)")
    plt.title("Monthly Revenue")
    plt.show()


def moving_averages(daily):
    """7-day MA (cancels the weekly cycle) and 30-day MA (long-term direction), both centred."""
    return pd.DataFrame({
        "Daily revenue": daily,
        "7-day MA": daily.rolling(7, center=True).mean(),
        "30-day MA": daily.rolling(30, center=True).mean(),
    })


def plot_trend(ma):
    plt.figure(figsize=(14, 5))
    plt.plot(ma.index, ma["Daily revenue"] / 1000, color="lightgray", label="Daily revenue")
    plt.plot(ma.index, ma["7-day MA"] / 1000, color="steelblue", label="7-day moving average")
    plt.plot(ma.index, ma["30-day MA"] / 1000, color="red", linewidth=2, label="30-day moving average")
    plt.ylabel("Revenue (thousands)")
    plt.title("Daily revenue and its trend")
    plt.legend()
    plt.show()


def trend_summary(daily, ma):
    """Key numbers of the trend: lowest/highest 30-day MA and first vs last quarter."""
    first_q, last_q = daily["2010-04":"2010-06"].mean(), daily["2011-01":"2011-03"].mean()
    return pd.Series({
        "Lowest 30-day MA": f"{ma['30-day MA'].min():,.0f} ({ma['30-day MA'].idxmin():%d %b %Y})",
        "Highest 30-day MA": f"{ma['30-day MA'].max():,.0f} ({ma['30-day MA'].idxmax():%d %b %Y})",
        "Average day Apr-Jun 2010": f"{first_q:,.0f}",
        "Average day Jan-Mar 2011": f"{last_q:,.0f}",
        "Growth": f"{last_q / first_q - 1:+.0%}",
    }).to_frame("Value")


# ---------------------------------------------------------------- 5.2 Decomposition

def decompose(daily, period=7):
    """Additive decomposition Y_t = T_t + S_t + e_t.

    - Trend T_t: centred moving average over one period (cancels the cycle)
    - Seasonality S_t: average deviation from the trend per position in the cycle
      (weekday), centred to sum to 0, so S_(t+period) = S_t
    - Noise e_t: what is left, Y_t - T_t - S_t
    Same method as statsmodels' seasonal_decompose, written out in pandas.
    """
    trend = daily.rolling(period, center=True).mean()
    position = daily.index.day_of_week if period == 7 else pd.RangeIndex(len(daily)) % period
    profile = (daily - trend).groupby(position).mean()
    profile -= profile.mean()
    seasonal = pd.Series(profile.loc[position].values, index=daily.index)
    return pd.DataFrame({
        "Observed (Y_t)": daily,
        "Trend (T_t)": trend,
        "Seasonality (S_t)": seasonal,
        "Noise (e_t)": daily - trend - seasonal,
    })


def plot_decomposition(dec):
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, name in zip(axes, dec.columns):
        ax.plot(dec.index, dec[name] / 1000)
        ax.set_title(name)
        ax.set_ylabel("Thousands")
    axes[3].axhline(0, color="gray", linewidth=0.8)
    plt.tight_layout()
    plt.show()


def variance_share(dec):
    """Rough share of the daily variance per component (they are not fully independent)."""
    observed = dec["Observed (Y_t)"].var()
    return (dec.drop(columns="Observed (Y_t)").var() / observed).round(2).to_frame("Share of variance")


def biggest_surprises(dec, n=6):
    """The n days with the largest noise: one-off events, not regular patterns."""
    expected = dec["Trend (T_t)"] + dec["Seasonality (S_t)"]
    table = pd.DataFrame({"Revenue": dec["Observed (Y_t)"], "Expected (T_t + S_t)": expected,
                          "Noise (e_t)": dec["Noise (e_t)"]})
    table = table.loc[dec["Noise (e_t)"].abs().nlargest(n).index].round(0)
    table.index = table.index.strftime("%a %d %b %Y")
    return table


# ---------------------------------------------------------------- 5.3 Seasonality: hourly

def _per_hour(df, n_days, **pivot):
    """Average per day for each opening hour."""
    return (df.pivot_table(index="Hour", aggfunc="sum", **pivot)
            .reindex(HOUR_ORDER, fill_value=0) / n_days)


def plot_hourly(df_copy, df_orders, beverages, n_days):
    orders = (df_orders.groupby("Hour").size() / n_days).reindex(HOUR_ORDER, fill_value=0)
    main_categories = df_copy[df_copy["Category"].isin(["FOOD", "BEVERAGE", "TOBACCO", "LIQUOR"])]
    categories = _per_hour(main_categories, n_days, columns="Category", values="Quantity")
    drinks = _per_hour(beverages[beverages["Drink type"] != "Other"], n_days,
                       columns="Drink type", values="Quantity")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].bar(HOUR_LABELS, orders.values, color="steelblue")
    axes[0].set_title("Average orders per hour")
    categories.set_axis(HOUR_LABELS).plot(ax=axes[1], marker="o")
    axes[1].set_title("Average units sold per hour by category")
    drinks.set_axis(HOUR_LABELS).plot(ax=axes[2], marker="o", color=["tab:blue", "tab:red"])
    axes[2].set_title("Average hot vs cold drinks per hour")
    for ax in axes:
        ax.set_xlabel("Hour of the day")
        ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plt.show()


def hourly_summary(df_orders, n_days):
    orders = (df_orders.groupby("Hour").size() / n_days).reindex(HOUR_ORDER, fill_value=0)
    return pd.Series({
        "Busiest hour": f"{orders.idxmax():02d}h ({orders.max():.1f} orders per day)",
        "Orders before 15h": f"{orders.loc[10:14].sum() / orders.sum():.0%}",
        "Orders from 19h until closing": f"{orders.loc[EVENING].sum() / orders.sum():.0%}",
        "Orders after midnight": f"{(df_orders['Hour'] < 6).mean():.0%}",
    }).to_frame("Value")


# ---------------------------------------------------------------- 5.3 Seasonality: weekly

def weekday_table(daily, df_orders, dec):
    """Average revenue, orders, order value and seasonal effect per weekday (calendar day)."""
    days_per_weekday = daily.groupby(daily.index.day_of_week).size()
    seasonal = dec["Seasonality (S_t)"]
    table = pd.DataFrame({
        "Avg revenue": daily.groupby(daily.index.day_of_week).mean(),
        "Avg orders": df_orders.groupby(df_orders["Date"].dt.day_of_week).size() / days_per_weekday,
        "Seasonal effect (S_t)": seasonal.groupby(seasonal.index.day_of_week).first(),
    })
    table["Avg order value"] = table["Avg revenue"] / table["Avg orders"]
    table.index = DAY_NAMES
    return table.round(0)


def plot_weekly(daily, df_orders):
    days_per_weekday = daily.groupby(daily.index.day_of_week).size()
    avg_revenue = daily.groupby(daily.index.day_of_week).mean()
    heat = (df_orders.pivot_table(index=df_orders["Date"].dt.day_of_week, columns="Hour",
                                  values="Bill Number", aggfunc="count")
            .reindex(columns=HOUR_ORDER).fillna(0).div(days_per_weekday, axis=0))
    heat.index, heat.columns = DAY_NAMES, HOUR_LABELS

    fig, axes = plt.subplots(1, 2, figsize=(18, 5), gridspec_kw={"width_ratios": [1, 2]})
    axes[0].bar(DAY_NAMES, avg_revenue / 1000, color=["steelblue"] * 4 + ["darkorange"] * 3)
    axes[0].axhline(daily.mean() / 1000, color="gray", linestyle="--", label="Average day")
    axes[0].set_title("Average revenue per weekday")
    axes[0].set_ylabel("Revenue (thousands)")
    axes[0].legend()
    sns.heatmap(heat, cmap="YlOrRd", ax=axes[1], cbar_kws={"label": "Orders per day"})
    axes[1].grid(False)   # no grid lines across the heatmap cells
    axes[1].set_title("Average orders per weekday and hour")
    plt.tight_layout()
    plt.show()


def business_day_revenue(df_orders, daily):
    """Revenue per business day: orders placed before 06h count for the previous evening
    (an order on Saturday at 01:00 is really Friday night)."""
    business_day = df_orders["Date"] - pd.to_timedelta((df_orders["Hour"] < 6).astype(int), unit="D")
    return df_orders.groupby(business_day)["Total"].sum().loc[daily.index.min():]


def calendar_vs_business_day(daily, business_daily):
    table = pd.DataFrame({
        "Avg revenue (calendar day)": daily.groupby(daily.index.day_of_week).mean(),
        "Avg revenue (business day)": business_daily.groupby(business_daily.index.day_of_week).mean(),
    })
    table.index = DAY_NAMES
    return table.round(0)


# ---------------------------------------------------------------- 5.3 Seasonality: monthly

def drinks_per_month(beverages):
    """Hot and cold drinks sold per month and the share of cold drinks."""
    hot_cold = beverages[beverages["Drink type"] != "Other"]
    table = hot_cold.pivot_table(index=hot_cold["Date"].dt.to_period("M"), columns="Drink type",
                                 values="Quantity", aggfunc="sum")
    table["Cold share"] = table["Cold"] / (table["Cold"] + table["Hot"])
    return table


def plot_drinks_per_month(table):
    labels = table.index.strftime("%b %y")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(labels, table["Hot"], marker="o", color="tab:red", label="Hot")
    axes[0].plot(labels, table["Cold"], marker="o", color="tab:blue", label="Cold")
    axes[0].set_title("Hot vs cold drinks sold per month")
    axes[0].set_ylabel("Units sold")
    axes[0].legend()
    axes[1].bar(labels, table["Cold share"] * 100, color="tab:blue")
    axes[1].axhline(50, color="gray", linestyle="--")
    axes[1].set_title("Cold drinks as % of hot + cold drinks")
    axes[1].set_ylabel("%")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- 5.4 Stationarity

def differenced_series(daily):
    """Original series, first difference Y_t - Y_(t-1), weekly difference Y_t - Y_(t-7)
    and the weekly difference of log revenue (log first, so the spread doesn't grow with the level)."""
    return {
        "Original (Y_t)": daily,
        "First difference (Y_t - Y_t-1)": daily.diff(),
        "Weekly difference (Y_t - Y_t-7)": daily.diff(7),
        "Log weekly difference (log Y_t - log Y_t-7)": np.log(daily).diff(7),
    }


def plot_differencing(series):
    fig, axes = plt.subplots(len(series), 1, figsize=(14, 3.3 * len(series)), sharex=True)
    for ax, (name, s) in zip(axes, series.items()):
        scale, unit = (1, "Log points") if name.startswith("Log") else (1000, "Thousands")
        ax.plot(s.index, s / scale, color="lightgray", label=name)
        ax.plot(s.index, s.rolling(30).mean() / scale, color="red", label="30-day rolling mean")
        ax.plot(s.index, s.rolling(30).std() / scale, color="steelblue", label="30-day rolling std")
        ax.set_ylabel(unit)
        ax.set_title(name)
        ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def stationarity_check(s):
    """A stationary series keeps the same mean and spread over time: compare both halves,
    plus the Augmented Dickey-Fuller test (H0: the series has a unit root, i.e. is NOT stationary)."""
    s = s.dropna()
    first, second = s.iloc[: len(s) // 2], s.iloc[len(s) // 2:]
    return pd.Series({"Mean 1st half": first.mean(), "Mean 2nd half": second.mean(),
                      "Std 1st half": first.std(), "Std 2nd half": second.std(),
                      "Std ratio (2nd / 1st half)": second.std() / first.std(),
                      "Autocorrelation lag 1": s.autocorr(1), "Autocorrelation lag 7": s.autocorr(7),
                      "ADF p-value": adfuller(s, autolag="AIC")[1]})


def stationarity_table(series):
    return pd.DataFrame({name: stationarity_check(s) for name, s in series.items()}).round(2)
