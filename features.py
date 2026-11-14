"""Step 7 - Feature engineering: new columns at line level (df_copy) and order level,
built from what we found in Steps 3-6 and needed for Part 2 (basket analysis) and Part 3."""
import pandas as pd

import time_series as ts

NEW_MENU_START = pd.Timestamp("2011-01-07")   # first day of the new menu (Step 6 anomalies)

# Days that behave differently from a normal day (Step 5 noise and Step 6 anomalies)
SPECIAL_DAYS = {
    "2010-05-09": "Mother's Day",
    "2010-12-24": "Christmas Eve",
    "2010-12-25": "Christmas",
    "2010-12-31": "New Year's Eve",
    "2011-01-01": "New Year's Day",
    "2011-01-25": "Party booking",
    "2011-02-14": "Valentine's Day",
}

# Promotion type from the item name (checked in order: the first matching pattern wins)
PROMO_PATTERNS = {
    "Buy 1 get 1": r"^(?:1\+1|B1G1)",
    "Multi-buy": r"\d ?\+ ?\d",
    "Multi-pack": r"^[2-9] (?:RED BULL|AXE TWIST)",
    "Combo": r" \+ ",
}
# Parts of a promo name to remove to get the base product ("1+1 VLN CAB SAUV (GLS)" -> "VLN CAB SAUV (GLS)")
PROMO_PARTS = r"^(?:1\+1|B1G1) ?|^[2-9] (?=RED BULL|AXE TWIST)| ?\(?\d\+\d\)?$"
# Deals that use a short name: map them to the regular product they are a deal on
BASE_NAMES = {
    "KF 1/2 LITER": "KF DRAUGHT (1/2LTR)", "KF 1 LITER": "KF DRAUGHT (1LTR)", "KF 2 LITER": "KF DRAUGHT PITCHER (2LTR)",
    "RED BULL": "RED BULL ENERGY DRINK", "SCHNEIDER": "SCHNEIDER WEISSE", "STELLA 1LTR": "STELLA ARTOIS MUG (1 LTR)",
    "HOEGAARDEN GLS": "HOEGAARDEN (GLS)", "HOEGAARDEN LTR MUGS": "HOEGAARDEN MUG (1 LITRE)",
    "4SEASON CLAS SAUV(BTL)": "4 SEASONS CLAS SAUV(BTL)", "4SEASON CLAS SAUV(GLS)": "4 SEASONS CLAS SAUV(GLS",
    "4SEASON CLAS SYRAH(GLS)": "4 SEASONS CLAS SYRAH(GLS)",
}


# ---------------------------------------------------------------- Product features

def add_promo(df):
    df["Promo"] = "No promo"
    for promo, pattern in reversed(PROMO_PATTERNS.items()):   # reversed so the first pattern wins
        df.loc[df["Item Desc"].str.contains(pattern), "Promo"] = promo
    return df


def add_base_product(df):
    """The product without its deal. Combos are made of several products, so they keep their name."""
    is_combo = df["Promo"] == "Combo"
    df["Base product"] = df["Item Desc"]
    base = df.loc[~is_combo, "Item Desc"].str.replace(PROMO_PARTS, "", regex=True).str.strip()
    df.loc[~is_combo, "Base product"] = base.replace(BASE_NAMES)
    return df


def promo_summary(df):
    summary = df.groupby("Promo").agg(Products=("Item Desc", "nunique"), Units=("Quantity", "sum"),
                                      Revenue=("Total", "sum"),
                                      Lines_with_discount=("Discount", lambda s: (s > 0).sum()))
    summary["Revenue share"] = (summary["Revenue"] / summary["Revenue"].sum()).map("{:.1%}".format)
    summary["Revenue"] = summary["Revenue"].map("{:,.0f}".format)
    return summary.sort_values("Units", ascending=False)


def promo_products(df):
    promo = df[df["Promo"] != "No promo"]
    return promo.groupby(["Promo", "Item Desc"]).agg(Base_product=("Base product", "first"),
                                                     Units=("Quantity", "sum"))


def add_drink_type(df):
    """Hot / Cold / Other drink for beverages (same keywords as Step 5), 'Not a drink' for the rest."""
    labelled = ts.label_drinks(df)["Drink type"]
    df["Drink type"] = labelled.reindex(df.index).fillna("Not a drink")
    return df


# ---------------------------------------------------------------- Time features

def add_time_features(df):
    """Business day (orders before 06h count for the previous evening), its weekday,
    weekend flag, special day and menu period. `Hour` and `Daypart` come from Steps 5-6."""
    df["Business day"] = df["Date"] - pd.to_timedelta((df["Hour"] < 6).astype(int), unit="D")
    df["Weekday"] = df["Business day"].dt.day_name().str[:3]
    df["Weekend"] = df["Business day"].dt.day_of_week >= 4          # Fri, Sat and Sun nights
    df["Special day"] = df["Business day"].dt.strftime("%Y-%m-%d").map(SPECIAL_DAYS).fillna("No")
    df["Menu"] = (df["Date"] >= NEW_MENU_START).map({False: "Old menu", True: "New menu"})
    return df


def add_bill_series(df):
    """Alcohol is billed on a separate series ('N' bills, Step 6 anomalies)."""
    df["Bill series"] = df["Bill Number"].str[0].map({"G": "Main bill", "N": "Alcohol bill"})
    return df


def add_line_features(df):
    """All line-level features in one go."""
    for step in [add_promo, add_base_product, add_drink_type, add_time_features, add_bill_series]:
        df = step(df)
    return df


# ---------------------------------------------------------------- Order features

def build_order_features(df):
    """One row per order with its size, value, timing and which categories it contains (for Part 2)."""
    first = df.groupby("Bill Number")[["Business day", "Weekday", "Weekend", "Daypart", "Hour",
                                       "Special day", "Menu", "Bill series"]].first()
    orders = df.groupby("Bill Number").agg(Products=("Item Desc", "nunique"), Units=("Quantity", "sum"),
                                           Order_value=("Total", "sum"))
    contains = pd.crosstab(df["Bill Number"], df["Category"]).gt(0)
    contains = contains[["FOOD", "BEVERAGE", "TOBACCO", "LIQUOR", "WINES"]].add_prefix("Has ")
    contains["Has promo"] = df.groupby("Bill Number")["Promo"].apply(lambda s: (s != "No promo").any())
    return first.join(orders).join(contains)


# ---------------------------------------------------------------- Overview

FEATURES = pd.DataFrame([
    ("Hour", "line + order", "Hour of the order (0-23)", "Hourly cycle (Step 5)"),
    ("Daypart", "line + order", "Late morning / Lunch / Afternoon / Evening / Night", "Category mix changes through the day (Step 6)"),
    ("Business day", "line + order", "Date, but orders before 06h count for the previous evening", "A night out doesn't stop at midnight (Step 5)"),
    ("Weekday", "line + order", "Weekday of the business day", "Weekly cycle (Step 5)"),
    ("Weekend", "line + order", "True for Fri, Sat and Sun nights", "Strongest nights (Step 5, test in Step 6)"),
    ("Special day", "line + order", "Name of the holiday/event, or 'No'", "Largest surprises in the decomposition (Step 5)"),
    ("Menu", "line + order", "Old menu / New menu (from 7 Jan 2011)", "Menu change found in Step 6"),
    ("Bill series", "line + order", "Main bill (G) / Alcohol bill (N)", "Alcohol is billed separately (Step 6)"),
    ("Promo", "line", "Buy 1 get 1 / Multi-buy / Multi-pack / Combo / No promo", "Deals are only in the name, not in Discount (Step 3)"),
    ("Base product", "line", "Product name without its deal", "Compare a product with and without its deal (Part 3)"),
    ("Drink type", "line", "Hot / Cold / Other drink / Not a drink", "Hot drinks drive the seasons (Step 5)"),
    ("Products, Units, Order_value", "order", "Size and value of the order", "Basket size for Part 2"),
    ("Has FOOD ... Has promo", "order", "Which categories / deals the order contains", "Cross-category baskets for Part 2"),
], columns=["Feature", "Level", "What it is", "Why we need it"]).set_index("Feature")
