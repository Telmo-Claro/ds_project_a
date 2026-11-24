"""Steps 2-3 - Loading and cleaning the café data.

Every function takes a DataFrame and returns a cleaned copy, so the notebook
can show the cleaning one step at a time and explain why each step is needed.
"""
import pandas as pd

MONEY_COLS = ["Rate", "Tax", "Discount", "Total"]
COLUMN_ORDER = ["Date", "Bill Number", "Item Desc", "Time", "Quantity", "Rate", "Tax", "Discount", "Total", "Category"]


def load_raw(path="cafe.csv"):
    """Read the raw export. It uses ';' as separator and is not UTF-8 (it contains
    bytes like 0xE1 in some item names), so we read it as latin-1."""
    return pd.read_csv(path, delimiter=";", encoding="latin-1")


# ---------------------------------------------------------------- Formatting

def strip_text(df):
    """Remove leading/trailing spaces from column names and text columns
    ("Bill Number " and the category "LIQUOR " have a trailing space)."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    for col in ["Bill Number", "Item Desc", "Category"]:
        df[col] = df[col].str.strip()
    return df


def convert_types(df):
    """Money columns use ',' as decimal separator -> floats; Date -> datetime."""
    df = df.copy()
    for col in MONEY_COLS:
        df[col] = pd.to_numeric(df[col].str.replace(",", ".", regex=False), errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%y")
    return df


def duplicate_rows(df):
    """All rows that appear more than once, sorted so the copies are next to each other."""
    dupes = df[df.duplicated(keep=False)]
    return dupes.sort_values(by=list(df.columns))


# ---------------------------------------------------------------- Item names

SAME_PRODUCT = {
    "MORROCAN SOY KEBABS": "MOROCCAN SOY KEBABS",   # spelling typo, renamed with the new menu
    "STELLA ARTOIS": "STELLA ARTOIS (GLS)",         # same beer, same price
}


def fix_item_names(df):
    """Junk characters, double spaces and different names for the same product."""
    df = df.copy()
    name = df["Item Desc"]
    name = name.str.replace(r"[^\x20-\x7E]", "", regex=True)          # junk characters, e.g. "SANGRIA (GLS)ááá"
    name = name.str.replace(r"\s+", " ", regex=True).str.strip()       # "N R G  HOOKAH" -> "N R G HOOKAH"
    df["Item Desc"] = name.replace(SAME_PRODUCT)
    return df


# ---------------------------------------------------------------- Categories

CATEGORY_TYPOS = {"LIQUOR & TPBACCO": "LIQUOR & TOBACCO"}

# Products filed in a category that doesn't fit, found by reading the item lists per category
WINES = ["RED SANGRIA (GLS)", "RED SANGRIA (CARAFE)", "WHITE SANGRIA (GLS)", "WHITE SANGRIA (CARAFE)",
         "ZINZI WHITE (GLS)", "ZINZI WHITE (BTL)", "ZINZI RED (GLS)",
         "LATE HARVEST SULA CHENIN (GLS)", "LATE HARVEST SULA CHENIN (BTL)", "J.PCHENET SPARKLING ROSE (BTL)"]
MISC_TO_CATEGORY = {
    # New products in December 2010 that the till booked as MISC
    "RED BULL SHEESHA": "TOBACCO", "MM SALSA SHEESHA": "TOBACCO", "PAAN SHEESHA": "TOBACCO",
    "ICE BERG SHEESHA": "TOBACCO", "GRAPPO SHEESHA": "TOBACCO",
    "KF DRAFT MUG": "LIQUOR", "GIN (SM)": "LIQUOR", "DARK RUM (SM)": "LIQUOR",
    "FISH FINGER": "FOOD", "ROAST CHICKEN SALAD": "FOOD",
    # Hookah flavours, beer deals and drinks that are always booked as MISC
    "ROSE FLAVORS SINGLE": "TOBACCO", "MANGO FLAVOUR SINGLE": "TOBACCO",
    "HOEGAARDEN GLS (2+1)": "LIQUOR", "HOEGAARDEN LTR MUGS (2+1)": "LIQUOR",
    "PEACH BULL": "BEVERAGE",
}


def main_category(df):
    """For products booked under more than one category: the category used most often."""
    counts = df.groupby(["Item Desc", "Category"]).size().reset_index(name="rows")
    multi = counts[counts.groupby("Item Desc")["Category"].transform("nunique") > 1]
    return multi.sort_values("rows").groupby("Item Desc")["Category"].last()


def fix_categories(df):
    """Fix the category typo and products filed in the wrong category."""
    df = df.copy()
    df["Category"] = df["Category"].replace(CATEGORY_TYPOS)

    # 1. Same product in several categories -> the category it is booked under most often
    #    (e.g. ORANGE ARRABIATA: 493x FOOD, 2x MISC). Mostly MISC bookings in December 2010.
    main = main_category(df)
    has_main = df["Item Desc"].isin(main.index)
    df.loc[has_main, "Category"] = df.loc[has_main, "Item Desc"].map(main)

    # 2. Milkshakes are drinks: 14 shakes were filed as FOOD, while other shakes are BEVERAGE
    shakes = (df["Category"] == "FOOD") & df["Item Desc"].str.contains("SHAKE")
    df.loc[shakes, "Category"] = "BEVERAGE"

    # 3. Wines filed as LIQUOR, TOBACCO or FOOD
    df.loc[df["Item Desc"].isin(WINES), "Category"] = "WINES"

    # 4. Products always booked as MISC that clearly belong to a category
    is_known = df["Item Desc"].isin(MISC_TO_CATEGORY)
    df.loc[is_known, "Category"] = df.loc[is_known, "Item Desc"].map(MISC_TO_CATEGORY)
    return df


def category_changes(before, after):
    """Units that moved from one category to another, to show what fix_categories did."""
    moved = pd.DataFrame({"Item Desc": after["Item Desc"], "From": before["Category"],
                          "To": after["Category"], "Quantity": after["Quantity"]})
    moved = moved[moved["From"] != moved["To"]]
    return (moved.groupby(["From", "To"]).agg(Products=("Item Desc", "nunique"), Units=("Quantity", "sum"))
            .sort_values("Units", ascending=False))


def category_units(df):
    return df.groupby("Category")["Quantity"].sum().sort_values(ascending=False)


# ---------------------------------------------------------------- Orders

def merge_repeated_items(df):
    """Merge rows of the same item in the same bill into one row
    (3 rows of "panini x1" -> 1 row of "panini x3")."""
    return (
        df.groupby(["Date", "Bill Number", "Item Desc", "Time", "Category"], as_index=False, sort=False)
        .agg(Quantity=("Quantity", "sum"), Rate=("Rate", "first"),
             Tax=("Tax", "sum"), Discount=("Discount", "sum"), Total=("Total", "sum"))
        [COLUMN_ORDER]
    )


def build_orders(df):
    """One row per order (bill): items bought, quantity and money summed."""
    return (
        df.groupby(["Bill Number", "Date", "Time"], as_index=False)
        .agg(Items=("Item Desc", lambda s: ", ".join(s)),
             Unique_Items=("Item Desc", "nunique"),
             Quantity=("Quantity", "sum"),
             Tax=("Tax", "sum"), Discount=("Discount", "sum"), Total=("Total", "sum"))
    )
