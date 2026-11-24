import os
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd


def print_apriori_explanation() -> None:
    """Prints a clear English explanation of the Apriori algorithm and MBA metrics."""
    print("=" * 82)
    print("APRIORI MARKET BASKET ANALYSIS (MBA) - METHODOLOGY & DECISION GUIDE")
    print("=" * 82)
    print(
        """
1. DATASET SELECTION & MBA FEATURE ENGINEERING:
   - Selected dataset: 'cafe_clean_lines.csv' (line-item level: 145,052 rows across
     69,982 unique bills) instead of 'cafe_clean_orders.csv' (aggregated order level).
   - Reason: Designing product-level combo deals requires knowing which individual
     items appear together on the same 'Bill Number'.
   - MBA-specific transformations:
     * Grouped items on 'Base product' to consolidate minor naming variants.
     * Deduplicated items per 'Bill Number' (binary presence: 1 if present on bill)
       so ordering multiple units of the same item does not skew pair probabilities.
     * Removed 'ADD ON S' and 'MERCHANDISE' to eliminate non-bundle noise.
     * Segmented by 'Daypart' (5 time windows) to capture time-specific ordering habits.

2. ALGORITHM CHOICE (APRIORI):
   - Apriori uses breadth-first search and the Apriori principle (pruning):
     if a single item fails 'min_support', any pair containing that item is pruned early.
   - Chosen for high transparency, step-by-step explainability, and direct control
     over threshold tuning on ~69.7k cleaned transactions.

3. METRICS & DIRECTIONALITY:
   - Support(A, B)      = Bills containing both A and B / Total bills in segment
   - Confidence(A -> B) = Bills containing both A and B / Bills containing A
     * Direction matters: A = anchor/base item, B = add-on/side item.
   - Lift(A, B)         = Support(A, B) / (Support(A) * Support(B))
     * Lift > 1 : Positive association (purchased together more often than chance).
     * Lift = 1 : Independent.
     * Lift < 1 : Negative association.
"""
    )
    print("=" * 82 + "\n")


def run_apriori_pairs(
    df_lines: pd.DataFrame,
    item_col: str = "Base product",
    min_support: float = 0.003,
    min_lift: float = 1.15,
    min_bills_floor: int = 15,
) -> pd.DataFrame:
    """Runs Apriori pruning and association rule metrics for product pairs."""
    n_bills = df_lines["Bill Number"].nunique()
    if n_bills == 0:
        return pd.DataFrame()

    min_count = max(min_bills_floor, int(np.ceil(min_support * n_bills)))

    # Build transaction baskets: unique items per bill
    baskets = (
        df_lines.groupby("Bill Number")[item_col]
        .apply(lambda s: sorted(set(s.dropna())))
        .tolist()
    )

    # Step 1: L1 (frequent single items)
    item_counts = Counter()
    for items in baskets:
        for it in items:
            item_counts[it] += 1

    frequent_items = {it for it, cnt in item_counts.items() if cnt >= min_count}

    # Step 2: C2 -> L2 (pairs built strictly from frequent items)
    pair_counts = Counter()
    for items in baskets:
        freq_in_bill = [it for it in items if it in frequent_items]
        if len(freq_in_bill) >= 2:
            for a, b in combinations(freq_in_bill, 2):
                pair_counts[(a, b)] += 1

    # Product metadata (dominant category and average unit price)
    meta = df_lines.groupby(item_col).agg(
        category=("Category", lambda s: s.mode()[0] if not s.mode().empty else "UNKNOWN"),
        avg_rate=("Rate", "mean"),
    )

    rows = []
    for (a, b), cnt in pair_counts.items():
        if cnt < min_count:
            continue

        supp_a = item_counts[a] / n_bills
        supp_b = item_counts[b] / n_bills
        supp_ab = cnt / n_bills

        conf_a_to_b = cnt / item_counts[a]
        conf_b_to_a = cnt / item_counts[b]
        lift = supp_ab / (supp_a * supp_b)

        if lift < min_lift:
            continue

        cat_a, cat_b = meta.loc[a, "category"], meta.loc[b, "category"]
        cross_cat = cat_a != cat_b
        price_sum = float(meta.loc[a, "avg_rate"] + meta.loc[b, "avg_rate"])

        # Define logical direction: higher-volume item as anchor -> add-on/side item
        if item_counts[a] >= item_counts[b]:
            anchor_item, addon_item = a, b
            conf_anchor_to_addon = conf_a_to_b * 100
        else:
            anchor_item, addon_item = b, a
            conf_anchor_to_addon = conf_b_to_a * 100

        rows.append(
            {
                "item_a": a,
                "item_b": b,
                "anchor_product": anchor_item,
                "addon_product": addon_item,
                "cat_a": cat_a,
                "cat_b": cat_b,
                "cross_category": cross_cat,
                "bills_together": cnt,
                "total_bills": n_bills,
                "support_pct": round(supp_ab * 100, 2),
                "conf_a_to_b_pct": round(conf_a_to_b * 100, 1),
                "conf_b_to_a_pct": round(conf_b_to_a * 100, 1),
                "conf_anchor_to_addon_pct": round(conf_anchor_to_addon, 1),
                "max_conf_pct": round(max(conf_a_to_b, conf_b_to_a) * 100, 1),
                "lift": round(lift, 2),
                "sum_rate": round(price_sum, 2),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["lift", "bills_together"], ascending=[False, False]).reset_index(drop=True)
    return out


def main() -> None:
    print_apriori_explanation()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "cafe_clean_lines.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"File not found at: {csv_path}. Please verify that 'cafe_clean_lines.csv' is inside the 'data' folder."
        )

    df = pd.read_csv(csv_path)
    df_clean = df[
        df["Base product"].notna()
        & (df["Base product"] != "ADD ON S")
        & (df["Category"] != "MERCHANDISE")
    ].copy()

    # 1. Overall Apriori analysis
    overall = run_apriori_pairs(df_clean, min_support=0.003, min_lift=1.15, min_bills_floor=25)
    print(f"Total overall rules found: {len(overall)}")

    # 2. Daypart Apriori analysis (5 dayparts)
    daypart_order = [
        "Late morning (10-12h)",
        "Lunch (12-15h)",
        "Afternoon (15-19h)",
        "Evening (19-23h)",
        "Night (23-03h)",
    ]
    daypart_list = []
    for dp in daypart_order:
        grp = df_clean[df_clean["Daypart"] == dp]
        res = run_apriori_pairs(grp, min_support=0.004, min_lift=1.15, min_bills_floor=10)
        if not res.empty:
            res.insert(0, "daypart", dp)
            res["rank_in_daypart"] = range(1, len(res) + 1)
            res["recommendation_tag"] = np.where(res["rank_in_daypart"] == 1, "BEST DEAL", "Runner-up")
            daypart_list.append(res)

    by_daypart = pd.concat(daypart_list, ignore_index=True) if daypart_list else pd.DataFrame()
    print(f"Total daypart rules found: {len(by_daypart)}\n")

    print("=== TOP 3 COMBO CANDIDATES PER DAYPART (RANK #1 = BEST DEAL) ===")
    if not by_daypart.empty:
        top_dp = by_daypart[by_daypart["rank_in_daypart"] <= 3]
        print(
            top_dp[
                [
                    "daypart",
                    "recommendation_tag",
                    "anchor_product",
                    "addon_product",
                    "bills_together",
                    "support_pct",
                    "conf_anchor_to_addon_pct",
                    "lift",
                    "sum_rate",
                ]
            ].to_string(index=False)
        )
        print()

    # 3. Export clean English CSVs for notebook and reporting
    out_overall = os.path.join(base_dir, "apriori_combideals_overall_en.csv")
    out_daypart = os.path.join(base_dir, "apriori_combideals_per_daypart_en.csv")

    overall.to_csv(out_overall, index=False, encoding="utf-8-sig")
    if not by_daypart.empty:
        by_daypart.to_csv(out_daypart, index=False, encoding="utf-8-sig")

    print("Export completed:")
    print("-", out_overall)
    print("-", out_daypart)


if __name__ == "__main__":
    main()