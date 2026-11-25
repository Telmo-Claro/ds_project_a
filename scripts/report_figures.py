"""Charts for the written report that the notebook doesn't draw itself.

Part 2 (market basket analysis) only prints tables, so this script reads the
rule files exported by the notebook (outputs/) and plots them.
Run from the project root:  python scripts/report_figures.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style  # noqa: F401  (same look as the notebook plots)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "report", "figures")
DAYPART_ORDER = ["Late morning (10-12h)", "Lunch (12-15h)", "Afternoon (15-19h)",
                 "Evening (19-23h)", "Night (23-03h)"]


def short(name, n=30):
    name = name.title()
    return name if len(name) <= n else name[: n - 1] + "…"


def plot_overall_rules(overall, n=10):
    top = overall.head(n).iloc[::-1]
    labels = [f"{short(a)} + {short(b)}" for a, b in zip(top["anchor_product"], top["addon_product"])]
    colors = ["#2a6496" if c else "#9aa5b1" for c in top["cross_category"]]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(labels, top["lift"], color=colors)
    ax.axvline(1, color="black", lw=0.8, ls="--")
    for y, (lift, bills) in enumerate(zip(top["lift"], top["bills_together"])):
        ax.text(lift + 0.03, y, f"lift {lift:.2f}  ({bills} bills)", va="center", fontsize=8)
    ax.set_xlim(0, top["lift"].max() * 1.35)
    ax.set_xlabel("Lift (1 = bought together only by chance)")
    ax.set_title(f"Top {n} product pairs of the year by lift\n(blue = pair mixes two categories)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_daypart_rules(by_daypart, n=2):
    top = by_daypart[by_daypart["rank_in_daypart"] <= n].copy()
    top["daypart"] = pd.Categorical(top["daypart"], DAYPART_ORDER, ordered=True)
    top = top.sort_values(["daypart", "rank_in_daypart"], ascending=[False, False])
    labels = [f"{dp.split(' (')[0]}\n{short(a)} + {short(b)}"
              for dp, a, b in zip(top["daypart"], top["anchor_product"], top["addon_product"])]
    colors = ["#2a6496" if r == 1 else "#9aa5b1" for r in top["rank_in_daypart"]]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(labels, top["lift"], color=colors)
    ax.axvline(1, color="black", lw=0.8, ls="--")
    for y, (lift, bills) in enumerate(zip(top["lift"], top["bills_together"])):
        ax.text(lift + 0.05, y, f"lift {lift:.2f}  ({bills} bills)", va="center", fontsize=8)
    ax.set_xlim(0, top["lift"].max() * 1.35)
    ax.set_xlabel("Lift")
    ax.set_title("Top 2 pairs per part of the day\n(blue = rank 1, grey = runner-up)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    overall = pd.read_csv(os.path.join(ROOT, "outputs", "apriori_combideals_overall_en.csv"))
    by_daypart = pd.read_csv(os.path.join(ROOT, "outputs", "apriori_combideals_per_daypart_en.csv"))
    for name, fig in [("mba_overall_rules.png", plot_overall_rules(overall)),
                      ("mba_daypart_rules.png", plot_daypart_rules(by_daypart))]:
        path = os.path.join(OUT_DIR, name)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
