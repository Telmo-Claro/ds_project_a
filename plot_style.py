"""Shared plot style, applied once when any plotting module is imported.

The light gray grid is on by default. Two ways to change it in the notebook:
- set_grid(False): turn the grid off for every plot from now on (set_grid(True) turns it back on)
- with grid(False): turn it off (or on) for the plots in that block only, e.g.
      with ps.grid(False):
          ds.plot_category_sales(category_sales, category_revenue)
"""
import matplotlib.pyplot as plt

# Light gray grid behind every plot (axisbelow keeps it under the bars and lines)
plt.rcParams.update({"grid.color": "lightgray", "grid.linewidth": 0.6, "axes.axisbelow": True})


def set_grid(on=True):
    """Turn the grid on or off for all the plots that follow."""
    plt.rcParams["axes.grid"] = on


def grid(on=True):
    """Turn the grid on or off only inside a `with` block; afterwards the default comes back."""
    return plt.rc_context({"axes.grid": on})


set_grid(True)
