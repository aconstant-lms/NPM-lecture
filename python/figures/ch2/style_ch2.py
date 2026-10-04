"""Common matplotlib style of the figures of Chapter 2.
Imported by the fig_*.py scripts of this folder: sets the fonts, defines the
colours BLUE, ORANGE, GREEN, GRAY and the output folder OUT."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "dejavusans",
})
BLUE, ORANGE, GREEN, GRAY = "#1F5AC8", "#D9822B", "#14963C", "#777777"
# Output folder, relative to python/figures/ch2 (run the scripts from there).
OUT = "../../../figures/ch2/"
