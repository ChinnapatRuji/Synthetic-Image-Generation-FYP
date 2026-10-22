# Charts for an evaluation results CSV made by evaluation.py.
# Usage:  python "Evaluation agent/visualize_results.py" [path/to/results.csv]
# Saves PNGs next to the CSV: <name>_pass_rate.png, <name>_confidence.png, <name>_confusion.png

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR / "results" / "img2img_eval.csv"
df = pd.read_csv(csv_path)
name = csv_path.stem

# Generated folder -> class label shown on the charts (same order as the TE dataset classes)
LABELS = {
    "ok": "OK",
    "ng_misalign": "NG-Misalign",
    "ng_no_solder": "NG-No solder",
    "ng_single": "NG-Single electrode",
}
PRED_LABELS = {"OK": "OK", "NG-Misalign": "NG-Misalign",
               "NG-No_solder": "NG-No solder", "NG-Single_electrode": "NG-Single electrode"}
ORDER = [LABELS[k] for k in LABELS if k in set(df["intended"])]
df["class"] = df["intended"].map(LABELS).fillna(df["intended"])
df["predicted_label"] = df["predicted"].map(PRED_LABELS).fillna(df["predicted"])
df["outcome"] = np.where(df["pass"], "pass", "fail")

# Palette (light): outcome = status colours (always shown with a label), text inks, recessive grid
OUTCOMES = {  # outcome -> (legend label, fill, text colour on the fill)
    "pass": ("Pass", "#0ca30c", "#ffffff"),
    "fail": ("Fail (other class)", "#d03b3b", "#ffffff"),
}
SURFACE = "#fcfcfb"
INK, INK_2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 11, "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": INK_2, "axes.edgecolor": BASE,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 200,
})


def tidy(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def save(fig, suffix):
    path = csv_path.with_name(f"{name}_{suffix}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


overall = df["pass"].mean()
n_total, n_pass = len(df), int(df["pass"].sum())

# 1. Outcome per class: 100% stacked bars (pass / fail), with average confidence at the end of each bar
counts = pd.crosstab(df["class"], df["outcome"]).reindex(index=ORDER, columns=list(OUTCOMES), fill_value=0)
shares = counts.div(counts.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(9, 0.75 * len(ORDER) + 1.6))
y = np.arange(len(ORDER))[::-1]
left = np.zeros(len(ORDER))
for outcome, (label, fill, text) in OUTCOMES.items():
    vals = shares[outcome].values
    ax.barh(y, vals, left=left, height=0.55, color=fill, label=label,
            edgecolor=SURFACE, linewidth=2)  # 2px surface gap between segments
    for yi, l, v, n in zip(y, left, vals, counts[outcome].values):
        if v >= 7:  # label only segments wide enough to hold it
            ax.text(l + v / 2, yi, f"{v:.0f}%" if v < 14 else f"{v:.0f}% ({n})",
                    ha="center", va="center", color=text, fontsize=9)
    left += vals
mean_conf = df.groupby("class")["confidence"].mean().reindex(ORDER)
for yi, c in zip(y, mean_conf.values):
    ax.text(102, yi, f"{c:.2f}", va="center", color=INK, fontsize=10, clip_on=False)
ax.text(102, len(ORDER) - 0.45, "avg conf.", va="center", color=MUTED, fontsize=8, clip_on=False)
ax.set_yticks(y, ORDER)
ax.set_xlim(0, 100)
ax.set_xticks(range(0, 101, 25), [f"{t}%" for t in range(0, 101, 25)])
tidy(ax)
ax.spines["bottom"].set_visible(False)
present = [o for o in OUTCOMES if counts[o].sum()]
handles = [h for h, o in zip(ax.get_legend_handles_labels()[0], OUTCOMES) if o in present]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=4, frameon=False, fontsize=9)
summary = f"{int(counts['fail'].sum())} fail  ·  average confidence {df['confidence'].mean():.2f}"
fig.suptitle(f"{n_pass} of {n_total} generated images passed ({overall:.0%})",
             x=0.02, y=0.99, ha="left", fontsize=13, fontweight="bold")
ax.set_title(f"Outcome by intended class  ·  {summary}",
             loc="left", color=INK_2, fontsize=10, pad=12)
save(fig, "pass_rate")

# 2. Confidence of every prediction, one row per class, coloured by pass / fail
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(9, 0.75 * len(ORDER) + 1.6))
for yi, cls in zip(y, ORDER):
    sub = df[df["class"] == cls]
    jitter = rng.uniform(-0.18, 0.18, len(sub))
    for outcome, (label, fill, _) in OUTCOMES.items():
        m = (sub["outcome"] == outcome).values
        ax.scatter(sub["confidence"].values[m], yi + jitter[m], s=36, color=fill, alpha=0.85,
                   edgecolors=SURFACE, linewidths=1, label=label if yi == y[0] else None, zorder=3)
    med = sub["confidence"].median()
    ax.plot([med, med], [yi - 0.3, yi + 0.3], color=INK, linewidth=2, zorder=4)
ax.set_yticks(y, ORDER)
ax.set_ylim(-0.6, len(ORDER) - 0.4)
ax.set_xlim(0, 1.02)
ax.set_xticks(np.linspace(0, 1, 5), [f"{t:.2f}" for t in np.linspace(0, 1, 5)])
ax.set_xlabel("Classifier confidence in its predicted class")
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
tidy(ax)
handles, labels = ax.get_legend_handles_labels()
handles.append(plt.Line2D([], [], color=INK, linewidth=2))
labels.append("Median")
ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=3, frameon=False, fontsize=9)
ax.set_title("Confidence of each prediction", loc="left", fontsize=13, fontweight="bold")
save(fig, "confidence")

# 3. Confusion matrix: intended class (rows) vs what the classifier predicted (columns)
cols = ORDER
cm = pd.crosstab(df["class"], df["predicted_label"]).reindex(index=ORDER, columns=cols, fill_value=0)
fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 1.2, 5.2))
cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
ax.imshow(cm.values, cmap=cmap, vmin=0, vmax=cm.values.max())
for i in range(len(ORDER)):
    for j in range(len(cols)):
        v = cm.values[i, j]
        ax.text(j, i, str(v), ha="center", va="center", fontsize=12,
                color="#ffffff" if v > cm.values.max() * 0.55 else (INK if v else MUTED))
ax.set_xticks(range(len(cols)), cols, rotation=25, ha="right")
ax.set_yticks(range(len(ORDER)), ORDER)
ax.set_xlabel("Predicted by classifier")
ax.set_ylabel("Intended (generated as)")
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_title("Where the misclassifications go", loc="left", fontsize=13, fontweight="bold")
save(fig, "confusion")
