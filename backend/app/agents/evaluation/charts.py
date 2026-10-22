"""Charts for an evaluation run: pass/fail per class, confidence per image, confusion matrix."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in workers

import matplotlib.pyplot as plt
import numpy as np

# Light palette: pass/fail use status colours (always with a label), text inks, recessive grid
PASS, FAIL = "#0ca30c", "#d03b3b"
SURFACE = "#fcfcfb"
INK, INK_2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

STYLE = {
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 11, "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": INK_2, "axes.edgecolor": BASE,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 200,
}

CHARTS = ("pass_rate", "confidence", "confusion")


def _tidy(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def pass_rate_chart(rows, class_names, path):
    """100% stacked pass/fail bar per class, with average confidence at the end of each bar."""
    ids = [c for c in class_names if any(r["class_id"] == c for r in rows)]
    labels = [class_names[c] for c in ids]
    y = np.arange(len(ids))[::-1]
    n_pass = sum(r["pass"] for r in rows)

    fig, ax = plt.subplots(figsize=(9, 0.75 * len(ids) + 1.6))
    for yi, class_id in zip(y, ids):
        subset = [r for r in rows if r["class_id"] == class_id]
        passed = sum(r["pass"] for r in subset)
        share = {"pass": passed / len(subset) * 100, "fail": (len(subset) - passed) / len(subset) * 100}
        count = {"pass": passed, "fail": len(subset) - passed}
        left = 0
        for outcome, color in (("pass", PASS), ("fail", FAIL)):
            v = share[outcome]
            ax.barh(yi, v, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=2)
            if v >= 7:
                ax.text(left + v / 2, yi, f"{v:.0f}%" if v < 14 else f"{v:.0f}% ({count[outcome]})",
                        ha="center", va="center", color="#ffffff", fontsize=9)
            left += v
        mean = sum(r["confidence"] for r in subset) / len(subset)
        ax.text(102, yi, f"{mean:.2f}", va="center", color=INK, fontsize=10, clip_on=False)

    ax.text(102, len(ids) - 0.45, "avg conf.", va="center", color=MUTED, fontsize=8, clip_on=False)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 25), [f"{t}%" for t in range(0, 101, 25)])
    _tidy(ax)
    ax.spines["bottom"].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PASS), plt.Rectangle((0, 0), 1, 1, color=FAIL)],
              labels=["Pass", "Fail (other class)"], loc="upper left", bbox_to_anchor=(0, -0.08),
              ncol=2, frameon=False, fontsize=9)
    avg = sum(r["confidence"] for r in rows) / len(rows)
    fig.suptitle(f"{n_pass} of {len(rows)} generated images passed ({n_pass / len(rows):.0%})",
                 x=0.02, y=0.99, ha="left", fontsize=13, fontweight="bold")
    ax.set_title(f"Outcome by intended class  ·  {len(rows) - n_pass} fail  ·  average confidence {avg:.2f}",
                 loc="left", color=INK_2, fontsize=10, pad=12)
    return _save(fig, path)


def confidence_chart(rows, class_names, path):
    """One dot per image at its confidence, coloured pass/fail, with the median per class."""
    ids = [c for c in class_names if any(r["class_id"] == c for r in rows)]
    y = np.arange(len(ids))[::-1]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(9, 0.75 * len(ids) + 1.6))
    for yi, class_id in zip(y, ids):
        subset = [r for r in rows if r["class_id"] == class_id]
        conf = np.array([r["confidence"] for r in subset])
        ok = np.array([r["pass"] for r in subset])
        jitter = rng.uniform(-0.18, 0.18, len(subset))
        for mask, color, label in ((ok, PASS, "Pass"), (~ok, FAIL, "Fail (other class)")):
            ax.scatter(conf[mask], yi + jitter[mask], s=36, color=color, alpha=0.85, edgecolors=SURFACE,
                       linewidths=1, label=label if yi == y[0] else None, zorder=3)
        med = float(np.median(conf))
        ax.plot([med, med], [yi - 0.3, yi + 0.3], color=INK, linewidth=2, zorder=4)

    ax.set_yticks(y, [class_names[c] for c in ids])
    ax.set_ylim(-0.6, len(ids) - 0.4)
    ax.set_xlim(0, 1.02)
    ax.set_xticks(np.linspace(0, 1, 5), [f"{t:.2f}" for t in np.linspace(0, 1, 5)])
    ax.set_xlabel("Classifier confidence in its predicted class")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _tidy(ax)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK, linewidth=2))
    labels.append("Median")
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=3, frameon=False, fontsize=9)
    ax.set_title("Confidence of each prediction", loc="left", fontsize=13, fontweight="bold")
    return _save(fig, path)


def confusion_chart(rows, class_names, path):
    """Intended class (rows) vs predicted class (columns)."""
    rows_ids = [c for c in class_names if any(r["class_id"] == c for r in rows)]
    col_ids = [c for c in class_names if any(r["class_id"] == c or r["predicted_class_id"] == c for r in rows)]
    cm = np.array([[sum(r["class_id"] == a and r["predicted_class_id"] == b for r in rows)
                    for b in col_ids] for a in rows_ids])

    fig, ax = plt.subplots(figsize=(1.3 * len(col_ids) + 1.2, 5.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=max(cm.max(), 1))
    for i in range(len(rows_ids)):
        for j in range(len(col_ids)):
            v = cm[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=12,
                    color="#ffffff" if v > cm.max() * 0.55 else (INK if v else MUTED))
    ax.set_xticks(range(len(col_ids)), [class_names[c] for c in col_ids], rotation=25, ha="right")
    ax.set_yticks(range(len(rows_ids)), [class_names[c] for c in rows_ids])
    ax.set_xlabel("Predicted by classifier")
    ax.set_ylabel("Intended (generated as)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Where the misclassifications go", loc="left", fontsize=13, fontweight="bold")
    return _save(fig, path)


def make_charts(rows, class_names: dict[str, str], out_dir: Path):
    """Write all charts as PNGs into out_dir. class_names: class_id -> display name, in display order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        return [
            pass_rate_chart(rows, class_names, out_dir / "pass_rate.png"),
            confidence_chart(rows, class_names, out_dir / "confidence.png"),
            confusion_chart(rows, class_names, out_dir / "confusion.png"),
        ]
