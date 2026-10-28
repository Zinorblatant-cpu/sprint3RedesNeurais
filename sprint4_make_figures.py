"""Sprint 4: figuras dos slides, a partir de outputs/sprint4/results.json (extraído do notebook executado no Colab)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "sprint4"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for f in ("segoeui.ttf", "segoeuib.ttf"):
    font_manager.fontManager.addfont(f"C:/Windows/Fonts/{f}")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_TRAIN, C_VAL = "#2a78d6", "#eb6834"
C_SEL, C_BASE, C_OTHER = "#4a3aa7", "#898781", "#b9b3e0"
BLOCK_FILL, HEAD_FILL, IO_FILL = "#dfe9f8", "#fbe4d8", "#f4f3ef"

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.grid": False,
})

R = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
M, CLASSES, SEL, BASE = R["models"], R["classes"], R["selected"], R["baseline"]
NAMES = list(M)


def n_(v):
    return f"{int(round(v)):,}".replace(",", ".")


def d_(v, k=1):
    return f"{v:.{k}f}".replace(".", ",")


def label(n):
    return n.replace("_baseline", " (baseline)")


def color(n):
    return C_SEL if n == SEL else C_BASE if n == BASE else C_OTHER


def save(fig, name, dpi=200):
    fig.savefig(FIG / name, dpi=dpi)
    plt.close(fig)


def box(ax, x, y, w, h, fill, title, body=None, title_size=9.5, body_size=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fill, ec=AXIS, lw=0.8))
    if body:
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=title_size, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.3, body, ha="center", va="center", fontsize=body_size, color=INK2, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size, fontweight="bold")


def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))


# ------------------------------------------------------------ slide 3: baseline
def fig_baseline():
    steps = [("Entrada", "224×224×3\n÷255, (x−μ)/σ", IO_FILL),
             ("Bloco 1", "Conv 3×3 · 16\nBN · ReLU\nMaxPool 2×2\n→ 112×112×16", BLOCK_FILL),
             ("Bloco 2", "Conv 3×3 · 32\nBN · ReLU\nMaxPool 2×2\n→ 56×56×32", BLOCK_FILL),
             ("Bloco 3", "Conv 3×3 · 64\nBN · ReLU\nMaxPool 2×2\n→ 28×28×64", BLOCK_FILL),
             ("GAP", "média de cada\nmapa 28×28\n→ 64", HEAD_FILL),
             ("Dense 64", "ReLU\nDropout 0,3", HEAD_FILL),
             ("Saída", "Dense 9\nsoftmax", "#dcf3ea")]
    fig, ax = plt.subplots(figsize=(9.4, 1.95))
    w, gap = 1.18, 0.32
    ax.set_xlim(0, len(steps) * (w + gap) - gap); ax.set_ylim(0, 1.6); ax.axis("off")
    for i, (t, b, c) in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.04), w, 1.52, boxstyle="round,pad=0,rounding_size=0.06", fc=c, ec=AXIS, lw=0.8))
        ax.text(x + w / 2, 1.34, t, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + w / 2, 0.66, b, ha="center", va="center", fontsize=8.2, color=INK2, linespacing=1.4)
        if i < len(steps) - 1:
            arrow(ax, x + w + 0.03, x + w + gap - 0.03, 0.8)
    fig.subplots_adjust(left=0.003, right=0.997, top=0.99, bottom=0.01)
    save(fig, "baseline.png")


# --------------------------------------------------------- slide 4: 4 arquiteturas
def fig_blocks():
    fig, ax = plt.subplots(figsize=(9.4, 3.0))
    row_h, bw, bh, gap = 0.74, 0.86, 0.52, 0.12
    ax.set_xlim(0, 10.6); ax.set_ylim(-0.05, len(NAMES) * row_h + 0.05); ax.axis("off")
    for r, n in enumerate(NAMES):
        m = M[n]
        y = (len(NAMES) - 1 - r) * row_h + (row_h - bh) / 2
        ax.text(0, y + bh / 2, label(n), va="center", fontsize=10, fontweight="bold", color=C_SEL if n == SEL else INK)
        x = 1.55
        box(ax, x, y, 0.62, bh, IO_FILL, "224²", title_size=8.5)
        x += 0.62
        side = 224
        for i, f in enumerate(m["filters"]):
            arrow(ax, x + 0.02, x + gap - 0.02, y + bh / 2)
            x += gap
            side //= 2
            box(ax, x, y, bw, bh, BLOCK_FILL, f"{f} filtros", f"→ {side}×{side}", title_size=8.6, body_size=7.8)
            x += bw
        arrow(ax, x + 0.02, x + gap - 0.02, y + bh / 2)
        x += gap
        box(ax, x, y, 1.05, bh, HEAD_FILL, "GAP → Dense 64", "→ 9 classes", title_size=8, body_size=7.8)
        ax.text(10.6, y + bh / 2 + 0.11, f"{n_(m['params'])} parâmetros", ha="right", va="center", fontsize=9,
                fontweight="bold", color=C_SEL if n == SEL else INK)
        ax.text(10.6, y + bh / 2 - 0.13, f"campo receptivo {m['receptive_field']} px", ha="right", va="center",
                fontsize=8.4, color=INK2)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    save(fig, "blocks.png")


# ---------------------------------------------------- slide 5: curvas Train × Val
def fig_curves():
    fig, axes = plt.subplots(2, len(NAMES), figsize=(9.6, 3.55), sharex=True, sharey="row")
    for j, n in enumerate(NAMES):
        h, best = M[n]["history"], M[n]["best_epoch"]
        ep = np.arange(1, len(h["loss"]) + 1)
        for i, (k, vk) in enumerate((("loss", "val_loss"), ("f1", "val_f1"))):
            ax = axes[i, j]
            ax.plot(ep, h[k], color=C_TRAIN, lw=1.7, label="Train")
            ax.plot(ep, h[vk], color=C_VAL, lw=1.7, label="Validation")
            ax.axvline(best, color=MUTED, lw=0.8, ls=":")
            ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=8)
        axes[0, j].set_title(f"{label(n)} · {M[n]['n_blocks']} blocos", fontsize=9.5, color=C_SEL if n == SEL else INK, pad=4)
        axes[1, j].set_xlabel("época", fontsize=8.5)
        axes[1, j].text(best - 0.6, 0.06, f"melhor: {best}", ha="right", fontsize=7.5, color=INK2)
    axes[0, 0].set_ylim(0, 2.2); axes[1, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("loss ponderada", fontsize=8.5); axes[1, 0].set_ylabel("F1 macro", fontsize=8.5)
    axes[1, 0].set_yticks([0, 0.25, 0.5, 0.75, 1]); axes[1, 0].set_yticklabels(["0", "0,25", "0,5", "0,75", "1"])
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right", handlelength=1.3)
    fig.subplots_adjust(left=0.062, right=0.995, top=0.92, bottom=0.12, wspace=0.08, hspace=0.18)
    save(fig, "curves.png")


# ---------------------------------------------- slide 6: comparação e por classe
def fig_compare():
    fig, ax = plt.subplots(figsize=(4.6, 4.5))
    f1 = np.array([M[n]["val"]["f1_macro"] for n in NAMES]) * 100
    acc = np.array([M[n]["val"]["accuracy"] for n in NAMES]) * 100
    x = np.arange(len(NAMES))
    ax.bar(x, f1, width=0.62, color=[color(n) for n in NAMES], zorder=2)
    ax.scatter(x, acc, s=30, color=INK, zorder=3, label="accuracy")
    for i, n in enumerate(NAMES):
        ax.text(i, f1[i] - 4.5, d_(f1[i]), ha="center", fontsize=9.5, fontweight="bold", color="white", zorder=4)
        if i:
            ax.text(i - 0.5, max(f1[i], acc[i]) + 4.5, f"+{d_(f1[i] - f1[i - 1])}", ha="center", fontsize=8.6, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{label(n).replace(' (baseline)', chr(10) + '(baseline)')}\n{n_(M[n]['params'])}" for n in NAMES], fontsize=8.4)
    ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Validation (%)", fontsize=8.8)
    ax.set_title("F1 macro (barra) e accuracy (ponto)", fontsize=10, pad=6)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.14)
    save(fig, "compare.png")


def fig_per_class():
    vals = np.array([M[n]["val"]["f1"] for n in NAMES]).T * 100          # classes × modelos
    fig, ax = plt.subplots(figsize=(4.4, 4.75))
    ax.imshow(vals, cmap="Purples", vmin=30, vmax=95, aspect="auto")
    for i in range(len(CLASSES)):
        for j in range(len(NAMES)):
            ax.text(j, i, d_(vals[i, j], 0), ha="center", va="center", fontsize=8.6,
                    color="white" if vals[i, j] > 70 else INK, fontweight="bold" if NAMES[j] == SEL else "normal")
    ax.set_xticks(range(len(NAMES))); ax.set_xticklabels([n.replace("_baseline", "") for n in NAMES], fontsize=8.8)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES, fontsize=8.6)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("F1 por classe no Validation (%)", fontsize=10, pad=22)
    fig.subplots_adjust(left=0.27, right=0.99, top=0.9, bottom=0.02)
    save(fig, "per_class.png")


if __name__ == "__main__":
    fig_baseline()
    fig_blocks()
    fig_curves()
    fig_compare()
    fig_per_class()
    print("figuras ok ->", FIG)
