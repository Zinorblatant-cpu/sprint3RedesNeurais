"""Gera as figuras PNG dos slides (lê outputs/stats.json, splits.csv, hist.npz e gaps.npy)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import dataprep  # noqa: E402

OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
DATASET = ROOT / "dataset"

for f in ("segoeui.ttf", "segoeuib.ttf"):
    font_manager.fontManager.addfont(f"C:/Windows/Fonts/{f}")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SPLIT_C = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
SPLIT_N = {"train": "Train", "val": "Validation", "test": "Test"}
C_SPECIES, C_NEG = "#4a3aa7", "#898781"   # espécie (violeta) vs. classe Negative (cinza)
SPLITS = ("train", "val", "test")

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.grid": False,
})

S = json.loads((OUT / "stats.json").read_text(encoding="utf-8"))
DF = pd.read_csv(OUT / "splits.csv")
H = np.load(OUT / "hist.npz")
GAPS = np.load(OUT / "gaps.npy")
CLASSES = S["counts"]["classes"]
SPECIES = [c for c in CLASSES if c != "Negative"]


def n_(v):
    return f"{int(round(v)):,}".replace(",", ".")


def d_(v, k=1):
    return f"{v:.{k}f}".replace(".", ",")


def kind_c(c):
    return C_NEG if c == "Negative" else C_SPECIES


def save(fig, name):
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


def clean(ax, grid_y=False):
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)
    if grid_y:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def img_path(row):
    return DATASET / row.split / row.folder / row.file


# ---------------------------------------------------------------- 1. exemplos
def fig_samples():
    rng = np.random.default_rng(3)
    tr = DF[DF.split == "train"]
    fig, axes = plt.subplots(3, 3, figsize=(5.3, 5.5))
    for ax, cls in zip(axes.ravel(), CLASSES):
        cand = tr[tr.species == cls]
        row = cand.iloc[int(rng.integers(len(cand)))]
        ax.imshow(Image.open(img_path(row)))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_edgecolor(kind_c(cls)); s.set_linewidth(3)
        ax.set_xlabel(cls, fontsize=9.5, color=INK, labelpad=3)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.05, wspace=0.07, hspace=0.24)
    save(fig, "samples.png")


# ------------------------------------------------------------ 3. balanceamento
def fig_balance():
    c = S["counts"]
    order = sorted(CLASSES, key=lambda k: -c["per_class"][k])
    fig = plt.figure(figsize=(12.3, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.13)

    ax = fig.add_subplot(gs[0])
    vals = [c["per_class"][k] for k in order]
    ax.bar(range(len(order)), vals, width=0.68, color=[kind_c(k) for k in order])
    for i, v in enumerate(vals):
        ax.text(i, v + 130, n_(v), ha="center", fontsize=8.5, fontweight="bold" if i == 0 else "normal")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([k.replace(" ", "\n") for k in order], fontsize=8)
    ax.set_ylim(0, 10400); ax.set_yticks([0, 3000, 6000, 9000]); ax.set_yticklabels(["0", "3.000", "6.000", "9.000"])
    ax.set_title(f"9 classes: Negative = {d_(c['negative_share'] * 100)}% ({d_(c['negative_vs_species_mean'])}× uma espécie)", pad=8)
    clean(ax, True)

    ax = fig.add_subplot(gs[1])
    sp = sorted(SPECIES, key=lambda k: -c["per_class"][k])
    v2 = [c["per_class"][k] for k in sp]
    ax.bar(range(len(sp)), v2, width=0.68, color=C_SPECIES)
    for i, v in enumerate(v2):
        ax.text(i, v + 20, n_(v), ha="center", fontsize=8.5)
    ax.set_xticks(range(len(sp))); ax.set_xticklabels([k.replace(" ", "\n") for k in sp], fontsize=7, rotation=28, ha="right")
    ax.set_ylim(0, 1450); ax.set_yticks([0, 500, 1000])
    ax.set_title(f"Só as 8 espécies: {n_(c['species_min'][1])} a {n_(c['species_max'][1])} ({d_(c['species_ratio'], 2)} : 1)", pad=8)
    clean(ax, True)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.88, bottom=0.17)
    save(fig, "balance.png")


# ------------------------------------------ 4. classes por split / stats de pixel
def fig_class_by_split():
    fig, ax = plt.subplots(figsize=(6.0, 3.35))
    h = 0.24
    for i, cls in enumerate(CLASSES):
        for j, s in enumerate(SPLITS):
            v = S["split"][s]["share_in_split"][cls] * 100
            y = i + (j - 1) * (h + 0.02)
            ax.barh(y, v, height=h, color=SPLIT_C[s], label=SPLIT_N[s] if i == 0 else None)
    neg = [S["split"][s]["share_in_split"]["Negative"] * 100 for s in SPLITS]
    same = max(neg) - min(neg) < 0.05
    ax.text(neg[0] + 0.8, len(CLASSES) - 1, f"{d_(neg[0])}% em cada" if same else " / ".join(d_(v) + "%" for v in neg),
            va="center", fontsize=8.5)
    sp_lo = min(S["split"][s]["share_in_split"][c] for s in SPLITS for c in SPECIES) * 100
    sp_hi = max(S["split"][s]["share_in_split"][c] for s in SPLITS for c in SPECIES) * 100
    ax.text(20, 3.5, f"cada espécie: {d_(sp_lo)}% a {d_(sp_hi)}%\nem qualquer conjunto", fontsize=9, va="center", color=INK2)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES, fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 66); ax.set_xlabel("% das imagens do conjunto", fontsize=9)
    ax.set_title("Composição de cada conjunto, por classe", pad=8)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.62), frameon=False, fontsize=9, handlelength=1, handleheight=1)
    fig.subplots_adjust(left=0.22, right=0.985, top=0.9, bottom=0.13)
    save(fig, "class_by_split.png")


def fig_pixel_stats():
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.35))
    for ax, key, title in ((axes[0], "mean_255", "Média por canal (0–255)"), (axes[1], "std_255", "Desvio-padrão por canal")):
        w = 0.26
        top = max(max(S["pixels"][s][key]) for s in SPLITS)
        for j, s in enumerate(SPLITS):
            v = S["pixels"][s][key]
            xs = np.arange(3) + (j - 1) * (w + 0.03)
            ax.bar(xs, v, width=w, color=SPLIT_C[s], label=SPLIT_N[s])
            for x, val in zip(xs, v):
                ax.text(x, val + top * 0.02, d_(val, 1), ha="center", va="bottom", fontsize=6.6, rotation=90)
        ax.set_xticks(range(3)); ax.set_xticklabels(["R", "G", "B"], fontsize=10.5)
        ax.set_ylim(0, top * 1.35); ax.set_title(title, pad=8, fontsize=10.5)
        clean(ax, True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.07, 1.0), ncol=3,
               frameon=False, fontsize=9.5, handlelength=1, handleheight=1, columnspacing=1.2)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.80, bottom=0.09, wspace=0.22)
    save(fig, "pixel_stats.png")


# ------------------------------------------------------- 5. histogramas RGB
def fig_rgb_hist():
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.15), gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.12, "wspace": 0.16})
    names = ["Canal R (vermelho)", "Canal G (verde)", "Canal B (azul)"]
    x = np.arange(64) * 4 + 2

    def norm(s, c):
        h = H[f"nat_{s}"][c].astype(float)
        return h.reshape(64, 4).sum(axis=1) / h.sum() * 100

    ymax = max(norm(s, c).max() for s in SPLITS for c in range(3)) * 1.12
    diffs = [np.abs(norm(s, c) - norm("train", c)).max() for s in ("val", "test") for c in range(3)]
    dmax, maxdiff = max(diffs) * 1.3, max(diffs)
    for c in range(3):
        top, bot = axes[0, c], axes[1, c]
        for s, lw in (("train", 3.0), ("val", 1.8), ("test", 1.1)):
            top.plot(x, norm(s, c), color=SPLIT_C[s], linewidth=lw, label=SPLIT_N[s])
        top.set_title(names[c], pad=6, fontsize=10.5)
        top.set_xlim(0, 255); top.set_ylim(0, ymax); top.set_xticks([0, 128, 255]); top.tick_params(labelbottom=False)
        clean(top, True)
        bot.axhline(0, color=SPLIT_C["train"], linewidth=2.2)
        for s in ("val", "test"):
            bot.plot(x, norm(s, c) - norm("train", c), color=SPLIT_C[s], linewidth=1.3)
        bot.set_xlim(0, 255); bot.set_ylim(-dmax, dmax); bot.set_xticks([0, 128, 255])
        clean(bot, True)
        bot.set_xlabel("intensidade do pixel", fontsize=9)
        if c == 0:
            top.set_ylabel("% dos pixels (faixas de 4 níveis)", fontsize=8.5)
            bot.set_ylabel("diferença vs Train\n(pontos percentuais)", fontsize=8.5)
        else:
            top.set_yticklabels([]); bot.set_yticklabels([])
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.975, 1.0), ncol=3,
               frameon=False, fontsize=9.5, handlelength=1.6, columnspacing=1.2)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.88, bottom=0.10)
    save(fig, "rgb_hist.png")
    return maxdiff


# ------------------------------------------ 6. câmeras, sessões e vazamento
def fig_acquisition():
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.3), gridspec_kw={"wspace": 0.25})
    ax = axes[0]
    inst = sorted(int(k) for k in S["raw"]["instruments"])
    w = 0.26
    for j, s in enumerate(SPLITS):
        vals = [S["instrument_by_split"][s].get(str(i), 0) * 100 for i in inst]
        ax.bar(np.arange(len(inst)) + (j - 1) * (w + 0.03), vals, width=w, color=SPLIT_C[s], label=SPLIT_N[s])
    ax.set_xticks(range(len(inst))); ax.set_xticklabels([f"câmera {i}" for i in inst], fontsize=8.5)
    ax.set_ylabel("% do conjunto"); ax.set_title(f"Câmeras: mesma mistura nos 3 conjuntos (p = {d_(S['chi2']['instrument']['p'], 2)})", pad=8, fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    clean(ax, True)

    ax = axes[1]
    order = sorted(CLASSES, key=lambda k: -S["leakage"]["sessions_per_class"][k])
    vals = [S["leakage"]["sessions_per_class"][k] for k in order]
    ax.barh(range(len(order)), vals, height=0.66, color=[kind_c(k) for k in order])
    for i, v in enumerate(vals):
        ax.text(v + 0.8, i, str(v), va="center", fontsize=8.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8.5); ax.invert_yaxis()
    ax.set_xlim(0, 62); ax.set_xlabel("sessões de captura (data + câmera)")
    ax.set_title(f"Sessões por classe (total: {S['raw']['n_sessions']})", pad=8, fontsize=10)
    clean(ax)

    ax = axes[2]
    g = np.clip(GAPS[np.isfinite(GAPS)], 0.5, 3000)
    ax.hist(g, bins=np.logspace(np.log10(0.5), np.log10(3000), 36), color=SPLIT_C["train"], edgecolor=SURFACE, linewidth=0.5)
    ax.set_xscale("log")
    med = S["leakage"]["median_gap_s"]
    ax.axvline(med, color=SPLIT_C["val"], lw=2)
    ax.text(med * 1.15, ax.get_ylim()[1] * 0.93, f"mediana: {n_(med)} s", fontsize=9, va="top")
    ax.set_xlabel("segundos até a foto de Train mais próxima", fontsize=9)
    ax.set_ylabel("nº de fotos de Val/Test")
    ax.set_title(f"{d_(S['leakage']['pct_same_session'] * 100)}% já têm foto de Train (mesma sessão e classe)", pad=8, fontsize=10)
    clean(ax, True)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.88, bottom=0.2)
    save(fig, "acquisition.png")


# --------------------------------------------------------- 7. estratégias
def fig_preprocess():
    tr = DF[DF.split == "train"].sort_values("file")
    picks = [tr[tr.species == "Lantana"].iloc[4], tr[tr.species == "Rubber vine"].iloc[9]]
    lost = S["size"]["crop_area_lost"]
    cols = ["Original\n256 × 256", "Center crop 224\n(Val/Test: escolhido)", "Resize 256 → 224\n(alternativa)", "Random crop + espelho\n(Train: augmentation)"]
    fig, axes = plt.subplots(2, 4, figsize=(7.3, 4.6))
    rng = np.random.default_rng(11)
    for r, row in enumerate(picks):
        img = Image.open(img_path(row)).convert("RGB")
        outs = [img, dataprep.center_crop(img, 224), img.resize((224, 224), Image.BICUBIC),
                ImageOps.mirror(dataprep.random_crop(img, 224, rng))]
        for c, o in enumerate(outs):
            ax = axes[r, c]
            ax.imshow(o)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_edgecolor(SPLIT_C["train"] if c == 1 else AXIS); s.set_linewidth(3.2 if c == 1 else 1)
        axes[r, 0].add_patch(Rectangle((16, 16), 224, 224, fill=False, edgecolor="#ffffff", lw=2.2, ls=(0, (4, 2))))
        axes[r, 0].set_xlabel(row.species, fontsize=8.5, labelpad=3)
        axes[r, 1].set_xlabel(f"perde {d_(lost * 100)}% da área", fontsize=8.5, labelpad=3)
        axes[r, 2].set_xlabel("0% perdido, reamostra (×0,875)", fontsize=8.5, labelpad=3)
        axes[r, 3].set_xlabel("posição sorteada a cada época", fontsize=8.5, labelpad=3)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.06, wspace=0.07, hspace=0.22)
    for c in range(4):
        pos = axes[0, c].get_position()
        fig.text(pos.x0 + pos.width / 2, 0.985, cols[c], ha="center", va="top", fontsize=8.6,
                 fontweight="bold" if c == 1 else "normal", linespacing=1.2)
    save(fig, "preprocess.png")


if __name__ == "__main__":
    fig_samples()
    fig_balance()
    fig_class_by_split()
    fig_pixel_stats()
    md = fig_rgb_hist()
    fig_acquisition()
    fig_preprocess()
    (OUT / "hist_maxdiff.json").write_text(json.dumps({"max_pp_diff": float(md)}))
    print("figuras ok; maior diferença de histograma (pp por faixa de 4 níveis):", round(md, 3))
