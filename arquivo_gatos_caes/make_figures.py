"""Gera as figuras PNG usadas nos slides (lê outputs/stats.json, splits.csv e hist.npz)."""
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
SP_C = {"gato": "#e87ba4", "cao": "#4a3aa7"}
SP_N = {"gato": "Gato", "cao": "Cão"}
SP_TXT = {"gato": INK, "cao": "#ffffff"}

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.grid": False,
})

S = json.loads((OUT / "stats.json").read_text(encoding="utf-8"))
DF = pd.read_csv(OUT / "splits.csv")
H = np.load(OUT / "hist.npz")


def n_(v):  # 5152 -> "5.152"
    return f"{int(v):,}".replace(",", ".")


def d_(v, k=1):  # 32.34 -> "32,3"
    return f"{v:.{k}f}".replace(".", ",")


def save(fig, name):
    fig.savefig(FIG / name, dpi=220)
    plt.close(fig)


def clean(ax, grid_y=False):
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)
    if grid_y:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def load_thumb(row, size=300):
    img = dataprep.load_rgb(DATASET / row.split / row.species / row.file)
    return dataprep.resize_center_crop(img, size)


# ---------------------------------------------------------------- 1. exemplos
def fig_samples():
    rng = np.random.default_rng(3)
    tr = DF[DF.split == "train"]
    picks = {
        "gato": ["Bengal", "Siamese", "Persian", "Maine_Coon", "Sphynx"],
        "cao": ["beagle", "pug", "samoyed", "boxer", "shiba_inu"],
    }
    fig, axes = plt.subplots(2, 5, figsize=(7.9, 4.35))
    for r, sp in enumerate(("gato", "cao")):
        for c, breed in enumerate(picks[sp]):
            cand = tr[tr.breed == breed]
            row = cand.iloc[int(rng.integers(len(cand)))]
            ax = axes[r, c]
            ax.imshow(load_thumb(row))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(SP_C[sp]); s.set_linewidth(3)
            ax.set_xlabel(breed.replace("_", " ").title(), fontsize=9.5, color=INK, labelpad=4)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.08, wspace=0.08, hspace=0.28)
    save(fig, "samples.png")


# ------------------------------------------------------------ 3. balanceamento
def fig_balance():
    c = S["counts"]
    fig = plt.figure(figsize=(12.3, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3.3], wspace=0.14)

    ax = fig.add_subplot(gs[0])
    vals = [c["species"]["gato"], c["species"]["cao"]]
    for i, sp in enumerate(("gato", "cao")):
        ax.bar(i, vals[i], width=0.58, color=SP_C[sp])
        ax.text(i, vals[i] + 90, f"{n_(vals[i])}\n({d_(vals[i] / c['total'] * 100)}%)",
                ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold", linespacing=1.15)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Gato", "Cão"], fontsize=10.5, color=INK)
    ax.set_ylim(0, 6300); ax.set_yticks([0, 2000, 4000])
    ax.set_yticklabels(["0", "2.000", "4.000"])
    ax.set_title(f"Gato × cão: {d_(c['imbalance_ratio'], 2)} : 1", pad=8)
    clean(ax, True)

    ax = fig.add_subplot(gs[1])
    order = sorted(c["breeds"], key=lambda b: (c["breed_species"][b] != "gato", b.lower()))
    for i, b in enumerate(order):
        sp = c["breed_species"][b]
        ax.bar(i, c["breeds"][b], width=0.72, color=SP_C[sp])
    ax.axhline(200, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(-0.6, 204, "200 imagens", ha="left", va="bottom", fontsize=8.5, color=INK2)
    for b in ("Bombay", "staffordshire_bull_terrier"):
        i = order.index(b)
        ax.text(i, c["breeds"][b] + 3, str(c["breeds"][b]), ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_ylim(0, 245); ax.set_yticks([0, 100, 200])
    ax.set_xlim(-0.8, len(order) - 0.2); ax.set_xticks([])
    ax.set_title(f"37 raças: {c['breed_min'][1]} a {c['breed_max'][1]} imagens cada", pad=8)
    n_gato = sum(1 for b in order if c["breed_species"][b] == "gato")
    ax.text((n_gato - 1) / 2, -14, f"{n_gato} raças de gato", ha="center", va="top", fontsize=9.5, color=INK)
    ax.text(n_gato + (len(order) - n_gato - 1) / 2, -14, f"{len(order) - n_gato} raças de cão",
            ha="center", va="top", fontsize=9.5, color=INK)
    clean(ax, True)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.88, bottom=0.14)
    save(fig, "balance.png")


# ------------------------------------------ 4. classes por split / stats de pixel
def fig_class_split():
    fig, ax = plt.subplots(figsize=(6.0, 2.95))
    splits = ["train", "val", "test"]
    for y, s in enumerate(splits):
        left = 0
        for sp in ("gato", "cao"):
            pct = S["split"][s]["species_pct"][sp] * 100
            n = S["split"][s]["species"][sp]
            ax.barh(y, pct, left=left, height=0.62, color=SP_C[sp], edgecolor=SURFACE, linewidth=2)
            ax.text(left + pct / 2, y, f"{SP_N[sp]} · {d_(pct)}%\n{n_(n)} imgs", ha="center", va="center",
                    fontsize=9.5, color=SP_TXT[sp], fontweight="bold", linespacing=1.15)
            left += pct
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"{SPLIT_N[s]}\nn = {n_(S['split'][s]['n'])}" for s in splits], fontsize=10, color=INK)
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xticks([])
    for s in ("bottom",):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Proporção gato × cão em cada conjunto", pad=8)
    fig.subplots_adjust(left=0.19, right=0.99, top=0.88, bottom=0.03)
    save(fig, "class_by_split.png")


def fig_pixel_stats():
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.95))
    ch = ["R", "G", "B"]
    for ax, key, title, ymax in ((axes[0], "mean_255", "Média por canal (0–255)", 150),
                                 (axes[1], "std_255", "Desvio-padrão por canal", 88)):
        w = 0.26
        for j, s in enumerate(("train", "val", "test")):
            v = S["pixels"][s][key]
            xs = np.arange(3) + (j - 1) * (w + 0.03)
            ax.bar(xs, v, width=w, color=SPLIT_C[s], label=SPLIT_N[s])
            for x, val in zip(xs, v):
                ax.text(x, val + 1.5, d_(val, 1), ha="center", va="bottom",
                        fontsize=6.6, color=INK, rotation=90)
        ax.set_xticks(range(3)); ax.set_xticklabels(ch, fontsize=10.5, color=INK)
        ax.set_ylim(0, ymax); ax.set_title(title, pad=8, fontsize=10.5)
        clean(ax, True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.07, 1.0), ncol=3,
               frameon=False, fontsize=9.5, handlelength=1, handleheight=1, columnspacing=1.2)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.79, bottom=0.09, wspace=0.22)
    save(fig, "pixel_stats.png")


# ------------------------------------------------------- 5. histogramas RGB
def fig_rgb_hist():
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.15), gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.12, "wspace": 0.16})
    names = ["Canal R (vermelho)", "Canal G (verde)", "Canal B (azul)"]
    x = np.arange(64) * 4 + 2

    def norm(s, c):
        h = H[f"nat_{s}"][c].astype(float)
        return h.reshape(64, 4).sum(axis=1) / h.sum() * 100

    maxdiff = 0
    for c in range(3):
        top, bot = axes[0, c], axes[1, c]
        for s, lw in (("train", 3.0), ("val", 1.8), ("test", 1.1)):
            top.plot(x, norm(s, c), color=SPLIT_C[s], linewidth=lw, label=SPLIT_N[s])
        top.set_title(names[c], pad=6, fontsize=10.5)
        top.set_xlim(0, 255); top.set_ylim(0, 5.0)
        top.set_xticks([0, 128, 255]); top.tick_params(labelbottom=False)
        clean(top, True)
        base = norm("train", c)
        bot.axhline(0, color=SPLIT_C["train"], linewidth=2.2)
        for s in ("val", "test"):
            d = norm(s, c) - base
            maxdiff = max(maxdiff, np.abs(d).max())
            bot.plot(x, d, color=SPLIT_C[s], linewidth=1.3)
        bot.set_xlim(0, 255); bot.set_ylim(-0.75, 0.75)
        bot.set_xticks([0, 128, 255]); bot.set_yticks([-0.5, 0, 0.5])
        clean(bot, True)
        bot.set_xlabel("intensidade do pixel", fontsize=9)
        if c == 0:
            top.set_ylabel("% dos pixels (faixas de 4 níveis)", fontsize=8.5)
            bot.set_ylabel("diferença vs Train\n(pontos percentuais)", fontsize=8.5)
        else:
            top.set_yticklabels([]); bot.set_yticklabels([])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.975, 1.0), ncol=3, frameon=False,
               fontsize=9.5, handlelength=1.6, columnspacing=1.2)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.88, bottom=0.10)
    save(fig, "rgb_hist.png")
    return maxdiff


# ------------------------------------------------------------- 6. resolução
def fig_resolution():
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.15), gridspec_kw={"wspace": 0.22})
    blue = SPLIT_C["train"]
    ax = axes[0]
    inside = (DF.width <= 520) & (DF.height <= 520)
    ax.scatter(DF.width[inside], DF.height[inside], s=7, color=blue, alpha=0.16, linewidths=0)
    ax.set_xlim(90, 525); ax.set_ylim(90, 525)
    ax.set_xlabel("largura (px)", fontsize=9.5); ax.set_ylabel("altura (px)", fontsize=9.5)
    ax.set_title("Largura × altura de cada imagem", pad=8)
    top = S["size"]["top_resolutions"][0]; topn = S["size"]["top_resolutions_n"][0]
    ax.annotate(f"{top.replace('x', '×')}: {n_(topn)} imgs\n({d_(topn / len(DF) * 100)}%)", xy=(500, 375), xytext=(118, 452),
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
    out = int((~inside).sum())
    ax.text(0.98, 0.03, f"{n_(out)} imgs maiores fora do recorte", transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    clean(ax, True)

    ax = axes[1]
    ax.hist(DF.aspect, bins=np.arange(0.4, 2.02, 0.03), color=blue, edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(1.01, ax.get_ylim()[1] * 0.93, "quadrada", fontsize=8.5, color=INK2)
    ax.set_xlabel("proporção largura / altura", fontsize=9.5); ax.set_ylabel("nº de imagens", fontsize=9.5)
    ax.set_title(f"Proporção: mediana {d_(S['size']['aspect']['median'], 2)} (4:3)", pad=8)
    clean(ax, True)

    ax = axes[2]
    ms = DF.min_side.clip(upper=800)
    ax.hist(ms, bins=np.arange(100, 820, 20), color=blue, edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(224, color=SPLIT_C["val"], linewidth=2)
    ax.text(440, ax.get_ylim()[1] * 0.78,
            f"linha laranja = alvo 224 px\n{n_(S['size']['n_smaller_than_target'])} imgs ({d_(S['size']['pct_smaller_than_target'] * 100)}%) ficam abaixo",
            fontsize=9, color=INK, va="top")
    ax.set_xlabel("lado menor da imagem (px; ≥ 800 agrupado)", fontsize=9.5); ax.set_ylabel("nº de imagens", fontsize=9.5)
    ax.set_title("Lado menor vs. alvo de entrada", pad=8)
    clean(ax, True)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.88, bottom=0.17)
    save(fig, "resolution.png")


# --------------------------------------------------------- 7. estratégias
def fig_preprocess():
    tr = DF[DF.split == "train"].sort_values("file")
    a = tr[(tr.species == "gato") & (tr.width == 500) & (tr.height == 375)].iloc[7]
    b = tr[(tr.species == "cao") & (tr.width == 375) & (tr.height == 500)].iloc[5]
    size = S["size"]["stretch_ratio_median"]
    fig, axes = plt.subplots(2, 4, figsize=(7.3, 5.05))
    cols = ["Original", "Resize direto\n(distorce a proporção)", "Resize + center crop\n(escolhido)", "Resize + padding\n(barras pretas)"]
    for r, row in enumerate((a, b)):
        img = dataprep.load_rgb(DATASET / row.split / row.species / row.file)
        w, h = img.size
        outs = [img, dataprep.resize_stretch(img, 224), dataprep.resize_center_crop(img, 224), dataprep.resize_pad(img, 224)]
        for c, o in enumerate(outs):
            ax = axes[r, c]
            ax.imshow(o)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_edgecolor(SPLIT_C["train"] if c == 2 else AXIS)
                s.set_linewidth(3.2 if c == 2 else 1)
        # janela mantida pelo crop sobre o original
        m = min(w, h)
        ax = axes[r, 0]
        ax.add_patch(Rectangle(((w - m) / 2, (h - m) / 2), m, m, fill=False, edgecolor="#ffffff", linewidth=2.2, linestyle=(0, (4, 2))))
        ax.set_xlabel(f"{w}×{h} px", fontsize=8.5, labelpad=3)
        axes[r, 1].set_xlabel(f"×{d_(max(w / h, h / w), 2)} de distorção", fontsize=8.5, labelpad=3)
        axes[r, 2].set_xlabel(f"perde {d_(dataprep.aspect_loss(w, h) * 100, 0)}% da área", fontsize=8.5, labelpad=3)
        axes[r, 3].set_xlabel(f"{d_(dataprep.aspect_loss(w, h) * 100, 0)}% da entrada vazia", fontsize=8.5, labelpad=3)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.885, bottom=0.06, wspace=0.07, hspace=0.24)
    for c in range(4):
        pos = axes[0, c].get_position()
        fig.text(pos.x0 + pos.width / 2, 0.985, cols[c], ha="center", va="top", fontsize=8.8,
                 fontweight="bold" if c == 2 else "normal", linespacing=1.2)
    save(fig, "preprocess.png")


if __name__ == "__main__":
    fig_samples()
    fig_balance()
    fig_class_split()
    fig_pixel_stats()
    md = fig_rgb_hist()
    fig_resolution()
    fig_preprocess()
    (OUT / "hist_maxdiff.json").write_text(json.dumps({"max_pp_diff": float(md)}))
    print("figuras ok; maior diferença de histograma (pp por faixa de 4 níveis):", round(md, 3))
