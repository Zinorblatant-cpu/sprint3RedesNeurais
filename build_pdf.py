"""Monta a apresentação (8 slides, PDF 16:9) a partir de outputs/stats.json e outputs/figures/."""
import json
import math
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
PDF = ROOT / "Sprint3_Preparacao_Dataset.pdf"
S = json.loads((OUT / "stats.json").read_text(encoding="utf-8"))

pdfmetrics.registerFont(TTFont("Seg", "C:/Windows/Fonts/segoeui.ttf"))
pdfmetrics.registerFont(TTFont("SegB", "C:/Windows/Fonts/segoeuib.ttf"))
registerFontFamily("Seg", normal="Seg", bold="SegB", italic="Seg", boldItalic="SegB")

W, H = 960, 540
M = 36
BG, INK, INK2, MUTED, LINE, CARD, SOFT = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff", "#f4f3ef"
C_TRAIN, C_VAL, C_TEST = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, WARN = "#0ca30c", "#fab219"
TOTAL = 8
MEMBERS = [
    ("Lucas Ferrari Lima", "563119"),
    ("Carlos Eduardo Pires Cervelli", "563462"),
    ("Felipe Krzyzanovski dos Santos Menezes", "564878"),
    ("Leonardo Lopes Oliveira", "565437"),
    ("Arthur de Souza Matos Dias", "566068"),
    ("Guilherme Carreri Giampietro", "565676"),
    ("Mateus Patrício Pereira", "564695"),
]


def n_(v):
    return f"{int(round(v)):,}".replace(",", ".")


def d_(v, k=1):
    return f"{v:.{k}f}".replace(".", ",")


def up(v, k=2):  # arredonda para cima (limites superiores honestos)
    return d_(math.ceil(v * 10**k) / 10**k, k)


def down(v, k=2):  # arredonda para baixo (limites inferiores honestos)
    return d_(math.floor(v * 10**k) / 10**k, k)


# ------------------------------------------------------------------ primitivas
def para(c, text, x, ytop, w, size=10.5, color=INK, bold=False, leading=None, align=0):
    st = ParagraphStyle("p", fontName="SegB" if bold else "Seg", fontSize=size,
                        leading=leading or size * 1.32, textColor=HexColor(color), alignment=align)
    p = Paragraph(text, st)
    _, h = p.wrap(w, 2000)
    p.drawOn(c, x, ytop - h)
    return h


def bullets(c, items, x, ytop, w, size=10, gap=5, color=INK, marker=INK2):
    y = ytop
    for it in items:
        c.setFillColor(HexColor(marker))
        c.circle(x + 3, y - size * 0.72, 1.9, stroke=0, fill=1)
        h = para(c, it, x + 13, y, w - 13, size=size, color=color)
        y -= h + gap
    return ytop - y


def card(c, x, ytop, w, h, fill=CARD, stroke=LINE):
    c.setFillColor(HexColor(fill))
    c.setStrokeColor(HexColor(stroke))
    c.setLineWidth(0.8)
    c.roundRect(x, ytop - h, w, h, 7, stroke=1, fill=1)


def label(c, text, x, ytop, color=MUTED):
    c.setFillColor(HexColor(color))
    c.setFont("SegB", 8)
    c.drawString(x, ytop - 8, text.upper())


def header(c, n, kicker, title):
    c.setFillColor(HexColor(BG))
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(HexColor(MUTED))
    c.setFont("SegB", 8.5)
    c.drawString(M, H - 34, kicker.upper())
    c.setFillColor(HexColor(INK))
    c.setFont("SegB", 25)
    c.drawString(M, H - 64, title)
    c.setStrokeColor(HexColor(LINE))
    c.setLineWidth(0.8)
    c.line(M, 26, W - M, 26)
    c.setFillColor(HexColor(MUTED))
    c.setFont("Seg", 8)
    c.drawString(M, 13, "Challenge Sprint 3 · Preparação e divisão do dataset · Redes Neurais Convolucionais")
    c.drawRightString(W - M, 13, f"{n} / {TOTAL}")


def image(c, name, x, ytop, w=None, h=None):
    path = FIG / name
    iw, ih = ImageReader(str(path)).getSize()
    if w is None:
        w = h * iw / ih
    if h is None:
        h = w * ih / iw
    c.drawImage(str(path), x, ytop - h, width=w, height=h, mask="auto")
    return w, h


def stat(c, x, ytop, w, h, big, small, big_size=24, color=INK):
    card(c, x, ytop, w, h)
    c.setFillColor(HexColor(color))
    c.setFont("SegB", big_size)
    c.drawString(x + 12, ytop - 12 - big_size * 0.8, big)
    para(c, small, x + 12, ytop - 18 - big_size, w - 24, size=9.2, color=INK2)


def status(c, x, y, kind):
    """ícone + rótulo: nunca só cor."""
    col = GOOD if kind == "ok" else WARN
    c.setFillColor(HexColor(col))
    c.circle(x + 9, y + 4, 9, stroke=0, fill=1)
    c.setStrokeColor(white if kind == "ok" else HexColor(INK))
    c.setLineWidth(2.1)
    c.setLineCap(1)
    if kind == "ok":
        c.line(x + 4.5, y + 4, x + 8, y + 0.5)
        c.line(x + 8, y + 0.5, x + 14, y + 8)
    else:
        c.line(x + 9, y + 8.5, x + 9, y + 3.5)
        c.line(x + 9, y + 0.6, x + 9, y + 0.7)


# ---------------------------------------------------------------- números-base
cnt, sp, px, cmp_, nm, sz, raw, lk = S["counts"], S["split"], S["pixels"], S["compare"], S["norm"], S["size"], S["raw"], S["leakage"]
N = cnt["total"]
NEG = cnt["negative_n"]
SPECIES = [k for k in cnt["classes"] if k != "Negative"]
ks_D = max(v["D"] for k in ("ks_brightness", "ks_contrast") for v in cmp_[k].values())
ks_p = min(v["p"] for k in ("ks_brightness", "ks_contrast") for v in cmp_[k].values())
w1_max = max(x for v in cmp_["wasserstein_rgb"].values() for x in v)
mean_diff_up, w1_up, ks_D_up, ks_p_dn = up(cmp_["max_mean_diff_255"], 1), up(w1_max, 1), up(ks_D, 3), down(ks_p, 2)
post_m = [x for s in ("val", "test") for x in nm["post_norm"][s]["mean"]]
post_s = [x for s in ("val", "test") for x in nm["post_norm"][s]["std"]]
abs_mean_max = up(max(abs(x) for x in post_m), 3)
w_tr = cnt["weights_train"]
w_sp_lo, w_sp_hi = min(w_tr[k] for k in SPECIES), max(w_tr[k] for k in SPECIES)
hist_diff = json.loads((OUT / "hist_maxdiff.json").read_text())["max_pp_diff"]
rv = cnt["per_class"]["Rubber vine"]
rv_split = [sp[s]["per_class"]["Rubber vine"] for s in ("train", "val", "test")]
n_val = sp["val"]["n"]
me90 = 1.96 * math.sqrt(0.09 / n_val) * 100
me50 = 1.96 * math.sqrt(0.25 / n_val) * 100
tm, ts = nm["train_mean"], nm["train_std"]
bril = {s: px[s]["brightness_mean"] for s in ("train", "val", "test")}


def build():
    c = canvas.Canvas(str(PDF), pagesize=(W, H))
    c.setTitle("Sprint 3 – Preparação e divisão do dataset (DeepWeeds)")
    c.setAuthor("; ".join(f"{n} (RM {rm})" for n, rm in MEMBERS))

    # ============================================================ 1. Dataset
    header(c, 1, "Sprint 3 · Preparação e divisão do dataset", "Dataset escolhido: DeepWeeds, espécies de plantas")
    top = H - 84
    stat(c, M, top, 300, 62, n_(N), f"imagens de campo ({raw['dup_removed']} duplicatas exatas; {n_(raw['n_missing'])} arquivos ausentes)")
    stat(c, M, top - 70, 300, 62, "9 classes", f"8 espécies-alvo + “Negative” (vegetação sem espécie-alvo)", big_size=22)
    stat(c, M, top - 140, 300, 62, "256 × 256 RGB", "todas as imagens, mesmo tamanho e 3 canais", big_size=20)
    card(c, M, top - 210, 300, 186)
    label(c, "Por que esta base", M + 12, top - 220)
    bullets(c, [
        f"<b>Foco em espécies:</b> ~1.000 fotos de cada uma das 8 espécies ({n_(cnt['species_min'][1])} a {n_(cnt['species_max'][1])}).",
        f"<b>Vegetação real de campo:</b> luz, fundo e ângulo variados ({raw['n_sessions']} sessões de captura, {raw['n_dates']} datas).",
        f"<b>Desbalanceamento genuíno:</b> “Negative” é {d_(cnt['negative_share'] * 100, 0)}% das imagens.",
        "<b>Licença CC BY 4.0</b> e download público.",
        "<b>Ligação com o projeto:</b> reconhecer espécies de vegetação é base para monitorar faixas de rodovia (Motiva); aqui, vegetação australiana como ponto de partida.",
    ], M + 12, top - 238, 276, size=9.6, gap=4)
    ix = M + 300 + 20
    gw, gh = image(c, "samples.png", ix, top, w=336)
    para(c, "Uma imagem de cada classe, sorteada do Train. Contorno violeta = espécie-alvo; cinza = Negative.",
         ix, top - gh - 4, gw, size=8.5, color=MUTED)
    cx = ix + gw + 16
    cw = W - M - cx
    card(c, cx, top, cw, 166)
    label(c, "Fonte e rótulos", cx + 12, top - 8)
    para(c, "GitHub · <b>AlexOlsen/DeepWeeds</b> (Olsen et al., <i>Scientific Reports</i>, 2019). Fotos tiradas em 8 locais do norte da Austrália.<br/><br/>"
            "<b>labels.csv</b> traz <i>Filename, Label, Species</i>. A data, a hora e a câmera vêm do nome do arquivo (<i>AAAAMMDD-HHMMSS-ID</i>).<br/><br/>"
            "Espécies: " + ", ".join(SPECIES) + ".",
         cx + 12, top - 24, cw - 24, size=9.2, color=INK)
    my = top - 166 - 12
    card(c, cx, my, cw, 178)
    label(c, "Integrantes", cx + 12, my - 8)
    y = my - 24
    for name, rm in MEMBERS:
        y -= para(c, f"<b>{name}</b> · RM {rm}", cx + 12, y, cw - 24, size=8.6, color=INK) + 3.5
    c.showPage()

    # ============================================================ 2. Divisão
    header(c, 2, "Divisão dos dados", "Train / Validation / Test = 70 / 15 / 15, estratificado")
    bx, by, bw, bh, gap = M, H - 84, W - 2 * M, 50, 3
    segs = [("train", "Train", C_TRAIN, "#ffffff"), ("val", "Validation", C_VAL, INK), ("test", "Test", C_TEST, INK)]
    avail = bw - gap * 2
    x = bx
    for key, name, col, txt in segs:
        w = avail * sp[key]["pct"]
        c.setFillColor(HexColor(col))
        c.roundRect(x, by - bh, w, bh, 6, stroke=0, fill=1)
        c.setFillColor(HexColor(txt))
        c.setFont("SegB", 12)
        if w > 300:
            c.drawCentredString(x + w / 2, by - bh / 2 - 4, f"{name} · {n_(sp[key]['n'])} imagens · {d_(sp[key]['pct'] * 100, 0)}%")
        else:
            c.drawCentredString(x + w / 2, by - 21, name)
            c.setFont("Seg", 10.5)
            c.drawCentredString(x + w / 2, by - 37, f"{n_(sp[key]['n'])} · {d_(sp[key]['pct'] * 100, 0)}%")
        x += w + gap
    ty = by - bh - 22
    tw = 540
    card(c, M, ty, tw, 168)
    cols = [12, 108, 168, 228, 322, 388]
    for cx, h_ in zip(cols, ["Conjunto", "Imagens", "%", "8 espécies", "Negative", "Para quê"]):
        c.setFillColor(HexColor(MUTED)); c.setFont("SegB", 8.5); c.drawString(M + cx, ty - 20, h_.upper())
    c.setStrokeColor(HexColor(LINE)); c.line(M + 12, ty - 27, M + tw - 12, ty - 27)
    uses = {"train": "Ajusta os pesos da CNN.", "val": "Hiperparâmetros e early stopping.", "test": "Nota final, usado uma vez."}
    ry = ty - 34
    for key, name, col, _ in segs:
        n_sp = sp[key]["n"] - sp[key]["per_class"]["Negative"]
        c.setFillColor(HexColor(col)); c.roundRect(M + 12, ry - 19, 6, 22, 2, stroke=0, fill=1)
        c.setFillColor(HexColor(INK)); c.setFont("SegB", 10.5); c.drawString(M + 24, ry - 12, name)
        c.setFont("Seg", 10.5)
        c.drawString(M + cols[1], ry - 12, n_(sp[key]["n"]))
        c.drawString(M + cols[2], ry - 12, f"{d_(sp[key]['pct'] * 100)}%")
        c.drawString(M + cols[3], ry - 12, n_(n_sp))
        c.drawString(M + cols[4], ry - 12, n_(sp[key]["per_class"]["Negative"]))
        para(c, uses[key], M + cols[5], ry - 2, tw - cols[5] - 12, size=9.2, color=INK2)
        ry -= 44
    para(c, f"Total: {n_(N)} imagens · a proporção de cada classe é a mesma nos 3 conjuntos (slide 4).", M, ty - 178, tw, size=9.2, color=INK2)
    rx = M + tw + 16
    rw = W - M - rx
    card(c, rx, ty, rw, 208)
    label(c, "Por que esta divisão", rx + 12, ty - 8)
    bullets(c, [
        f"<b>70% treino:</b> ~{n_(sp['train']['n'])} imagens para 9 classes: dados de sobra para a CNN aprender folha, textura e porte.",
        f"<b>15% + 15%:</b> ~{n_(n_val)} imagens em Val e em Test dão margem de ≈ ±{d_(me90)} p.p. (IC 95%) para acurácia de ~90% (±{d_(me50)} p.p. no pior caso).",
        f"<b>Estratificada por classe</b> (seed 42): até a menor espécie (Rubber vine, {n_(rv)}) fica com {rv_split[0]} / {rv_split[1]} / {rv_split[2]} imagens.",
    ], rx + 12, ty - 26, rw - 24, size=10, gap=6)
    card(c, M, ty - 214, W - 2 * M, 92, fill=SOFT)
    label(c, "Como foi feito (reprodutível)", M + 12, ty - 222)
    bullets(c, [
        f"Conferência: {n_(raw['n_labels'])} linhas no labels.csv × {n_(N + raw['n_missing'])} arquivos; hash MD5 sem duplicatas exatas; todas as imagens abrem como RGB.",
        "Divisão em dois passos com <font name='SegB'>train_test_split(stratify=classe, random_state=42)</font>: 70% / 30%, e os 30% em 50/50.",
        "Cópia para <font name='SegB'>dataset/{train,val,test}/&lt;Classe&gt;/</font> (pronto para ImageFolder) e manifesto <font name='SegB'>splits.csv</font>.",
    ], M + 12, ty - 238, W - 2 * M - 24, size=9.8, gap=4)
    c.showPage()

    # ============================================================ 3. Balanceamento
    header(c, 3, "Balanceamento das classes", "Espécies balanceadas, mas “Negative” domina")
    top = H - 82
    _, ih = image(c, "balance.png", M, top, w=W - 2 * M)
    cy = top - ih - 14
    ch = cy - 42
    cw = (W - 2 * M - 16) / 2
    card(c, M, cy, cw, ch)
    label(c, "O dataset está balanceado?", M + 14, cy - 8)
    status(c, M + 14, cy - 44, "ok")
    para(c, f"<b>Só as 8 espécies: sim.</b> {n_(cnt['species_min'][1])} a {n_(cnt['species_max'][1])} imagens por espécie "
            f"(razão {d_(cnt['species_ratio'], 2)} : 1).", M + 40, cy - 28, cw - 54, size=10.6)
    status(c, M + 14, cy - 96, "warn")
    para(c, f"<b>Com “Negative”: não.</b> {n_(NEG)} imagens = <b>{d_(cnt['negative_share'] * 100)}%</b> do total, "
            f"{d_(cnt['negative_vs_species_mean'])}× uma espécie média. Responder sempre “Negative” já acerta {d_(cnt['negative_share'] * 100)}% sem aprender nada.",
         M + 40, cy - 80, cw - 54, size=10.6)
    x2 = M + cw + 16
    card(c, x2, cy, cw, ch)
    label(c, "Como vamos tratar", x2 + 14, cy - 8)
    bullets(c, [
        f"<b>Pesos de classe na loss</b> (n / k·n<sub>classe</sub>, calculado no Train): Negative ≈ {d_(w_tr['Negative'], 2)}; espécies ≈ {d_(w_sp_lo, 2)} a {d_(w_sp_hi, 2)}.",
        "<b>Métricas por classe:</b> recall e F1 de cada espécie, F1 macro e matriz de confusão, em vez de só acurácia.",
        "<b>Mesma proporção em Val e Test</b> (split estratificado): métricas comparáveis.",
        f"<b>Sem undersampling agressivo:</b> descartaria milhares de fotos de vegetação real, que ensinam o que <i>não</i> é espécie-alvo. Augmentation só no Train.",
        "<b>Se os pesos não bastarem:</b> oversampling das espécies ou focal loss (a testar no treino).",
    ], x2 + 14, cy - 28, cw - 28, size=10, gap=4)
    c.showPage()

    # ============================================================ 4. Comparação (classes + pixels)
    header(c, 4, "Comparação entre Train, Validation e Test (1/2)", "Classes e estatísticas de pixel são comparáveis")
    top = H - 82
    w1_, h1 = image(c, "class_by_split.png", M, top, w=(W - 2 * M) * 0.485)
    image(c, "pixel_stats.png", M + w1_ + 14, top, h=h1)
    my = top - h1 - 12
    mw = (W - 2 * M - 3 * 12) / 4
    ms = [
        (f"p ≈ {d_(S['chi2']['class']['p'], 2)}", "teste χ²: composição de classes × conjunto. Nenhuma diferença detectável."),
        (f"≤ {mean_diff_up} de 255", f"maior diferença entre as médias RGB de dois conjuntos ({d_(cmp_['max_mean_diff_255'] / 255 * 100, 2)}% da escala)."),
        (f"D ≤ {ks_D_up}", f"KS 2 amostras no brilho e no contraste por imagem (p ≥ {ks_p_dn}). D = distância máxima entre as curvas acumuladas."),
        (f"≤ {w1_up} nível", "distância de Wasserstein entre os histogramas RGB (em níveis de intensidade, de 0 a 255)."),
    ]
    for i, (big, small) in enumerate(ms):
        stat(c, M + i * (mw + 12), my, mw, 92, big, small, big_size=18)
    card(c, M, my - 104, W - 2 * M, 60, fill=SOFT)
    para(c, "<b>Leitura:</b> o objetivo não é deixar os conjuntos idênticos, e sim garantir que nenhum tenha um “visual” muito diferente. "
            f"As diferenças são de ~{d_(cmp_['max_mean_diff_255'], 1)} nível em 255, ruído esperado de amostrar ~{n_(n_val)} imagens. "
            "Ressalva: como o split é aleatório, as sessões de captura são compartilhadas entre os conjuntos (slide 6).",
         M + 14, my - 116, W - 2 * M - 28, size=10.4, color=INK)
    c.showPage()

    # ============================================================ 5. Histogramas RGB
    header(c, 5, "Comparação entre Train, Validation e Test (2/2)", "Histogramas de intensidade e canais RGB")
    top = H - 80
    fw, fh = image(c, "rgb_hist.png", M, top, h=H - 80 - 44)
    rx = M + fw + 18
    rw = W - M - rx
    card(c, rx, top, rw, 170)
    label(c, "O que a figura mostra", rx + 12, top - 8)
    bullets(c, [
        "As curvas de Train, Validation e Test <b>praticamente coincidem</b> nos 3 canais.",
        f"Embaixo, a diferença vs Train <b>não passa de ≈ {up(hist_diff, 1)} ponto percentual</b> por faixa de 4 níveis, sem padrão sistemático.",
    ], rx + 12, top - 24, rw - 24, size=9.6, gap=5)
    card(c, rx, top - 182, rw, 176)
    label(c, "Números que sustentam", rx + 12, top - 190)
    bullets(c, [
        f"<b>Intervalo dos pixels:</b> 0 a 255 (uint8) nos 3 conjuntos.",
        f"<b>Brilho médio por imagem:</b> {d_(bril['train'])} · {d_(bril['val'])} · {d_(bril['test'])} (Train · Val · Test).",
        f"<b>Perfil comum:</b> imagens escuras (média ≈ {d_(px['train']['mean_255'][0], 0)}–{d_(px['train']['mean_255'][1], 0)} de 255) e um pico em 255 (pixels estourados).",
    ], rx + 12, top - 206, rw - 24, size=9.4, gap=4)
    card(c, rx, top - 370, rw, 40, fill=SOFT)
    para(c, "<b>Conclusão:</b> Val e Test têm o mesmo perfil de cor do Train.", rx + 12, top - 378, rw - 24, size=9.6)
    c.showPage()

    # ============================================================ 6. Dimensão e aquisição
    header(c, 6, "Dimensão e preparação das imagens (1/2)", "Tamanho uniforme; o risco está na sessão de captura")
    top = H - 82
    _, ih = image(c, "acquisition.png", M, top, w=W - 2 * M)
    cy = top - ih - 10
    cw = (W - 2 * M - 3 * 12) / 4
    hh = 66
    inst = raw["instruments"]
    stat(c, M, cy, cw, hh, "256 × 256", f"Train, Val e Test: {n_(sp['train']['n'])} / {n_(sp['val']['n'])} / {n_(sp['test']['n'])} imagens, todas do mesmo tamanho.", big_size=18)
    stat(c, M + (cw + 12), cy, cw, hh, "3 canais", f"todas RGB (JPEG); intervalo dos pixels 0–255 (uint8).", big_size=18)
    stat(c, M + 2 * (cw + 12), cy, cw, hh, f"{raw['n_sessions']} sessões", f"{raw['n_dates']} datas, de {raw['date_min'][:4]} a {raw['date_max'][:4]}, com câmeras 0 a 4.", big_size=18)
    stat(c, M + 3 * (cw + 12), cy, cw, hh, f"{d_(lk['pct_same_session'] * 100)}%", f"das fotos de Val/Test têm foto de Train da mesma sessão e classe (mediana {n_(lk['median_gap_s'])} s).", big_size=18)
    iy = cy - hh - 10
    card(c, M, iy, W - 2 * M, iy - 40, fill=SOFT)
    label(c, "O que isso implica", M + 14, iy - 6)
    bullets(c, [
        "<b>Resize e padding: dispensados.</b> As imagens já são quadradas e iguais.",
        f"<b>Espécie e sessão andam juntas</b> (Parkinsonia: {lk['sessions_per_class']['Parkinsonia']} sessões): a CNN pode aprender o fundo, e não a planta.",
        "<b>Split por sessão: descartado.</b> Em 4 sementes, a Lantana ficou sem imagens no Test. Mantivemos o split por classe e declaramos o risco.",
    ], M + 14, iy - 22, W - 2 * M - 28, size=9.6, gap=3)
    c.showPage()

    # ============================================================ 7. Padronização
    header(c, 7, "Dimensão e preparação das imagens (2/2)", "Padronização para a CNN: 224 × 224 × 3")
    top = H - 80
    fw, fh = image(c, "preprocess.png", M, top, w=500)
    para(c, "Exemplos do Train (Lantana e Rubber vine). Contorno azul = o que Val/Test recebem.", M, top - fh - 4, fw, size=8.5, color=MUTED)
    rx = M + fw + 18
    rw = W - M - rx
    steps = [("1", "Resize", "não é necessário: todas já são 256 × 256"),
             ("2", "Crop 224 × 224", "Train: posição aleatória + espelho (a planta não tem lado “certo”). Val/Test: centro."),
             ("3", "Normalização", "x / 255 → (x − μ) / σ por canal, com μ e σ do Train")]
    y = top
    for num, name, desc in steps:
        card(c, rx, y, rw, 38)
        c.setFillColor(HexColor(C_TRAIN)); c.circle(rx + 18, y - 19, 10, stroke=0, fill=1)
        c.setFillColor(white); c.setFont("SegB", 11); c.drawCentredString(rx + 18, y - 23, num)
        para(c, f"<b>{name}:</b> {desc}", rx + 36, y - 6, rw - 46, size=9.2)
        y -= 44
    para(c, "→ entrada da CNN: 224 × 224 × 3 (float32)", rx, y - 2, rw, size=9.6, color=INK2, bold=True)
    y -= 24
    label(c, "Por que essa estratégia", rx, y)
    y -= 14
    hb = bullets(c, [
        "<b>Sem resize e sem padding:</b> imagens quadradas e iguais, sem distorção nem barras.",
        f"<b>Crop sem reamostrar:</b> Train e Val/Test na mesma escala de pixel. Custo: {d_(sz['crop_area_lost'] * 100)}% da área nas bordas em Val/Test (o crop aleatório expõe essas bordas no Train). Alternativa: resize 256 → 224, sem perda de área, mas reamostra.",
        "<b>224 px:</b> padrão de backbones pré-treinados e do artigo original.",
    ], rx, y, rw, size=9.4, gap=4)
    ny = y - hb - 8
    card(c, rx, ny, rw, 92, fill=SOFT)
    label(c, "Normalização escolhida", rx + 12, ny - 6)
    para(c, f"<b>Só do Train:</b> μ = ({'; '.join(d_(v, 3) for v in tm)}), σ = ({'; '.join(d_(v, 3) for v in ts)}). "
            f"Em Val/Test: |μ| ≤ {abs_mean_max} e σ entre {down(min(post_s), 3)} e {up(max(post_s), 3)}. "
            f"O μ é bem menor que o do ImageNet (fotos escuras): com backbone pré-treinado, usar as constantes do ImageNet.",
         rx + 12, ny - 20, rw - 24, size=9.2)
    c.showPage()

    # ============================================================ 8. Análise final
    header(c, 8, "Análise final", "O dataset está pronto para treinar uma CNN?")
    top = H - 84
    cw = (W - 2 * M - 16) / 2
    chh = 190
    cards_ = [
        ("O dataset está balanceado?", "warn", "PARCIALMENTE", [
            f"<b>8 espécies:</b> sim ({n_(cnt['species_min'][1])}–{n_(cnt['species_max'][1])}; {d_(cnt['species_ratio'], 2)} : 1).",
            f"<b>Com “Negative”:</b> não ({n_(NEG)} = {d_(cnt['negative_share'] * 100)}% do total).",
            f"<b>Tratamento:</b> pesos de classe (Negative ≈ {d_(w_tr['Negative'], 2)}; espécies ≈ {d_(w_sp_lo, 2)}–{d_(w_sp_hi, 2)}), F1 macro e recall por classe.",
        ]),
        ("Train, Val e Test são semelhantes?", "ok", "SIM", [
            f"Classes com a mesma proporção (χ² p ≈ {d_(S['chi2']['class']['p'], 2)}) e mesma mistura de câmeras (p = {d_(S['chi2']['instrument']['p'], 2)}).",
            f"Médias RGB a ≤ {mean_diff_up} de 255; KS D ≤ {ks_D_up}; histogramas sobrepostos; pixels de 0 a 255.",
            "Semelhantes em boa parte porque o split é aleatório e divide as mesmas sessões entre os conjuntos.",
        ]),
        ("As imagens estão padronizadas?", "ok", "SIM", [
            "Todas RGB e 256 × 256; entrada da CNN em 224 × 224 × 3 via crop, sem resize nem padding.",
            f"Normalização com μ = ({', '.join(d_(v, 3) for v in tm)}) e σ = ({', '.join(d_(v, 3) for v in ts)}), só do Train.",
            f"Após normalizar, Val/Test têm |μ| ≤ {abs_mean_max} e σ entre {down(min(post_s), 3)} e {up(max(post_s), 3)}.",
        ]),
        ("Está pronto para o treinamento?", "ok", "SIM, COM 3 CUIDADOS", [
            "Aplicar <b>pesos de classe</b> e acompanhar <b>F1 macro</b> e o recall de cada espécie.",
            f"Usar o Test uma única vez. Ele divide sessões com o Train ({d_(lk['pct_same_session'] * 100)}%), então a acurácia tende a ser otimista para locais novos.",
            "Analisar os erros por câmera/sessão: espécie e local estão associados (ex.: Parkinsonia, 6 sessões).",
        ]),
    ]
    for i, (q, kind, tag, lines) in enumerate(cards_):
        x = M + (i % 2) * (cw + 16)
        y = top - (i // 2) * (chh + 12)
        card(c, x, y, cw, chh)
        label(c, q, x + 14, y - 6)
        status(c, x + 14, y - 47, kind)
        c.setFillColor(HexColor(INK)); c.setFont("SegB", 15); c.drawString(x + 40, y - 45, tag)
        bullets(c, lines, x + 14, y - 64, cw - 28, size=10, gap=5)
    c.showPage()
    c.save()
    print("PDF ok ->", PDF)


if __name__ == "__main__":
    build()
