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
BG, INK, INK2, MUTED, LINE, CARD = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
C_TRAIN, C_VAL, C_TEST = "#2a78d6", "#eb6834", "#1baf7a"
C_GATO, C_CAO = "#e87ba4", "#4a3aa7"
GOOD, WARN = "#0ca30c", "#fab219"
TOTAL = 8


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
cnt, sp, px, cmp_, nm, sz, raw = S["counts"], S["split"], S["pixels"], S["compare"], S["norm"], S["size"], S["raw"]
N = cnt["total"]
gato, cao = cnt["species"]["gato"], cnt["species"]["cao"]
ks_D = max(v["D"] for k in ("ks_brightness", "ks_contrast") for v in cmp_[k].values())
ks_p = min(v["p"] for k in ("ks_brightness", "ks_contrast") for v in cmp_[k].values())
w1_max = max(x for v in cmp_["wasserstein_rgb"].values() for x in v)
tr_n = sp["train"]["n"]
w_gato = tr_n / (2 * sp["train"]["species"]["gato"])
w_cao = tr_n / (2 * sp["train"]["species"]["cao"])
post_m = [x for s in ("val", "test") for x in nm["post_norm"][s]["mean"]]
post_s = [x for s in ("val", "test") for x in nm["post_norm"][s]["std"]]
top_res, top_n = sz["top_resolutions"], sz["top_resolutions_n"]
f2 = lambda v: up(v, 3)
mean_diff_up = up(cmp_["max_mean_diff_255"], 2)
w1_up = up(w1_max, 1)
ks_p_dn = down(ks_p, 2)
abs_mean_max = up(max(abs(x) for x in post_m), 3)


def build():
    c = canvas.Canvas(str(PDF), pagesize=(W, H))
    c.setTitle("Sprint 3 – Preparação e divisão do dataset")
    c.setAuthor("Challenge Sprint 3")

    # ============================================================ 1. Dataset
    header(c, 1, "Sprint 3 · Preparação e divisão do dataset", "Dataset escolhido: fotos de gatos e cães")
    top = H - 84
    stat(c, M, top, 300, 62, n_(N), f"imagens únicas ({n_(raw['n_files'])} baixadas − {raw['dup_removed']} duplicatas exatas)")
    stat(c, M, top - 70, 300, 62, "2 classes", f"Gato ({n_(gato)}) e Cão ({n_(cao)}) · problema binário", big_size=22)
    stat(c, M, top - 140, 300, 62, "37 raças", f"12 de gato e 25 de cão, com {cnt['breed_min'][1]}–{cnt['breed_max'][1]} imagens cada", big_size=22)
    card(c, M, top - 210, 300, 186)
    label(c, "Por que esta base", M + 12, top - 220)
    bullets(c, [
        f"<b>Volume:</b> {n_(N)} imagens bastam para dividir em Train/Val/Test e ainda medir cada conjunto.",
        "<b>Variedade real:</b> fotos com poses, fundos, iluminação e resoluções muito diferentes.",
        "<b>Desbalanceamento genuíno</b> gato × cão para discutir (não foi criado artificialmente).",
        "<b>Rótulo gratuito e confiável</b>, derivado do nome do arquivo.",
    ], M + 12, top - 238, 276, size=10.2, gap=6)
    ix = M + 300 + 20
    iw = W - M - ix
    _, ih = image(c, "samples.png", ix, top, w=iw)
    para(c, "Exemplos do conjunto de treino (miniaturas recortadas ao centro só para exibição). "
            "Contorno rosa = gato, violeta = cão.", ix, top - ih - 4, iw, size=8.5, color=MUTED)
    cy = top - ih - 30
    card(c, ix, cy, iw, cy - 40)
    label(c, "Fonte e rótulos", ix + 12, cy - 6)
    para(c, "Kaggle · <b>yashdogra/cats-and-dogs</b> (mesma estrutura de nomes do Oxford-IIIT Pet). "
            "O nome do arquivo dá o rótulo: <b>Abyssinian_1.jpg → gato / Abyssinian</b>; "
            "<b>beagle_42.jpg → cão / beagle</b> (raça de gato começa com maiúscula).",
         ix + 12, cy - 22, iw - 24, size=10.4, color=INK)
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
        one_line = w > 300
        if one_line:
            c.drawCentredString(x + w / 2, by - bh / 2 - 4, f"{name} · {n_(sp[key]['n'])} imagens · {d_(sp[key]['pct'] * 100, 0)}%")
        else:
            c.drawCentredString(x + w / 2, by - 21, name)
            c.setFont("Seg", 10.5)
            c.drawCentredString(x + w / 2, by - 37, f"{n_(sp[key]['n'])} · {d_(sp[key]['pct'] * 100, 0)}%")
        x += w + gap
    # tabela
    ty = by - bh - 22
    tw = 540
    card(c, M, ty, tw, 168)
    cols = [12, 108, 172, 234, 300, 364]
    heads = ["Conjunto", "Imagens", "%", "Gatos", "Cães", "Para quê"]
    c.setFillColor(HexColor(MUTED))
    c.setFont("SegB", 8.5)
    for cx, h_ in zip(cols, heads):
        c.drawString(M + cx, ty - 20, h_.upper())
    c.setStrokeColor(HexColor(LINE))
    c.line(M + 12, ty - 27, M + tw - 12, ty - 27)
    uses = {"train": "Ajusta os pesos da CNN.",
            "val": "Escolhe hiperparâmetros, early stopping.",
            "test": "Nota final. Usado uma única vez, no fim."}
    ry = ty - 34
    for key, name, col, _ in segs:
        c.setFillColor(HexColor(col))
        c.roundRect(M + 12, ry - 19, 6, 22, 2, stroke=0, fill=1)
        c.setFillColor(HexColor(INK))
        c.setFont("SegB", 10.5)
        c.drawString(M + 24, ry - 12, name)
        c.setFont("Seg", 10.5)
        c.drawString(M + cols[1], ry - 12, n_(sp[key]["n"]))
        c.drawString(M + cols[2], ry - 12, f"{d_(sp[key]['pct'] * 100)}%")
        c.drawString(M + cols[3], ry - 12, n_(sp[key]["species"]["gato"]))
        c.drawString(M + cols[4], ry - 12, n_(sp[key]["species"]["cao"]))
        para(c, uses[key], M + cols[5], ry - 2, tw - cols[5] - 12, size=9.2, color=INK2)
        ry -= 44
    para(c, f"Total: {n_(N)} imagens · proporção gato/cão em cada conjunto: "
            f"{d_(sp['train']['species_pct']['gato'] * 100)}% / {d_(sp['val']['species_pct']['gato'] * 100)}% / "
            f"{d_(sp['test']['species_pct']['gato'] * 100)}% gatos.", M, ty - 178, tw, size=9.2, color=INK2)
    # justificativa
    rx = M + tw + 16
    rw = W - M - rx
    card(c, rx, ty, rw, 168 + 40)
    label(c, "Por que esta divisão", rx + 12, ty - 8)
    bullets(c, [
        "<b>70% treino:</b> cerca de 5 mil imagens para a CNN aprender features de pelagem, focinho e orelhas.",
        "<b>15% + 15%:</b> ~1.100 imagens em Val e Test dão margem de erro de ≈ ±1,8 p.p. (IC 95%) para acurácia de ~90% (±3 p.p. no pior caso).",
        "<b>Estratificada por raça</b> (seed 42): cada raça é dividida 70/15/15, então espécie e raça ficam proporcionais nos 3 conjuntos.",
        "<b>Sem vazamento:</b> duplicatas exatas removidas <i>antes</i> do split.",
    ], rx + 12, ty - 26, rw - 24, size=10.2, gap=6)
    card(c, M, ty - 214, W - 2 * M, 92, fill="#f4f3ef")
    label(c, "Como foi feito (reprodutível)", M + 12, ty - 222)
    bullets(c, [
        f"Hash MD5 dos {n_(raw['n_files'])} arquivos → {raw['dup_groups']} grupos idênticos; mantém 1 de cada (nenhum grupo com raças diferentes).",
        f"Divisão em dois passos com <font name='SegB'>train_test_split(stratify=raça, random_state=42)</font>: 70% / 30%, e os 30% em 50/50.",
        "Cópia para <font name='SegB'>dataset/{train,val,test}/{gato,cao}/</font> (pronto para ImageFolder / image_dataset_from_directory) e manifesto <font name='SegB'>splits.csv</font>.",
    ], M + 12, ty - 238, W - 2 * M - 24, size=9.8, gap=4)
    c.showPage()

    # ============================================================ 3. Balanceamento
    header(c, 3, "Balanceamento das classes", "Balanceado por raça, desbalanceado entre gato e cão")
    top = H - 82
    _, ih = image(c, "balance.png", M, top, w=W - 2 * M)
    cy = top - ih - 14
    ch = cy - 42
    cw = (W - 2 * M - 16) / 2
    card(c, M, cy, cw, ch)
    label(c, "O dataset está balanceado?", M + 14, cy - 8)
    status(c, M + 14, cy - 44, "ok")
    para(c, f"<b>Por raça (37 classes): sim.</b> {cnt['breed_min'][1]} a {cnt['breed_max'][1]} imagens por raça "
            f"(razão {d_(cnt['breed_max'][1] / cnt['breed_min'][1], 2)} : 1).", M + 40, cy - 28, cw - 54, size=10.8)
    status(c, M + 14, cy - 96, "warn")
    para(c, f"<b>Por espécie (gato × cão): não.</b> {n_(gato)} × {n_(cao)} = <b>{d_(cnt['imbalance_ratio'], 2)} : 1</b>. "
            f"Um modelo que responde sempre “cão” já acerta {d_(cao / N * 100)}% sem aprender nada.",
         M + 40, cy - 80, cw - 54, size=10.8)
    x2 = M + cw + 16
    card(c, x2, cy, cw, ch)
    label(c, "Como vamos tratar", x2 + 14, cy - 8)
    bullets(c, [
        f"<b>Pesos de classe na loss</b> (n / 2·n<sub>classe</sub>, calculado no Train): gato ≈ {d_(w_gato, 2)} · cão ≈ {d_(w_cao, 2)}.",
        "<b>Métricas que enxergam a minoria:</b> F1 e recall por classe + matriz de confusão, em vez de só acurácia.",
        f"<b>Mesma proporção em Val e Test</b> (split estratificado): métricas comparáveis entre os conjuntos.",
        f"<b>Sem undersampling:</b> descartaria ~{n_(cao - gato)} fotos de cães. Augmentation só no Train.",
    ], x2 + 14, cy - 28, cw - 28, size=10.4, gap=6)
    c.showPage()

    # ============================================================ 4. Comparação (classes + pixels)
    header(c, 4, "Comparação entre Train, Validation e Test (1/2)", "Classes e estatísticas de pixel são comparáveis")
    top = H - 82
    w1_, h1 = image(c, "class_by_split.png", M, top, w=(W - 2 * M) * 0.485)
    image(c, "pixel_stats.png", M + w1_ + 14, top, h=h1)
    my = top - h1 - 12
    mw = (W - 2 * M - 3 * 12) / 4
    ms = [
        (f"p = {d_(S['chi2']['species']['p'], 3)}", "teste χ²: proporção gato/cão × conjunto. Sem diferença detectável (raça: p = 1,0)."),
        (f"≤ {mean_diff_up} de 255", f"maior diferença entre as médias RGB de dois conjuntos ({d_(cmp_['max_mean_diff_255'] / 255 * 100, 2)}% da escala)."),
        (f"D ≤ {f2(ks_D)}", f"KS 2 amostras no brilho e no contraste por imagem (p ≥ {ks_p_dn}). D é a distância máxima entre as curvas acumuladas."),
        (f"≤ {w1_up} níveis", "distância de Wasserstein entre os histogramas RGB (em níveis de intensidade, de 0 a 255)."),
    ]
    for i, (big, small) in enumerate(ms):
        stat(c, M + i * (mw + 12), my, mw, 92, big, small, big_size=18)
    card(c, M, my - 106, W - 2 * M, 62, fill="#f4f3ef")
    para(c, "<b>Leitura:</b> o objetivo não é deixar os conjuntos idênticos, e sim garantir que nenhum tenha um “visual” muito diferente. "
            "As pequenas diferenças (≈ 1 nível de 255) são o ruído esperado de amostrar ~1.100 imagens; nada aponta para viés de seleção.",
         M + 14, my - 118, W - 2 * M - 28, size=10.8, color=INK)
    c.showPage()

    # ============================================================ 5. Histogramas RGB
    header(c, 5, "Comparação entre Train, Validation e Test (2/2)", "Histogramas de intensidade e canais RGB")
    top = H - 80
    fw, fh = image(c, "rgb_hist.png", M, top, h=H - 80 - 44)
    rx = M + fw + 18
    rw = W - M - rx
    card(c, rx, top, rw, 178)
    label(c, "O que a figura mostra", rx + 12, top - 8)
    bullets(c, [
        "As curvas de Train, Validation e Test <b>quase se sobrepõem</b> nos 3 canais.",
        "Embaixo, a diferença vs Train fica em <b>até ≈ 0,6 ponto percentual</b> por faixa de 4 níveis, concentrada nas pontas (0 e 255) e num pico de Validation perto de 128, compatível com ruído de amostragem.",
    ], rx + 12, top - 24, rw - 24, size=9.4, gap=4)
    card(c, rx, top - 190, rw, 168)
    label(c, "Números que sustentam", rx + 12, top - 198)
    b_ = {s: px[s]["brightness_mean"] for s in ("train", "val", "test")}
    bullets(c, [
        f"<b>Intervalo dos pixels:</b> 0 a 255 (uint8) nos 3 conjuntos: min = {px['train']['min']}, max = {px['train']['max']}.",
        f"<b>Brilho médio por imagem:</b> {d_(b_['train'])} · {d_(b_['val'])} · {d_(b_['test'])} (Train · Val · Test), σ ≈ 30.",
        f"<b>Perfil comum:</b> B mais baixo que R e G (média B ≈ {d_(px['train']['mean_255'][2], 0)}, R ≈ {d_(px['train']['mean_255'][0], 0)}): predominam tons quentes.",
    ], rx + 12, top - 214, rw - 24, size=9.2, gap=3)
    card(c, rx, top - 370, rw, 40, fill="#f4f3ef")
    para(c, "<b>Conclusão:</b> Val e Test representam o mesmo tipo de imagem do Train.", rx + 12, top - 378, rw - 24, size=9.6)
    c.showPage()

    # ============================================================ 6. Dimensão
    header(c, 6, "Dimensão e preparação das imagens (1/2)", "Resoluções muito variadas, sempre 3 canais")
    top = H - 82
    _, ih = image(c, "resolution.png", M, top, w=W - 2 * M)
    cy = top - ih - 12
    cw = (W - 2 * M - 3 * 12) / 4
    hh = 92
    stat(c, M, cy, cw, hh, f"{n_(sz['n_unique_resolutions'])}", f"resoluções distintas. Mais comuns: {top_res[0].replace('x', '×')} ({d_(top_n[0] / N * 100)}%) e {top_res[1].replace('x', '×')} ({d_(top_n[1] / N * 100)}%).", big_size=22)
    stat(c, M + (cw + 12), cy, cw, hh, f"{sz['width']['min']}–{n_(sz['width']['max'])} px", f"largura (mediana {n_(sz['width']['median'])}). Altura: {sz['height']['min']}–{n_(sz['height']['max'])} px (mediana {n_(sz['height']['median'])}).", big_size=22)
    stat(c, M + 2 * (cw + 12), cy, cw, hh, "3 canais", f"todas as imagens saem em RGB. {sum(v for k, v in raw['modes'].items() if k != 'RGB')} eram GIF/PNG/tons de cinza/RGBA disfarçados de .jpg e foram convertidos.", big_size=22)
    stat(c, M + 3 * (cw + 12), cy, cw, hh, "0 – 255", f"intervalo dos pixels (uint8) nos 3 conjuntos; ainda sem escala. Proporção típica 4:3 ou 3:4 (só {d_(sz['pct_near_square'] * 100)}% ≈ quadradas).", big_size=22)
    iy = cy - hh - 12
    card(c, M, iy, W - 2 * M, iy - 40, fill="#f4f3ef")
    label(c, "O que isso implica para a CNN", M + 14, iy - 8)
    bullets(c, [
        "<b>Entrada de tamanho fixo:</b> para formar batches e alimentar as camadas densas finais, todas as imagens precisam ter o mesmo tamanho.",
        f"<b>Proporção quase nunca quadrada</b> ({d_(100 - sz['pct_near_square'] * 100)}% das imagens): distorcer, cortar ou preencher têm custos diferentes. A escolha está justificada no próximo slide.",
        f"<b>Extremos:</b> imagens de até {n_(sz['width']['max'])} px serão reduzidas e {sz['n_smaller_than_target']} pequenas serão ampliadas, o que altera um pouco a nitidez.",
    ], M + 14, iy - 24, W - 2 * M - 28, size=10, gap=4)
    c.showPage()

    # ============================================================ 7. Padronização
    header(c, 7, "Dimensão e preparação das imagens (2/2)", "Padronização para a CNN: 224 × 224 × 3")
    top = H - 80
    fw, fh = image(c, "preprocess.png", M, top, h=H - 80 - 44)
    rx = M + fw + 18
    rw = W - M - rx
    steps = [("1", "Resize", "lado menor → 224 px (bicúbico), mantendo a proporção"),
             ("2", "Center crop", "recorta 224 × 224 no centro"),
             ("3", "Normalização", "x / 255 → (x − μ) / σ por canal, com μ e σ do Train")]
    y = top
    for num, name, desc in steps:
        card(c, rx, y, rw, 38)
        c.setFillColor(HexColor(C_TRAIN))
        c.circle(rx + 18, y - 19, 10, stroke=0, fill=1)
        c.setFillColor(white)
        c.setFont("SegB", 11)
        c.drawCentredString(rx + 18, y - 23, num)
        para(c, f"<b>{name}:</b> {desc}", rx + 36, y - 6, rw - 46, size=9.2)
        y -= 44
    para(c, "→ entrada da CNN: 224 × 224 × 3 (float32)", rx, y - 2, rw, size=9.6, color=INK2, bold=True)
    y -= 26
    label(c, "Por que essa estratégia", rx, y)
    y -= 14
    bullets(c, [
        f"<b>Resize com proporção fixa:</b> esticar direto distorce a imagem (×{d_(sz['stretch_ratio_median'], 2)} na mediana) e deforma focinho e orelhas.",
        f"<b>Crop em vez de padding:</b> ambos gastam a mesma fração da entrada (média {d_(sz['aspect_loss']['mean'] * 100)}%). O crop descarta só bordas; o padding preenche com pixels pretos sem informação. Premissa: o animal fica perto do centro (vale nos exemplos vistos).",
        f"<b>224 px:</b> tamanho padrão de backbones pré-treinados. Só {sz['n_smaller_than_target']} imagens ({d_(sz['pct_smaller_than_target'] * 100)}%) têm lado menor &lt; 224 e serão ampliadas.",
        "<b>Val/Test:</b> sempre o mesmo pipeline determinístico. Augmentation (flip, RandomResizedCrop) só no Train.",
    ], rx, y, rw, size=8.8, gap=3)
    ny = 44 + 96
    card(c, rx, ny, rw, 96, fill="#f4f3ef")
    label(c, "Normalização escolhida", rx + 12, ny - 6)
    para(c, f"<b>Só do Train</b> (sem vazamento): μ = ({'; '.join(d_(v, 3) for v in nm['train_mean'])}) e "
            f"σ = ({'; '.join(d_(v, 3) for v in nm['train_std'])}). Aplicada em Val/Test, dá |μ| ≤ {abs_mean_max} e σ entre "
            f"{d_(min(post_s), 3)} e {d_(max(post_s), 3)}. Com backbone pré-treinado, trocar pelas constantes do ImageNet "
            f"(μ = 0,485; 0,456; 0,406 · σ = 0,229; 0,224; 0,225).", rx + 12, ny - 20, rw - 24, size=8.8)
    c.showPage()

    # ============================================================ 8. Análise final
    header(c, 8, "Análise final", "O dataset está pronto para treinar uma CNN?")
    top = H - 84
    cw = (W - 2 * M - 16) / 2
    chh = 190
    tm = nm["train_mean"]
    ts = nm["train_std"]
    cards_ = [
        ("O dataset está balanceado?", "warn", "PARCIALMENTE", [
            f"<b>Raças:</b> sim ({cnt['breed_min'][1]}–{cnt['breed_max'][1]} por raça).",
            f"<b>Gato × cão:</b> não ({d_(cnt['imbalance_ratio'], 2)} : 1; {d_(gato / N * 100)}% × {d_(cao / N * 100)}%).",
            f"<b>Tratamento:</b> pesos de classe (gato ≈ {d_(w_gato, 2)}, cão ≈ {d_(w_cao, 2)}), F1/recall por classe, sem undersampling.",
        ]),
        ("Train, Val e Test são semelhantes?", "ok", "SIM", [
            f"Proporção de classes idêntica (χ² p = {d_(S['chi2']['species']['p'], 3)}).",
            f"Médias RGB a ≤ {mean_diff_up} de 255; KS D ≤ {f2(ks_D)}; histogramas quase sobrepostos.",
            "Pequenas diferenças são ruído de amostragem, não viés.",
        ]),
        ("As imagens estão padronizadas?", "ok", "SIM", [
            "Todas em RGB, 224 × 224 × 3, via resize (proporção fixa) + center crop.",
            f"Normalização com μ = ({', '.join(d_(v, 3) for v in tm)}) e σ = ({', '.join(d_(v, 3) for v in ts)}), só do Train.",
            f"Após normalizar, Val/Test ficam com |μ| ≤ {abs_mean_max} e σ entre {d_(min(post_s), 3)} e {d_(max(post_s), 3)}.",
        ]),
        ("Está pronto para o treinamento?", "ok", "SIM, COM 3 CUIDADOS", [
            "Aplicar pesos de classe e acompanhar F1/recall, não só acurácia.",
            "Usar o Test uma única vez, no final.",
            f"Só duplicatas <i>exatas</i> foram removidas: quase-duplicatas podem existir. E {sz['n_smaller_than_target']} imagens pequenas serão ampliadas.",
        ]),
    ]
    for i, (q, kind, tag, lines) in enumerate(cards_):
        x = M + (i % 2) * (cw + 16)
        y = top - (i // 2) * (chh + 12)
        card(c, x, y, cw, chh)
        label(c, q, x + 14, y - 6)
        status(c, x + 14, y - 47, kind)
        c.setFillColor(HexColor(INK))
        c.setFont("SegB", 15)
        c.drawString(x + 40, y - 45, tag)
        bullets(c, lines, x + 14, y - 64, cw - 28, size=10.4, gap=6)
    c.showPage()
    c.save()
    print("PDF ok ->", PDF)


if __name__ == "__main__":
    build()
