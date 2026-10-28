"""Sprint 4: monta a apresentação (8 slides, PDF 16:9) a partir de outputs/sprint4/results.json e figures/.

Reaproveita as primitivas visuais da Sprint 3 (build_pdf.py). Nenhum modelo é executado aqui.
"""
import json
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

import build_pdf as s3
from build_pdf import (BG, C_TEST, C_TRAIN, C_VAL, INK, INK2, LINE, MEMBERS, MUTED, SOFT, H, M, W,
                       bullets, card, d_, label, n_, para, stat, status)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "sprint4"
FIG = OUT / "figures"
PDF = ROOT / "Sprint4_Arquiteturas_CNN.pdf"
C_SEL = "#4a3aa7"
TOTAL = 8

R = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
S3 = s3.S                                   # stats.json da Sprint 3
MOD, CLASSES, T = R["models"], R["classes"], R["test"]
NAMES = list(MOD)
BASE, SEL = R["baseline"], R["selected"]
sel, base = MOD[SEL], MOD[BASE]
f1 = {n: MOD[n]["val"]["f1_macro"] for n in NAMES}
dbase = {n: MOD[n]["delta_vs_baseline"] for n in NAMES}          # Δ F1 vs baseline, como no notebook
steps = [(a, b, f1[b] - f1[a]) for a, b in zip(NAMES, NAMES[1:])]
big = max(steps, key=lambda s: abs(s[2]))
second = sorted(NAMES, key=lambda n: -f1[n])[1]
last5 = {n: sum(MOD[n]["history"]["val_f1"][-5:]) / 5 for n in NAMES}
better_cls = [c for i, c in enumerate(CLASSES) if MOD[SEL]["val"]["f1"][i] > MOD[second]["val"]["f1"][i]]
gain_cls = sorted(((MOD[SEL]["val"]["f1"][i] - MOD[second]["val"]["f1"][i], c) for i, c in enumerate(CLASSES)), reverse=True)
neg = CLASSES.index("Negative")
worst = min(range(len(CLASSES)), key=lambda i: T["recall"][i])
val_vs_train = all(MOD[n]["history"]["val_f1"][MOD[n]["best_epoch"] - 1] >= MOD[n]["history"]["f1"][MOD[n]["best_epoch"] - 1]
                   for n in NAMES)
sp = S3["split"]


def pc(v, k=1):
    return d_(v * 100, k) + "%"


def pp(v, k=1):
    return ("+" if v >= 0 else "−") + d_(abs(v) * 100, k) + " p.p."


def nm(n):
    return n.replace("_baseline", " (baseline)")


def header(c, n, kicker, title):
    c.setFillColor(HexColor(BG)); c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(HexColor(MUTED)); c.setFont("SegB", 8.5); c.drawString(M, H - 34, kicker.upper())
    c.setFillColor(HexColor(INK)); c.setFont("SegB", 25); c.drawString(M, H - 64, title)
    c.setStrokeColor(HexColor(LINE)); c.setLineWidth(0.8); c.line(M, 26, W - M, 26)
    c.setFillColor(HexColor(MUTED)); c.setFont("Seg", 8)
    c.drawString(M, 13, "Challenge Sprint 4 · Investigação experimental de arquiteturas de CNN · DeepWeeds")
    c.drawRightString(W - M, 13, f"{n} / {TOTAL}")


def image(c, path, x, ytop, w=None, h=None):
    iw, ih = ImageReader(str(path)).getSize()
    w = w if w is not None else h * iw / ih
    h = h if h is not None else w * ih / iw
    c.drawImage(str(path), x, ytop - h, width=w, height=h, mask="auto")
    return w, h


def table(c, x, ytop, cols, header_row, rows, size=9.4, row_h=21, bold_row=None, colors=None):
    """Tabela simples: cols = larguras; colors[i] pinta a marca da linha i."""
    c.setFillColor(HexColor(MUTED)); c.setFont("SegB", 7.8)
    cx = x
    for w_, t in zip(cols, header_row):
        c.drawString(cx, ytop - 10, t.upper()); cx += w_
    c.setStrokeColor(HexColor(LINE)); c.setLineWidth(0.8); c.line(x, ytop - 16, x + sum(cols), ytop - 16)
    y = ytop - 16
    for i, row in enumerate(rows):
        if colors:
            c.setFillColor(HexColor(colors[i])); c.roundRect(x - 9, y - row_h + 4, 4, row_h - 8, 1.5, stroke=0, fill=1)
        cx = x
        for w_, t in zip(cols, row):
            para(c, t, cx, y - 4, w_ - 6, size=size, bold=(i == bold_row))
            cx += w_
        y -= row_h
        c.setStrokeColor(HexColor("#eeede8")); c.line(x, y, x + sum(cols), y)
    return ytop - y


def build():
    c = canvas.Canvas(str(PDF), pagesize=(W, H))
    c.setTitle("Sprint 4 – Investigação experimental de arquiteturas de CNN (DeepWeeds)")
    c.setAuthor("; ".join(f"{n} (RM {rm})" for n, rm in MEMBERS))

    # ============================================================ 1. Problema
    header(c, 1, "Sprint 4 · Problema e modalidade", "Classificar a espécie de planta em cada foto")
    top = H - 84
    stat(c, M, top, 300, 62, "Classificação", "uma classe por imagem, entre 9: 8 espécies invasoras + “Negative”", big_size=21)
    stat(c, M, top - 70, 300, 62, "224 × 224 × 3", "entrada: foto de campo RGB (256 × 256 com crop), normalizada", big_size=21)
    stat(c, M, top - 140, 300, 62, "9 probabilidades", "saída softmax; a previsão é a classe mais provável", big_size=21)
    card(c, M, top - 210, 300, 196)
    label(c, "Métrica principal: F1 macro", M + 12, top - 220)
    bullets(c, [
        f"“Negative” é <b>{pc(S3['counts']['negative_share'], 0)}</b> das fotos: responder sempre “Negative” já daria {pc(S3['counts']['negative_share'], 0)} de accuracy.",
        "<b>F1 macro</b> = média do F1 das 9 classes, cada uma com o mesmo peso: uma espécie mal reconhecida derruba a nota.",
        "Também reportamos <b>accuracy, precision e recall</b> (macro e por classe) e a <b>matriz de confusão</b>.",
        "Não foram necessárias anotações extras: a base já traz um rótulo por imagem.",
    ], M + 12, top - 238, 276, size=9.6, gap=4)
    ix = M + 320
    gw, gh = image(c, ROOT / "outputs" / "figures" / "samples.png", ix, top, w=300)
    para(c, "Uma foto de cada classe (Train). Contorno violeta = espécie-alvo; cinza = Negative.", ix, top - gh - 4, gw, size=8.5, color=MUTED)
    cx = ix + gw + 16
    cw = W - M - cx
    card(c, cx, top, cw, 176, fill=SOFT)
    label(c, "Pergunta da Sprint 4", cx + 12, top - 8)
    para(c, "<b>Quantos blocos convolucionais</b> uma CNN treinada do zero precisa para reconhecer essas espécies?<br/><br/>"
            "Testamos <b>2, 3, 4 e 5 blocos</b>, mudando só a profundidade, com os mesmos dados, pré-processamento e protocolo de treino.",
         cx + 12, top - 24, cw - 24, size=9.6)
    my = top - 176 - 12
    card(c, cx, my, cw, my - 40)
    label(c, "Integrantes", cx + 12, my - 8)
    y = my - 24
    for name, rm in MEMBERS:
        y -= para(c, f"<b>{name}</b> · RM {rm}", cx + 12, y, cw - 24, size=8.4) + 3
    c.showPage()

    # ============================================================ 2. Dataset
    header(c, 2, "Dataset e continuidade da Sprint 3", "Mesmas imagens, mesma divisão, mesmo pré-processamento")
    bx, by, bw, bh, gap = M, H - 84, W - 2 * M, 44, 3
    x = bx
    for key, name, col, txt in [("train", "Train", C_TRAIN, "#ffffff"), ("val", "Validation", C_VAL, INK), ("test", "Test", C_TEST, INK)]:
        w_ = (bw - 2 * gap) * sp[key]["pct"]
        c.setFillColor(HexColor(col)); c.roundRect(x, by - bh, w_, bh, 6, stroke=0, fill=1)
        c.setFillColor(HexColor(txt)); c.setFont("SegB", 12)
        if w_ > 300:
            c.drawCentredString(x + w_ / 2, by - bh / 2 - 4, f"{name} · {n_(sp[key]['n'])} imagens · 70%")
        else:
            c.drawCentredString(x + w_ / 2, by - 19, name)
            c.setFont("Seg", 10.5); c.drawCentredString(x + w_ / 2, by - 34, f"{n_(sp[key]['n'])} · 15%")
        x += w_ + gap
    ty = by - bh - 14
    tw = 520
    card(c, M, ty, tw, 128)
    uses = {"train": "ajusta os pesos de cada CNN", "val": "escolhe a época e a arquitetura", "test": "nota final, lido uma única vez"}
    table(c, M + 22, ty - 10, [90, 66, 70, 70, 190], ["Conjunto", "Imagens", "Espécies", "Negative", "Papel nesta sprint"],
          [[nm_, n_(sp[k]["n"]), n_(sp[k]["n"] - sp[k]["per_class"]["Negative"]), n_(sp[k]["per_class"]["Negative"]), uses[k]]
           for k, nm_ in (("train", "Train"), ("val", "Validation"), ("test", "Test"))],
          colors=[C_TRAIN, C_VAL, C_TEST], row_h=24)
    rx, rw = M + tw + 16, W - 2 * M - tw - 16
    card(c, rx, ty, rw, 128)
    label(c, "Continuidade garantida", rx + 12, ty - 8)
    bullets(c, [
        f"A divisão <b>não foi refeita</b>: o split do <font name='SegB'>splits.csv</font> da Sprint 3 vai embutido no notebook.",
        f"No Colab, o notebook confere os {n_(S3['counts']['total'])} nomes de arquivo (MD5) e as contagens antes de treinar.",
        "O Test só é carregado depois da escolha da arquitetura.",
    ], rx + 12, ty - 24, rw - 24, size=9.4, gap=4)
    py = ty - 128 - 12
    label(c, "Tamanho de entrada e pré-processamento (igual nos 4 modelos)", M, py - 2)
    fw, fh = image(c, ROOT / "outputs" / "figures" / "preprocess.png", M, py - 16, h=py - 16 - 46)
    para(c, "Exemplos do Train. Contorno azul = crop central usado em Val/Test.", M, py - 16 - fh - 3, fw, size=7.8, color=MUTED)
    chips = [("1", "Crop 224 × 224, sem resize", "Train: posição aleatória + espelho horizontal. Val/Test: centro da foto 256 × 256."),
             ("2", "Normalização dentro do modelo", "÷255 e (x − μ)/σ por canal, com μ e σ medidos só no Train na Sprint 3."),
             ("3", "Pesos de classe na loss", f"n / (k · n<sub>classe</sub>), no Train: Negative ≈ {d_(S3['counts']['weights_train']['Negative'], 2)}; espécies ≈ 1,73 a 1,93."),
             ("4", "Mesmo protocolo", "Mesmos dados, augmentation, loss, otimizador, épocas e seed em todos os experimentos.")]
    cx = M + fw + 14
    chw = W - M - cx
    chh = (py - 16 - 32 - 3 * 8) / 4
    for i, (num, t, desc) in enumerate(chips):
        y = py - 16 - i * (chh + 8)
        card(c, cx, y, chw, chh, fill=SOFT if i % 2 else "#ffffff")
        c.setFillColor(HexColor(C_TRAIN)); c.circle(cx + 18, y - chh / 2, 10, stroke=0, fill=1)
        c.setFillColor(white); c.setFont("SegB", 10.5); c.drawCentredString(cx + 18, y - chh / 2 - 3.6, num)
        para(c, f"<b>{t}:</b> {desc}", cx + 38, y - 9, chw - 50, size=9.6)
    c.showPage()

    # ============================================================ 3. Baseline
    header(c, 3, "Arquitetura baseline", f"Ponto de partida: 3 blocos convolucionais, {n_(base['params'])} parâmetros")
    top = H - 82
    _, ih = image(c, FIG / "baseline.png", M, top, w=W - 2 * M)
    ty = top - ih - 12
    tw = 470
    card(c, M, ty, tw, ty - 40)
    label(c, "model.summary() resumido (Keras)", M + 12, ty - 8)
    rows = [["Rescaling + Normalization", "224 × 224 × 3", "0"],
            ["Bloco 1: Conv2D 16 + BN + ReLU + MaxPool", "112 × 112 × 16", "496"],
            ["Bloco 2: Conv2D 32 + BN + ReLU + MaxPool", "56 × 56 × 32", "4.736"],
            ["Bloco 3: Conv2D 64 + BN + ReLU + MaxPool", "28 × 28 × 64", "18.688"],
            ["GlobalAveragePooling2D", "64", "0"],
            ["Dense 64 (ReLU) + Dropout 0,3", "64", "4.160"],
            ["Dense 9 (softmax)", "9", "585"],
            ["<b>Total</b> (28.441 treináveis)", "", f"<b>{n_(base['params'])}</b>"]]
    table(c, M + 12, ty - 24, [250, 110, 80], ["Camada", "Saída", "Parâmetros"], rows, size=9, row_h=19.5)
    rx, rw = M + tw + 16, W - 2 * M - tw - 16
    stat(c, rx, ty, (rw - 10) / 2, 64, f"{base['receptive_field']} px", "campo receptivo: cada ativação final “vê” 22 × 22 px da foto", big_size=19)
    stat(c, rx + (rw - 10) / 2 + 10, ty, (rw - 10) / 2, 64, "GAP", "média de cada mapa: 64 números, poucos parâmetros na cabeça", big_size=19)
    py = ty - 74
    card(c, rx, py, rw, py - 40)
    label(c, "Protocolo de treino (fixo nos 4 modelos)", rx + 12, py - 8)
    bullets(c, [
        "Adam, lr 1e-3, reduzido à metade quando a loss de Val para de cair por 3 épocas.",
        "Batch 64, até 30 épocas; early stopping pelo F1 macro de Val (paciência 8).",
        "Guardamos os pesos da <b>época de maior F1 macro no Validation</b>.",
        "Entropia cruzada ponderada; seed 42; GPU T4 no Google Colab (mixed precision).",
        "Kernel 3×3, MaxPool 2×2, BatchNorm, GAP e Dense 64 iguais em todos os modelos.",
    ], rx + 12, py - 24, rw - 24, size=9.4, gap=3)
    c.showPage()

    # ============================================================ 4. Experimentos
    header(c, 4, "Experimentos de arquitetura", "Uma única mudança: o número de blocos convolucionais")
    top = H - 82
    _, ih = image(c, FIG / "blocks.png", M + 44, top, w=W - 2 * M - 88)
    cy = top - ih - 10
    cw = (W - 2 * M - 16) / 2
    card(c, M, cy, cw, cy - 40)
    label(c, "Por que cada modelo foi testado", M + 14, cy - 8)
    bullets(c, [
        f"<b>B2 (−1 bloco):</b> rede rasa basta? Campo receptivo de {MOD['B2']['receptive_field']} px: só textura; esperávamos <i>underfitting</i>.",
        "<b>B3 (baseline):</b> ponto de partida simples, com 3 blocos.",
        f"<b>B4 (+1 bloco):</b> campo receptivo de {MOD['B4']['receptive_field']} px: ver folhas inteiras generaliza melhor?",
        f"<b>B5 (+2 blocos):</b> até onde aprofundar? {n_(MOD['B5']['params'])} parâmetros (14× a baseline): risco de <i>overfitting</i>.",
    ], M + 14, cy - 26, cw - 28, size=9.4, gap=4)
    x2 = M + cw + 16
    card(c, x2, cy, cw, cy - 40, fill=SOFT)
    label(c, "Como mantivemos a comparação controlada", x2 + 14, cy - 8)
    bullets(c, [
        "<b>Só muda a profundidade:</b> filtros começam em 16 e dobram a cada bloco; o bloco novo é a única diferença.",
        "Cada bloco a mais <b>dobra o campo receptivo</b> e <b>reduz o mapa final à metade</b>.",
        "Mesmo split, pré-processamento, loss, otimizador, épocas, seed e early stopping.",
        "<b>Regra de escolha fixada antes:</b> maior F1 macro no Val; empate (&lt; 1 p.p.) → menos parâmetros.",
    ], x2 + 14, cy - 26, cw - 28, size=9.4, gap=4)
    c.showPage()

    # ============================================================ 5. Comparação no Validation
    header(c, 5, "Comparação no Validation", "Cada bloco a mais melhorou o F1 macro, sem overfitting")
    top = H - 80
    obs = {"B2": f"Underfitting: F1 de Train só {d_(MOD['B2']['history']['f1'][-1], 2)}, ainda subindo na época 30",
           "B3_baseline": f"Ponto de partida; underfitting (F1 Train {d_(MOD['B3_baseline']['history']['f1'][-1], 2)})",
           "B4": f"{pp(dbase['B4'])} vs baseline; Val ≥ Train, sem overfitting",
           "B5": f"{pp(dbase['B5'])} vs baseline; sem overfitting; Val oscila ±2 p.p."}
    rows = [[f"<b>{nm(n)}</b>", "Arquitetura inicial (3 blocos)" if n == BASE else
             {2: "−1 bloco", 4: "+1 bloco", 5: "+2 blocos"}[MOD[n]["n_blocks"]] + f" ({MOD[n]['n_blocks']} blocos)",
             n_(MOD[n]["params"]), pc(MOD[n]["val"]["accuracy"]), f"<b>{pc(f1[n])}</b>", f"{MOD[n]['best_epoch']}/30", obs[n]]
            for n in NAMES]
    card(c, M, top, W - 2 * M, 132)
    table(c, M + 22, top - 8, [96, 150, 74, 66, 66, 50, 340],
          ["Experimento", "Mudança principal", "Parâmetros", "Accuracy", "F1 macro", "Época", "Observação (curvas)"],
          rows, size=9, row_h=26, colors=[C_SEL if n == SEL else "#898781" if n == BASE else "#b9b3e0" for n in NAMES])
    fy = top - 132 - 8
    fw, fh = image(c, FIG / "curves.png", M, fy, h=fy - 32)
    nx = M + fw + 12
    card(c, nx, fy, W - M - nx, fy - 32, fill=SOFT)
    label(c, "Leitura", nx + 10, fy - 8)
    bullets(c, [
        "<b>Train abaixo do Val</b> porque é medido com augmentation e dropout ligados.",
        "<b>Nenhum sinal de overfitting:</b> as curvas de Val acompanham as de Train.",
        "<b>B2 e B3:</b> curvas baixas e ainda subindo, falta capacidade.",
        "Linha pontilhada = época cujos pesos foram guardados.",
    ], nx + 10, fy - 24, W - M - nx - 20, size=8.6, gap=3)
    c.showPage()

    # ============================================================ 6. Arquitetura selecionada
    header(c, 6, "Arquitetura selecionada", f"{nm(SEL)}: {sel['n_blocks']} blocos, escolhido pelo Validation")
    top = H - 82
    sw = (W - 2 * M - 3 * 12) / 4
    stat(c, M, top, sw, 70, pc(f1[SEL]), f"F1 macro no Validation ({pp(dbase[SEL])} sobre a baseline)", big_size=22, color=C_SEL)
    stat(c, M + (sw + 12), top, sw, 70, pp(f1[SEL] - f1[second]), f"sobre o 2º melhor ({nm(second)}): fora da margem de empate de 1 p.p.", big_size=22)
    stat(c, M + 2 * (sw + 12), top, sw, 70, f"{pc(last5[SEL])} × {pc(last5[second])}",
         f"média das 5 últimas épocas ({nm(SEL)} × {nm(second)}): não é um pico isolado", big_size=17)
    stat(c, M + 3 * (sw + 12), top, sw, 70, n_(sel["params"]), f"parâmetros ({sel['params'] / base['params']:.0f}× a baseline, ~1,6 MB): ainda uma rede pequena", big_size=22)
    fy = top - 82
    w1, h1 = image(c, FIG / "compare.png", M, fy - 12, w=300)
    w2, _ = image(c, FIG / "per_class.png", M + w1 + 8, fy, w=282)
    rx = M + w1 + w2 + 20
    card(c, rx, fy, W - M - rx, fy - 34)
    label(c, "Por que " + nm(SEL), rx + 12, fy - 8)
    bullets(c, [
        f"<b>Maior F1 macro no Validation</b> ({pc(f1[SEL])}), que é a regra definida antes do treino.",
        f"Melhora o F1 em <b>{len(better_cls)} de 9 classes</b> sobre o {nm(second)}; o maior ganho é em {gain_cls[0][1]} ({pp(gain_cls[0][0], 0)}).",
        "<b>Sem overfitting:</b> Val ≥ Train na melhor época." if val_vs_train else "Curvas sem sinal forte de overfitting.",
        f"Campo receptivo de {sel['receptive_field']} px: cada ativação vê uma região maior da planta.",
        f"<b>Classes mais equilibradas:</b> a pior classe sobe de {pc(min(MOD[second]['val']['f1']), 0)} ({nm(second)}) para {pc(min(sel['val']['f1']), 0)} ({nm(SEL)}).",
        "<b>Não foi escolhido por ser maior:</b> se B4 empatasse (&lt; 1 p.p.), ficaria B4.",
    ], rx + 12, fy - 24, W - M - rx - 24, size=9.8, gap=6)
    c.showPage()

    # ============================================================ 7. Test
    header(c, 7, "Avaliação final no Test (uma única vez)", f"{nm(SEL)} no Test: F1 macro {pc(T['f1_macro'])}, accuracy {pc(T['accuracy'])}")
    top = H - 82
    for i, (big_, small, col) in enumerate([
            (pc(T["accuracy"]), f"accuracy ({n_(T['acertos'])} de {n_(T['n'])} fotos)", INK),
            (pc(T["precision_macro"]), "precision macro", INK),
            (pc(T["recall_macro"]), "recall macro", INK),
            (pc(T["f1_macro"]), f"F1 macro ({pp(T['f1_minus_val'])} vs Validation)", C_SEL)]):
        stat(c, M + i * (sw + 12), top, sw, 60, big_, small, big_size=22, color=col)
    fy = top - 70
    cw_, ch_ = image(c, FIG / "colab_confusion.png", M, fy, h=fy - 30)
    ex = M + cw_ + 12
    ew = W - M - ex
    _, eh = image(c, FIG / "colab_examples.png", ex, fy, w=ew)
    ry = fy - eh - 8
    card(c, ex, ry, ew, ry - 30, fill=SOFT)
    label(c, "O que os erros mostram", ex + 12, ry - 8)
    mc = T["maior_confusao"]
    bullets(c, [
        f"<b>Negative:</b> precision {pc(T['precision'][neg])}, mas recall {pc(T['recall'][neg])}. O modelo erra mais por falso alarme de espécie do que por deixar uma passar, o que é coerente com os pesos de classe na loss. Os 6 erros sorteados acima são todos desse tipo.",
        f"<b>Maior confusão:</b> {pc(mc['pct'], 0)} das fotos de {mc['real']} foram previstas como {mc['previsto']}.",
        f"<b>Mais difícil:</b> {CLASSES[worst]} (recall {pc(T['recall'][worst])}); ela e Chinee apple se confundem nos dois sentidos.",
        f"<b>Confiança média:</b> {pc(T['conf_acertos'])} nos acertos × {pc(T['conf_erros'])} nos erros: os erros vêm com menos certeza.",
    ], ex + 12, ry - 24, ew - 24, size=9.6, gap=4)
    c.showPage()

    # ============================================================ 8. Conclusão
    header(c, 8, "Conclusão", "Profundidade foi decisiva; o ganho por bloco diminui")
    top = H - 84
    cw = (W - 2 * M - 16) / 2
    chh = 196
    cards_ = [
        ("Qual alteração teve maior impacto?", "ok", f"DE {MOD[big[0]]['n_blocks']} PARA {MOD[big[1]]['n_blocks']} BLOCOS: {pp(big[2])}", [
            "Passos de profundidade (F1 macro, Val): " + " · ".join(f"{MOD[a]['n_blocks']}→{MOD[b]['n_blocks']}: <b>{pp(d)}</b>" for a, b, d in steps) + ".",
            f"O ganho cai no último passo (retorno decrescente), enquanto os parâmetros quase quadruplicam a cada bloco.",
            f"Explicação provável: o campo receptivo passa de {base['receptive_field']} para {MOD['B4']['receptive_field']} px, então cada ativação vê uma região maior da planta, não só textura local.",
        ]),
        ("Arquitetura final", "ok", f"{nm(SEL).upper()} · {sel['n_blocks']} BLOCOS · {n_(sel['params'])} PARÂMETROS", [
            f"Validation: F1 macro <b>{pc(f1[SEL])}</b>, accuracy {pc(sel['val']['accuracy'])}.",
            f"Test (uma vez): F1 macro <b>{pc(T['f1_macro'])}</b>, accuracy {pc(T['accuracy'])}, recall {pc(T['recall_macro'])}.",
            "Escolhida pelo F1 no Validation, não pelo tamanho; nenhuma das 4 redes mostrou overfitting em 30 épocas.",
        ]),
        ("Limitações observadas", "warn", "4 PONTOS DE ATENÇÃO", [
            "<b>Uma seed por modelo</b>, e o F1 de Val oscila ±2 p.p. entre épocas. Por isso usamos a regra de empate e conferimos a média das 5 últimas épocas.",
            f"<b>Test otimista para locais novos:</b> {pc(S3['leakage']['pct_same_session'])} das fotos de Val/Test têm foto de Train da mesma sessão de captura (Sprint 3).",
            f"<b>Falso alarme:</b> recall de Negative é {pc(T['recall'][neg])}. A classe mais difícil é {CLASSES[worst]} (recall {pc(T['recall'][worst])}).",
            "<b>Só a profundidade foi testada:</b> filtros, kernel, pooling e cabeça ficaram fixos.",
        ]),
        ("O que testaríamos na próxima etapa", "ok", "PRÓXIMOS EXPERIMENTOS", [
            "6 blocos, ou 2 convoluções por bloco, para ver onde a profundidade satura.",
            "Sobre o B5: mais filtros (32 → 512) e kernel 5×5, uma mudança por vez.",
            "Mais épocas e 3 seeds por modelo (média ± desvio).",
            "Split por sessão de captura e transfer learning (MobileNet/ResNet) como teto.",
        ]),
    ]
    for i, (q, kind, tag, lines) in enumerate(cards_):
        x = M + (i % 2) * (cw + 16)
        y = top - (i // 2) * (chh + 12)
        card(c, x, y, cw, chh)
        label(c, q, x + 14, y - 6)
        status(c, x + 14, y - 45, kind)
        c.setFillColor(HexColor(INK)); c.setFont("SegB", 12.5); c.drawString(x + 40, y - 43, tag)
        bullets(c, lines, x + 14, y - 64, cw - 28, size=10.4, gap=6)
    c.showPage()
    c.save()
    print("PDF ok ->", PDF)


if __name__ == "__main__":
    build()
