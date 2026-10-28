"""Sprint 4: extrai do notebook executado no Colab (Sprint4_Arquiteturas_CNN.ipynb) os números e as figuras.

Nenhum modelo é executado: só lê as saídas já gravadas no .ipynb e grava
outputs/sprint4/results.json e outputs/sprint4/figures/colab_*.png.
"""
import base64
import json
import re
from io import StringIO
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parent
NB = ROOT / "Sprint4_Arquiteturas_CNN.ipynb"
OUT = ROOT / "outputs" / "sprint4"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

EPOCH = re.compile(r"época\s+(\d+)\s+train loss ([\d.]+) acc ([\d.]+) F1 ([\d.]+) \| val loss ([\d.]+) acc ([\d.]+) F1 ([\d.]+)")
HEAD = re.compile(r"^\[(\w+)\] (\d+) blocos, ([\d.]+) parâmetros", re.M)
DONE = re.compile(r"^\[(\w+)\] melhor época (\d+)/(\d+): .*\(([\d.]+) min\)", re.M)


def pct(s):          # "78,9%" -> 0.789
    return float(str(s).replace("%", "").replace(",", ".")) / 100


def pp_value(s):    # "+12,0 p.p." ou "−8,2 p,p," -> 0.12 / -0.082 (o notebook do Colab escreveu "p,p,")
    m = re.match(r"([+−-])\s*([\d,]+)", s.strip())
    return (-1 if m.group(1) in "−-" else 1) * float(m.group(2).replace(",", ".")) / 100


def outputs(cell):
    for o in cell.outputs:
        yield o, o.get("data", {})


def html_tables(cell):
    """DataFrames exibidos na célula, como texto (sem conversão numérica) e com cabeçalho de um nível."""
    tables = []
    for _, d in outputs(cell):
        if "text/html" in d:
            t = pd.read_html(StringIO(d["text/html"]), thousands=None)[0]
            if isinstance(t.columns, pd.MultiIndex):    # linha extra com o nome do índice
                t.columns = [b if a.startswith("Unnamed") else a for a, b in t.columns]
            tables.append(t)
    return tables


def cell_with(nb, text):
    """Primeira célula de código cujo código-fonte contém `text`."""
    return next(c for c in nb.cells if c.cell_type == "code" and text in c.source)


def main():
    nb = nbformat.read(NB, 4)
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    assert not errors, f"o notebook tem {len(errors)} célula(s) com erro"

    # ---- treino: log por época de cada modelo
    log = "".join(o.get("text", "") for o, _ in outputs(cell_with(nb, "results = {name: train(name, nb)")))
    models, current = {}, None
    for line in log.splitlines():
        if m := HEAD.match(line):
            current = m.group(1)
            models[current] = {"name": current, "n_blocks": int(m.group(2)), "params": int(m.group(3).replace(".", "")),
                               "history": {k: [] for k in ("loss", "accuracy", "f1", "val_loss", "val_accuracy", "val_f1")}}
        elif m := EPOCH.search(line):
            for k, v in zip(models[current]["history"], m.groups()[1:]):
                models[current]["history"][k].append(float(v))
        elif m := DONE.match(line):
            models[m.group(1)].update(best_epoch=int(m.group(2)), epochs_run=int(m.group(3)), minutes=float(m.group(4)))
    for r in models.values():
        r["filters"] = [16 * 2 ** i for i in range(r["n_blocks"])]
        r["receptive_field"] = {2: 10, 3: 22, 4: 46, 5: 94}[r["n_blocks"]]
        assert len(r["history"]["loss"]) == r["epochs_run"], r["name"]

    # ---- Validation: tabela comparativa e F1 por classe
    comp, f1_cls = html_tables(cell_with(nb, 'table.to_csv(OUT / "comparacao_validation.csv"'))[:2]
    comp = comp.set_index(comp.columns[0])
    f1_cls = f1_cls.set_index(f1_cls.columns[0])
    classes = f1_cls.index.tolist()
    for name, r in models.items():
        row = comp.loc[name]
        r["val"] = {"accuracy": pct(row["Accuracy"]), "precision_macro": pct(row["Precision"]),
                    "recall_macro": pct(row["Recall"]), "f1_macro": pct(row["F1 macro"]),
                    "f1": [pct(v) for v in f1_cls[name]]}
        r["observacao"] = row["Observação"]
        d = str(row["Δ F1"])
        r["delta_vs_baseline"] = 0.0 if d in ("—", "nan") else pp_value(d)

    # ---- seleção e Validation por classe do escolhido
    sel_cell = cell_with(nb, "SELECTED, tied = select(results")
    sel_md = next(d["text/markdown"] for _, d in outputs(sel_cell) if "text/markdown" in d)
    selected = re.search(r"Selecionado: \*\*(\w+)\*\*", sel_md).group(1)
    sel_name = next(n for n in models if n.replace("_", " ") == selected or n == selected)
    vt = html_tables(sel_cell)[0].set_index("Validation, por classe")
    models[sel_name]["val"].update(precision=[pct(v) for v in vt.Precision], recall=[pct(v) for v in vt.Recall])

    # ---- Test (uma única avaliação, feita no Colab)
    test_cell = cell_with(nb, 'test_path = OUT / "test.json"')
    tmd = next(d["text/markdown"] for _, d in outputs(test_cell) if "text/markdown" in d)
    acc, prec, rec, f1 = re.findall(r"\| ([\d,]+%) \| ([\d,]+%) \| ([\d,]+%) \| \*\*([\d,]+%)\*\* \|", tmd)[0]
    tt = html_tables(test_cell)[0].set_index("Test, por classe")
    test_minus_val = pp_value(re.search(r"\(([+−-][\d,]+) p[.,]p[.,] de F1 no Test\)", tmd).group(1))
    test = {"name": sel_name, "n": int(tt.Imagens.sum()), "accuracy": pct(acc), "precision_macro": pct(prec),
            "recall_macro": pct(rec), "f1_macro": pct(f1), "precision": [pct(v) for v in tt.Precision],
            "recall": [pct(v) for v in tt.Recall], "f1": [pct(v) for v in tt.F1], "support": tt.Imagens.astype(int).tolist(),
            "f1_minus_val": test_minus_val}
    conf_cell = cell_with(nb, "cm = np.array(test[\"confusion\"])")
    conf_txt = "".join(o.get("text", "") for o, _ in outputs(conf_cell))
    m = re.search(r"Maior confusão: (\d+)% das imagens de '(.+?)' foram previstas como '(.+?)'", conf_txt)
    test["maior_confusao"] = {"pct": int(m.group(1)) / 100, "real": m.group(2), "previsto": m.group(3)}
    ex_cell = cell_with(nb, "ok, err = pred.index")
    ex_txt = "".join(o.get("text", "") for o, _ in outputs(ex_cell))
    m = re.search(r"Acertos: ([\d.]+) de ([\d.]+) \| erros: ([\d.]+) \| confiança média nos acertos ([\d,]+%), nos erros ([\d,]+%)", ex_txt)
    test.update(acertos=int(m.group(1).replace(".", "")), erros=int(m.group(3).replace(".", "")),
                conf_acertos=pct(m.group(4)), conf_erros=pct(m.group(5)))

    # ---- figuras do Colab que dependem das previsões do Test
    for cell, name in ((conf_cell, "colab_confusion.png"), (ex_cell, "colab_examples.png")):
        png = next(d["image/png"] for _, d in outputs(cell) if "image/png" in d)
        (FIG / name).write_bytes(base64.b64decode(png))

    res = {"classes": classes, "baseline": "B3_baseline", "selected": sel_name, "models": models, "test": test}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("ok ->", OUT / "results.json")
    for n, r in models.items():
        print(f"  {n:<12} {r['params']:>8} parâm. | Val F1 {r['val']['f1_macro']:.3f} acc {r['val']['accuracy']:.3f} | "
              f"melhor época {r['best_epoch']}/{r['epochs_run']}")
    print(f"  Test ({sel_name}): F1 {test['f1_macro']:.3f} acc {test['accuracy']:.3f} n={test['n']}")


if __name__ == "__main__":
    main()
