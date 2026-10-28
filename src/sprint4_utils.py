"""Sprint 4: funções puras de apoio (sem TensorFlow). O notebook do Colab recebe uma cópia deste código."""
import base64
import hashlib
import zlib

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

SPLIT_CODE = {"train": "0", "val": "1", "test": "2"}


# ---------- divisão da Sprint 3 embutida no notebook ----------

def encode_split(splits):
    """['train', 'val', ...] -> texto curto (um dígito por imagem, zlib + base64)."""
    return base64.b64encode(zlib.compress("".join(SPLIT_CODE[s] for s in splits).encode(), 9)).decode()


def decode_split(blob):
    """Inverso de encode_split."""
    names = {v: k for k, v in SPLIT_CODE.items()}
    return [names[c] for c in zlib.decompress(base64.b64decode(blob)).decode()]


def files_md5(files):
    """Impressão digital da lista de arquivos, na ordem: confere se o labels.csv é o mesmo da Sprint 3."""
    return hashlib.md5("\n".join(files).encode()).hexdigest()


# ---------- arquitetura e dados ----------

def receptive_field(n_blocks, kernel=3):
    """Lado (px) da região vista por uma ativação do último mapa, com [Conv k×k + MaxPool 2×2] por bloco."""
    rf, jump = 1, 1
    for _ in range(n_blocks):
        rf += (kernel - 1) * jump + jump
        jump *= 2
    return rf


def class_weights(labels, n_classes):
    """Peso da classe = n / (k · n_classe), calculado no Train (definido na Sprint 3)."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes)
    return len(labels) / (n_classes * counts)


def metrics(y_true, y_pred, n_classes):
    """Accuracy, precision/recall/F1 macro e por classe, e matriz de confusão."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(n_classes), zero_division=0)
    return {"accuracy": float((y_true == y_pred).mean()),
            "precision_macro": float(p.mean()), "recall_macro": float(r.mean()), "f1_macro": float(f.mean()),
            "precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(),
            "confusion": confusion_matrix(y_true, y_pred, labels=range(n_classes)).tolist()}


# ---------- leitura dos resultados (só Validation) ----------

def select(results, margin=0.01):
    """Maior F1 macro no Validation; entre os que ficam a menos de `margin` do melhor, o de menos parâmetros.

    results: {nome: {"params": int, "val": {"f1_macro": float}}}. Devolve (nome escolhido, [empatados]).
    """
    top = max(r["val"]["f1_macro"] for r in results.values())
    tied = [n for n, r in results.items() if top - r["val"]["f1_macro"] < margin]
    return min(tied, key=lambda n: results[n]["params"]), tied


def diagnose(history, best_epoch):
    """Leitura das curvas na melhor época e depois dela (heurística; conferir com os gráficos).

    history: listas por época de accuracy, val_accuracy e val_loss. best_epoch começa em 1.
    """
    i = best_epoch - 1
    gap = history["accuracy"][i] - history["val_accuracy"][i]
    after = history["val_loss"][i:]
    rising = len(after) > 1 and after[-1] > 1.10 * min(history["val_loss"])
    if gap > 0.08 or (gap > 0.04 and rising):
        return "overfitting"
    if history["accuracy"][i] < 0.80:
        return "underfitting"
    return "bom ajuste"
