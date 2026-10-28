"""Funções puras de preparação do dataset (rótulos, split, padronização)."""
import hashlib
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

SPLITS = ("train", "val", "test")


# ---------- rótulos ----------

def parse_label(filename):
    """'Abyssinian_1.jpg' -> ('gato', 'Abyssinian'); 'beagle_42.jpg' -> ('cao', 'beagle').

    Convenção do Oxford-IIIT Pet: raças de gato começam com maiúscula.
    """
    breed = Path(filename).stem.rsplit("_", 1)[0]
    return ("gato" if breed[0].isupper() else "cao"), breed


# ---------- duplicatas ----------

def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def drop_duplicates(paths):
    """Mantém a primeira ocorrência de cada conteúdo idêntico. Retorna (mantidos, removidos)."""
    seen, kept, removed = set(), [], []
    for p in paths:
        digest = file_md5(p)
        if digest in seen:
            removed.append(p)
        else:
            seen.add(digest)
            kept.append(p)
    return kept, removed


# ---------- split estratificado ----------

def assign_splits(strata, ratios=(0.70, 0.15, 0.15), seed=42):
    """Devolve o nome do split ('train'/'val'/'test') de cada item, estratificado por `strata`."""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"as razões devem somar 1, recebido {sum(ratios)}")
    train_r, val_r, test_r = ratios
    idx = np.arange(len(strata))
    strata = np.asarray(strata)

    train_idx, rest_idx = train_test_split(
        idx, test_size=val_r + test_r, stratify=strata, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=test_r / (val_r + test_r),
        stratify=strata[rest_idx],
        random_state=seed,
    )

    out = [None] * len(strata)
    for name, members in zip(SPLITS, (train_idx, val_idx, test_idx)):
        for i in members:
            out[i] = name
    return out


# ---------- padronização das imagens ----------

def load_rgb(path):
    """Abre qualquer imagem (JPEG, PNG/GIF disfarçado, L, P, RGBA) como RGB de 3 canais."""
    with Image.open(path) as im:
        return im.convert("RGB")


def resize_center_crop(img, size):
    """Redimensiona o lado menor para `size` (mantém proporção) e recorta o centro size×size."""
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - size) // 2, (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


def resize_pad(img, size, fill=(0, 0, 0)):
    """Redimensiona o lado maior para `size` (mantém proporção) e completa com `fill` (letterbox)."""
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def resize_stretch(img, size):
    """Redimensiona direto para size×size, distorcendo a proporção."""
    return img.resize((size, size), Image.BICUBIC)


def aspect_loss(w, h):
    """Fração da área original descartada (crop) ou preenchida com vazio (padding)
    ao levar uma imagem w×h para um quadrado: 1 - lado_menor/lado_maior."""
    return 1 - min(w, h) / max(w, h)


# ---------- normalização ----------

def normalize(arr, mean, std):
    """uint8 HWC [0,255] -> float32 HWC, escalado para [0,1] e padronizado por canal."""
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if (std <= 0).any():
        raise ValueError("o desvio-padrão deve ser > 0 em todos os canais")
    x = np.asarray(arr, dtype=np.float32) / 255.0
    return ((x - mean) / std).astype(np.float32)


# ---------- DeepWeeds: nomes de arquivo, labels.csv e pesos de classe ----------

_CAPTURE_RE = re.compile(r"^(\d{8})-(\d{6})-(\d+)\.\w+$")


def parse_capture_id(filename):
    """'20170320-093423-1.jpg' -> {'date': '20170320', 'time': '093423', 'instrument': 1}.

    Padrão da DeepWeeds: YYYYMMDD-HHMMSS-ID, com ID = câmera (0 a 3) do equipamento em campo.
    """
    m = _CAPTURE_RE.match(Path(filename).name)
    if not m:
        raise ValueError(f"nome fora do padrão YYYYMMDD-HHMMSS-ID: {filename!r}")
    return {"date": m.group(1), "time": m.group(2), "instrument": int(m.group(3))}


def capture_session(filename):
    """Sessão de captura = data + instrumento; fotos da mesma sessão são muito parecidas entre si."""
    info = parse_capture_id(filename)
    return f"{info['date']}-{info['instrument']}"


def load_labels(path):
    """Lê o labels.csv (Filename, Label, Species) e acrescenta is_negative, date, instrument e session."""
    raw = pd.read_csv(path)
    missing = {"Filename", "Label", "Species"} - set(raw.columns)
    if missing:
        raise ValueError(f"colunas ausentes em {path}: {sorted(missing)}")
    df = pd.DataFrame({
        "file": raw["Filename"].astype(str),
        "label": raw["Label"].astype(int),
        "species": raw["Species"].astype(str),
    })
    df["is_negative"] = df.species.str.lower() == "negative"
    info = pd.DataFrame([parse_capture_id(f) for f in df.file])
    df["date"], df["instrument"] = info["date"], info["instrument"]
    df["session"] = df["date"] + "-" + df["instrument"].astype(str)
    return df


def balanced_class_weights(labels):
    """Peso de cada classe = n_total / (n_classes · n_classe): a classe rara pesa mais na loss."""
    counts = Counter(labels)
    if not counts:
        raise ValueError("a lista de rótulos está vazia")
    n, k = sum(counts.values()), len(counts)
    return {c: n / (k * v) for c, v in counts.items()}


def capture_seconds(filename):
    """Instante da captura, em segundos desde 2000-01-01 (inteiro; evita a unidade variável de datetime64)."""
    info = parse_capture_id(filename)
    dt = datetime.strptime(info["date"] + info["time"], "%Y%m%d%H%M%S")
    return int((dt - datetime(2000, 1, 1)).total_seconds())


def nearest_train_gap(df, time_col="t", group_cols=("session", "label"), split_col="split"):
    """Para cada foto fora do treino, o menor intervalo (s) até uma foto de treino do mesmo grupo.

    NaN = a própria foto é de treino; inf = o grupo (sessão + classe) não tem nenhuma foto de treino.
    Serve para medir o risco de vazamento por fotos quase iguais entre Train e Val/Test.
    """
    times = df[time_col].to_numpy(dtype=float)
    is_train = (df[split_col] == "train").to_numpy()
    gaps = np.full(len(df), np.nan)
    for idx in df.groupby(list(group_cols)).indices.values():
        train_t = np.sort(times[idx][is_train[idx]])
        for p in idx[~is_train[idx]]:
            if train_t.size == 0:
                gaps[p] = np.inf
                continue
            i = np.searchsorted(train_t, times[p])
            cand = []
            if i < train_t.size:
                cand.append(train_t[i] - times[p])
            if i > 0:
                cand.append(times[p] - train_t[i - 1])
            gaps[p] = min(cand)
    return gaps


# ---------- recortes sem reamostragem ----------

def center_crop(img, size):
    """Recorta o centro size×size sem reamostrar: os pixels do recorte são idênticos aos da origem."""
    w, h = img.size
    if size > w or size > h:
        raise ValueError(f"recorte {size} maior que a imagem {w}x{h}")
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def random_crop(img, size, rng):
    """Recorte size×size numa posição sorteada por `rng` (numpy Generator). Uso: augmentation no Train."""
    w, h = img.size
    if size > w or size > h:
        raise ValueError(f"recorte {size} maior que a imagem {w}x{h}")
    left = int(rng.integers(0, w - size + 1))
    top = int(rng.integers(0, h - size + 1))
    return img.crop((left, top, left + size, top + size))


def crop_area_lost(side, size):
    """Fração da área descartada ao recortar size×size de uma imagem quadrada de lado `side`."""
    return 1 - (size / side) ** 2
