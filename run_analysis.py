"""Sprint 3 (DeepWeeds): valida a base, divide em Train/Val/Test e mede as características das imagens.

Entradas : data/labels.csv e data/images/*.jpg  (DeepWeeds, https://github.com/AlexOlsen/DeepWeeds)
Saídas   : dataset/{train,val,test}/<Classe>/*.jpg   imagens separadas
           outputs/splits.csv   manifesto (arquivo, classe, sessão, split...)
           outputs/stats.json   todos os números usados nos slides
           outputs/hist.npz     histogramas RGB por split; outputs/gaps.npy distâncias no tempo
"""
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import dataprep  # noqa: E402

IMAGES = ROOT / "data" / "images"
LABELS = ROOT / "data" / "labels.csv"
DATASET = ROOT / "dataset"
OUT = ROOT / "outputs"
SIZE = 224
RATIOS = (0.70, 0.15, 0.15)
SEED = 42
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BINS = np.arange(256)


def hist_rgb(arr):
    return np.stack([np.bincount(arr[..., c].ravel(), minlength=256) for c in range(3)])


def moments(h):
    """média e desvio-padrão por canal (escala 0-255) a partir de histogramas (3, 256)."""
    n = h.sum(axis=1)
    mean = (h * BINS).sum(axis=1) / n
    var = (h * BINS**2).sum(axis=1) / n - mean**2
    return mean, np.sqrt(var)


def analyse(path):
    with Image.open(path) as im:
        fmt, mode = im.format, im.mode
    img = dataprep.load_rgb(path)
    h_nat = hist_rgb(np.asarray(img))
    h_pre = hist_rgb(np.asarray(dataprep.center_crop(img, SIZE)))
    mean, _ = moments(h_nat)
    all_mean = mean.mean()
    all_var = (h_nat.sum(axis=0) * BINS**2).sum() / h_nat.sum() - all_mean**2
    return {"format": fmt, "mode": mode, "width": img.width, "height": img.height,
            "brightness": 0.299 * mean[0] + 0.587 * mean[1] + 0.114 * mean[2],
            "contrast": float(np.sqrt(all_var)), "h_nat": h_nat, "h_pre": h_pre}


def folder_name(species):
    return species.replace(" ", "_")


def main():
    OUT.mkdir(exist_ok=True)
    df = dataprep.load_labels(LABELS)
    paths = [IMAGES / f for f in df.file]
    missing = [p.name for p in paths if not p.exists()]
    extra = len({p.name for p in IMAGES.glob("*.jpg")} - set(df.file))
    print(f"{len(df)} rótulos | {len(missing)} arquivos ausentes | {extra} imagens sem rótulo")
    assert not missing, missing[:5]

    # duplicatas exatas (MD5) e rótulos conflitantes
    kept, removed = dataprep.drop_duplicates(paths)
    dup_removed = len(removed)
    if dup_removed:
        df = df[df.file.isin({p.name for p in kept})].reset_index(drop=True)
        paths = [IMAGES / f for f in df.file]
    print(f"duplicatas exatas removidas: {dup_removed}")

    df["t"] = [dataprep.capture_seconds(f) for f in df.file]
    df["split"] = dataprep.assign_splits(df.label.tolist(), ratios=RATIOS, seed=SEED)
    df["folder"] = df.species.map(folder_name)

    for row in df.itertuples():
        dest = DATASET / row.split / row.folder
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / row.file).exists():
            shutil.copy2(IMAGES / row.file, dest / row.file)

    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(analyse, paths, chunksize=64))
    hn = np.stack([r.pop("h_nat") for r in rows])
    hp = np.stack([r.pop("h_pre") for r in rows])
    df = pd.concat([df, pd.DataFrame(rows)], axis=1)
    df["min_side"] = df[["width", "height"]].min(axis=1)
    df.drop(columns=["t"]).to_csv(OUT / "splits.csv", index=False)

    mask = {s: (df.split == s).to_numpy() for s in dataprep.SPLITS}
    H_nat = {s: hn[m].sum(axis=0) for s, m in mask.items()}
    H_pre = {s: hp[m].sum(axis=0) for s, m in mask.items()}
    np.savez(OUT / "hist.npz", **{f"nat_{s}": h for s, h in H_nat.items()},
             **{f"pre_{s}": h for s, h in H_pre.items()})

    classes = df.drop_duplicates("label").sort_values("label")[["label", "species"]].values.tolist()
    class_names = [c[1] for c in classes]
    cnt = df.species.value_counts()
    species_cnt = cnt.drop("Negative")
    res = {"seed": SEED, "ratios": RATIOS, "size": SIZE}

    res["raw"] = {
        "n_labels": len(df) + dup_removed, "n_missing": len(missing), "dup_removed": dup_removed,
        "n_unique": len(df), "modes": dict(Counter(df["mode"])), "formats": dict(Counter(df["format"])),
        "sizes": {f"{w}x{h}": int(n) for (w, h), n in Counter(zip(df.width, df.height)).items()},
        "instruments": {int(k): int(v) for k, v in df.instrument.value_counts().sort_index().items()},
        "n_sessions": int(df.session.nunique()), "n_dates": int(df.date.nunique()),
        "date_min": df.date.min(), "date_max": df.date.max(),
    }

    weights = dataprep.balanced_class_weights(df.species.tolist())
    res["counts"] = {
        "total": len(df), "classes": class_names,
        "per_class": {c: int(cnt[c]) for c in class_names},
        "negative_n": int(cnt["Negative"]), "negative_share": float(cnt["Negative"] / len(df)),
        "species_min": [species_cnt.idxmin(), int(species_cnt.min())],
        "species_max": [species_cnt.idxmax(), int(species_cnt.max())],
        "species_ratio": float(species_cnt.max() / species_cnt.min()),
        "negative_vs_species_mean": float(cnt["Negative"] / species_cnt.mean()),
        "negative_vs_species_min": float(cnt["Negative"] / species_cnt.min()),
        "weights_full": {c: float(weights[c]) for c in class_names},
    }
    tr_species = df[(df.split == "train")].species.tolist()
    w_tr = dataprep.balanced_class_weights(tr_species)
    res["counts"]["weights_train"] = {c: float(w_tr[c]) for c in class_names}

    res["split"] = {}
    for s in dataprep.SPLITS:
        d = df[df.split == s]
        c = d.species.value_counts()
        res["split"][s] = {
            "n": len(d), "pct": len(d) / len(df),
            "per_class": {k: int(c.get(k, 0)) for k in class_names},
            "share_in_split": {k: float(c.get(k, 0) / len(d)) for k in class_names},
            "share_of_class": {k: float(c.get(k, 0) / cnt[k]) for k in class_names},
            "n_sessions": int(d.session.nunique()),
        }
    ct_class = pd.crosstab(df.split, df.species)
    ct_inst = pd.crosstab(df.split, df.instrument)
    c1, c2 = stats.chi2_contingency(ct_class), stats.chi2_contingency(ct_inst)
    res["chi2"] = {"class": {"stat": float(c1[0]), "p": float(c1[1])},
                   "instrument": {"stat": float(c2[0]), "p": float(c2[1])}}
    res["instrument_by_split"] = {s: {int(k): float(v) for k, v in
                                      (df[df.split == s].instrument.value_counts(normalize=True).sort_index()).items()}
                                  for s in dataprep.SPLITS}

    res["size"] = {"unique_sizes": list(res["raw"]["sizes"]), "all_uniform": len(res["raw"]["sizes"]) == 1,
                   "side": int(df.width.iloc[0]), "crop": SIZE,
                   "crop_area_lost": dataprep.crop_area_lost(int(df.width.iloc[0]), SIZE)}

    res["pixels"] = {}
    for s in dataprep.SPLITS:
        mean, std = moments(H_nat[s])
        present = np.nonzero(H_nat[s].sum(axis=0))[0]
        d = df[df.split == s]
        res["pixels"][s] = {"mean_255": mean.tolist(), "std_255": std.tolist(),
                            "min": int(present.min()), "max": int(present.max()),
                            "brightness_mean": float(d.brightness.mean()), "brightness_std": float(d.brightness.std()),
                            "contrast_mean": float(d.contrast.mean())}

    def w1(a, b):
        return [float(stats.wasserstein_distance(BINS, BINS, a[c], b[c])) for c in range(3)]

    res["compare"] = {"ks_brightness": {}, "ks_contrast": {}, "wasserstein_rgb": {}, "max_mean_diff_255": 0.0}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        key = f"{a}_vs_{b}"
        da, db = df[df.split == a], df[df.split == b]
        for feat, name in (("brightness", "ks_brightness"), ("contrast", "ks_contrast")):
            r = stats.ks_2samp(da[feat], db[feat])
            res["compare"][name][key] = {"D": float(r.statistic), "p": float(r.pvalue)}
        res["compare"]["wasserstein_rgb"][key] = w1(H_nat[a], H_nat[b])
        diff = np.abs(np.array(res["pixels"][a]["mean_255"]) - np.array(res["pixels"][b]["mean_255"])).max()
        res["compare"]["max_mean_diff_255"] = max(res["compare"]["max_mean_diff_255"], float(diff))

    tr_mean, tr_std = moments(H_pre["train"])
    res["norm"] = {"train_mean": (tr_mean / 255).tolist(), "train_std": (tr_std / 255).tolist(),
                   "imagenet_mean": list(IMAGENET_MEAN), "imagenet_std": list(IMAGENET_STD), "post_norm": {}}
    for s in dataprep.SPLITS:
        m, sd = moments(H_pre[s])
        res["norm"]["post_norm"][s] = {
            "mean": ((m / 255 - np.array(res["norm"]["train_mean"])) / np.array(res["norm"]["train_std"])).tolist(),
            "std": ((sd / 255) / np.array(res["norm"]["train_std"])).tolist()}

    # risco de vazamento: fotos de Val/Test muito próximas (no tempo/sessão) de fotos de Treino da mesma classe
    gaps = dataprep.nearest_train_gap(df)
    ev = gaps[~np.isnan(gaps)]
    np.save(OUT / "gaps.npy", ev)
    fin = ev[np.isfinite(ev)]
    per_class_sessions = df.groupby("species").session.nunique()
    top_share = df.groupby(["species", "session"]).size().groupby(level=0).apply(lambda s: s.max() / s.sum())
    res["leakage"] = {
        "n_eval": int(len(ev)), "pct_same_session": float(np.isfinite(ev).mean()),
        **{f"pct_within_{lim}s": float((ev <= lim).mean()) for lim in (5, 30, 60, 300)},
        "median_gap_s": float(np.median(fin)),
        "median_interval_s": float(df.sort_values(["session", "t"]).groupby("session").t.diff().dropna().median()),
        "sessions_per_class": {c: int(per_class_sessions[c]) for c in class_names},
        "top_session_share": {c: float(top_share[c]) for c in class_names},
    }
    (OUT / "stats.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("ok ->", OUT / "stats.json")


if __name__ == "__main__":
    main()
