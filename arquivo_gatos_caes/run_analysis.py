"""Sprint 3 - prepara o dataset (dedup + split estratificado) e mede as características das imagens.

Saídas:
  dataset/{train,val,test}/{gato,cao}/*.jpg   imagens separadas
  outputs/splits.csv                          manifesto (arquivo, espécie, raça, split, dimensões...)
  outputs/stats.json                          todos os números usados nos slides
  outputs/hist.npz                            histogramas RGB agregados por split
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

SRC = Path.home() / ".cache/kagglehub/datasets/yashdogra/cats-and-dogs/versions/1/images"
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
    """Worker: lê a imagem uma vez e devolve metadados + histogramas (nativo e pré-processado)."""
    with Image.open(path) as im:
        fmt, mode = im.format, im.mode
    img = dataprep.load_rgb(path)
    arr = np.asarray(img)
    h_nat = hist_rgb(arr)
    h_pre = hist_rgb(np.asarray(dataprep.resize_center_crop(img, SIZE)))
    mean, std = moments(h_nat)
    gray = 0.299 * mean[0] + 0.587 * mean[1] + 0.114 * mean[2]
    all_mean = mean.mean()
    all_var = (h_nat.sum(axis=0) * BINS**2).sum() / h_nat.sum() - all_mean**2
    return {
        "file": Path(path).name,
        "format": fmt,
        "mode": mode,
        "width": img.width,
        "height": img.height,
        "mean_r": mean[0],
        "mean_g": mean[1],
        "mean_b": mean[2],
        "brightness": gray,
        "contrast": float(np.sqrt(all_var)),
        "h_nat": h_nat,
        "h_pre": h_pre,
    }


def main():
    OUT.mkdir(exist_ok=True)
    all_files = sorted(SRC.glob("*.jpg"))
    kept, removed = dataprep.drop_duplicates(all_files)
    print(f"{len(all_files)} arquivos, {len(removed)} duplicatas exatas removidas -> {len(kept)}")

    # duplicatas com rótulos conflitantes (mesmo conteúdo, raças diferentes)
    by_hash = {}
    for p in all_files:
        by_hash.setdefault(dataprep.file_md5(p), []).append(p.name)
    dup_groups = [v for v in by_hash.values() if len(v) > 1]
    conflicts = [g for g in dup_groups if len({dataprep.parse_label(n)[1] for n in g}) > 1]

    labels = [dataprep.parse_label(p.name) for p in kept]
    species = [l[0] for l in labels]
    breeds = [l[1] for l in labels]
    splits = dataprep.assign_splits(breeds, ratios=RATIOS, seed=SEED)

    # cópia física para dataset/<split>/<espécie>/
    for p, sp, split in zip(kept, species, splits):
        dest = DATASET / split / sp
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / p.name).exists():
            shutil.copy2(p, dest / p.name)

    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(analyse, kept, chunksize=32))
    hn = np.stack([r.pop("h_nat") for r in rows])
    hp = np.stack([r.pop("h_pre") for r in rows])

    df = pd.DataFrame(rows)
    df["species"], df["breed"], df["split"] = species, breeds, splits
    df["aspect"] = df.width / df.height
    df["aspect_loss"] = [dataprep.aspect_loss(w, h) for w, h in zip(df.width, df.height)]
    df["min_side"] = df[["width", "height"]].min(axis=1)
    df.to_csv(OUT / "splits.csv", index=False)

    split_mask = {s: (df.split == s).to_numpy() for s in dataprep.SPLITS}
    H_nat = {s: hn[m].sum(axis=0) for s, m in split_mask.items()}
    H_pre = {s: hp[m].sum(axis=0) for s, m in split_mask.items()}
    np.savez(OUT / "hist.npz", **{f"nat_{s}": h for s, h in H_nat.items()},
             **{f"pre_{s}": h for s, h in H_pre.items()})

    # ---------- estatísticas ----------
    res = {"seed": SEED, "ratios": RATIOS, "size": SIZE}
    res["raw"] = {
        "n_files": len(all_files),
        "dup_groups": len(dup_groups),
        "dup_removed": len(removed),
        "dup_label_conflicts": len(conflicts),
        "n_unique": len(kept),
        "modes": dict(Counter(df["mode"])),
        "formats": dict(Counter(df["format"])),
    }
    sp_counts = df.species.value_counts()
    br_counts = df.breed.value_counts()
    res["counts"] = {
        "total": len(df),
        "species": sp_counts.to_dict(),
        "imbalance_ratio": float(sp_counts.max() / sp_counts.min()),
        "n_breeds": int(br_counts.size),
        "breed_min": [br_counts.idxmin(), int(br_counts.min())],
        "breed_max": [br_counts.idxmax(), int(br_counts.max())],
        "breeds": br_counts.to_dict(),
        "breed_species": {b: dataprep.parse_label(b + "_1.jpg")[0] for b in br_counts.index},
    }

    res["split"] = {}
    for s in dataprep.SPLITS:
        d = df[df.split == s]
        c = d.species.value_counts()
        res["split"][s] = {
            "n": len(d),
            "pct": len(d) / len(df),
            "species": c.to_dict(),
            "species_pct": (c / len(d)).to_dict(),
            "breeds": d.breed.value_counts().to_dict(),
        }

    ct_species = pd.crosstab(df.split, df.species)
    ct_breed = pd.crosstab(df.split, df.breed)
    res["chi2"] = {
        "species": {"stat": float(stats.chi2_contingency(ct_species)[0]),
                    "p": float(stats.chi2_contingency(ct_species)[1])},
        "breed": {"stat": float(stats.chi2_contingency(ct_breed)[0]),
                  "p": float(stats.chi2_contingency(ct_breed)[1])},
    }

    q = lambda s: s.quantile([0.05, 0.5, 0.95]).round(1).tolist()
    res["size"] = {
        "n_unique_resolutions": int(df.groupby(["width", "height"]).ngroups),
        "width": {"min": int(df.width.min()), "median": float(df.width.median()),
                  "max": int(df.width.max()), "p5_p50_p95": q(df.width)},
        "height": {"min": int(df.height.min()), "median": float(df.height.median()),
                   "max": int(df.height.max()), "p5_p50_p95": q(df.height)},
        "min_side": {"min": int(df.min_side.min()), "median": float(df.min_side.median()),
                     "max": int(df.min_side.max())},
        "aspect": {"median": float(df.aspect.median()), "p5_p50_p95": q(df.aspect)},
        "top_resolutions": [f"{w}x{h}" for (w, h), _ in
                            Counter(zip(df.width, df.height)).most_common(5)],
        "top_resolutions_n": [n for _, n in Counter(zip(df.width, df.height)).most_common(5)],
        "pct_smaller_than_target": float((df.min_side < SIZE).mean()),
        "n_smaller_than_target": int((df.min_side < SIZE).sum()),
        "aspect_loss": {"mean": float(df.aspect_loss.mean()),
                        "median": float(df.aspect_loss.median()),
                        "p90": float(df.aspect_loss.quantile(0.9)),
                        "max": float(df.aspect_loss.max())},
        "pct_near_square": float((df.aspect_loss < 0.05).mean()),
        "stretch_ratio_median": float(np.maximum(df.aspect, 1 / df.aspect).median()),
        "channels": {"3": int(len(df))},
        "per_split": {s: {"width_mean": float(df[df.split == s].width.mean()),
                          "height_mean": float(df[df.split == s].height.mean()),
                          "aspect_mean": float(df[df.split == s].aspect.mean()),
                          "min_side_median": float(df[df.split == s].min_side.median())}
                      for s in dataprep.SPLITS},
    }

    res["pixels"] = {}
    for s in dataprep.SPLITS:
        mean, std = moments(H_nat[s])
        present = np.nonzero(H_nat[s].sum(axis=0))[0]
        d = df[df.split == s]
        res["pixels"][s] = {
            "mean_255": mean.tolist(), "std_255": std.tolist(),
            "mean_01": (mean / 255).tolist(), "std_01": (std / 255).tolist(),
            "min": int(present.min()), "max": int(present.max()),
            "brightness_mean": float(d.brightness.mean()),
            "brightness_std": float(d.brightness.std()),
            "contrast_mean": float(d.contrast.mean()),
        }

    def w1(a, b):
        return [float(stats.wasserstein_distance(BINS, BINS, a[c], b[c])) for c in range(3)]

    res["compare"] = {"ks_brightness": {}, "ks_contrast": {}, "wasserstein_rgb": {}, "max_mean_diff_255": 0.0}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        key = f"{a}_vs_{b}"
        da, db = df[df.split == a], df[df.split == b]
        for feat, name in [("brightness", "ks_brightness"), ("contrast", "ks_contrast")]:
            r = stats.ks_2samp(da[feat], db[feat])
            res["compare"][name][key] = {"D": float(r.statistic), "p": float(r.pvalue)}
        res["compare"]["wasserstein_rgb"][key] = w1(H_nat[a], H_nat[b])
        diff = np.abs(np.array(res["pixels"][a]["mean_255"]) - np.array(res["pixels"][b]["mean_255"])).max()
        res["compare"]["max_mean_diff_255"] = max(res["compare"]["max_mean_diff_255"], float(diff))

    # normalização: constantes só do TREINO (sem vazamento), medidas nas imagens já recortadas 224x224
    tr_mean, tr_std = moments(H_pre["train"])
    res["norm"] = {
        "train_mean": (tr_mean / 255).tolist(), "train_std": (tr_std / 255).tolist(),
        "imagenet_mean": list(IMAGENET_MEAN), "imagenet_std": list(IMAGENET_STD),
        "post_norm": {},
    }
    for s in dataprep.SPLITS:
        m, sd = moments(H_pre[s])
        res["norm"]["post_norm"][s] = {
            "mean": ((m / 255 - np.array(res["norm"]["train_mean"])) / np.array(res["norm"]["train_std"])).tolist(),
            "std": ((sd / 255) / np.array(res["norm"]["train_std"])).tolist(),
        }

    (OUT / "stats.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("ok ->", OUT / "stats.json")


if __name__ == "__main__":
    main()
