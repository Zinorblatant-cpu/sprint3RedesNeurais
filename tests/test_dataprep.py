import collections
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dataprep  # noqa: E402


# ---------- parse_label ----------

def test_parse_label_gato_tem_inicial_maiuscula():
    assert dataprep.parse_label("Abyssinian_1.jpg") == ("gato", "Abyssinian")


def test_parse_label_cao_tem_inicial_minuscula():
    assert dataprep.parse_label("beagle_42.jpg") == ("cao", "beagle")


def test_parse_label_raca_com_varias_palavras():
    assert dataprep.parse_label("american_pit_bull_terrier_12.jpg") == (
        "cao",
        "american_pit_bull_terrier",
    )
    assert dataprep.parse_label("Maine_Coon_7.jpg") == ("gato", "Maine_Coon")


# ---------- duplicatas ----------

def _write_img(path, color, size=(8, 8)):
    Image.new("RGB", size, color).save(path, format="PNG")


def test_drop_duplicates_mantem_um_de_cada_grupo(tmp_path):
    _write_img(tmp_path / "a.png", (255, 0, 0))
    _write_img(tmp_path / "b.png", (255, 0, 0))  # idêntico a a.png
    _write_img(tmp_path / "c.png", (0, 255, 0))
    paths = sorted(tmp_path.glob("*.png"))

    kept, removed = dataprep.drop_duplicates(paths)

    assert [p.name for p in kept] == ["a.png", "c.png"]
    assert [p.name for p in removed] == ["b.png"]


def test_drop_duplicates_sem_duplicatas_nao_remove_nada(tmp_path):
    _write_img(tmp_path / "a.png", (1, 2, 3))
    _write_img(tmp_path / "b.png", (4, 5, 6))
    kept, removed = dataprep.drop_duplicates(sorted(tmp_path.glob("*.png")))
    assert len(kept) == 2 and removed == []


# ---------- split estratificado ----------

def _strata():
    # 3 raças com tamanhos diferentes (desbalanceado de propósito)
    return ["a"] * 200 + ["b"] * 100 + ["c"] * 50


def test_assign_splits_cobre_todos_os_itens_com_rotulos_validos():
    splits = dataprep.assign_splits(_strata(), seed=42)
    assert len(splits) == 350
    assert set(splits) == {"train", "val", "test"}


def test_assign_splits_respeita_proporcoes_70_15_15():
    splits = dataprep.assign_splits(_strata(), ratios=(0.70, 0.15, 0.15), seed=42)
    counts = collections.Counter(splits)
    n = len(splits)
    assert counts["train"] / n == pytest.approx(0.70, abs=0.02)
    assert counts["val"] / n == pytest.approx(0.15, abs=0.02)
    assert counts["test"] / n == pytest.approx(0.15, abs=0.02)


def test_assign_splits_e_estratificado_por_classe():
    strata = _strata()
    splits = dataprep.assign_splits(strata, ratios=(0.70, 0.15, 0.15), seed=42)
    for cls in ("a", "b", "c"):
        idx = [i for i, s in enumerate(strata) if s == cls]
        counts = collections.Counter(splits[i] for i in idx)
        assert counts["train"] / len(idx) == pytest.approx(0.70, abs=0.03)
        assert counts["val"] / len(idx) == pytest.approx(0.15, abs=0.03)
        assert counts["test"] / len(idx) == pytest.approx(0.15, abs=0.03)


def test_assign_splits_e_deterministico_com_mesma_seed():
    a = dataprep.assign_splits(_strata(), seed=7)
    b = dataprep.assign_splits(_strata(), seed=7)
    c = dataprep.assign_splits(_strata(), seed=8)
    assert a == b
    assert a != c


def test_assign_splits_rejeita_razoes_que_nao_somam_1():
    with pytest.raises(ValueError):
        dataprep.assign_splits(_strata(), ratios=(0.5, 0.2, 0.2))


# ---------- padronização das imagens ----------

def _rgb(w, h, color=(10, 20, 30)):
    return Image.new("RGB", (w, h), color)


@pytest.mark.parametrize("size", [(500, 375), (375, 500), (224, 224), (800, 200)])
def test_resize_center_crop_sempre_gera_quadrado_alvo(size):
    out = dataprep.resize_center_crop(_rgb(*size), 224)
    assert out.size == (224, 224)


def test_resize_center_crop_preserva_proporcao_e_corta_as_bordas():
    # metade esquerda vermelha, centro verde, direita azul (imagem larga)
    img = Image.new("RGB", (600, 200))
    for x in range(600):
        col = (255, 0, 0) if x < 200 else ((0, 255, 0) if x < 400 else (0, 0, 255))
        for y in range(200):
            img.putpixel((x, y), col)
    out = np.array(dataprep.resize_center_crop(img, 100))
    # só a faixa central (verde) deve sobreviver ao crop; ignora 5px de borda
    # onde a interpolação bicúbica ainda mistura vizinhos
    inner = out[:, 5:-5]
    assert (inner[:, :, 1] > 200).all()
    assert (inner[:, :, 0] < 50).all() and (inner[:, :, 2] < 50).all()


def test_resize_pad_gera_quadrado_e_nao_perde_conteudo():
    out = dataprep.resize_pad(_rgb(400, 200, (200, 200, 200)), 100, fill=(0, 0, 0))
    arr = np.array(out)
    assert out.size == (100, 100)
    assert (arr[0] == 0).all() and (arr[-1] == 0).all()  # barras no topo/base
    assert (arr[50] == 200).all()  # centro tem a imagem inteira, sem corte


def test_resize_stretch_gera_quadrado():
    assert dataprep.resize_stretch(_rgb(400, 200), 64).size == (64, 64)


def test_load_rgb_converte_modos_nao_rgb_para_3_canais(tmp_path):
    for mode, name in [("L", "g.png"), ("RGBA", "r.png"), ("P", "p.png")]:
        Image.new(mode, (4, 4)).save(tmp_path / name)
        img = dataprep.load_rgb(tmp_path / name)
        assert img.mode == "RGB"
        assert np.array(img).shape == (4, 4, 3)


def test_aspect_loss_e_zero_para_quadrada_e_cresce_com_a_proporcao():
    assert dataprep.aspect_loss(300, 300) == 0
    assert dataprep.aspect_loss(400, 300) == pytest.approx(0.25)
    assert dataprep.aspect_loss(300, 400) == pytest.approx(0.25)


# ---------- normalização ----------

def test_normalize_gera_float32_hwc_com_media_zero_e_desvio_um():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    mean = arr.reshape(-1, 3).mean(axis=0) / 255.0
    std = arr.reshape(-1, 3).std(axis=0) / 255.0

    out = dataprep.normalize(arr, mean, std)

    assert out.dtype == np.float32 and out.shape == (32, 32, 3)
    assert out.reshape(-1, 3).mean(axis=0) == pytest.approx([0, 0, 0], abs=1e-4)
    assert out.reshape(-1, 3).std(axis=0) == pytest.approx([1, 1, 1], abs=1e-4)


def test_normalize_escala_0_255_para_0_1_antes_de_padronizar():
    arr = np.full((2, 2, 3), 255, dtype=np.uint8)
    out = dataprep.normalize(arr, mean=(0, 0, 0), std=(1, 1, 1))
    assert out.max() == pytest.approx(1.0) and out.min() == pytest.approx(1.0)


def test_normalize_rejeita_desvio_zero():
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        dataprep.normalize(arr, mean=(0, 0, 0), std=(0, 1, 1))
