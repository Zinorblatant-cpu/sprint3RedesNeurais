import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dataprep  # noqa: E402


# ---------- nome do arquivo: YYYYMMDD-HHMMSS-ID.jpg ----------

def test_parse_capture_id_extrai_data_hora_e_instrumento():
    assert dataprep.parse_capture_id("20170320-093423-1.jpg") == {
        "date": "20170320",
        "time": "093423",
        "instrument": 1,
    }


def test_parse_capture_id_aceita_caminho_completo():
    assert dataprep.parse_capture_id("images/20160928-140314-0.jpg")["instrument"] == 0


@pytest.mark.parametrize("bad", ["Abyssinian_1.jpg", "2017032-093423-1.jpg", "20170320-093423.jpg", "20170320-093423-x.jpg"])
def test_parse_capture_id_rejeita_nomes_fora_do_padrao(bad):
    with pytest.raises(ValueError):
        dataprep.parse_capture_id(bad)


def test_capture_session_junta_data_e_instrumento():
    assert dataprep.capture_session("20170320-093423-1.jpg") == "20170320-1"
    # mesma data e câmera => mesma sessão, independente da hora
    assert dataprep.capture_session("20170320-101010-1.jpg") == "20170320-1"
    assert dataprep.capture_session("20170320-093423-2.jpg") != "20170320-1"


# ---------- labels.csv (Filename,Label,Species) ----------

def _csv(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text(
        "Filename,Label,Species\n"
        "20160928-140314-0.jpg,0,Chinee apple\n"
        "20160928-140337-0.jpg,0,Chinee apple\n"
        "20170101-101010-2.jpg,8,Negative\n",
        encoding="utf-8",
    )
    return p


def test_load_labels_normaliza_colunas_e_tipos(tmp_path):
    df = dataprep.load_labels(_csv(tmp_path))
    assert list(df.columns)[:3] == ["file", "label", "species"]
    assert len(df) == 3
    assert df.label.dtype.kind == "i"


def test_load_labels_marca_a_classe_negativa(tmp_path):
    df = dataprep.load_labels(_csv(tmp_path))
    assert df.is_negative.tolist() == [False, False, True]


def test_load_labels_acrescenta_sessao_de_captura(tmp_path):
    df = dataprep.load_labels(_csv(tmp_path))
    assert df.session.tolist() == ["20160928-0", "20160928-0", "20170101-2"]


def test_load_labels_falha_se_faltar_coluna(tmp_path):
    p = tmp_path / "ruim.csv"
    p.write_text("Filename,Label\na.jpg,0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        dataprep.load_labels(p)


# ---------- pesos de classe ----------

def test_balanced_class_weights_segue_n_total_sobre_k_vezes_n_classe():
    labels = ["a"] * 6 + ["b"] * 3 + ["c"] * 1
    w = dataprep.balanced_class_weights(labels)
    assert w["a"] == pytest.approx(10 / (3 * 6))
    assert w["b"] == pytest.approx(10 / (3 * 3))
    assert w["c"] == pytest.approx(10 / (3 * 1))


def test_balanced_class_weights_classe_minoritaria_pesa_mais():
    w = dataprep.balanced_class_weights(["cao"] * 90 + ["gato"] * 10)
    assert w["gato"] > w["cao"]
    # com pesos balanceados, cada classe contribui com o mesmo peso total
    assert 90 * w["cao"] == pytest.approx(10 * w["gato"])


def test_balanced_class_weights_rejeita_lista_vazia():
    with pytest.raises(ValueError):
        dataprep.balanced_class_weights([])


# ---------- tempo de captura e proximidade com o treino ----------

def test_capture_seconds_diferenca_em_segundos_entre_fotos():
    a = dataprep.capture_seconds("20170320-093423-1.jpg")
    b = dataprep.capture_seconds("20170320-093439-1.jpg")
    assert b - a == 16


def test_capture_seconds_atravessa_virada_de_dia_e_mes():
    a = dataprep.capture_seconds("20170331-235950-0.jpg")
    b = dataprep.capture_seconds("20170401-000010-0.jpg")
    assert b - a == 20


def _frame():
    return pd.DataFrame({
        "session": ["s1"] * 5 + ["s2"] * 2,
        "label": [0] * 7,
        "t": [0, 10, 100, 130, 500, 0, 50],
        "split": ["train", "val", "train", "test", "train", "val", "test"],
    })


def test_nearest_train_gap_mede_ate_a_foto_de_treino_mais_proxima():
    gaps = dataprep.nearest_train_gap(_frame())
    # val em t=10 (s1): treino em 0 e 100 -> 10 ; test em t=130: treino em 100 e 500 -> 30
    assert gaps[1] == 10
    assert gaps[3] == 30


def test_nearest_train_gap_imagens_de_treino_ficam_com_nan():
    gaps = dataprep.nearest_train_gap(_frame())
    assert pd.isna(gaps[0]) and pd.isna(gaps[2]) and pd.isna(gaps[4])


def test_nearest_train_gap_sessao_sem_treino_e_infinito():
    gaps = dataprep.nearest_train_gap(_frame())
    # s2 só tem val/test, nenhuma foto de treino na mesma sessão e classe
    assert gaps[5] == float("inf") and gaps[6] == float("inf")


def test_nearest_train_gap_respeita_a_classe():
    df = _frame()
    df.loc[1, "label"] = 1  # o val em s1 agora é de outra classe: não há treino dela
    gaps = dataprep.nearest_train_gap(df)
    assert gaps[1] == float("inf")


# ---------- recorte sem reamostragem (imagens já uniformes 256x256) ----------

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def _grad(size=16):
    """imagem em que cada pixel tem valor único: permite checar recortes pixel a pixel."""
    a = np.arange(size * size * 3, dtype=np.uint32).reshape(size, size, 3) % 251
    return Image.fromarray(a.astype(np.uint8), "RGB")


def test_center_crop_devolve_exatamente_os_pixels_do_centro():
    img = _grad(16)
    out = dataprep.center_crop(img, 8)
    assert out.size == (8, 8)
    assert np.array_equal(np.array(out), np.array(img)[4:12, 4:12])  # sem interpolação


def test_center_crop_rejeita_alvo_maior_que_a_imagem():
    with pytest.raises(ValueError):
        dataprep.center_crop(_grad(16), 32)


def test_random_crop_tem_o_tamanho_pedido_e_vem_de_dentro_da_imagem():
    img = _grad(16)
    rng = np.random.default_rng(0)
    for _ in range(20):
        out = np.array(dataprep.random_crop(img, 8, rng))
        assert out.shape == (8, 8, 3)
        # o recorte precisa existir em alguma posição da imagem original
        src = np.array(img)
        found = any(np.array_equal(out, src[y:y + 8, x:x + 8]) for y in range(9) for x in range(9))
        assert found


def test_random_crop_e_deterministico_com_a_mesma_semente():
    img = _grad(16)
    a = np.array(dataprep.random_crop(img, 8, np.random.default_rng(3)))
    b = np.array(dataprep.random_crop(img, 8, np.random.default_rng(3)))
    assert np.array_equal(a, b)


def test_area_perdida_no_crop_para_imagem_quadrada():
    assert dataprep.crop_area_lost(256, 224) == pytest.approx(1 - (224 / 256) ** 2)
    assert dataprep.crop_area_lost(224, 224) == 0
