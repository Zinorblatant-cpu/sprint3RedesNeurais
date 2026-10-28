import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import sprint4_utils as u  # noqa: E402


# ---------- divisão embutida ----------

def test_split_codifica_e_decodifica_sem_perda():
    splits = ["train", "val", "test", "train", "train"]
    assert u.decode_split(u.encode_split(splits)) == splits


def test_split_da_sprint3_volta_identico():
    sp = pd.read_csv(ROOT / "splits.csv")
    blob = u.encode_split(sp.split)
    assert len(blob) < 6000
    assert u.decode_split(blob) == sp.split.tolist()


def test_md5_depende_da_ordem_dos_arquivos():
    assert u.files_md5(["a.jpg", "b.jpg"]) != u.files_md5(["b.jpg", "a.jpg"])


# ---------- arquitetura e dados ----------

@pytest.mark.parametrize("blocks, rf", [(1, 4), (2, 10), (3, 22), (4, 46), (5, 94)])
def test_campo_receptivo(blocks, rf):
    assert u.receptive_field(blocks) == rf


def test_pesos_de_classe():
    assert u.class_weights([0, 0, 0, 1], 2).tolist() == pytest.approx([4 / 6, 2.0])


def test_metricas():
    m = u.metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert m["accuracy"] == 0.75
    assert m["confusion"] == [[1, 1], [0, 2]]
    assert m["recall"] == [0.5, 1.0]


# ---------- seleção e diagnóstico ----------

def _r(f1, params):
    return {"params": params, "val": {"f1_macro": f1}}


def test_select_maior_f1():
    assert u.select({"a": _r(0.70, 10), "b": _r(0.80, 500)}) == ("b", ["b"])


def test_select_empate_fica_com_menos_parametros():
    assert u.select({"a": _r(0.800, 500), "b": _r(0.795, 100), "c": _r(0.70, 10)}) == ("b", ["a", "b"])


def _h(acc, val_acc, val_loss):
    return {"accuracy": acc, "val_accuracy": val_acc, "val_loss": val_loss}


def test_diagnose_overfitting_pelo_gap():
    assert u.diagnose(_h([0.7, 0.95], [0.7, 0.80], [1.0, 0.9]), 2) == "overfitting"


def test_diagnose_overfitting_pela_loss_subindo():
    assert u.diagnose(_h([0.90, 0.92, 0.95], [0.86, 0.87, 0.86], [0.5, 0.4, 0.6]), 2) == "overfitting"


def test_diagnose_underfitting():
    assert u.diagnose(_h([0.5, 0.6], [0.55, 0.62], [1.2, 1.1]), 2) == "underfitting"


def test_diagnose_bom_ajuste():
    assert u.diagnose(_h([0.85, 0.90], [0.84, 0.88], [0.6, 0.5]), 2) == "bom ajuste"
