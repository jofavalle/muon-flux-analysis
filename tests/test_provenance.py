"""Tests de trazabilidad: versionado _vN y serializacion de los productos."""

import json

import numpy as np
import pytest

from lib_muon import provenance


class TestVersionado:
    def test_la_primera_version_es_1(self, tmp_path):
        assert provenance.siguiente_version(tmp_path, "07_prueba") == 1
        assert provenance.ultima_version(tmp_path, "07_prueba") is None

    def test_ninguna_corrida_sobrescribe_la_anterior(self, tmp_path):
        for esperada in (1, 2, 3):
            ruta_json, ruta_txt = provenance.escribir_producto(
                tmp_path, "07_prueba", datos={"valor": esperada}, texto="x"
            )
            assert ruta_json.name == f"07_prueba_v{esperada}.json"
            assert ruta_txt.name == f"07_prueba_v{esperada}.txt"
        # Las tres siguen existiendo con su contenido original.
        for esperada in (1, 2, 3):
            contenido = json.loads(
                (tmp_path / f"07_prueba_v{esperada}.json").read_text()
            )
            assert contenido["valor"] == esperada

    def test_leer_producto_devuelve_el_mas_reciente(self, tmp_path):
        for valor in (10, 20, 30):
            provenance.escribir_producto(tmp_path, "07_prueba",
                                         datos={"valor": valor}, texto="x")
        assert provenance.leer_producto(tmp_path, "07_prueba")["valor"] == 30

    def test_leer_producto_ausente_dice_que_paso_falta(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Ejecuta antes el paso"):
            provenance.leer_producto(tmp_path, "02_rates")


class TestSerializacion:
    def test_los_tipos_de_numpy_se_escriben(self, tmp_path):
        datos = {
            "entero": np.int64(7),
            "flotante": np.float64(1.5),
            "arreglo": np.array([1.0, 2.0, 3.0]),
            "booleano": np.bool_(True),
        }
        ruta_json, _ = provenance.escribir_producto(tmp_path, "07_prueba",
                                                    datos=datos, texto="x")
        leido = json.loads(ruta_json.read_text())
        assert leido["entero"] == 7
        assert leido["flotante"] == 1.5
        assert leido["arreglo"] == [1.0, 2.0, 3.0]
        assert leido["booleano"] is True

    def test_infinito_y_nan_se_escriben_como_null(self, tmp_path):
        """JSON no tiene Infinity ni NaN. Un ajuste sin valor finito es null,
        no un cero: la diferencia importa, porque -logL infinito ES el
        resultado del modelo cos^n puro."""
        datos = {
            "nll_infinito": float("inf"),
            "sin_valor": float("nan"),
            "anidado": {"lista": [1.0, float("inf"), 3.0]},
            "arreglo": np.array([np.inf, 1.0]),
        }
        ruta_json, _ = provenance.escribir_producto(tmp_path, "07_prueba",
                                                    datos=datos, texto="x")
        crudo = ruta_json.read_text()
        assert "Infinity" not in crudo and "NaN" not in crudo
        leido = json.loads(crudo)
        assert leido["nll_infinito"] is None
        assert leido["sin_valor"] is None
        assert leido["anidado"]["lista"] == [1.0, None, 3.0]
        assert leido["arreglo"] == [None, 1.0]


class TestProcedencia:
    def test_todo_producto_lleva_commit_y_parametros(self, tmp_path):
        ruta_json, ruta_txt = provenance.escribir_producto(
            tmp_path, "07_prueba", datos={"valor": 1}, texto="contenido",
            parametros={"bin_seconds": 10.0},
        )
        sello = json.loads(ruta_json.read_text())["procedencia"]
        assert "git" in sello and "commit" in sello["git"]
        assert sello["parametros"]["bin_seconds"] == 10.0
        assert sello["semilla"] == provenance.SEMILLA_POR_DEFECTO
        assert "numpy" in sello and "python" in sello
        # El txt repite el commit en su cabecera para poder leerlo sin json.
        assert "commit:" in ruta_txt.read_text()
