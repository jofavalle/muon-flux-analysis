"""Test de extremo a extremo: el pipeline entero sobre los datos reales.

Cabe en CI porque el conjunto son 256 kB y esa es justamente la razon de
versionar los datos crudos. Comprueba que los ocho scripts corren en orden, que
cada uno escribe lo que su docstring promete, y que los numeros que el
repositorio publica salen de esa corrida y no de otro sitio.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

PIPELINE = [
    ("00_ingest.py", 0),
    ("01_validation.py", 0),
    ("02_rates.py", 0),
    ("03_timeseries.py", 0),
    ("04_normalization.py", 0),
    ("05_fits.py", 0),
    ("06_systematics.py", 0),
]


@pytest.fixture(scope="module")
def corrida_completa(tmp_path_factory):
    """Corre el pipeline entero en un directorio limpio."""
    trabajo = tmp_path_factory.mktemp("pipeline")
    shutil.copytree(RAIZ / "data", trabajo / "data")
    shutil.copytree(RAIZ / "scripts", trabajo / "scripts")

    salidas = {}
    for script, esperado in PIPELINE:
        proceso = subprocess.run(
            [PYTHON, f"scripts/{script}"],
            cwd=trabajo, capture_output=True, text=True, timeout=300,
        )
        assert proceso.returncode == esperado, (
            f"{script} salio con {proceso.returncode}, se esperaba {esperado}\n"
            f"stdout:\n{proceso.stdout}\nstderr:\n{proceso.stderr}"
        )
        salidas[script] = proceso.stdout
    return trabajo, salidas


def leer(trabajo, subdirectorio, base):
    ruta = trabajo / "results" / subdirectorio / f"{base}_v1.json"
    assert ruta.is_file(), f"falta {ruta}"
    return json.loads(ruta.read_text())


class TestElPipelineCorreEntero:
    def test_los_siete_scripts_terminan_bien(self, corrida_completa):
        _, salidas = corrida_completa
        assert len(salidas) == len(PIPELINE)

    def test_cada_paso_escribe_su_producto(self, corrida_completa):
        trabajo, _ = corrida_completa
        for base in ("00_ingest", "01_validation", "02_rates", "03_timeseries",
                     "04_normalization", "06_systematics"):
            assert (trabajo / "results" / "tables" / f"{base}_v1.json").is_file()
            assert (trabajo / "results" / "tables" / f"{base}_v1.txt").is_file()
        assert (trabajo / "results" / "fits" / "05_fits_v1.json").is_file()

    def test_las_figuras_salen_pareadas_en_pdf_y_png(self, corrida_completa):
        trabajo, _ = corrida_completa
        figuras = trabajo / "results" / "figures"
        pdfs = sorted(p.stem for p in figuras.glob("*.pdf"))
        pngs = sorted(p.stem for p in figuras.glob("*.png"))
        assert pdfs == pngs, "cada figura debe existir en PDF y en PNG"
        assert len(pdfs) >= 12


class TestLosNumerosPublicados:
    """Ancla las cifras que el README y el informe citan."""

    def test_recuento_de_eventos(self, corrida_completa):
        trabajo, _ = corrida_completa
        assert leer(trabajo, "tables", "00_ingest")["n_eventos_total"] == 4760

    def test_una_sola_linea_malformada(self, corrida_completa):
        trabajo, _ = corrida_completa
        assert leer(trabajo, "tables", "00_ingest")["n_lineas_rechazadas_total"] == 1

    def test_cos_n_puro_tiene_verosimilitud_nula(self, corrida_completa):
        """El resultado central del analisis: no es que ajuste mal, es que su
        verosimilitud es cero. En el JSON eso se escribe como null."""
        trabajo, _ = corrida_completa
        ajustes = leer(trabajo, "fits", "05_fits")["ajustes"]
        for clave in ("B_cosn_libre", "C_cos2_fijo"):
            assert ajustes[clave]["nll"] is None
            assert ajustes[clave]["nll_infinito"] is True
            assert ajustes[clave]["aic"] is None
        assert ajustes["A_constante"]["nll"] is not None

    def test_el_coseno_de_90_esta_anclado_a_cero(self, corrida_completa):
        trabajo, _ = corrida_completa
        cosenos = leer(trabajo, "fits", "05_fits")["coseno_anclado"]
        assert cosenos[-1] == 0.0

    def test_la_modulacion_no_sobrevive_al_sistematico(self, corrida_completa):
        """La conclusion publicada. Si esto falla, la conclusion cambio."""
        trabajo, _ = corrida_completa
        sistematicas = leer(trabajo, "tables", "06_systematics")
        critico = sistematicas["sistematico_critico_para_significancia"]
        medido = sistematicas["sistematico_dominante"]
        assert medido > critico, (
            "la sistematica medida ya no supera a la que anula la significancia: "
            "la conclusion del repositorio habria cambiado"
        )

    def test_hay_cota_superior_sobre_la_fraccion_modulada(self, corrida_completa):
        trabajo, _ = corrida_completa
        sistematicas = leer(trabajo, "tables", "06_systematics")
        cota = sistematicas["cota_fraccion_modulada_con_sistematico"]
        assert 0.0 < cota < 1.0
        assert cota > sistematicas["cota_fraccion_modulada_solo_estadistica"], (
            "incluir una sistematica no puede estrechar la cota"
        )


class TestReproducibilidad:
    def test_los_productos_llevan_sello_de_procedencia(self, corrida_completa):
        """El pipeline corre aqui fuera de un repositorio git, asi que el
        commit se registra como null. Esa es la conducta correcta: decir que no
        se sabe, no inventar un hash."""
        trabajo, _ = corrida_completa
        sello = leer(trabajo, "fits", "05_fits")["procedencia"]
        assert "commit" in sello["git"]
        assert sello["git"]["commit"] is None
        assert sello["parametros"] != {}
        assert sello["semilla"] is not None
        assert sello["numpy"] and sello["python"]

    def test_dentro_del_repositorio_si_se_captura_el_commit(self):
        """Comprobacion complementaria: en el repositorio real, el sello lleva
        el hash y el estado del arbol."""
        sello = subprocess.run(
            [PYTHON, "-c",
             "import sys; sys.path.insert(0, 'scripts');"
             "from lib_muon import provenance; import json;"
             "print(json.dumps(provenance.hash_commit()))"],
            cwd=RAIZ, capture_output=True, text=True, timeout=60, check=True,
        ).stdout
        datos = json.loads(sello)
        assert datos["commit"] is not None and len(datos["commit"]) == 40
        assert datos["arbol_limpio"] in (True, False)

    def test_una_segunda_corrida_no_pisa_la_primera(self, corrida_completa):
        trabajo, _ = corrida_completa
        subprocess.run([PYTHON, "scripts/02_rates.py"], cwd=trabajo,
                       capture_output=True, text=True, timeout=120, check=True)
        tablas = trabajo / "results" / "tables"
        assert (tablas / "02_rates_v1.json").is_file()
        assert (tablas / "02_rates_v2.json").is_file()
