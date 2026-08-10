"""Prueba de humo de muon-angle-analysis.

Comprueba lo unico que se puede comprobar sin el pipeline: que los ocho
archivos crudos estan donde el pipeline los espera y no estan vacios. Las
comprobaciones de verdad viven en los demas archivos de tests/.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ANGULOS = [0, 5, 15, 25, 35, 45, 90, 360]


def test_los_ocho_archivos_crudos_existen():
    for angulo in ANGULOS:
        ruta = RAIZ / "data" / "raw" / f"{angulo:03d}deg"
        assert ruta.is_file(), f"falta el archivo crudo {ruta}"
        assert ruta.stat().st_size > 0, f"archivo crudo vacio: {ruta}"
