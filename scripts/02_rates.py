#!/usr/bin/env python3
"""
02_rates.py

Tasa de conteo de cada corrida, corregida por tiempo vivo.

Pasos:
  1. Lectura de las ocho corridas crudas.
  2. Ventana de observacion como tN - t1, sin truncar a multiplos del tamano de
     bin. En el analisis anterior la duracion se truncaba a multiplos de
     --bin-seconds, de modo que un parametro cosmetico movia la tasa publicada:
     1.8118 1/s con bin de 1 s frente a 1.8194 1/s con bin de 60 s.
  3. Tiempo muerto de la columna DeadTime[us], que antes se parseaba y no se
     usaba nunca. La fraccion muerta es del 9.09 % al 9.28 %.
  4. Tasa por el estimador insesgado (N-1)/T_vivo. Para N eventos en [t1, tN]
     hay N-1 intervalos, no N.
  5. Comparacion explicita contra la tasa de reloj de pared, para dejar ver
     cuanto mueve cada correccion.

Aviso que el informe repite: como la fraccion muerta es practicamente identica
en las ocho corridas, la correccion de tiempo vivo se cancela casi entera en
las tasas NORMALIZADAS. Importa para la tasa absoluta, no para la forma
angular. Corregir es lo correcto, pero no arregla nada que estuviera roto en
la comparacion entre angulos.

Salidas exactas:
  - results/tables/02_rates_vN.json   (consumido por 04, 05 y 06)
  - results/tables/02_rates_vN.txt

Codigo de salida: 0 si las ocho tasas se calculan; 1 si alguna corrida no tiene
eventos suficientes.

Uso tipico, desde la raiz del repositorio y con .venv activo:
  python scripts/02_rates.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import io, provenance, rates  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Tasas corregidas por tiempo vivo.")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()

    corridas = io.load_all(args.data_dir)
    tasas = {}
    for angulo in sorted(corridas):
        try:
            tasas[angulo] = rates.calcular_tasa(corridas[angulo])
        except ValueError as error:
            print(f"[02] ERROR en {angulo} grados: {error}", file=sys.stderr)
            return 1

    lineas = [
        f"{'ang':>4} {'N':>5} {'N-1':>5} {'ventana':>9} {'vivo':>9} {'muerta%':>8} "
        f"{'R_pared':>9} {'R_viva':>9} {'sigma':>8} {'subida%':>8}",
        "-" * 88,
    ]
    for angulo, t in tasas.items():
        subida = 100 * (t.tasa_viva / t.tasa_pared - 1)
        lineas.append(
            f"{angulo:>4} {t.n_eventos:>5} {t.n_intervalos:>5} {t.ventana_s:>9.2f} "
            f"{t.tiempo_vivo_s:>9.2f} {100 * t.fraccion_muerta:>8.2f} "
            f"{t.tasa_pared:>9.4f} {t.tasa_viva:>9.4f} {t.error_tasa_viva:>8.4f} "
            f"{subida:>8.2f}"
        )

    fracciones = np.array([t.fraccion_muerta for t in tasas.values()])
    lineas += [
        "-" * 88,
        f"Fraccion muerta entre {100 * fracciones.min():.2f} % y "
        f"{100 * fracciones.max():.2f} %, con dispersion de "
        f"{100 * fracciones.std():.3f} puntos porcentuales.",
        "Es practicamente identica en las ocho corridas, luego la correccion de",
        "tiempo vivo se cancela casi entera en los cocientes entre angulos: sube",
        "todas las tasas un 10 % y deja la forma angular donde estaba. Importa",
        "para la tasa absoluta, no para la dependencia con el angulo.",
    ]
    texto = "\n".join(lineas)
    print(texto)

    ruta_json, ruta_txt = provenance.escribir_producto(
        args.out, "02_rates",
        datos={"tasas": {str(a): t.como_dict() for a, t in tasas.items()}},
        texto=texto,
        parametros={"data_dir": args.data_dir, "solo_coincidentes": True},
    )
    print(f"[02] escrito {ruta_json}")
    print(f"[02] escrito {ruta_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
