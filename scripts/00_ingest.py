#!/usr/bin/env python3
"""
00_ingest.py

Lectura e inventario de las ocho corridas crudas del detector CosmicWatch,
primer paso del pipeline de muon-angle-analysis.

Para cada archivo de data/raw se hace:
  1. Parseo linea a linea, contando y conservando el texto de las que no son
     legibles. El script anterior las descartaba en silencio con
     `except Exception: continue`, y hay una de verdad: data/raw/015deg esta
     truncado por arriba y su primera linea es un fragmento del preambulo.
  2. Decision del orden real de las columnas de presion y temperatura, que NO
     es el que declara la cabecera. Se decide una vez por archivo contra
     rangos fisicos y con la mediana, no fila a fila por magnitud.
  3. Marcado de las filas con la lectura de arranque del BMP280
     (73157.02 Pa / 23.13 C), que aparecen antes de que el sensor se estabilice.
  4. Recuento de eventos coincidentes frente al total.
  5. Contabilidad temporal: ventana [t1, tN], tiempo muerto acumulado de la
     columna DeadTime[us] y tiempo vivo resultante.

Salidas exactas:
  - results/tables/00_ingest_vN.json   (inventario por archivo, legible por 01)
  - results/tables/00_ingest_vN.txt    (el mismo inventario en tabla)

Codigo de salida: 0 si los ocho archivos se leen; 1 si alguno no se puede leer
o no se puede decidir el orden de sus columnas.

Uso tipico, desde la raiz del repositorio y con .venv activo:
  python scripts/00_ingest.py
  python scripts/00_ingest.py --data-dir data/raw --out results/tables
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import io, provenance  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--data-dir", default="data/raw",
                    help="directorio con los archivos NNNdeg (por defecto data/raw)")
    ap.add_argument("--out", default="results/tables",
                    help="directorio de salida (por defecto results/tables)")
    args = ap.parse_args()

    print(f"[00] leyendo corridas de {args.data_dir}")
    try:
        corridas = io.load_all(args.data_dir)
    except (ValueError, OSError) as error:
        print(f"[00] ERROR: {error}", file=sys.stderr)
        return 1

    inventario = []
    alertas = []
    for angulo in sorted(corridas):
        c = corridas[angulo]
        calidad = c.calidad
        inventario.append({
            "angulo_deg": angulo,
            "archivo": Path(c.ruta).name,
            "n_eventos": len(c),
            "n_coincidentes": int((c.coincidente == 1).sum()),
            "ventana_s": c.ventana_s,
            "tiempo_muerto_s": c.tiempo_muerto_s,
            "tiempo_vivo_s": c.tiempo_vivo_s,
            "fraccion_muerta": c.fraccion_muerta,
            "presion_mediana_pa": float(c.presion_pa[~c.arranque_sensor].mean())
            if (~c.arranque_sensor).any() else None,
            "temperatura_mediana_c": float(c.temperatura_c[~c.arranque_sensor].mean())
            if (~c.arranque_sensor).any() else None,
            "calidad": calidad.como_dict(),
        })
        if calidad.lineas_rechazadas:
            alertas.append(
                f"{angulo:>3} grados: {calidad.lineas_rechazadas} linea(s) "
                f"rechazada(s): {calidad.textos_rechazados}"
            )

    total_eventos = sum(fila["n_eventos"] for fila in inventario)
    total_rechazadas = sum(fila["calidad"]["lineas_rechazadas"] for fila in inventario)
    total_arranque = sum(fila["calidad"]["filas_arranque_sensor"] for fila in inventario)

    lineas = [
        f"{'ang':>4} {'archivo':>8} {'N':>5} {'coinc':>6} {'ventana':>9} "
        f"{'muerto':>8} {'vivo':>9} {'muerta%':>8} {'malas':>6} {'arranque':>9}",
        "-" * 88,
    ]
    for fila in inventario:
        lineas.append(
            f"{fila['angulo_deg']:>4} {fila['archivo']:>8} {fila['n_eventos']:>5} "
            f"{fila['n_coincidentes']:>6} {fila['ventana_s']:>9.2f} "
            f"{fila['tiempo_muerto_s']:>8.2f} {fila['tiempo_vivo_s']:>9.2f} "
            f"{100 * fila['fraccion_muerta']:>8.2f} "
            f"{fila['calidad']['lineas_rechazadas']:>6} "
            f"{fila['calidad']['filas_arranque_sensor']:>9}"
        )
    lineas += [
        "-" * 88,
        f"total: {total_eventos} eventos en {len(inventario)} corridas, "
        f"{total_rechazadas} linea(s) rechazada(s), "
        f"{total_arranque} fila(s) con la lectura de arranque del sensor.",
        "",
        "Orden de columnas: la cabecera declara",
        f"  {io.CABECERA_DECLARADA}",
        "y los datos vienen en el orden",
        f"  {io.CABECERA_REAL}",
        "es decir con la presion antes que la temperatura. Verificado archivo a "
        "archivo contra rangos fisicos.",
    ]
    if alertas:
        lineas += ["", "Alertas:"] + [f"  - {a}" for a in alertas]

    texto = "\n".join(lineas)
    print(texto)

    ruta_json, ruta_txt = provenance.escribir_producto(
        args.out, "00_ingest",
        datos={
            "n_corridas": len(inventario),
            "n_eventos_total": total_eventos,
            "n_lineas_rechazadas_total": total_rechazadas,
            "n_filas_arranque_total": total_arranque,
            "cabecera_declarada": io.CABECERA_DECLARADA,
            "cabecera_real": io.CABECERA_REAL,
            "inventario": inventario,
        },
        texto=texto,
        parametros={"data_dir": args.data_dir},
    )
    print(f"[00] escrito {ruta_json}")
    print(f"[00] escrito {ruta_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
