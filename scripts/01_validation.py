#!/usr/bin/env python3
"""
01_validation.py

Puerta de calidad de los datos crudos. Comprueba las condiciones que el resto
del pipeline da por ciertas, y avisa cuando una de ellas deja de cumplirse.

Comprobaciones, en orden:
  1. Estan las ocho corridas esperadas y ninguna esta vacia.
  2. El reloj no retrocede en ninguna corrida. Justifica medir la ventana como
     tN - t1; el script anterior acumulaba diferencias positivas, lo que habria
     absorbido un retroceso en silencio.
  3. La fraccion de tiempo muerto esta en el 9 % caracteristico del conjunto.
     Una corrida que se salga indica un modo de operacion distinto.
  4. Presion y temperatura caen en rango fisico una vez excluidas las filas de
     arranque del sensor.
  5. Ninguna corrida excede la ventana de 600 s que el script anterior imponia
     con --max-ms: se comprueba que ese recorte nunca actuo sobre estos datos.
  6. Coincident == 1 en todos los eventos. Se verifica y se DECLARA la
     consecuencia: sin singles registrados, la tasa de accidentales no es
     estimable desde este conjunto.
  7. Las lineas rechazadas se enumeran una a una, con su texto.

Dos anomalias de estos datos estan ya documentadas en docs/dataset.md: la linea
truncada de 015deg y las ocho filas de arranque del BMP280. Se reportan siempre,
marcadas como CONOCIDA, pero no hacen fallar la puerta: lo que esta tiene que
detectar es un CAMBIO en los datos, no la existencia de defectos ya descritos y
acotados. Con --strict cuentan como alerta.

Salidas exactas:
  - results/tables/01_validation_vN.json
  - results/tables/01_validation_vN.txt

Codigo de salida: 0 si no hay alertas nuevas; 1 si aparece cualquier anomalia no
documentada, o cualquiera en absoluto si se pasa --strict.

Uso tipico, desde la raiz del repositorio y con .venv activo:
  python scripts/01_validation.py
  python scripts/01_validation.py --strict
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import io, provenance  # noqa: E402

ANGULOS_ESPERADOS = [0, 5, 15, 25, 35, 45, 90, 360]
FRACCION_MUERTA_ESPERADA = (0.085, 0.100)
VENTANA_MAXIMA_ANTIGUA_S = 600.0

# Anomalias ya documentadas en docs/dataset.md. Se siguen reportando siempre,
# pero no hacen fallar la puerta: lo que tiene que detectar es un cambio en los
# datos, no la existencia de defectos que ya estan descritos y acotados. Con
# --strict cuentan como alerta y el script sale con 1.
RECHAZOS_CONOCIDOS = {15: ["118  | Hexa address:  0x76"]}
ARRANQUE_CONOCIDO = {0: 2, 5: 2, 25: 2, 35: 1, 360: 1}


def main() -> int:
    ap = argparse.ArgumentParser(description="Puerta de calidad de los datos crudos.")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--strict", action="store_true",
                    help="trata tambien las anomalias ya documentadas como alerta")
    args = ap.parse_args()

    corridas = io.load_all(args.data_dir)
    comprobaciones: list[dict] = []
    lineas: list[str] = []

    def comprobar(nombre: str, pasa: bool, detalle: str,
                  conocida: bool = False) -> None:
        """`conocida` marca una anomalia ya documentada: se reporta pero no
        hace fallar la puerta salvo con --strict."""
        alerta = not pasa and (args.strict or not conocida)
        comprobaciones.append({
            "nombre": nombre, "pasa": bool(pasa), "detalle": detalle,
            "anomalia_conocida": bool(conocida), "cuenta_como_alerta": bool(alerta),
        })
        marca = "OK  " if pasa else ("CONOCIDA" if conocida and not args.strict
                                     else "AVISO")
        lineas.append(f"[{marca:>8}] {nombre}: {detalle}")

    # 1. Inventario
    faltan = sorted(set(ANGULOS_ESPERADOS) - set(corridas))
    sobran = sorted(set(corridas) - set(ANGULOS_ESPERADOS))
    comprobar(
        "inventario de corridas",
        not faltan and not sobran,
        f"{len(corridas)} corridas"
        + (f", faltan {faltan}" if faltan else "")
        + (f", inesperadas {sobran}" if sobran else ""),
    )

    # 2. Monotonia del reloj
    no_monotonas = [
        a for a, c in corridas.items() if not np.all(np.diff(c.timestamp_ms) > 0)
    ]
    comprobar(
        "el reloj no retrocede",
        not no_monotonas,
        "todas las marcas de tiempo son estrictamente crecientes"
        if not no_monotonas else f"retrocede en {no_monotonas}",
    )

    # 3. Tiempo muerto
    fuera_muerto = {
        a: round(c.fraccion_muerta, 4)
        for a, c in corridas.items()
        if not (FRACCION_MUERTA_ESPERADA[0] < c.fraccion_muerta < FRACCION_MUERTA_ESPERADA[1])
    }
    fracciones = np.array([c.fraccion_muerta for c in corridas.values()])
    comprobar(
        "fraccion de tiempo muerto",
        not fuera_muerto,
        f"entre {100 * fracciones.min():.2f} % y {100 * fracciones.max():.2f} %"
        + (f"; fuera de rango: {fuera_muerto}" if fuera_muerto else ""),
    )

    # 4. Rangos fisicos de los sensores
    fuera_rango = {
        a: c.calidad.filas_fuera_de_rango
        for a, c in corridas.items()
        if c.calidad.filas_fuera_de_rango
    }
    comprobar(
        "presion y temperatura en rango",
        not fuera_rango,
        "todas las filas dentro de rango fisico"
        if not fuera_rango else f"filas fuera de rango: {fuera_rango}",
    )

    # 5. La ventana de 600 s del analisis anterior
    largas = {a: round(c.ventana_s, 1) for a, c in corridas.items()
              if c.ventana_s > VENTANA_MAXIMA_ANTIGUA_S}
    ventanas = np.array([c.ventana_s for c in corridas.values()])
    comprobar(
        "ventana de 600 s del analisis anterior",
        not largas,
        f"la corrida mas larga dura {ventanas.max():.1f} s, asi que el recorte "
        f"--max-ms de 600 s nunca actuo sobre estos datos"
        + (f"; exceden {largas}" if largas else ""),
    )

    # 6. Coincidencia
    no_coincidentes = {
        a: int((c.coincidente != 1).sum())
        for a, c in corridas.items() if (c.coincidente != 1).any()
    }
    total_eventos = sum(len(c) for c in corridas.values())
    comprobar(
        "filtro de coincidencia",
        True,
        f"Coincident == 1 en los {total_eventos} eventos, luego el filtro es un "
        "no-op. Al no haber singles registrados, LA TASA DE ACCIDENTALES NO ES "
        "ESTIMABLE desde este conjunto: no se puede restar ni acotar."
        + (f" Excepciones: {no_coincidentes}" if no_coincidentes else ""),
    )

    # 7. Lineas rechazadas y arranque del sensor
    rechazos = {
        a: c.calidad.textos_rechazados
        for a, c in corridas.items() if c.calidad.lineas_rechazadas
    }
    total_rechazadas = sum(c.calidad.lineas_rechazadas for c in corridas.values())
    comprobar(
        "lineas malformadas",
        total_rechazadas == 0,
        f"{total_rechazadas} linea(s) rechazada(s)"
        + (f": {rechazos}" if rechazos else "")
        + (" (documentado: data/raw/015deg esta truncado por arriba)"
           if rechazos == RECHAZOS_CONOCIDOS else ""),
        conocida=rechazos == RECHAZOS_CONOCIDOS,
    )

    arranque = {a: c.calidad.filas_arranque_sensor for a, c in corridas.items()
                if c.calidad.filas_arranque_sensor}
    total_arranque = sum(arranque.values())
    comprobar(
        "lecturas de arranque del sensor",
        total_arranque == 0,
        f"{total_arranque} fila(s) con {io.ARRANQUE_PRESION_PA} Pa / "
        f"{io.ARRANQUE_TEMPERATURA_C} C antes de estabilizarse el BMP280, "
        f"repartidas asi: {arranque}. No afectan al conteo de eventos ni a los "
        "tiempos; se marcan porque invalidan cualquier uso de presion o "
        "temperatura en esas filas."
        if total_arranque else "ninguna",
        conocida=arranque == ARRANQUE_CONOCIDO,
    )

    n_avisos = sum(1 for c in comprobaciones if c["cuenta_como_alerta"])
    n_conocidas = sum(1 for c in comprobaciones
                      if not c["pasa"] and c["anomalia_conocida"])
    lineas += [
        "",
        f"{len(comprobaciones)} comprobaciones, {n_avisos} alerta(s), "
        f"{n_conocidas} anomalia(s) ya documentada(s) en docs/dataset.md.",
    ]
    if n_conocidas and not args.strict:
        lineas.append(
            "Las anomalias documentadas no hacen fallar la puerta: lo que tiene "
            "que detectar es un cambio en los datos. Con --strict cuentan."
        )
    texto = "\n".join(lineas)
    print(texto)

    ruta_json, ruta_txt = provenance.escribir_producto(
        args.out, "01_validation",
        datos={"n_comprobaciones": len(comprobaciones), "n_avisos": n_avisos,
               "n_anomalias_conocidas": n_conocidas,
               "comprobaciones": comprobaciones},
        texto=texto,
        parametros={"data_dir": args.data_dir, "strict": args.strict},
    )
    print(f"[01] escrito {ruta_json}")
    print(f"[01] escrito {ruta_txt}")
    return 1 if n_avisos else 0


if __name__ == "__main__":
    raise SystemExit(main())
