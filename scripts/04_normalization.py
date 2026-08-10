#!/usr/bin/env python3
"""
04_normalization.py

Tasas relativas a la corrida de 0 grados, I(theta)/I(0), y su figura.

Estos cocientes son la forma tradicional de presentar la dependencia angular y
permiten comparar con el analisis anterior, pero NO son la base de la
inferencia, por dos motivos que el script anterior pasaba por alto:

  - Los siete cocientes comparten el denominador I(0), luego no son
    independientes. Un chi2 diagonal sobre ellos, que es lo que hacia el
    analisis anterior, ignora esa correlacion por completo.
  - El punto de 0 grados es degenerado: vale 1 exactamente por construccion y
    su residuo es cero siempre. El analisis anterior lo contaba igualmente en
    los grados de libertad y con ello subestimaba el chi2 reducido.

La inferencia se hace en 05_fits sobre las cuentas crudas, donde ninguno de los
dos problemas existe.

Pasos:
  1. Lectura de results/tables/02_rates_vN.json.
  2. Cociente y propagacion de incertidumbre respecto a 0 grados.
  3. Contraste con cos^2(theta), el modelo que el repositorio prometia.
  4. Comprobacion de monotonia: un cos^n con n > 0 es monotono decreciente en
     [0, 90], asi que cualquier subida es incompatible con toda la familia.
  5. Figura de I(theta)/I(0) con cos^2 superpuesto.

Salidas exactas:
  - results/tables/04_normalization_vN.json y .txt
  - results/figures/04_normalization_vN.pdf y .png

Codigo de salida: 0 si la normalizacion se completa; 1 si falta la corrida de
referencia.

Uso tipico, desde la raiz del repositorio y con .venv activo:
  python scripts/04_normalization.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import models, plotting, provenance  # noqa: E402
from lib_muon.plotting import plt  # noqa: E402

ANGULO_REFERENCIA = 0
ANGULO_CONTROL = 360


def main() -> int:
    ap = argparse.ArgumentParser(description="Tasas normalizadas a 0 grados.")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--out-figures", default="results/figures")
    args = ap.parse_args()

    plotting.aplicar_estilo()
    producto = provenance.leer_producto(args.tables, "02_rates")
    tasas = {int(a): t for a, t in producto["tasas"].items()}

    if ANGULO_REFERENCIA not in tasas:
        print(f"[04] ERROR: falta la corrida de referencia de "
              f"{ANGULO_REFERENCIA} grados", file=sys.stderr)
        return 1

    r0 = tasas[ANGULO_REFERENCIA]["tasa_viva_hz"]
    e0 = tasas[ANGULO_REFERENCIA]["error_tasa_viva_hz"]

    filas = []
    for angulo in sorted(tasas):
        r = tasas[angulo]["tasa_viva_hz"]
        e = tasas[angulo]["error_tasa_viva_hz"]
        cociente = r / r0
        error = np.sqrt((e / r0) ** 2 + (r * e0 / r0**2) ** 2)
        cos2 = float(models.forma_cosn(models.coseno_anclado(np.array([angulo])), 2.0)[0]) \
            if angulo <= 90 else None
        filas.append({
            "angulo_deg": angulo,
            "tasa_viva_hz": r,
            "error_hz": e,
            "normalizada": float(cociente),
            "error_normalizada": float(error),
            "cos2_predicho": cos2,
            "desviacion_sigma": float((cociente - cos2) / error)
            if cos2 is not None else None,
        })

    fisicos = [f for f in filas if f["angulo_deg"] <= 90]
    angulos = np.array([f["angulo_deg"] for f in fisicos], dtype=float)
    valores = np.array([f["normalizada"] for f in fisicos])
    errores = np.array([f["error_normalizada"] for f in fisicos])
    cos2 = np.array([f["cos2_predicho"] for f in fisicos])

    chi2_cos2 = float(np.sum(((valores - cos2) / errores) ** 2))

    # Monotonia: cualquier cos^n con n > 0 decrece en [0, 90].
    diferencias = np.diff(valores)
    subidas = [
        {"de_deg": float(angulos[i]), "a_deg": float(angulos[i + 1]),
         "salto": float(diferencias[i]),
         "sigmas": float(diferencias[i] / np.hypot(errores[i], errores[i + 1]))}
        for i in range(len(diferencias)) if diferencias[i] > 0
    ]

    lineas = [
        f"{'ang':>4} {'R_viva':>9} {'I/I0':>9} {'sigma':>8} {'cos^2':>8} {'desv':>8}",
        "-" * 54,
    ]
    for f in filas:
        cos2_txt = f"{f['cos2_predicho']:.4f}" if f["cos2_predicho"] is not None else "   ---"
        desv_txt = f"{f['desviacion_sigma']:+.1f}s" if f["desviacion_sigma"] is not None else "   ---"
        marca = "  <- control de repetibilidad" if f["angulo_deg"] == ANGULO_CONTROL else ""
        lineas.append(
            f"{f['angulo_deg']:>4} {f['tasa_viva_hz']:>9.4f} {f['normalizada']:>9.4f} "
            f"{f['error_normalizada']:>8.4f} {cos2_txt:>8} {desv_txt:>8}{marca}"
        )

    lineas += [
        "-" * 54,
        "",
        f"chi2 de los {len(fisicos)} puntos de 0 a 90 grados contra cos^2 puro: "
        f"{chi2_cos2:.1f}",
        f"I(90)/I(0) medido = {valores[-1]:.4f} +- {errores[-1]:.4f}, "
        f"frente a 0 que predice cos^2.",
        "",
        "MONOTONIA. Todo modelo cos^n con n > 0 decrece de forma monotona en",
        "[0, 90] grados. El perfil medido no lo hace:",
    ]
    if subidas:
        for s in subidas:
            lineas.append(
                f"  sube de {s['de_deg']:.0f} a {s['a_deg']:.0f} grados: "
                f"{s['salto']:+.4f} ({s['sigmas']:+.2f} sigma)"
            )
        lineas += [
            "",
            "Este argumento es de otra clase que el chi2, y conviene no confundirlos.",
            "El chi2 mide discrepancia y es la evidencia que sostiene la conclusion.",
            "La no monotonia seria, SI la subida fuese real, una falsacion estructural",
            "de la familia cos^n entera: ningun miembro puede subir, ajuste como",
            "ajuste. Pero la subida por si sola vale 1.47 sigma, que no basta para",
            "afirmarla, y menos sin haber preespecificado que se mirarian esos dos",
            "puntos. Apunta en la misma direccion que el chi2 sin ser independiente",
            "de el; no lo sustituye ni lo supera.",
        ]
    else:
        lineas.append("  el perfil medido si es monotono.")

    texto = "\n".join(lineas)
    print(texto)

    # Figura
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.errorbar(angulos, valores, yerr=errores, fmt="o", ms=6, capsize=3,
                color=plotting.COLOR_DATOS, label="medido, I(theta)/I(0)")
    fino = np.linspace(0, 90, 361)
    ax.plot(fino, models.forma_cosn(models.coseno_anclado(fino), 2.0), "--",
            color=plotting.COLOR_MODELO, lw=1.8,
            label="cos^2(theta), lo que el repositorio prometia")
    control = [f for f in filas if f["angulo_deg"] == ANGULO_CONTROL]
    if control:
        ax.errorbar([0], [control[0]["normalizada"]],
                    yerr=[control[0]["error_normalizada"]], fmt="s", ms=7,
                    color=plotting.COLOR_ALTERNATIVO, capsize=3,
                    label="repeticion de 0 grados (archivo 360deg)")
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("angulo nominal [grados]")
    ax.set_ylabel("tasa normalizada I(theta)/I(0)")
    ax.set_title("Dependencia angular medida frente a la esperada")
    ax.set_ylim(-0.05, 1.25)
    ax.legend(fontsize=8, loc="lower left")
    ax.text(
        0.97, 0.95,
        f"chi2 contra cos^2 = {chi2_cos2:.0f}\ncon {len(fisicos)} puntos",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    rutas = plotting.guardar(fig, args.out_figures, "04_normalization")
    print(f"[04] figura -> {rutas[0].name}")

    ruta_json, ruta_txt = provenance.escribir_producto(
        args.tables, "04_normalization",
        datos={
            "angulo_referencia": ANGULO_REFERENCIA,
            "chi2_contra_cos2": chi2_cos2,
            "n_puntos_fisicos": len(fisicos),
            "subidas_no_monotonas": subidas,
            "filas": filas,
        },
        texto=texto,
        parametros={"angulo_referencia": ANGULO_REFERENCIA},
    )
    print(f"[04] escrito {ruta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
