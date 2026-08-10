#!/usr/bin/env python3
"""
03_timeseries.py

Series temporales de tasa instantanea, una figura por corrida y una combinada.

Sirven para lo unico para lo que sirve un binning: mirar si la tasa se mantuvo
estable durante la corrida o si hubo tramos anomalos. Ningun numero fisico del
analisis sale de aqui; las tasas se calculan en 02_rates sobre la ventana
completa y sin bins.

Pasos:
  1. Lectura de las corridas y binning en intervalos de --bin-seconds,
     descartando el bin parcial final, cuya tasa aparente esta inflada.
  2. Media movil centrada y banda de una desviacion tipica local.
  3. Contraste de la dispersion observada contra la esperada de Poisson: para
     una tasa constante R y bins de ancho b, sigma/media vale 1/sqrt(R*b). Un
     cociente muy por encima de 1 indicaria inestabilidad de la corrida.
  4. Figura combinada con las ocho corridas para comparar niveles.

Salidas exactas:
  - results/figures/03_timeseries_{angulo:03d}deg_vN.pdf y .png  (ocho pares)
  - results/figures/03_timeseries_all_vN.pdf y .png
  - results/tables/03_timeseries_vN.json y .txt

Codigo de salida: 0 siempre que las corridas se lean; 1 si alguna no admite ni
un bin completo.

Uso tipico, desde la raiz del repositorio y con .venv activo:
  python scripts/03_timeseries.py
  python scripts/03_timeseries.py --bin-seconds 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import binning, io, plotting, provenance  # noqa: E402
from lib_muon.plotting import plt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Series temporales de tasa.")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-figures", default="results/figures")
    ap.add_argument("--out-tables", default="results/tables")
    ap.add_argument("--bin-seconds", type=float, default=10.0,
                    help="ancho de bin para las series (por defecto 10 s). "
                         "Parametro puramente grafico: no entra en ningun "
                         "resultado fisico.")
    ap.add_argument("--rolling-seconds", type=float, default=60.0,
                    help="ventana de la media movil (por defecto 60 s)")
    args = ap.parse_args()

    plotting.aplicar_estilo()
    corridas = io.load_all(args.data_dir)
    resumen = []

    for angulo in sorted(corridas):
        corrida = corridas[angulo]
        tiempos = corrida.tiempo_relativo_s[corrida.coincidente == 1]
        centros, tasas, errores = binning.tasas_por_bin(
            tiempos, bin_s=args.bin_seconds, inicio_s=0.0,
            fin_s=float(tiempos[-1]),
        )
        if len(centros) == 0:
            print(f"[03] ERROR: {angulo} grados no admite ni un bin completo "
                  f"de {args.bin_seconds} s", file=sys.stderr)
            return 1

        ventana_bins = max(int(round(args.rolling_seconds / args.bin_seconds)), 1)
        medias, desviaciones = binning.media_movil(tasas, ventana_bins)

        media = float(np.mean(tasas))
        dispersion = float(np.std(tasas))
        esperada_poisson = np.sqrt(media / args.bin_seconds)
        cociente = dispersion / esperada_poisson if esperada_poisson > 0 else np.nan

        fig, ax = plt.subplots(figsize=(8.0, 3.4))
        ax.errorbar(centros, tasas, yerr=errores, fmt="o", ms=3, lw=0.8,
                    color=plotting.COLOR_DATOS, alpha=0.8,
                    label=f"tasa por bin de {args.bin_seconds:g} s")
        ax.plot(centros, medias, color=plotting.COLOR_MODELO, lw=1.6,
                label=f"media movil ({args.rolling_seconds:g} s)")
        ax.fill_between(centros, medias - desviaciones, medias + desviaciones,
                        color=plotting.COLOR_BANDA, alpha=0.35,
                        label="+-1 sigma local")
        ax.set_xlabel("tiempo desde el primer evento [s]")
        ax.set_ylabel("tasa [1/s]")
        ax.set_title(f"Corrida de {angulo} grados")
        ax.legend(fontsize=8, loc="upper right")
        ax.text(
            0.02, 0.05,
            f"N = {len(tiempos)}\nmedia = {media:.2f} 1/s\n"
            f"dispersion/Poisson = {cociente:.2f}",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        fig.tight_layout()
        rutas = plotting.guardar(fig, args.out_figures,
                                 f"03_timeseries_{angulo:03d}deg")
        print(f"[03] {angulo:>3} grados: {len(centros)} bins, "
              f"dispersion/Poisson = {cociente:.2f} -> {rutas[0].name}")

        resumen.append({
            "angulo_deg": angulo,
            "n_bins": len(centros),
            "media_hz": media,
            "dispersion_hz": dispersion,
            "dispersion_poisson_esperada_hz": float(esperada_poisson),
            "cociente_dispersion": float(cociente),
        })

    # Figura combinada
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    mapa = plt.get_cmap("viridis")
    angulos = sorted(corridas)
    for i, angulo in enumerate(angulos):
        corrida = corridas[angulo]
        tiempos = corrida.tiempo_relativo_s[corrida.coincidente == 1]
        centros, tasas, _ = binning.tasas_por_bin(
            tiempos, bin_s=args.bin_seconds, inicio_s=0.0, fin_s=float(tiempos[-1])
        )
        ventana_bins = max(int(round(args.rolling_seconds / args.bin_seconds)), 1)
        medias, _ = binning.media_movil(tasas, ventana_bins)
        ax.plot(centros, medias, lw=1.5, color=mapa(i / max(len(angulos) - 1, 1)),
                label=f"{angulo} grados")
    ax.set_xlabel("tiempo desde el primer evento [s]")
    ax.set_ylabel(f"tasa suavizada ({args.rolling_seconds:g} s) [1/s]")
    ax.set_title("Estabilidad temporal de las ocho corridas")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    rutas_todas = plotting.guardar(fig, args.out_figures, "03_timeseries_all")
    print(f"[03] combinada -> {rutas_todas[0].name}")

    cocientes = np.array([r["cociente_dispersion"] for r in resumen])
    texto = "\n".join([
        f"Series temporales con bins de {args.bin_seconds:g} s y media movil de "
        f"{args.rolling_seconds:g} s.",
        "",
        f"{'ang':>4} {'bins':>5} {'media':>9} {'disp':>9} {'Poisson':>9} {'coc':>6}",
        "-" * 50,
        *[
            f"{r['angulo_deg']:>4} {r['n_bins']:>5} {r['media_hz']:>9.4f} "
            f"{r['dispersion_hz']:>9.4f} "
            f"{r['dispersion_poisson_esperada_hz']:>9.4f} "
            f"{r['cociente_dispersion']:>6.2f}"
            for r in resumen
        ],
        "-" * 50,
        f"Cociente dispersion/Poisson entre {cocientes.min():.2f} y "
        f"{cocientes.max():.2f}.",
        "Un valor proximo a 1 indica que la corrida es compatible con una tasa",
        "constante mas fluctuacion de Poisson, es decir que no hay tramos",
        "anomalos dentro de una misma corrida. La dispersion ENTRE corridas es",
        "otra cosa y se trata en 06_systematics.",
    ])

    ruta_json, ruta_txt = provenance.escribir_producto(
        args.out_tables, "03_timeseries",
        datos={"series": resumen},
        texto=texto,
        parametros={"bin_seconds": args.bin_seconds,
                    "rolling_seconds": args.rolling_seconds},
    )
    print(f"[03] escrito {ruta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
