#!/usr/bin/env python3
"""
05_fits.py

Inferencia sobre la dependencia angular, por maxima verosimilitud de Poisson
sobre las cuentas crudas.

Por que no chi2 sobre los cocientes normalizados, que es lo que hacia el
analisis anterior: los siete cocientes comparten el denominador I(0) y no son
independientes, de modo que un chi2 diagonal esta mal planteado; y el punto de
0 grados es degenerado tras normalizar, con residuo nulo por construccion, pero
sumaba a los grados de libertad. Sobre las cuentas crudas ninguno de los dos
problemas existe. El observable es el numero de intervalos entre eventos, N-1,
que es Poisson de media R*T_vivo.

Modelos ajustados, todos con cos(90) anclado a cero exacto:
  A. Constante:      mu_i = R * T_i.                        1 parametro.
  B. cos^n libre:    mu_i = R * T_i * cos^n(theta_i).       2 parametros.
  C. cos^2 fijo:     mu_i = R * T_i * cos^2(theta_i).       1 parametro.
  D. Senal + fondo:  mu_i = T_i * (S*cos^n(theta_i) + B).   3 parametros.
  E. Senal + fondo con n = 2 fijo.                          2 parametros.

Resultado central, y no es un detalle de implementacion: con cos(90) = 0
exacto, los modelos B y C predicen exactamente cero cuentas a 90 grados, donde
se observaron 446. Su verosimilitud es rigurosamente nula y su -logL diverge,
asi que NO tienen AIC ni BIC finitos con los que competir. El unico motivo por
el que el analisis anterior podia ajustar un cos^n era que evaluaba
cos(deg2rad(90))**n = (6.123e-17)**n, que para n pequeno vale del orden de 1.
La familia senal mas fondo es la que corresponde: B recoge en un solo numero lo
que a 90 grados predicen las tres hipotesis del analisis (aceptancia
geometrica ancha, componente accidental y convencion angular no documentada),
que estos datos no permiten separar.

Pasos:
  1. Lectura de results/tables/02_rates_vN.json y construccion del observable.
  2. Ajuste de los cinco modelos.
  3. Incertidumbres por perfil de verosimilitud, Delta(-2lnL) = 1 para el 68 %.
  4. Bondad de ajuste por deviance y p-valor de chi2.
  5. Comparacion por AIC, BIC y razon de verosimilitudes.
  6. Cota superior al 95 % sobre la fraccion de flujo modulada como cos^2.
  7. Figura de ajuste con panel de residuos de Pearson.

Salidas exactas:
  - results/fits/05_fits_vN.json y .txt
  - results/figures/05_fits_vN.pdf y .png

Codigo de salida: 0 si el ajuste del modelo constante y el de senal mas fondo
convergen; 1 si alguno de los dos falla. Que B y C tengan verosimilitud nula NO
es un fallo: es el resultado, y se registra como tal.

Uso tipico, desde la raiz del repositorio y con .venv activo:
  python scripts/05_fits.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import models, plotting, provenance  # noqa: E402
from lib_muon.plotting import plt  # noqa: E402

ANGULO_CONTROL = 360
# Delta(-logL) para una cota unilateral al 95 %: Delta(-2lnL) = 2.706.
DELTA_COTA_95 = 1.3530


def main() -> int:
    ap = argparse.ArgumentParser(description="Ajustes por verosimilitud Poisson.")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--out-fits", default="results/fits")
    ap.add_argument("--out-figures", default="results/figures")
    args = ap.parse_args()

    plotting.aplicar_estilo()
    producto = provenance.leer_producto(args.tables, "02_rates")
    tasas = {int(a): t for a, t in producto["tasas"].items()}

    # El archivo de 360 grados es un control de repetibilidad de 0 grados, no
    # un angulo fisico distinto: entra en 06_systematics, no en el ajuste.
    angulos = np.array(sorted(a for a in tasas if a != ANGULO_CONTROL), dtype=float)
    cuentas = np.array([tasas[int(a)]["n_intervalos"] for a in angulos], dtype=float)
    tiempos = np.array([tasas[int(a)]["tiempo_vivo_s"] for a in angulos])
    cos_t = models.coseno_anclado(angulos)

    print(f"[05] {len(angulos)} puntos de {angulos[0]:.0f} a {angulos[-1]:.0f} "
          f"grados; el control de {ANGULO_CONTROL} grados queda para 06.")
    print(f"[05] cos(theta) anclado: {np.array2string(cos_t, precision=6)}")

    ajustes = {
        "A_constante": models.ajustar_constante(cuentas, tiempos),
        "B_cosn_libre": models.ajustar_cosn_puro(cuentas, tiempos, cos_t),
        "C_cos2_fijo": models.ajustar_cosn_fijo(cuentas, tiempos, cos_t, n=2.0),
        "D_senal_fondo": models.ajustar_senal_fondo(cuentas, tiempos, cos_t),
        "E_senal_fondo_n2": models.ajustar_senal_fondo(cuentas, tiempos, cos_t, n=2.0),
    }

    if not np.isfinite(ajustes["A_constante"].nll):
        print("[05] ERROR: el modelo constante no converge", file=sys.stderr)
        return 1
    if not np.isfinite(ajustes["E_senal_fondo_n2"].nll):
        print("[05] ERROR: el modelo senal mas fondo no converge", file=sys.stderr)
        return 1

    # --- Bondad de ajuste de cada modelo -----------------------------------
    bondad = {}
    for clave, ajuste in ajustes.items():
        d, p = models.p_bondad_ajuste(cuentas, ajuste.mu, ajuste.gdl)
        bondad[clave] = {"deviance": d, "gdl": ajuste.gdl, "p_valor": p}

    # --- Incertidumbre del parametro de tasa del modelo constante ----------
    constante = ajustes["A_constante"]

    def nll_de_r(r):
        return models.nll_poisson(r * tiempos, cuentas)

    r_bajo, r_alto = models.perfil_verosimilitud(
        nll_de_r, constante.parametros["R_hz"], constante.nll,
        limite_inferior=1e-6, limite_superior=10.0,
    )

    # --- Fraccion modulada con n = 2 fijo, y su cota superior --------------
    e = ajustes["E_senal_fondo_n2"]
    forma2 = models.forma_cosn(cos_t, 2.0)

    def nll_perfil_s(s):
        """-logL minimizado sobre B con S fijo."""
        if s < 0:
            return np.inf
        r = optimize.minimize_scalar(
            lambda b: models.nll_poisson(tiempos * (s * forma2 + b), cuentas)
            if b > 0 else np.inf,
            bounds=(1e-9, 20.0), method="bounded",
        )
        return float(r.fun)

    def b_optimo_dado_s(s):
        r = optimize.minimize_scalar(
            lambda b: models.nll_poisson(tiempos * (s * forma2 + b), cuentas)
            if b > 0 else np.inf,
            bounds=(1e-9, 20.0), method="bounded",
        )
        return float(r.x)

    s_mejor = e.parametros["S_hz"]
    s_bajo_68, s_alto_68 = models.perfil_verosimilitud(
        nll_perfil_s, s_mejor, e.nll, delta=0.5,
        limite_inferior=0.0, limite_superior=10.0,
    )
    _, s_cota_95 = models.perfil_verosimilitud(
        nll_perfil_s, s_mejor, e.nll, delta=DELTA_COTA_95,
        limite_inferior=0.0, limite_superior=10.0,
    )
    b_en_cota = b_optimo_dado_s(s_cota_95) if np.isfinite(s_cota_95) else np.nan
    fraccion_cota = (s_cota_95 / (s_cota_95 + b_en_cota)
                     if np.isfinite(s_cota_95) else np.nan)

    # --- Comparacion de modelos --------------------------------------------
    d_lr, p_lr = models.razon_verosimilitudes(
        constante.nll, e.nll, gdl_extra=e.n_parametros - constante.n_parametros
    )

    # --- Salida legible ------------------------------------------------------
    lineas = [
        "OBSERVABLE: numero de intervalos entre eventos, N-1, Poisson de media "
        "R*T_vivo.",
        f"{len(angulos)} puntos, de {angulos[0]:.0f} a {angulos[-1]:.0f} grados. "
        f"El control de {ANGULO_CONTROL} grados se trata en 06_systematics.",
        "",
        f"{'ang':>4} {'N-1':>6} {'T_vivo':>9} {'cos(theta)':>11}",
        "-" * 34,
    ]
    for a, n, t, c in zip(angulos, cuentas, tiempos, cos_t):
        lineas.append(f"{a:>4.0f} {n:>6.0f} {t:>9.2f} {c:>11.6f}")

    lineas += [
        "",
        "=" * 78,
        "AJUSTES",
        "=" * 78,
    ]
    for clave, ajuste in ajustes.items():
        b = bondad[clave]
        if np.isfinite(ajuste.nll):
            parametros = ", ".join(
                f"{k} = {v:.4f}" for k, v in ajuste.parametros.items()
            )
            lineas += [
                f"{clave}  ({ajuste.n_parametros} parametros)",
                f"   {parametros}",
                f"   -logL = {ajuste.nll:.3f}   AIC = {ajuste.aic:.2f}   "
                f"BIC = {ajuste.bic:.2f}",
                f"   deviance = {b['deviance']:.2f} con {b['gdl']} gdl, "
                f"p = {b['p_valor']:.4f}",
            ]
        else:
            lineas += [
                f"{clave}  ({ajuste.n_parametros} parametros)",
                "   -logL = INFINITO. El modelo predice exactamente 0 cuentas a "
                f"90 grados y se observaron {cuentas[-1]:.0f}.",
                "   Su verosimilitud es rigurosamente cero: no hay AIC ni BIC que "
                "comparar.",
                "   Esto es consecuencia de anclar cos(90) = 0. El analisis "
                "anterior podia ajustarlo",
                "   solo porque evaluaba (6.123e-17)**n, que para n pequeno vale "
                "del orden de 1.",
            ]
        lineas.append("")

    lineas += [
        "=" * 78,
        "RESULTADOS",
        "=" * 78,
        "",
        "1. Tasa unica (modelo constante)",
        f"   R = {constante.parametros['R_hz']:.4f} 1/s, "
        f"intervalo del 68 % [{r_bajo:.4f}, {r_alto:.4f}]",
        f"   deviance = {bondad['A_constante']['deviance']:.2f} con "
        f"{bondad['A_constante']['gdl']} gdl, "
        f"p = {bondad['A_constante']['p_valor']:.4f}",
    ]
    if bondad["A_constante"]["p_valor"] < 0.05:
        lineas += [
            "   NO es un buen ajuste bajo Poisson pura. Es el mejor soportado de",
            "   los cinco, pero decir solo eso seria enganoso: los datos estan mas",
            "   dispersos de lo que permite la estadistica de conteo sola. La",
            "   dispersion sobrante se cuantifica en 06_systematics, donde se ve",
            "   que es del tamano del sistematico entre corridas que mide el",
            "   propio control de 360 grados.",
        ]
    lineas += [
        "",
        "2. Cuanta modulacion angular admiten los datos (modelo E, n = 2 fijo)",
        f"   S = {e.parametros['S_hz']:.4f} 1/s "
        f"[{s_bajo_68:.4f}, {s_alto_68:.4f}] al 68 %",
        f"   B = {e.parametros['B_hz']:.4f} 1/s",
        f"   fraccion modulada a 0 grados = {e.parametros['fraccion_modulada']:.4f}",
        f"   COTA SUPERIOR AL 95 %: S < {s_cota_95:.4f} 1/s, es decir una",
        f"   fraccion modulada por debajo del {100 * fraccion_cota:.1f} %.",
        "",
        "3. Comparacion con la hipotesis de tasa constante",
        f"   Delta(-2lnL) = {d_lr:.3f} con "
        f"{e.n_parametros - constante.n_parametros} gdl, p = {p_lr:.4f}",
    ]
    if p_lr > 0.05:
        lineas.append(
            "   Anadir una componente modulada NO mejora significativamente el "
            "ajuste."
        )
    else:
        lineas += [
            "   Bajo Poisson pura la componente modulada SI mejora el ajuste.",
            "",
            "   ESTE P-VALOR NO ES LA CONCLUSION, y tomarlo por tal seria el error",
            "   principal que este analisis existe para no repetir. Supone que la",
            "   unica fuente de incertidumbre es la estadistica de conteo, y este",
            "   conjunto contiene la prueba de que no lo es: el archivo de 360",
            "   grados repite la medida de 0 grados y difiere un 7.2 %. Basta un",
            "   sistematico entre corridas del 3.9 % para que esta preferencia deje",
            "   de ser significativa, es decir bastante menos de lo que el propio",
            "   control mide. La reevaluacion esta en 06_systematics y es la que",
            "   manda.",
        ]
    lineas += [
        "   Advertencia estadistica adicional: S esta acotado a valores no",
        "   negativos, luego el parametro esta en el borde del espacio y la",
        "   distribucion asintotica no es exactamente chi2.",
        "",
        "4. Residuos de Pearson frente al modelo constante",
    ]
    residuos_constante = models.residuos_pearson(cuentas, constante.mu)
    for a, n, m, r in zip(angulos, cuentas, constante.mu, residuos_constante):
        lineas.append(
            f"   {a:>3.0f} grados: N-1 = {n:>4.0f}, esperado = {m:>7.1f}, "
            f"residuo = {r:>+6.2f} sigma"
        )

    texto = "\n".join(lineas)
    print(texto)

    # --- Figura con panel de residuos --------------------------------------
    # constrained_layout en vez de tight_layout: este ultimo no sabe tratar un
    # gridspec con hspace fijado y avisa de que el resultado puede salir mal.
    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(7.6, 6.2), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [3, 1]},
    )

    tasa_obs = cuentas / tiempos
    error_obs = tasa_obs / np.sqrt(cuentas)
    ax.errorbar(angulos, tasa_obs, yerr=error_obs, fmt="o", ms=6, capsize=3,
                color=plotting.COLOR_DATOS, label="medido", zorder=5)

    fino = np.linspace(0, 90, 361)
    cos_fino = models.coseno_anclado(fino)
    ax.axhline(constante.parametros["R_hz"], color=plotting.COLOR_ALTERNATIVO,
               lw=1.8, label=f"constante, R = {constante.parametros['R_hz']:.3f} 1/s")
    ax.fill_between(fino, r_bajo, r_alto, color=plotting.COLOR_ALTERNATIVO,
                    alpha=0.15)
    ax.plot(fino,
            e.parametros["S_hz"] * models.forma_cosn(cos_fino, 2.0)
            + e.parametros["B_hz"],
            "-", color=plotting.COLOR_MODELO, lw=1.8,
            label=f"S cos^2 + B, S = {e.parametros['S_hz']:.3f} 1/s")
    ax.plot(fino,
            s_cota_95 * models.forma_cosn(cos_fino, 2.0) + b_en_cota,
            ":", color=plotting.COLOR_MODELO, lw=1.5,
            label=f"cota al 95 %, S < {s_cota_95:.3f} 1/s")
    escala = constante.parametros["R_hz"]
    ax.plot(fino, escala * models.forma_cosn(cos_fino, 2.0), "--", color="0.45",
            lw=1.5, label="cos^2 puro (verosimilitud nula)")
    ax.set_ylabel("tasa corregida por tiempo vivo [1/s]")
    ax.set_title("Dependencia angular del flujo: lo medido y lo que admite")
    ax.set_ylim(0, 2.6)
    ax.legend(fontsize=8, loc="lower left")

    ax_res.errorbar(angulos, residuos_constante, yerr=1.0, fmt="o", ms=5,
                    capsize=3, color=plotting.COLOR_DATOS)
    ax_res.axhline(0, color=plotting.COLOR_ALTERNATIVO, lw=1.4)
    ax_res.axhspan(-2, 2, color=plotting.COLOR_ALTERNATIVO, alpha=0.12)
    ax_res.set_xlabel("angulo nominal [grados]")
    ax_res.set_ylabel("residuo\n[sigma]")
    ax_res.set_ylim(-4, 4)
    rutas = plotting.guardar(fig, args.out_figures, "05_fits")
    print(f"[05] figura -> {rutas[0].name}")

    datos = {
        "angulos_deg": angulos,
        "cuentas": cuentas,
        "tiempos_vivos_s": tiempos,
        "coseno_anclado": cos_t,
        "eps_coseno": models.EPS_COSENO,
        "ajustes": {k: {**a.como_dict(), **bondad[k]} for k, a in ajustes.items()},
        "constante": {
            "R_hz": constante.parametros["R_hz"],
            "intervalo_68": [r_bajo, r_alto],
        },
        "modulacion_n2": {
            "S_hz": e.parametros["S_hz"],
            "S_intervalo_68": [s_bajo_68, s_alto_68],
            "B_hz": e.parametros["B_hz"],
            "fraccion_modulada": e.parametros["fraccion_modulada"],
            "S_cota_95": s_cota_95,
            "B_en_cota_95": b_en_cota,
            "fraccion_modulada_cota_95": fraccion_cota,
        },
        "razon_verosimilitudes_constante_vs_modulado": {
            "delta_2lnL": d_lr, "gdl": e.n_parametros - constante.n_parametros,
            "p_valor": p_lr,
        },
        "residuos_pearson_constante": residuos_constante,
    }
    ruta_json, ruta_txt = provenance.escribir_producto(
        args.out_fits, "05_fits", datos=datos, texto=texto,
        parametros={"angulo_control_excluido": ANGULO_CONTROL,
                    "delta_cota_95": DELTA_COTA_95},
    )
    print(f"[05] escrito {ruta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
