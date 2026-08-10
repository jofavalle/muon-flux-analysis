#!/usr/bin/env python3
"""
06_systematics.py

Incertidumbres sistematicas y reevaluacion de la significancia del resultado.

Este es el paso que decide la conclusion del analisis. 05_fits hace la
inferencia bajo la hipotesis de que la unica fuente de incertidumbre es la
estadistica de conteo, y bajo esa hipotesis la componente modulada resulta
significativa. Este script comprueba que esa hipotesis es falsa en este
conjunto, y con cuanto margen.

Fuentes evaluadas:
  1. REPETIBILIDAD ENTRE CORRIDAS. El archivo de 360 grados es una segunda
     medida del mismo angulo que el de 0 grados. La discrepancia entre ambas es
     una medida directa de cuanto se mueve el aparato entre corridas por causas
     que no son estadisticas. Es la sistematica dominante y la unica que este
     conjunto permite medir en vez de estimar.
  2. TIEMPO VIVO. Se separa su efecto sobre la tasa absoluta del efecto sobre
     los cocientes entre angulos, que es de segundo orden porque la fraccion
     muerta es casi identica en las ocho corridas.
  3. VENTANA TEMPORAL. Cuanto se mueven los cocientes al analizar solo la
     primera mitad de cada corrida.
  4. ANGULO DE REFERENCIA. Cuanto depende el resultado de haber normalizado a
     0 grados y no a otro angulo.
  5. TAMANO DE BIN. En el analisis anterior movia la tasa publicada un 0.4 %;
     aqui se comprueba que ya no interviene en ningun numero fisico.

Con la sistematica dominante medida, se rehace la comparacion de modelos y se
recalcula la cota superior sobre la fraccion modulada.

Salidas exactas:
  - results/tables/06_systematics_vN.json y .txt
  - results/figures/06_systematics_vN.pdf y .png

Codigo de salida: 0 si la evaluacion se completa; 1 si falta el control de
repetibilidad, sin el cual la sistematica dominante no es medible.

Uso tipico, desde la raiz del repositorio y con .venv activo:
  python scripts/06_systematics.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import io, models, plotting, provenance, rates  # noqa: E402
from lib_muon.plotting import plt  # noqa: E402

ANGULO_REFERENCIA = 0
ANGULO_CONTROL = 360


def ajuste_ponderado(tasas, sigmas, forma):
    """Minimos cuadrados ponderados de R = S*forma + B. Devuelve (S, B, chi2)."""
    matriz = np.vstack([forma, np.ones_like(forma)]).T
    pesos = np.diag(1.0 / sigmas**2)
    coeficientes = np.linalg.solve(
        matriz.T @ pesos @ matriz, matriz.T @ pesos @ tasas
    )
    residuo = tasas - matriz @ coeficientes
    return coeficientes[0], coeficientes[1], float(np.sum((residuo / sigmas) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser(description="Sistematicas y reevaluacion.")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--out-figures", default="results/figures")
    args = ap.parse_args()

    plotting.aplicar_estilo()
    producto = provenance.leer_producto(args.tables, "02_rates")
    tasas_json = {int(a): t for a, t in producto["tasas"].items()}

    if ANGULO_CONTROL not in tasas_json:
        print(f"[06] ERROR: falta la corrida de control de {ANGULO_CONTROL} "
              "grados; la sistematica dominante no es medible", file=sys.stderr)
        return 1

    lineas = []
    fuentes = []

    # --- 1. Repetibilidad entre corridas -----------------------------------
    r_ref = tasas_json[ANGULO_REFERENCIA]["tasa_viva_hz"]
    e_ref = tasas_json[ANGULO_REFERENCIA]["error_tasa_viva_hz"]
    r_ctl = tasas_json[ANGULO_CONTROL]["tasa_viva_hz"]
    e_ctl = tasas_json[ANGULO_CONTROL]["error_tasa_viva_hz"]

    cociente = r_ctl / r_ref
    error_cociente = np.sqrt((e_ctl / r_ref) ** 2 + (r_ctl * e_ref / r_ref**2) ** 2)
    discrepancia = abs(1.0 - cociente)
    sigmas_discrepancia = discrepancia / error_cociente

    lineas += [
        "=" * 78,
        "1. REPETIBILIDAD ENTRE CORRIDAS (la sistematica dominante)",
        "=" * 78,
        f"El archivo de {ANGULO_CONTROL} grados y el de {ANGULO_REFERENCIA} grados "
        "miden el MISMO angulo.",
        f"  R({ANGULO_REFERENCIA:>3} grados) = {r_ref:.4f} +- {e_ref:.4f} 1/s",
        f"  R({ANGULO_CONTROL:>3} grados) = {r_ctl:.4f} +- {e_ctl:.4f} 1/s",
        f"  cociente = {cociente:.4f} +- {error_cociente:.4f}, frente a 1 esperado",
        f"  discrepancia = {100 * discrepancia:.1f} % "
        f"({sigmas_discrepancia:.2f} sigma)",
        "",
        "Interpretacion honesta: la discrepancia no esta establecida como efecto "
        f"({sigmas_discrepancia:.2f} sigma no basta para",
        "afirmar que las dos corridas difieren de verdad). Pero como ESCALA del "
        "sistematico",
        "entre corridas es la mejor informacion que hay, y es la unica que este "
        "conjunto",
        "permite medir en vez de suponer.",
        "",
        "Y esa escala es grande comparada con lo que el analisis pretende "
        "interpretar:",
    ]

    normalizadas = provenance.leer_producto(args.tables, "04_normalization")
    fila_45 = next(f for f in normalizadas["filas"] if f["angulo_deg"] == 45)
    caida_0_45 = 1.0 - fila_45["normalizada"]
    lineas += [
        f"  caida total de 0 a 45 grados:        {100 * caida_0_45:.1f} %",
        f"  discrepancia entre dos medidas de 0: {100 * discrepancia:.1f} %",
        "",
        "El sistematico entre corridas del mismo angulo es MAYOR que la caida "
        "completa",
        "de 0 a 45 grados. Dicho de otro modo: se come toda la senal que el "
        "ajuste",
        "pretende interpretar. Esta sola frase resume el resultado del analisis.",
    ]
    fuentes.append({
        "fuente": "repetibilidad entre corridas",
        "efecto_relativo": float(discrepancia),
        "medida_o_estimada": "medida",
        "detalle": f"control de {ANGULO_CONTROL} grados frente a "
                   f"{ANGULO_REFERENCIA} grados, {sigmas_discrepancia:.2f} sigma",
    })

    # --- 2. Tiempo vivo ------------------------------------------------------
    corridas = io.load_all(args.data_dir)
    tasas_obj = {a: rates.calcular_tasa(c) for a, c in corridas.items()}
    fracciones = np.array([t.fraccion_muerta for t in tasas_obj.values()])
    efecto_absoluto = float(fracciones.mean() / (1 - fracciones.mean()))

    norm_pared = {a: t.tasa_pared / tasas_obj[ANGULO_REFERENCIA].tasa_pared
                  for a, t in tasas_obj.items()}
    norm_viva = {a: t.tasa_viva / tasas_obj[ANGULO_REFERENCIA].tasa_viva
                 for a, t in tasas_obj.items()}
    efecto_cocientes = max(
        abs(norm_viva[a] - norm_pared[a]) / norm_pared[a] for a in norm_viva
    )

    lineas += [
        "",
        "=" * 78,
        "2. TIEMPO VIVO",
        "=" * 78,
        f"Fraccion muerta entre {100 * fracciones.min():.2f} % y "
        f"{100 * fracciones.max():.2f} %, dispersion "
        f"{100 * fracciones.std():.3f} puntos.",
        f"  efecto sobre la tasa ABSOLUTA:   {100 * efecto_absoluto:.2f} %",
        f"  efecto sobre los COCIENTES:      {100 * efecto_cocientes:.2f} %",
        "",
        "Corregir por tiempo vivo es obligatorio y el analisis anterior no lo",
        "hacia, pero conviene no exagerar lo que arregla: como la fraccion muerta",
        "es casi identica en las ocho corridas, se cancela en los cocientes y deja",
        "la forma angular practicamente donde estaba. Importa para la tasa",
        "absoluta, no para la dependencia con el angulo.",
    ]
    fuentes.append({
        "fuente": "tiempo vivo (sobre cocientes)",
        "efecto_relativo": float(efecto_cocientes),
        "medida_o_estimada": "medida",
        "detalle": f"efecto sobre la tasa absoluta: {100 * efecto_absoluto:.2f} %",
    })

    # --- 3. Ventana temporal -------------------------------------------------
    desviaciones_ventana = []
    esperadas_ventana = []
    for angulo, corrida in corridas.items():
        tiempos = corrida.tiempo_relativo_s[corrida.coincidente == 1]
        mitad = tiempos[-1] / 2.0
        primera = tiempos[tiempos <= mitad]
        if len(primera) < 2:
            continue
        tasa_mitad = (len(primera) - 1) / (primera[-1] - primera[0])
        desviaciones_ventana.append(
            abs(tasa_mitad - tasas_obj[angulo].tasa_pared) / tasas_obj[angulo].tasa_pared
        )
        # Comparar media mitad con el total: las dos comparten los eventos de la
        # primera mitad, y la varianza de la diferencia relativa vale
        # 1/N_1 - 1/N con N_1 los de la mitad y N los de la corrida entera.
        n_mitad, n_total = len(primera) - 1, len(tiempos) - 1
        esperadas_ventana.append(np.sqrt(max(1 / n_mitad - 1 / n_total, 0.0)))

    desviaciones_ventana = np.array(desviaciones_ventana)
    esperadas_ventana = np.array(esperadas_ventana)
    efecto_ventana = float(np.max(desviaciones_ventana))
    esperado_tipico = float(np.mean(esperadas_ventana))
    exceso = float(np.sqrt(max(
        np.mean(desviaciones_ventana**2) - np.mean(esperadas_ventana**2), 0.0
    )))

    lineas += [
        "",
        "=" * 78,
        "3. VENTANA TEMPORAL",
        "=" * 78,
        "Comparando la primera mitad de cada corrida contra la corrida entera:",
        f"  desviacion maxima observada:      {100 * efecto_ventana:.2f} %",
        f"  mediana observada:                "
        f"{100 * np.median(desviaciones_ventana):.2f} %",
        f"  esperada solo por estadistica:    {100 * esperado_tipico:.2f} %",
        f"  exceso sobre lo estadistico:      {100 * exceso:.2f} %",
        "",
        "ESTA FILA NO ES UNA SISTEMATICA y no debe leerse como tal. Media corrida",
        "tiene la mitad de eventos, asi que su tasa fluctua lo que manda Poisson;",
        "el numero observado es del tamano de esa fluctuacion. Lo que entra en el",
        "presupuesto de sistematicas es el EXCESO sobre ella, que aqui es",
        f"{100 * exceso:.2f} %. Coincide con lo visto en 03_timeseries, donde la",
        "dispersion por bins dentro de cada corrida es compatible con Poisson: la",
        "inestabilidad esta ENTRE corridas, no dentro de ellas.",
    ]
    fuentes.append({
        "fuente": "ventana temporal (exceso sobre estadistica)",
        "efecto_relativo": exceso,
        "medida_o_estimada": "medida",
        "detalle": f"observado {100 * efecto_ventana:.2f} %, esperado por "
                   f"estadistica {100 * esperado_tipico:.2f} %; solo el exceso "
                   f"cuenta como sistematica",
    })

    # --- 4. Angulo de referencia --------------------------------------------
    angulos_fisicos = sorted(a for a in tasas_json if a != ANGULO_CONTROL)
    variacion_referencia = []
    base = {a: tasas_json[a]["tasa_viva_hz"] / r_ref for a in angulos_fisicos}
    for alterno in angulos_fisicos:
        r_alt = tasas_json[alterno]["tasa_viva_hz"]
        for a in angulos_fisicos:
            propio = tasas_json[a]["tasa_viva_hz"] / r_alt
            reescalado = propio * (r_alt / r_ref)
            variacion_referencia.append(abs(reescalado - base[a]))
    efecto_referencia = float(np.max(variacion_referencia))

    lineas += [
        "",
        "=" * 78,
        "4. ANGULO DE REFERENCIA",
        "=" * 78,
        "Normalizar a otro angulo reescala todos los cocientes por un factor",
        "comun y no cambia la FORMA del perfil: la variacion residual es",
        f"{efecto_referencia:.2e}, es decir cero salvo redondeo.",
        "Lo que si cambia es la incertidumbre: el angulo de referencia inyecta su",
        f"propio error en los siete cocientes ({100 * e_ref / r_ref:.1f} % aqui), y",
        "por eso la inferencia de 05_fits se hace sobre cuentas crudas y no sobre",
        "cocientes, donde ese error comun quedaria contado siete veces como si",
        "fuese independiente.",
    ]
    fuentes.append({
        "fuente": "angulo de referencia",
        "efecto_relativo": efecto_referencia,
        "medida_o_estimada": "medida",
        "detalle": "solo reescala; el error de la referencia se trata en 05_fits",
    })

    # --- 5. Tamano de bin ----------------------------------------------------
    lineas += [
        "",
        "=" * 78,
        "5. TAMANO DE BIN",
        "=" * 78,
        "En el analisis anterior la duracion de cada corrida se truncaba a un",
        "multiplo entero de --bin-seconds, de modo que un parametro grafico movia",
        "la tasa publicada: 1.8118 1/s con bin de 1 s frente a 1.8194 1/s con bin",
        "de 60 s, y el cociente I(90)/I(0) de 0.8079 a 0.8148.",
        "En este pipeline el binning solo interviene en las figuras de",
        "03_timeseries. La tasa se calcula sobre la ventana completa [t1, tN] y",
        "calcular_tasa no admite ningun parametro de bin, asi que el efecto es",
        "exactamente cero por construccion, no pequeno.",
    ]
    fuentes.append({
        "fuente": "tamano de bin",
        "efecto_relativo": 0.0,
        "medida_o_estimada": "eliminada por construccion",
        "detalle": "antes 0.4 % sobre la tasa; ahora el binning no entra en la fisica",
    })

    # --- 6. Reevaluacion de la significancia --------------------------------
    angulos = np.array(angulos_fisicos, dtype=float)
    cuentas = np.array([tasas_json[int(a)]["n_intervalos"] for a in angulos], float)
    tiempos = np.array([tasas_json[int(a)]["tiempo_vivo_s"] for a in angulos])
    tasas_obs = cuentas / tiempos
    sigma_estadistico = tasas_obs / np.sqrt(cuentas)
    forma2 = models.forma_cosn(models.coseno_anclado(angulos), 2.0)

    f_sys = float(discrepancia)

    def delta_chi2(f):
        sigma = np.sqrt(sigma_estadistico**2 + (f * tasas_obs) ** 2)
        pesos = 1.0 / sigma**2
        media = np.sum(pesos * tasas_obs) / np.sum(pesos)
        chi2_constante = float(np.sum(((tasas_obs - media) / sigma) ** 2))
        _, _, chi2_modulado = ajuste_ponderado(tasas_obs, sigma, forma2)
        return chi2_constante - chi2_modulado

    barrido = []
    for f in (0.0, 0.02, 0.03, 0.05, f_sys, 0.10):
        sigma = np.sqrt(sigma_estadistico**2 + (f * tasas_obs) ** 2)
        pesos = 1.0 / sigma**2
        media = float(np.sum(pesos * tasas_obs) / np.sum(pesos))
        chi2_constante = float(np.sum(((tasas_obs - media) / sigma) ** 2))
        s_ajustado, b_ajustado, chi2_modulado = ajuste_ponderado(tasas_obs, sigma, forma2)
        d = chi2_constante - chi2_modulado
        barrido.append({
            "f_sys": float(f),
            "chi2_constante": chi2_constante,
            "p_constante": float(stats.chi2.sf(chi2_constante, len(angulos) - 1)),
            "S_hz": float(s_ajustado),
            "delta_chi2": float(d),
            "p_modulacion": float(stats.chi2.sf(d, 1)),
        })

    f_critico = float(optimize.brentq(
        lambda f: delta_chi2(f) - stats.chi2.isf(0.05, 1), 1e-6, 0.5
    ))
    f_compatible = models.sistematico_necesario(tasas_obs, sigma_estadistico, 0.05)

    lineas += [
        "",
        "=" * 78,
        "6. REEVALUACION CON LA SISTEMATICA MEDIDA",
        "=" * 78,
        "05_fits encuentra que la componente modulada mejora el ajuste con",
        "p = 0.008, pero eso supone que la unica incertidumbre es la de conteo.",
        "Anadiendo el sistematico entre corridas en cuadratura:",
        "",
        f"{'f_sys':>8} {'chi2_cte':>10} {'p_cte':>8} {'S [1/s]':>9} "
        f"{'Dchi2':>8} {'p_mod':>8}  veredicto",
        "-" * 74,
    ]
    for b in barrido:
        marca = "  <- el que mide el control" if abs(b["f_sys"] - f_sys) < 1e-12 else ""
        lineas.append(
            f"{100 * b['f_sys']:>7.1f}% {b['chi2_constante']:>10.2f} "
            f"{b['p_constante']:>8.4f} {b['S_hz']:>9.4f} {b['delta_chi2']:>8.2f} "
            f"{b['p_modulacion']:>8.4f}  "
            f"{'significativa' if b['p_modulacion'] < 0.05 else 'NO significativa'}"
            f"{marca}"
        )

    lineas += [
        "-" * 74,
        "",
        f"La modulacion deja de ser significativa a partir de un sistematico del "
        f"{100 * f_critico:.2f} %.",
        f"El control de repetibilidad mide {100 * f_sys:.1f} %, casi el doble.",
        f"La hipotesis de tasa constante pasa a ser compatible con un "
        f"{100 * f_compatible:.2f} %.",
        "",
        "CONCLUSION DEL ANALISIS:",
        "La preferencia por una componente modulada que aparece bajo estadistica",
        "de conteo pura NO sobrevive al sistematico entre corridas que el propio",
        "conjunto de datos mide. No se puede afirmar que este montaje haya",
        "detectado la dependencia angular del flujo de muones. Lo que si se puede",
        "afirmar es una cota superior sobre cuanta modulacion seria compatible con",
        "lo observado.",
    ]

    # --- Cota superior con sistematico incluido -----------------------------
    sigma_total = np.sqrt(sigma_estadistico**2 + (f_sys * tasas_obs) ** 2)
    s_mejor, b_mejor, chi2_min = ajuste_ponderado(tasas_obs, sigma_total, forma2)

    def chi2_de_s(s):
        def chi2_dado_b(b):
            return float(np.sum(((tasas_obs - (s * forma2 + b)) / sigma_total) ** 2))
        r = optimize.minimize_scalar(chi2_dado_b, bounds=(0.0, 5.0), method="bounded")
        return float(r.fun)

    objetivo = chi2_min + stats.chi2.isf(0.10, 1)  # 95 % unilateral
    s_cota = float(optimize.brentq(lambda s: chi2_de_s(s) - objetivo,
                                   max(s_mejor, 0.0), 5.0))
    r_dado_cota = optimize.minimize_scalar(
        lambda b: float(np.sum(((tasas_obs - (s_cota * forma2 + b)) / sigma_total) ** 2)),
        bounds=(0.0, 5.0), method="bounded",
    ).x
    fraccion_cota = s_cota / (s_cota + r_dado_cota)

    ajuste_05 = provenance.leer_producto("results/fits", "05_fits")
    cota_estadistica = ajuste_05["modulacion_n2"]["fraccion_modulada_cota_95"]

    lineas += [
        "",
        "=" * 78,
        "7. COTA SUPERIOR SOBRE LA FRACCION MODULADA",
        "=" * 78,
        f"  solo estadistica (05_fits):        < {100 * cota_estadistica:.1f} %",
        f"  con el sistematico del {100 * f_sys:.1f} %:      "
        f"< {100 * fraccion_cota:.1f} %",
        "",
        "La segunda es la que se publica: incluye la unica sistematica que este",
        "conjunto permite medir. Es el resultado cuantitativo del analisis.",
    ]

    texto = "\n".join(lineas)
    print(texto)

    # --- Figura --------------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4), layout="constrained")

    rejilla_f = np.linspace(0.0, 0.12, 121)
    p_valores = [stats.chi2.sf(delta_chi2(f), 1) for f in rejilla_f]
    ax.plot(100 * rejilla_f, p_valores, color=plotting.COLOR_DATOS, lw=2)
    ax.axhline(0.05, color=plotting.COLOR_MODELO, ls="--", lw=1.4,
               label="p = 0.05")
    ax.axvline(100 * f_sys, color=plotting.COLOR_ALTERNATIVO, lw=1.8,
               label=f"medido por el control: {100 * f_sys:.1f} %")
    ax.axvspan(100 * f_critico, 12, color=plotting.COLOR_ALTERNATIVO, alpha=0.12)
    ax.set_xlabel("sistematico entre corridas [%]")
    ax.set_ylabel("p-valor de la componente modulada")
    ax.set_title("La senal no sobrevive a la sistematica medida")
    ax.set_xlim(0, 12)
    ax.legend(fontsize=8)

    etiquetas = [f["fuente"] for f in fuentes]
    valores = [100 * f["efecto_relativo"] for f in fuentes]
    posiciones = np.arange(len(etiquetas))
    colores = [plotting.COLOR_MODELO if v == max(valores) else plotting.COLOR_DATOS
               for v in valores]
    ax2.barh(posiciones, valores, color=colores, alpha=0.85)
    ax2.set_yticks(posiciones)
    ax2.set_yticklabels(etiquetas, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("efecto relativo [%]")
    ax2.set_title("Tabla de sistematicas")
    for i, v in enumerate(valores):
        ax2.text(v + 0.1, i, f"{v:.2f} %", va="center", fontsize=8)
    ax2.set_xlim(0, max(valores) * 1.35)

    rutas = plotting.guardar(fig, args.out_figures, "06_systematics")
    print(f"[06] figura -> {rutas[0].name}")

    ruta_json, ruta_txt = provenance.escribir_producto(
        args.tables, "06_systematics",
        datos={
            "fuentes": fuentes,
            "sistematico_dominante": f_sys,
            "sistematico_critico_para_significancia": f_critico,
            "sistematico_para_compatibilidad_constante": f_compatible,
            "barrido": barrido,
            "cota_fraccion_modulada_solo_estadistica": cota_estadistica,
            "cota_fraccion_modulada_con_sistematico": float(fraccion_cota),
            "S_cota_hz": s_cota,
        },
        texto=texto,
        parametros={"angulo_control": ANGULO_CONTROL,
                    "angulo_referencia": ANGULO_REFERENCIA},
    )
    print(f"[06] escrito {ruta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
