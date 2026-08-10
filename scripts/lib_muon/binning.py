"""Binning temporal, escrito una sola vez.

En el script original esta logica aparecia tres veces casi literal
(`plot_per_second`, `plot_time_vs_rate_all`, `compute_mean_rate`), de modo que
cualquier correccion habia que aplicarla tres veces. Peor: la tercera copia
truncaba la duracion de la corrida a un multiplo entero del tamano de bin, con
lo que un parametro puramente cosmetico movia la tasa media publicada.
Comprobado sobre los datos reales: la tasa a 0 grados pasa de 1.8118 1/s con
bin de 1 s a 1.8194 1/s con bin de 60 s, y el cociente I(90)/I(0) se mueve de
0.8079 a 0.8148. Aqui el binning solo sirve para dibujar series temporales y no
interviene en ningun numero fisico.
"""

from __future__ import annotations

import numpy as np


def bins_completos(
    tiempos_s: np.ndarray, bin_s: float, inicio_s: float = 0.0,
    fin_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cuenta eventos en bins de ancho fijo, descartando el bin parcial final.

    Devuelve (centros, cuentas, ancho) con ancho constante e igual a `bin_s`.
    El bin parcial final se descarta porque su tasa aparente esta inflada: un
    bin de 0.2 s con un evento da 5 1/s. Esto solo afecta a las figuras de
    serie temporal; las tasas fisicas se calculan en `rates` sobre la ventana
    completa, sin binning.
    """
    if bin_s <= 0:
        raise ValueError(f"bin_s debe ser positivo, es {bin_s}")
    if fin_s is None:
        fin_s = float(tiempos_s[-1]) if len(tiempos_s) else inicio_s

    dentro = tiempos_s[(tiempos_s >= inicio_s) & (tiempos_s <= fin_s)]
    if len(dentro) == 0:
        vacio = np.array([], dtype=float)
        return vacio, vacio, vacio

    n_bins = int(np.floor((fin_s - inicio_s) / bin_s))
    if n_bins < 1:
        vacio = np.array([], dtype=float)
        return vacio, vacio, vacio

    bordes = inicio_s + np.arange(n_bins + 1) * bin_s
    cuentas, _ = np.histogram(dentro, bins=bordes)
    centros = bordes[:-1] + bin_s / 2.0
    anchos = np.full(n_bins, bin_s, dtype=float)
    return centros, cuentas.astype(float), anchos


def tasas_por_bin(
    tiempos_s: np.ndarray, bin_s: float, inicio_s: float = 0.0,
    fin_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Serie temporal de tasa instantanea con su error de Poisson.

    Devuelve (centros, tasa, error). El error de un bin con cero cuentas se
    toma como 1/ancho, cota superior de 68 % para una Poisson que observa cero:
    devolver cero seria afirmar certeza absoluta donde no hay dato.
    """
    centros, cuentas, anchos = bins_completos(tiempos_s, bin_s, inicio_s, fin_s)
    if len(centros) == 0:
        vacio = np.array([], dtype=float)
        return vacio, vacio, vacio
    tasas = cuentas / anchos
    errores = np.where(cuentas > 0, np.sqrt(cuentas), 1.0) / anchos
    return centros, tasas, errores


def media_movil(valores: np.ndarray, ventana_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Media movil centrada y desviacion tipica local.

    Sustituye al `rolling` de pandas del script original. Los bordes usan la
    ventana truncada disponible, igual que `min_periods=1`.
    """
    ventana_bins = max(int(ventana_bins), 1)
    n = len(valores)
    medias = np.empty(n, dtype=float)
    desviaciones = np.empty(n, dtype=float)
    medio = ventana_bins // 2
    for i in range(n):
        lo = max(0, i - medio)
        hi = min(n, i + ventana_bins - medio)
        trozo = valores[lo:hi]
        medias[i] = float(np.mean(trozo))
        desviaciones[i] = float(np.std(trozo))
    return medias, desviaciones
