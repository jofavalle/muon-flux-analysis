"""Modelos angulares e inferencia por maxima verosimilitud de Poisson.

Por que verosimilitud sobre cuentas crudas y no chi2 sobre cocientes, que es
lo que hacia el script original:

- Los siete cocientes I(theta)/I(0) comparten el denominador. NO son
  independientes, y los chi2 diagonales del script original ignoraban esa
  correlacion por completo. Sobre las cuentas crudas el problema no existe.
- Con 445 a 766 cuentas por punto la aproximacion gaussiana es buena, pero la
  verosimilitud de Poisson es exacta y no cuesta mas.
- El punto de 0 grados es degenerado tras normalizar: su residuo es cero por
  construccion, y el script original lo contaba igualmente en los grados de
  libertad, subestimando el chi2 reducido.

**El anclaje de cos(90).** `np.cos(np.deg2rad(90))` vale 6.123e-17, no cero.
Elevado a un exponente pequeno da valores de orden 1: 6.12e-17**0.01 = 0.688.
Por eso la busqueda en rejilla del script original encontraba su minimo pegado
al borde inferior y publicaba n = 0.006. Aqui se trabaja en espacio de
cos(theta) y todo |cos| < EPS_COSENO se ancla a cero exacto.

**Consecuencia del anclaje, que no es un detalle.** Con cos(90) = 0 exacto,
cualquier cos^n con n > 0 predice exactamente cero cuentas a 90 grados. Como
alli se observaron 447, la verosimilitud del cos^n puro es rigurosamente cero
para todo n > 0: el modelo no es que ajuste mal, es imposible. El unico motivo
por el que el script original podia ajustarlo era el 6.123e-17. Para acotar
cuanta modulacion admiten los datos hay que usar la familia senal mas fondo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln

# Por debajo de esto, un coseno es cero. 1e-12 esta muy por encima del
# 6.123e-17 que produce cos(deg2rad(90)) y muy por debajo de cualquier coseno
# de un angulo fisicamente distinguible de 90 grados: cos(89.9 grados) = 1.7e-3.
EPS_COSENO = 1e-12

# Tolerancias de Nelder-Mead. `fatol` es absoluto sobre -logL, que aqui vale
# entre 20 y 60: pedir 1e-12 seria exigir precision relativa de 1e-14, al borde
# del epsilon de doble precision, y el optimizador terminaria por agotar
# iteraciones con el parametro ya bien recuperado. 1e-9 sobra para que
# Delta(-2lnL) = 1 sea inequivoco.
OPCIONES_NELDER_MEAD = {
    "xatol": 1e-9,
    "fatol": 1e-9,
    "maxiter": 20000,
    "maxfev": 20000,
}


def coseno_anclado(angulos_deg: np.ndarray, eps: float = EPS_COSENO) -> np.ndarray:
    """cos(theta) con los ceros anclados a cero exacto.

    Este es el arreglo del defecto que motivo la reconstruccion del analisis.
    """
    cos_t = np.cos(np.deg2rad(np.asarray(angulos_deg, dtype=float)))
    return np.where(np.abs(cos_t) < eps, 0.0, cos_t)


def forma_cosn(cos_t: np.ndarray, n: float) -> np.ndarray:
    """cos^n(theta), definido tambien en cos = 0 y para n = 0.

    Con la convencion 0**0 = 1 de numpy, n = 0 da la forma constante.
    """
    cos_t = np.asarray(cos_t, dtype=float)
    positivos = cos_t > 0
    salida = np.zeros_like(cos_t)
    salida[positivos] = cos_t[positivos] ** n
    if n == 0:
        salida[~positivos] = 1.0
    return salida


def nll_poisson(mu: np.ndarray, cuentas: np.ndarray) -> float:
    """-log L de Poisson, con el termino log(k!) incluido.

    Devuelve infinito si algun mu es cero donde se observaron cuentas: ese
    modelo tiene probabilidad rigurosamente nula, y decirlo es el resultado.
    """
    mu = np.asarray(mu, dtype=float)
    cuentas = np.asarray(cuentas, dtype=float)
    if np.any(mu < 0):
        return np.inf
    if np.any((mu == 0) & (cuentas > 0)):
        return np.inf
    seguro = np.where(mu > 0, mu, 1.0)
    return float(np.sum(seguro - cuentas * np.log(seguro) + gammaln(cuentas + 1.0)))


@dataclass
class Ajuste:
    """Resultado de un ajuste, con lo necesario para compararlo con otro."""

    nombre: str
    parametros: dict[str, float]
    nll: float
    n_parametros: int
    n_puntos: int
    mu: np.ndarray = field(repr=False)
    convergio: bool = True

    @property
    def aic(self) -> float:
        return 2 * self.n_parametros + 2 * self.nll

    @property
    def bic(self) -> float:
        return self.n_parametros * np.log(self.n_puntos) + 2 * self.nll

    @property
    def gdl(self) -> int:
        return self.n_puntos - self.n_parametros

    def como_dict(self) -> dict:
        return {
            "nombre": self.nombre,
            "parametros": self.parametros,
            "nll": self.nll if np.isfinite(self.nll) else None,
            "nll_infinito": not np.isfinite(self.nll),
            "n_parametros": self.n_parametros,
            "n_puntos": self.n_puntos,
            "gdl": self.gdl,
            "aic": self.aic if np.isfinite(self.aic) else None,
            "bic": self.bic if np.isfinite(self.bic) else None,
            "convergio": self.convergio,
        }


def ajustar_constante(cuentas: np.ndarray, tiempos: np.ndarray) -> Ajuste:
    """Tasa unica: mu_i = R * T_i. Un parametro.

    El estimador de maxima verosimilitud tiene forma cerrada, R = sum(N)/sum(T).
    """
    cuentas = np.asarray(cuentas, dtype=float)
    tiempos = np.asarray(tiempos, dtype=float)
    r = float(cuentas.sum() / tiempos.sum())
    mu = r * tiempos
    return Ajuste(
        nombre="constante",
        parametros={"R_hz": r},
        nll=nll_poisson(mu, cuentas),
        n_parametros=1,
        n_puntos=len(cuentas),
        mu=mu,
    )


def ajustar_cosn_puro(
    cuentas: np.ndarray, tiempos: np.ndarray, cos_t: np.ndarray
) -> Ajuste:
    """mu_i = R * T_i * cos^n(theta_i). Dos parametros.

    Con cos(90) anclado a cero este modelo tiene verosimilitud nula en cuanto
    haya un punto con cos = 0 y cuentas observadas, que es exactamente el caso
    de este conjunto de datos. Se ajusta igualmente para dejar constancia
    reproducible de que el resultado es -logL = infinito, no un numero feo.
    """
    cuentas = np.asarray(cuentas, dtype=float)
    tiempos = np.asarray(tiempos, dtype=float)
    imposible = np.any((cos_t <= 0) & (cuentas > 0))

    def objetivo(p):
        log_r, n = p
        if not (0.0 <= n <= 10.0):
            return np.inf
        return nll_poisson(np.exp(log_r) * tiempos * forma_cosn(cos_t, n), cuentas)

    if imposible:
        # Cualquier n > 0 da -logL infinito. El optimo formal es n = 0, que es
        # el modelo constante disfrazado; se reporta como no convergido.
        r = float(cuentas.sum() / tiempos.sum())
        return Ajuste(
            nombre="cos^n puro",
            parametros={"R_hz": r, "n": 0.0},
            nll=np.inf,
            n_parametros=2,
            n_puntos=len(cuentas),
            mu=r * tiempos * forma_cosn(cos_t, 0.0),
            convergio=False,
        )

    r0 = float(cuentas.sum() / tiempos.sum())
    res = optimize.minimize(
        objetivo, x0=[np.log(r0), 1.0], method="Nelder-Mead",
        options=OPCIONES_NELDER_MEAD,
    )
    log_r, n = res.x
    mu = np.exp(log_r) * tiempos * forma_cosn(cos_t, n)
    return Ajuste(
        nombre="cos^n puro",
        parametros={"R_hz": float(np.exp(log_r)), "n": float(n)},
        nll=float(res.fun),
        n_parametros=2,
        n_puntos=len(cuentas),
        mu=mu,
        convergio=bool(res.success),
    )


def ajustar_cosn_fijo(
    cuentas: np.ndarray, tiempos: np.ndarray, cos_t: np.ndarray, n: float = 2.0
) -> Ajuste:
    """mu_i = R * T_i * cos^n con n fijo. Un parametro.

    Con n = 2 es la hipotesis fisica que el repositorio prometia.
    """
    cuentas = np.asarray(cuentas, dtype=float)
    tiempos = np.asarray(tiempos, dtype=float)
    forma = forma_cosn(cos_t, n)
    denominador = float(np.sum(tiempos * forma))
    r = float(cuentas.sum() / denominador) if denominador > 0 else np.nan
    mu = r * tiempos * forma
    return Ajuste(
        nombre=f"cos^{n:g} fijo",
        parametros={"R_hz": r, "n": float(n)},
        nll=nll_poisson(mu, cuentas),
        n_parametros=1,
        n_puntos=len(cuentas),
        mu=mu,
        convergio=bool(np.isfinite(nll_poisson(mu, cuentas))),
    )


def ajustar_senal_fondo(
    cuentas: np.ndarray,
    tiempos: np.ndarray,
    cos_t: np.ndarray,
    n: float | None = None,
) -> Ajuste:
    """mu_i = T_i * (S * cos^n(theta_i) + B).

    La unica familia que admite a la vez una componente modulada y el hecho de
    que a 90 grados se observen cuentas. B recoge en un solo numero lo que las
    tres hipotesis del analisis predicen a 90 grados: aceptancia geometrica
    ancha, componente accidental y convencion angular no documentada. Los datos
    no permiten separarlas, y el informe no pretende que si.

    Con `n` dado, se ajustan S y B (dos parametros). Con `n = None`, tambien n.
    """
    cuentas = np.asarray(cuentas, dtype=float)
    tiempos = np.asarray(tiempos, dtype=float)
    n_libre = n is None
    tasa_tipica = float(cuentas.sum() / tiempos.sum())

    def mu_de(s, exponente, b):
        return tiempos * (s * forma_cosn(cos_t, exponente) + b)

    def objetivo(p):
        if n_libre:
            s, exponente, b = p[0], p[1], p[2]
        else:
            s, b = p[0], p[1]
            exponente = n
        if s < 0 or b <= 0 or exponente < 0 or exponente > 10:
            return np.inf
        return nll_poisson(mu_de(s, exponente, b), cuentas)

    mejor = None
    semillas_s = (0.05, 0.2, 0.5)
    semillas_n = (0.5, 1.0, 2.0) if n_libre else (n,)
    for s0 in semillas_s:
        for n0 in semillas_n:
            x0 = [s0 * tasa_tipica, n0, (1 - s0) * tasa_tipica] if n_libre else [
                s0 * tasa_tipica, (1 - s0) * tasa_tipica
            ]
            res = optimize.minimize(
                objetivo, x0=x0, method="Nelder-Mead",
                options=OPCIONES_NELDER_MEAD,
            )
            if mejor is None or res.fun < mejor.fun:
                mejor = res

    if n_libre:
        s, exponente, b = mejor.x
    else:
        s, b = mejor.x
        exponente = float(n)

    parametros = {"S_hz": float(s), "B_hz": float(b), "n": float(exponente)}
    total = s + b
    parametros["fraccion_modulada"] = float(s / total) if total > 0 else np.nan
    return Ajuste(
        nombre="senal+fondo" + ("" if n_libre else f" (n={n:g} fijo)"),
        parametros=parametros,
        nll=float(mejor.fun),
        n_parametros=3 if n_libre else 2,
        n_puntos=len(cuentas),
        mu=mu_de(s, exponente, b),
        convergio=bool(mejor.success),
    )


def perfil_verosimilitud(
    nll_de_parametro,
    valor_optimo: float,
    nll_optimo: float,
    delta: float = 0.5,
    limite_inferior: float = 0.0,
    limite_superior: float = 10.0,
) -> tuple[float, float]:
    """Intervalo por perfil de verosimilitud alrededor del optimo.

    `delta` es el incremento de -logL que define el intervalo: 0.5 para el
    68 % (Delta(-2lnL) = 1), 1.353 para una cota unilateral al 95 %.
    Devuelve (inferior, superior); un extremo es NaN si no hay cruce dentro de
    los limites, lo que significa que el dato no acota ese lado.
    """
    objetivo = nll_optimo + delta

    def desviacion(x):
        return nll_de_parametro(x) - objetivo

    inferior = np.nan
    if valor_optimo > limite_inferior and np.isfinite(desviacion(limite_inferior)):
        if desviacion(limite_inferior) > 0:
            inferior = float(optimize.brentq(desviacion, limite_inferior, valor_optimo))
        else:
            inferior = limite_inferior

    superior = np.nan
    if np.isfinite(desviacion(limite_superior)) and desviacion(limite_superior) > 0:
        superior = float(optimize.brentq(desviacion, valor_optimo, limite_superior))
    else:
        # Busca el cruce ampliando desde el optimo; si no lo hay, no acota.
        alto = valor_optimo
        for _ in range(60):
            alto = alto * 2 if alto > 0 else 1e-3
            if alto > limite_superior:
                break
            d = desviacion(alto)
            if np.isfinite(d) and d > 0:
                superior = float(optimize.brentq(desviacion, valor_optimo, alto))
                break
    return inferior, superior


def deviance(cuentas: np.ndarray, mu: np.ndarray) -> float:
    """Deviance de Poisson: 2*(logL_saturado - logL_modelo).

    Se distribuye aproximadamente como chi2 con gdl grados de libertad y es el
    estadistico de bondad de ajuste correcto para cuentas.
    """
    cuentas = np.asarray(cuentas, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if np.any((mu <= 0) & (cuentas > 0)):
        return np.inf
    termino = np.where(cuentas > 0, cuentas * np.log(cuentas / mu), 0.0)
    return float(2 * np.sum(termino - (cuentas - mu)))


def residuos_pearson(cuentas: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """(N - mu)/sqrt(mu), en unidades de sigma."""
    cuentas = np.asarray(cuentas, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return np.where(mu > 0, (cuentas - mu) / np.sqrt(np.where(mu > 0, mu, 1.0)), np.nan)


def p_bondad_ajuste(cuentas: np.ndarray, mu: np.ndarray, gdl: int) -> tuple[float, float]:
    """Deviance y su p-valor bajo chi2 con `gdl` grados de libertad."""
    d = deviance(cuentas, mu)
    if not np.isfinite(d) or gdl <= 0:
        return d, 0.0
    return d, float(stats.chi2.sf(d, gdl))


def razon_verosimilitudes(
    nll_nulo: float, nll_alternativo: float, gdl_extra: int
) -> tuple[float, float]:
    """Test de Wilks entre modelos anidados.

    Devuelve (Delta(-2lnL), p). El nulo es el modelo con menos parametros.
    Advertencia estadistica: si el parametro extra esta en el borde del
    espacio (S >= 0 con S_verdadero = 0), la distribucion asintotica no es
    chi2 sino una mezcla, y el p-valor de chi2 es conservador.
    """
    if not np.isfinite(nll_nulo) or not np.isfinite(nll_alternativo):
        return np.inf, 0.0
    d = 2 * (nll_nulo - nll_alternativo)
    if d <= 0 or gdl_extra <= 0:
        return float(d), 1.0
    return float(d), float(stats.chi2.sf(d, gdl_extra))


def chi2_con_sistematico(
    tasas: np.ndarray, errores: np.ndarray, fraccion_sistematica: float
) -> tuple[float, float, float]:
    """chi2 de una tasa comun anadiendo un sistematico relativo en cuadratura.

    Sirve para responder cuanto sistematico entre corridas hace falta para que
    los siete puntos sean compatibles con una tasa unica. Devuelve
    (tasa_ponderada, chi2, p).
    """
    tasas = np.asarray(tasas, dtype=float)
    errores = np.asarray(errores, dtype=float)
    sigma = np.sqrt(errores**2 + (fraccion_sistematica * tasas) ** 2)
    pesos = 1.0 / sigma**2
    media = float(np.sum(pesos * tasas) / np.sum(pesos))
    chi2 = float(np.sum(((tasas - media) / sigma) ** 2))
    gdl = len(tasas) - 1
    return media, chi2, float(stats.chi2.sf(chi2, gdl))


def sistematico_necesario(
    tasas: np.ndarray, errores: np.ndarray, p_objetivo: float = 0.05
) -> float:
    """Fraccion sistematica minima para que la hipotesis de tasa unica no se
    rechace al nivel `p_objetivo`. Devuelve 0 si ya es compatible sin ella."""
    _, chi2_sin, p_sin = chi2_con_sistematico(tasas, errores, 0.0)
    if p_sin >= p_objetivo:
        return 0.0

    def desviacion(f):
        return chi2_con_sistematico(tasas, errores, f)[2] - p_objetivo

    return float(optimize.brentq(desviacion, 1e-9, 1.0))
