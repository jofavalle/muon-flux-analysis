"""Tasas de conteo corregidas por tiempo vivo.

Tres correcciones sobre el script original, todas ellas cambian el numero
publicado:

1. **Tiempo vivo.** La columna `DeadTime[us]` se parseaba y no se usaba nunca.
   La fraccion muerta es del 9.09 % al 9.28 % en los ocho archivos, asi que la
   tasa publicada era un 10 % baja de forma sistematica.
2. **Estimador insesgado.** Para N eventos observados en la ventana [t1, tN]
   hay N-1 intervalos entre eventos, no N. El estimador insesgado de la tasa es
   (N-1)/T, no N/T. El sesgo es 1/N, del 0.15 % al 0.22 % aqui: pequeno frente
   a la estadistica, pero gratis de corregir.
3. **Sin dependencia del binning.** La ventana es [t1, tN] completa, no
   truncada a un multiplo del tamano de bin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import Corrida


@dataclass
class Tasa:
    """Tasa de una corrida, con lo necesario para rehacer el calculo."""

    angulo_deg: int
    n_eventos: int
    n_intervalos: int
    ventana_s: float
    tiempo_muerto_s: float
    tiempo_vivo_s: float
    fraccion_muerta: float
    tasa_pared: float
    tasa_viva: float
    error_tasa_viva: float

    def como_dict(self) -> dict:
        return {
            "angulo_deg": self.angulo_deg,
            "n_eventos": self.n_eventos,
            "n_intervalos": self.n_intervalos,
            "ventana_s": self.ventana_s,
            "tiempo_muerto_s": self.tiempo_muerto_s,
            "tiempo_vivo_s": self.tiempo_vivo_s,
            "fraccion_muerta": self.fraccion_muerta,
            "tasa_pared_hz": self.tasa_pared,
            "tasa_viva_hz": self.tasa_viva,
            "error_tasa_viva_hz": self.error_tasa_viva,
        }


def calcular_tasa(corrida: Corrida, solo_coincidentes: bool = True) -> Tasa:
    """Tasa de una corrida corregida por tiempo vivo.

    `solo_coincidentes` se conserva por fidelidad al analisis original, pero es
    un no-op sobre estos datos: `Coincident == 1` en los 4760 eventos de los
    ocho archivos. Al no haber singles registrados, la tasa de accidentales no
    es estimable desde este conjunto; ver docs/dataset.md.
    """
    tiempos = corrida.tiempo_relativo_s
    if solo_coincidentes:
        tiempos = tiempos[corrida.coincidente == 1]
    if len(tiempos) < 2:
        raise ValueError(
            f"la corrida de {corrida.angulo_deg} grados tiene {len(tiempos)} "
            "eventos: hacen falta al menos 2 para definir una tasa"
        )

    n = len(tiempos)
    ventana = float(tiempos[-1] - tiempos[0])
    muerto = corrida.tiempo_muerto_s
    vivo = ventana - muerto
    if vivo <= 0:
        raise ValueError(
            f"tiempo vivo no positivo en {corrida.angulo_deg} grados: "
            f"ventana {ventana:.2f} s, tiempo muerto {muerto:.2f} s"
        )

    n_intervalos = n - 1
    tasa_viva = n_intervalos / vivo
    # n_intervalos es Poisson de media R*T_vivo con T_vivo fijo, luego el error
    # relativo de la tasa es 1/sqrt(n_intervalos).
    error = tasa_viva / np.sqrt(n_intervalos)

    return Tasa(
        angulo_deg=corrida.angulo_deg,
        n_eventos=n,
        n_intervalos=n_intervalos,
        ventana_s=ventana,
        tiempo_muerto_s=muerto,
        tiempo_vivo_s=vivo,
        fraccion_muerta=muerto / ventana,
        tasa_pared=n_intervalos / ventana,
        tasa_viva=tasa_viva,
        error_tasa_viva=error,
    )


def normalizar(tasas: dict[int, Tasa], angulo_ref: int = 0) -> dict[int, tuple[float, float]]:
    """Tasas relativas a la del angulo de referencia, con su error propagado.

    Advertencia que el script original no hacia: los cocientes comparten el
    denominador, luego NO son independientes y un chi2 diagonal sobre ellos
    esta mal planteado. Estos valores sirven para dibujar y para comparar con
    el analisis anterior; la inferencia se hace en `models` sobre las cuentas
    crudas, donde el problema no existe.
    """
    if angulo_ref not in tasas:
        raise ValueError(f"falta el angulo de referencia {angulo_ref} en las tasas")
    ref = tasas[angulo_ref]
    r0, e0 = ref.tasa_viva, ref.error_tasa_viva
    salida = {}
    for angulo, t in tasas.items():
        cociente = t.tasa_viva / r0
        error = np.sqrt((t.error_tasa_viva / r0) ** 2 + (t.tasa_viva * e0 / r0**2) ** 2)
        salida[angulo] = (float(cociente), float(error))
    return salida
