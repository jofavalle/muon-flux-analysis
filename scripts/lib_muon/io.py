"""Lectura de los archivos crudos del detector CosmicWatch.

Dos trampas del formato, ninguna deducible leyendo el codigo original:

1. **La cabecera miente sobre el orden de las columnas.** Declara
   `SiPM[mV] Temp[C] Pressure[Pa]` y los datos traen la presion (~93600 Pa)
   ANTES que la temperatura (~28 C). El script original lo compensaba con una
   heuristica de magnitud por fila (umbrales 1000 y 200) sin documentarlo ni
   dejar rastro. Aqui la decision se toma una vez por archivo, contra rangos
   fisicos declarados, y queda registrada en el informe de calidad.

2. **Las lineas malformadas existen de verdad.** `data/raw/015deg` esta
   truncado por arriba: le faltan tres lineas de cabecera y su primera linea
   es el fragmento ` 118  | Hexa address:  0x76`. El script original hacia
   `except Exception: continue` y la descartaba en silencio. Aqui se cuenta,
   se guarda su texto y se reporta.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Cabecera que los archivos declaran. Se conserva literal para poder
# contrastarla y dejar constancia de que no describe los datos.
CABECERA_DECLARADA = (
    "Event TimeStamp[ms] ADC1 ADC2 SiPM[mV] Temp[C] Pressure[Pa] "
    "DeadTime[us] Coincident Name"
)

# Orden real de las columnas 6 y 7, verificado contra los ocho archivos.
CABECERA_REAL = (
    "Event TimeStamp[ms] ADC1 ADC2 SiPM[mV] Pressure[Pa] Temp[C] "
    "DeadTime[us] Coincident Name"
)

N_COLUMNAS = 10

# Rangos fisicos admisibles, usados para decidir el orden de las columnas 6 y 7
# y para marcar lecturas fuera de rango. Presion a nivel del mar 101325 Pa; el
# conjunto ronda 93600 Pa, coherente con altitud moderada.
RANGO_PRESION_PA = (50_000.0, 110_000.0)
RANGO_TEMPERATURA_C = (-20.0, 60.0)

# Lectura que el BMP280 emite antes de estabilizarse. Aparece en 8 filas
# repartidas por cinco de los ocho archivos, siempre al principio de la corrida.
ARRANQUE_PRESION_PA = 73157.02
ARRANQUE_TEMPERATURA_C = 23.13
TOLERANCIA_ARRANQUE = 0.01


@dataclass
class InformeCalidad:
    """Lo que salio mal al leer un archivo. Se reporta, nunca se descarta."""

    ruta: str
    lineas_totales: int = 0
    lineas_datos: int = 0
    lineas_rechazadas: int = 0
    textos_rechazados: list[str] = field(default_factory=list)
    cabecera_presente: bool = False
    columnas_invertidas: bool = False
    filas_arranque_sensor: int = 0
    indices_arranque_sensor: list[int] = field(default_factory=list)
    filas_fuera_de_rango: int = 0

    def como_dict(self) -> dict:
        return {
            "ruta": self.ruta,
            "lineas_totales": self.lineas_totales,
            "lineas_datos": self.lineas_datos,
            "lineas_rechazadas": self.lineas_rechazadas,
            "textos_rechazados": self.textos_rechazados,
            "cabecera_presente": self.cabecera_presente,
            "columnas_invertidas": self.columnas_invertidas,
            "filas_arranque_sensor": self.filas_arranque_sensor,
            "indices_arranque_sensor": self.indices_arranque_sensor,
            "filas_fuera_de_rango": self.filas_fuera_de_rango,
        }


@dataclass
class Corrida:
    """Una corrida: un archivo, un angulo nominal."""

    angulo_deg: int
    ruta: str
    evento: np.ndarray
    timestamp_ms: np.ndarray
    adc1: np.ndarray
    adc2: np.ndarray
    sipm_mv: np.ndarray
    presion_pa: np.ndarray
    temperatura_c: np.ndarray
    deadtime_us: np.ndarray
    coincidente: np.ndarray
    arranque_sensor: np.ndarray
    calidad: InformeCalidad

    def __len__(self) -> int:
        return len(self.timestamp_ms)

    @property
    def tiempo_relativo_s(self) -> np.ndarray:
        """Tiempo desde el primer evento, en segundos.

        El script original acumulaba diferencias positivas con `cumsum`, lo que
        equivale a esto salvo si el reloj retrocede. No retrocede en ninguno de
        los ocho archivos (se comprueba en 01_validation), y la resta directa
        es la definicion honesta: si el reloj retrocediera habria que verlo, no
        absorberlo en silencio.
        """
        return (self.timestamp_ms - self.timestamp_ms[0]) / 1000.0

    @property
    def ventana_s(self) -> float:
        """Duracion de la ventana observada [t1, tN], en segundos."""
        return float(self.tiempo_relativo_s[-1])

    @property
    def tiempo_muerto_s(self) -> float:
        """Tiempo muerto acumulado de la corrida, en segundos.

        La columna DeadTime[us] se parseaba en el script original y no se usaba
        nunca: la tasa publicada era de reloj de pared, no de tiempo vivo.
        """
        return float(self.deadtime_us.sum()) / 1e6

    @property
    def tiempo_vivo_s(self) -> float:
        return self.ventana_s - self.tiempo_muerto_s

    @property
    def fraccion_muerta(self) -> float:
        return self.tiempo_muerto_s / self.ventana_s


def parse_line(linea: str) -> tuple:
    """Parsea una linea de datos. Lanza ValueError si no lo es.

    Devuelve las diez columnas en su orden literal en el archivo, sin decidir
    todavia cual de las dos intermedias es presion y cual temperatura: esa
    decision se toma por archivo en `load_run`.
    """
    partes = re.split(r"\s+", linea.strip())
    if len(partes) != N_COLUMNAS:
        raise ValueError(
            f"se esperaban {N_COLUMNAS} columnas y hay {len(partes)}"
        )
    return (
        int(partes[0]),
        int(partes[1]),
        int(partes[2]),
        int(partes[3]),
        float(partes[4]),
        float(partes[5]),
        float(partes[6]),
        float(partes[7]),
        int(partes[8]),
        partes[9],
    )


def _decidir_orden_columnas(col6: np.ndarray, col7: np.ndarray) -> bool:
    """Decide si las columnas 6 y 7 vienen como (presion, temperatura).

    Devuelve True si estan invertidas respecto a lo que declara la cabecera,
    es decir si la columna 6 es la presion. La decision se toma una vez por
    archivo con la mediana, no fila a fila: una fila anomala no debe poder
    cambiar la interpretacion de las columnas.
    """
    m6, m7 = float(np.median(col6)), float(np.median(col7))
    seis_es_presion = (
        RANGO_PRESION_PA[0] <= m6 <= RANGO_PRESION_PA[1]
        and RANGO_TEMPERATURA_C[0] <= m7 <= RANGO_TEMPERATURA_C[1]
    )
    siete_es_presion = (
        RANGO_PRESION_PA[0] <= m7 <= RANGO_PRESION_PA[1]
        and RANGO_TEMPERATURA_C[0] <= m6 <= RANGO_TEMPERATURA_C[1]
    )
    if seis_es_presion and not siete_es_presion:
        return True
    if siete_es_presion and not seis_es_presion:
        return False
    raise ValueError(
        "no se puede decidir cual columna es la presion y cual la temperatura: "
        f"medianas col6 = {m6:.2f}, col7 = {m7:.2f}. Rangos admisibles "
        f"presion {RANGO_PRESION_PA} Pa, temperatura {RANGO_TEMPERATURA_C} C."
    )


def load_run(ruta: str | Path, angulo_deg: int | None = None) -> Corrida:
    """Lee un archivo crudo y devuelve una Corrida con su informe de calidad."""
    ruta = Path(ruta)
    if angulo_deg is None:
        emparejado = re.match(r"^(\d+)deg$", ruta.name)
        if not emparejado:
            raise ValueError(
                f"no se deduce el angulo del nombre {ruta.name!r}; "
                "se espera NNNdeg (por ejemplo 045deg)"
            )
        angulo_deg = int(emparejado.group(1))

    calidad = InformeCalidad(ruta=str(ruta))
    filas: list[tuple] = []

    with open(ruta, "r", errors="ignore") as f:
        for linea in f:
            calidad.lineas_totales += 1
            desnuda = linea.strip()
            if not desnuda:
                continue
            if desnuda.startswith("#"):
                if CABECERA_DECLARADA.split()[0] in desnuda and "TimeStamp" in desnuda:
                    calidad.cabecera_presente = True
                continue
            try:
                filas.append(parse_line(linea))
            except ValueError:
                calidad.lineas_rechazadas += 1
                if len(calidad.textos_rechazados) < 10:
                    calidad.textos_rechazados.append(desnuda)

    if not filas:
        raise ValueError(f"ninguna linea de datos legible en {ruta}")
    calidad.lineas_datos = len(filas)

    evento = np.array([f[0] for f in filas], dtype=np.int64)
    timestamp_ms = np.array([f[1] for f in filas], dtype=np.float64)
    adc1 = np.array([f[2] for f in filas], dtype=np.int64)
    adc2 = np.array([f[3] for f in filas], dtype=np.int64)
    sipm_mv = np.array([f[4] for f in filas], dtype=np.float64)
    col6 = np.array([f[5] for f in filas], dtype=np.float64)
    col7 = np.array([f[6] for f in filas], dtype=np.float64)
    deadtime_us = np.array([f[7] for f in filas], dtype=np.float64)
    coincidente = np.array([f[8] for f in filas], dtype=np.int64)

    calidad.columnas_invertidas = _decidir_orden_columnas(col6, col7)
    if calidad.columnas_invertidas:
        presion_pa, temperatura_c = col6, col7
    else:
        presion_pa, temperatura_c = col7, col6

    arranque = (
        np.abs(presion_pa - ARRANQUE_PRESION_PA) < TOLERANCIA_ARRANQUE
    ) & (np.abs(temperatura_c - ARRANQUE_TEMPERATURA_C) < TOLERANCIA_ARRANQUE)
    calidad.filas_arranque_sensor = int(arranque.sum())
    calidad.indices_arranque_sensor = [int(i) for i in np.flatnonzero(arranque)]

    fuera = (
        (presion_pa < RANGO_PRESION_PA[0])
        | (presion_pa > RANGO_PRESION_PA[1])
        | (temperatura_c < RANGO_TEMPERATURA_C[0])
        | (temperatura_c > RANGO_TEMPERATURA_C[1])
    )
    calidad.filas_fuera_de_rango = int(fuera.sum())

    return Corrida(
        angulo_deg=angulo_deg,
        ruta=str(ruta),
        evento=evento,
        timestamp_ms=timestamp_ms,
        adc1=adc1,
        adc2=adc2,
        sipm_mv=sipm_mv,
        presion_pa=presion_pa,
        temperatura_c=temperatura_c,
        deadtime_us=deadtime_us,
        coincidente=coincidente,
        arranque_sensor=arranque,
        calidad=calidad,
    )


def load_all(directorio: str | Path = "data/raw") -> dict[int, Corrida]:
    """Lee las ocho corridas de un directorio, indexadas por angulo nominal."""
    directorio = Path(directorio)
    corridas = {}
    for ruta in sorted(directorio.iterdir()):
        if re.match(r"^\d+deg$", ruta.name):
            corrida = load_run(ruta)
            corridas[corrida.angulo_deg] = corrida
    if not corridas:
        raise ValueError(f"no hay archivos NNNdeg en {directorio}")
    return corridas
