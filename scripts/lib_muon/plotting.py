"""Estilo y guardado de figuras.

Dos decisiones que evitan fallos en CI y en revision:

- **Backend Agg fijado antes de importar pyplot.** Sin el, matplotlib intenta
  abrir un backend interactivo y el CI falla por no tener DISPLAY.
- **API explicita `fig, ax = plt.subplots()`**, no la maquina de estados de
  pyplot que usaba el script original. Con la maquina de estados, dos figuras
  abiertas a la vez se pisan y el error no salta: sale una figura mal.

Cada figura se guarda en PDF y PNG con el mismo sufijo _vN, igual que las
tablas.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from .provenance import siguiente_version  # noqa: E402

ESTILO = {
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "legend.framealpha": 0.9,
    "figure.autolayout": False,
}

COLOR_DATOS = "#1b3a5c"
COLOR_MODELO = "#c1440e"
COLOR_ALTERNATIVO = "#2a7f62"
COLOR_BANDA = "#7fa8c9"


def aplicar_estilo() -> None:
    plt.rcParams.update(ESTILO)


def guardar(fig, directorio: str | Path, base: str) -> list[Path]:
    """Guarda la figura en PDF y PNG con el siguiente _vN libre y la cierra.

    La version se decide con el PDF y se reutiliza para el PNG, de modo que el
    par siempre lleva el mismo numero.
    """
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    version = max(
        siguiente_version(directorio, base, "pdf"),
        siguiente_version(directorio, base, "png"),
    )
    rutas = []
    for extension in ("pdf", "png"):
        ruta = directorio / f"{base}_v{version}.{extension}"
        fig.savefig(ruta)
        rutas.append(ruta)
    plt.close(fig)
    return rutas
