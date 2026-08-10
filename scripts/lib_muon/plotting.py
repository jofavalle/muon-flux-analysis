"""Figure style and saving.

Two decisions that avoid failures in CI and in review:

- **Agg backend fixed before importing pyplot.** Without it matplotlib tries to
  open an interactive backend and CI fails for want of a DISPLAY.
- **Explicit API `fig, ax = plt.subplots()`**, not the pyplot state machine the
  original script used. With the state machine, two figures open at once step
  on each other and nothing raises: you simply get a wrong figure.

Every figure is saved as PDF and PNG with the same _vN suffix as the tables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from .provenance import next_version  # noqa: E402

STYLE = {
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

COLOUR_DATA = "#1b3a5c"
COLOUR_MODEL = "#c1440e"
COLOUR_ALTERNATE = "#2a7f62"
COLOUR_BAND = "#7fa8c9"


def apply_style() -> None:
    plt.rcParams.update(STYLE)


def save(fig, directory: str | Path, base: str) -> list[Path]:
    """Save the figure as PDF and PNG with the next free _vN, then close it.

    The version is decided from the PDF and reused for the PNG, so the pair
    always carries the same number.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    version = max(
        next_version(directory, base, "pdf"),
        next_version(directory, base, "png"),
    )
    paths = []
    for extension in ("pdf", "png"):
        path = directory / f"{base}_v{version}.{extension}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths
