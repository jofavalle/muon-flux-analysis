"""Shared library of the muon-angle-analysis pipeline.

Everything used by two or more numbered scripts lives here. In particular the
binning logic, which in the original script was written three times almost
verbatim and forced any fix to be applied three times over.
"""

from . import (  # noqa: F401
    acquisition, altitude, binning, geometry, health, io, models, plotting,
    provenance, rates,
)

__all__ = ["acquisition", "altitude", "binning", "geometry", "health", "io",
           "models", "plotting", "provenance", "rates"]
