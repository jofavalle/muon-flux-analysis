"""Libreria compartida del pipeline de muon-angle-analysis.

Todo lo que usan dos o mas scripts numerados vive aqui. En particular la
logica de binning, que en el script original estaba escrita tres veces casi
literal y obligaba a aplicar cualquier correccion tres veces.
"""

from . import binning, io, models, plotting, provenance, rates  # noqa: F401

__all__ = ["binning", "io", "models", "plotting", "provenance", "rates"]
