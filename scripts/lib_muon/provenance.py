"""Trazabilidad de los productos: versionado _vN y sello de procedencia.

Dos reglas del repositorio se implementan aqui:

- **Ningun script sobrescribe resultados.** Cada corrida escribe con sufijo
  `_vN`, con N el siguiente libre. Un numero que ya se cito en el informe
  tiene que seguir existiendo.
- **Cada JSON lleva el hash del commit, los parametros y la semilla**, de modo
  que cualquier cifra del informe es trazable hasta la corrida que la produjo.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SEMILLA_POR_DEFECTO = 20260810


def hash_commit() -> dict[str, Any]:
    """Hash del commit actual y si el arbol tiene cambios sin comitear.

    Un resultado producido con el arbol sucio no es reproducible desde el
    hash, y eso hay que registrarlo, no esconderlo.
    """
    def git(*argumentos: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *argumentos], capture_output=True, text=True, check=True,
                timeout=10,
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    commit = git("rev-parse", "HEAD")
    estado = git("status", "--porcelain")
    return {
        "commit": commit,
        "rama": git("rev-parse", "--abbrev-ref", "HEAD"),
        "arbol_limpio": estado == "" if estado is not None else None,
    }


def sello(parametros: dict[str, Any] | None = None,
          semilla: int = SEMILLA_POR_DEFECTO) -> dict[str, Any]:
    """Bloque de procedencia que se incrusta en todo JSON de results/."""
    return {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": hash_commit(),
        "semilla": semilla,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "plataforma": platform.platform(),
        "parametros": parametros or {},
    }


def siguiente_version(directorio: str | Path, base: str, extension: str = "json") -> int:
    """Primer sufijo _vN libre para `base` en `directorio`."""
    directorio = Path(directorio)
    if not directorio.is_dir():
        return 1
    patron = re.compile(rf"^{re.escape(base)}_v(\d+)\.{re.escape(extension)}$")
    usados = [
        int(m.group(1))
        for ruta in directorio.iterdir()
        if (m := patron.match(ruta.name))
    ]
    return max(usados) + 1 if usados else 1


def _json_serializable(objeto: Any) -> Any:
    if isinstance(objeto, (np.integer,)):
        return int(objeto)
    if isinstance(objeto, (np.floating,)):
        valor = float(objeto)
        return valor if np.isfinite(valor) else None
    if isinstance(objeto, np.ndarray):
        return [_json_serializable(x) for x in objeto.tolist()]
    if isinstance(objeto, float) and not np.isfinite(objeto):
        return None
    raise TypeError(f"no serializable a JSON: {type(objeto)}")


def escribir_producto(
    directorio: str | Path,
    base: str,
    datos: dict[str, Any],
    texto: str,
    parametros: dict[str, Any] | None = None,
    semilla: int = SEMILLA_POR_DEFECTO,
) -> tuple[Path, Path]:
    """Escribe el par .json y .txt con el siguiente _vN libre.

    Productos pareados a proposito: el JSON para volver a leerlo desde otro
    script, el TXT para leerlo con los ojos sin herramientas.
    """
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    version = siguiente_version(directorio, base)

    ruta_json = directorio / f"{base}_v{version}.json"
    ruta_txt = directorio / f"{base}_v{version}.txt"

    contenido = {"procedencia": sello(parametros, semilla), **datos}
    ruta_json.write_text(
        json.dumps(contenido, indent=2, ensure_ascii=False, default=_json_serializable),
        encoding="utf-8",
    )

    sello_texto = contenido["procedencia"]
    cabecera = (
        f"{base}_v{version}\n"
        f"generado: {sello_texto['generado_utc']}\n"
        f"commit:   {sello_texto['git']['commit']} "
        f"({'limpio' if sello_texto['git']['arbol_limpio'] else 'ARBOL SUCIO'})\n"
        f"{'=' * 78}\n"
    )
    ruta_txt.write_text(cabecera + texto.rstrip() + "\n", encoding="utf-8")
    return ruta_json, ruta_txt
