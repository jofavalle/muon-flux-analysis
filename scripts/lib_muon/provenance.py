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


def ultima_version(directorio: str | Path, base: str, extension: str = "json") -> int | None:
    """Mayor sufijo _vN existente para `base`, o None si no hay ninguno."""
    siguiente = siguiente_version(directorio, base, extension)
    return siguiente - 1 if siguiente > 1 else None


def leer_producto(directorio: str | Path, base: str) -> dict[str, Any]:
    """Lee el JSON mas reciente de `base`, que es como se encadena el pipeline.

    Lanza FileNotFoundError con el paso que falta, de modo que ejecutar un
    script fuera de orden diga que hay que correr antes en vez de fallar con
    un error de mas abajo.
    """
    directorio = Path(directorio)
    version = ultima_version(directorio, base)
    if version is None:
        raise FileNotFoundError(
            f"no hay ningun {base}_vN.json en {directorio}. "
            f"Ejecuta antes el paso {base.split('_')[0]} del pipeline."
        )
    ruta = directorio / f"{base}_v{version}.json"
    return json.loads(ruta.read_text(encoding="utf-8"))


def _limpiar_no_finitos(valor: Any) -> Any:
    """JSON no tiene infinito ni NaN. Se escriben como null, que es lo que
    significan aqui: un ajuste sin valor finito, no un cero.

    Recorre el arbol entero antes de serializar. No basta con el `default=` de
    json.dumps, que no se invoca para floats: json escribiria `Infinity` y
    `NaN`, que no son JSON valido y que ningun otro lector aceptaria.
    """
    if isinstance(valor, (np.floating, float)):
        v = float(valor)
        return v if np.isfinite(v) else None
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.bool_):
        return bool(valor)
    if isinstance(valor, np.ndarray):
        return _limpiar_no_finitos(valor.tolist())
    if isinstance(valor, dict):
        return {k: _limpiar_no_finitos(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_limpiar_no_finitos(x) for x in valor]
    return valor


def _json_serializable(objeto: Any) -> Any:
    """Solo se invoca para tipos que json no sabe serializar por si mismo."""
    if isinstance(objeto, np.integer):
        return int(objeto)
    if isinstance(objeto, np.floating):
        return _limpiar_no_finitos(float(objeto))
    if isinstance(objeto, np.ndarray):
        return _limpiar_no_finitos(objeto.tolist())
    if isinstance(objeto, np.bool_):
        return bool(objeto)
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

    contenido = _limpiar_no_finitos({"procedencia": sello(parametros, semilla),
                                     **datos})
    ruta_json.write_text(
        json.dumps(contenido, indent=2, ensure_ascii=False, allow_nan=False,
                   default=_json_serializable),
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
