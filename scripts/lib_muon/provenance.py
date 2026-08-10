"""Traceability of the products: _vN versioning and provenance stamp.

Two repository rules are implemented here:

- **No script overwrites results.** Every run writes with a `_vN` suffix, N
  being the next free index. A number already cited in the report must keep
  existing.
- **Every JSON carries the commit hash, the parameters and the seed**, so any
  figure in the report is traceable back to the run that produced it.
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

DEFAULT_SEED = 20260810

# Repository root, used to record paths portably.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def portable_path(path: str | Path) -> str:
    """A path as recorded in the products: relative to the repository root.

    Two reasons, and both matter for a public repository:

    - **Reproducibility.** An absolute path makes a product depend on where the
      repository happens to sit. Two correct runs of the same pipeline on the
      same data would disagree merely because one ran in /home/alice and the
      other in /home/bob, and `check_reproducibility.py` would report a
      difference that is not one.
    - **Privacy.** These products are committed and published. An absolute path
      publishes the directory layout, and usually the username, of whoever ran
      the pipeline. Nobody reading the analysis needs that.

    A path outside the repository is reduced to its filename, for the same
    reason: what lies outside is the runner's business and nobody else's.
    """
    path = Path(path)
    try:
        return str(path.resolve().relative_to(_REPOSITORY_ROOT))
    except ValueError:
        return path.name


def _portable_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Rewrite any parameter that names an existing path into a portable one.

    Applied automatically rather than left to each script: the scripts take
    directories as arguments, and one of them forgetting to normalise would
    quietly publish a home directory into a committed file. Detection is by
    whether the value actually resolves to something on disk, which is
    deterministic, rather than by guessing from the parameter's name.
    """
    portable = {}
    for key, value in parameters.items():
        if isinstance(value, str) and value:
            try:
                if Path(value).exists():
                    portable[key] = portable_path(value)
                    continue
            except OSError:
                pass
        portable[key] = value
    return portable


def commit_hash() -> dict[str, Any]:
    """Current commit hash and whether the inputs were committed.

    A result produced from uncommitted code is not reproducible from the hash,
    and that must be recorded, not hidden.

    `inputs_clean` deliberately ignores anything under `results/`. What has to
    be committed for a result to be reproducible is what PRODUCED it, the code
    and the raw data. The products themselves cannot be, since a pipeline dirties
    its own output directory as it writes: step 00 creates a file, and from step
    01 onwards every product would otherwise stamp itself as dirty just for
    existing. Reporting that would be noise, and noise in a provenance field is
    worse than no field, because it trains the reader to ignore it.
    """
    def git(*arguments: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *arguments], capture_output=True, text=True, check=True,
                timeout=10,
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    status = git("status", "--porcelain")
    if status is None:
        inputs_clean = None
    else:
        dirty = [
            line[3:] for line in status.splitlines()
            if line[3:] and not line[3:].startswith("results/")
        ]
        inputs_clean = not dirty
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "inputs_clean": inputs_clean,
    }


def stamp(parameters: dict[str, Any] | None = None,
          seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Provenance block embedded in every JSON under results/."""
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": commit_hash(),
        "seed": seed,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "parameters": _portable_parameters(parameters or {}),
    }


def next_version(directory: str | Path, base: str, extension: str = "json") -> int:
    """First free _vN suffix for `base` in `directory`."""
    directory = Path(directory)
    if not directory.is_dir():
        return 1
    pattern = re.compile(rf"^{re.escape(base)}_v(\d+)\.{re.escape(extension)}$")
    used = [
        int(m.group(1))
        for path in directory.iterdir()
        if (m := pattern.match(path.name))
    ]
    return max(used) + 1 if used else 1


def latest_version(directory: str | Path, base: str, extension: str = "json") -> int | None:
    """Highest existing _vN suffix for `base`, or None if there is none."""
    following = next_version(directory, base, extension)
    return following - 1 if following > 1 else None


def read_product(directory: str | Path, base: str) -> dict[str, Any]:
    """Read the most recent JSON for `base`, which is how the pipeline chains.

    Raises FileNotFoundError naming the missing step, so running a script out
    of order says what to run first instead of failing further down.
    """
    directory = Path(directory)
    version = latest_version(directory, base)
    if version is None:
        raise FileNotFoundError(
            f"no {base}_vN.json found in {directory}. "
            f"Run pipeline step {base.split('_')[0]} first."
        )
    return json.loads((directory / f"{base}_v{version}.json").read_text(encoding="utf-8"))


def _strip_non_finite(value: Any) -> Any:
    """JSON has no infinity and no NaN. They are written as null, which is what
    they mean here: a fit with no finite value, not a zero.

    Walks the whole tree before serialising. The `default=` hook of json.dumps
    is not enough, since it is never invoked for floats: json would write
    `Infinity` and `NaN`, which are not valid JSON and which no other reader
    would accept.
    """
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _strip_non_finite(value.tolist())
    if isinstance(value, dict):
        return {k: _strip_non_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strip_non_finite(x) for x in value]
    return value


def _json_default(obj: Any) -> Any:
    """Only invoked for types json cannot serialise on its own."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _strip_non_finite(float(obj))
    if isinstance(obj, np.ndarray):
        return _strip_non_finite(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def write_product(
    directory: str | Path,
    base: str,
    data: dict[str, Any],
    text: str,
    parameters: dict[str, Any] | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[Path, Path]:
    """Write the paired .json and .txt with the next free _vN.

    Paired on purpose: the JSON to be read back by another script, the TXT to
    be read with your eyes and no tools.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    version = next_version(directory, base)

    json_path = directory / f"{base}_v{version}.json"
    text_path = directory / f"{base}_v{version}.txt"

    content = _strip_non_finite({"provenance": stamp(parameters, seed), **data})
    json_path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False,
                   default=_json_default),
        encoding="utf-8",
    )

    prov = content["provenance"]
    header = (
        f"{base}_v{version}\n"
        f"generated: {prov['generated_utc']}\n"
        f"commit:    {prov['git']['commit']} "
        f"({'committed inputs' if prov['git']['inputs_clean'] else 'UNCOMMITTED INPUTS'})\n"
        f"{'=' * 78}\n"
    )
    text_path.write_text(header + text.rstrip() + "\n", encoding="utf-8")
    return json_path, text_path
