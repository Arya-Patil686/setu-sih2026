"""Optional ISIS cube reader.

Tier A only, and behind a capability check. The specification is explicit that ISIS
must not be a prerequisite: a heavyweight install is the single most likely cause of a
failed live demo, so this module degrades to a clear error rather than an import crash.
"""

from __future__ import annotations

from pathlib import Path

from setu.types import Product


def isis_available() -> bool:
    try:
        import kalasiris  # noqa: F401
        return True
    except Exception:
        return False


def read_isis_cube(path: str | Path, sensor: str | None = None) -> Product:
    """Read an ISIS cube via `kalasiris`, or explain what is missing."""
    if not isis_available():
        raise RuntimeError(
            "ISIS support is optional and is not installed. SETU's Tier B sensor model "
            "(corner fit plus terrain parallax) is the default path and needs no ISIS. "
            "Install `kalasiris` and a full ISIS environment only if you want Tier A."
        )
    import kalasiris as isis  # noqa: F401
    import pvl

    from setu.io.pds3 import read_pds3

    path = Path(path)
    header = pvl.load(str(path))
    if "IsisCube" not in header:
        raise ValueError(f"{path}: not an ISIS cube")
    return read_pds3(path, sensor=sensor)
