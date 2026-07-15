# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Install a compatibility shim so Warp >=1.13 and ``omni.replicator.core`` coexist.

Why this exists
---------------
The perf gate must run all four gate tasks in a *single* environment. Two of them
pull in conflicting Warp expectations:

* Newton rough-terrain tasks need ``wp.tile_query_valid`` (added in Warp 1.13.0).
* ``omni.replicator.core-1.13.4`` (the RTX/camera path used by the shadow-vision
  task) was built against the pre-1.13 layout and still references
  ``wp.context`` and a handful of ``warp.types.*`` symbols that Warp 1.13.0
  relocated into ``warp._src``.

No single PyPI Warp release satisfies both, so we pin Warp 1.13.0 and re-expose
the relocated symbols. This is a temporary bridge: it can be deleted once Isaac
Sim ships a replicator built for Warp >=1.13.

What it does
------------
Idempotently appends a small shim block to the installed ``warp/__init__.py``
that re-publishes ``warp.context`` (both as an attribute and in ``sys.modules``)
and any missing ``warp.types`` helpers from ``warp._src``. Safe no-op when the
symbols already resolve or the shim is already present.

Usage::

    ./isaaclab.sh -p tools/perf_smoke/warp_replicator_shim.py          # install
    ./isaaclab.sh -p tools/perf_smoke/warp_replicator_shim.py --check  # verify only
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_MARKER = "# >>> isaaclab perf-gate warp/replicator compatibility shim >>>"
_END_MARKER = "# <<< isaaclab perf-gate warp/replicator compatibility shim <<<"

_SHIM = f"""

{_MARKER}
# Bridge omni.replicator.core (built for the pre-1.13 Warp layout) onto Warp
# >=1.13, which moved internals into warp._src. Installed by
# tools/perf_smoke/warp_replicator_shim.py. Remove once replicator targets
# Warp >=1.13 natively.
try:
    import sys as _sys

    from warp import _src as _src

    context = _src.context  # noqa: F811  (re-export for `wp.context.*`)
    _sys.modules.setdefault("warp.context", _src.context)  # for `import warp.context`

    import warp.types as _pub_types  # noqa

    for _n in ("array", "type_size_in_bytes", "warp_type_to_np_dtype", "np_dtype_to_warp_type"):
        if not hasattr(_pub_types, _n) and hasattr(_src.types, _n):
            setattr(_pub_types, _n, getattr(_src.types, _n))
except Exception:  # pragma: no cover - the shim must never break `import warp`
    pass
{_END_MARKER}
"""


def _warp_init_path() -> Path:
    """Locate the installed ``warp/__init__.py`` without importing warp."""
    spec = importlib.util.find_spec("warp")
    if spec is None or not spec.origin:
        raise SystemExit("warp is not importable in this interpreter")
    return Path(spec.origin)


def _is_installed(init_path: Path) -> bool:
    return _MARKER in init_path.read_text(encoding="utf-8")


def install() -> bool:
    """Append the shim if absent. Returns True if it wrote, False if already present."""
    init_path = _warp_init_path()
    if _is_installed(init_path):
        print(f"shim already present: {init_path}")
        return False
    with open(init_path, "a", encoding="utf-8") as f:
        f.write(_SHIM)
    print(f"shim installed: {init_path}")
    return True


def check() -> bool:
    """Verify the relocated symbols resolve. Returns True when the env is usable."""
    import warp as wp  # noqa

    ok = True
    try:
        import warp.context  # noqa
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: import warp.context -> {e}")
        ok = False
    if not hasattr(wp, "context"):
        print("FAIL: wp.context attribute missing")
        ok = False
    if ok:
        print(f"OK: warp {wp.__version__} exposes context for replicator")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true", help="Only verify the shim resolves; do not edit files.")
    args = parser.parse_args(argv)
    if args.check:
        return 0 if check() else 1
    install()
    return 0


if __name__ == "__main__":
    sys.exit(main())
