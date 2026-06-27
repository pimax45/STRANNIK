"""Support both package-module and direct-script execution."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from terrain_nav.cli import main
else:
    from .cli import main

raise SystemExit(main())
