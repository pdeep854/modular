"""``python -m flux_klein_host`` runs the pipeline driver."""

from __future__ import annotations

from .pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
