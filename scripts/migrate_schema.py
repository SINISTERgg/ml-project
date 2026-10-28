"""migrate_schema.py — rewrite on-disk CSVs to the current schema.

Renames legacy SSC/HSC columns to 10th/12th names and adds the master's
pathway fields. Safe to re-run (idempotent).

Usage:
    python scripts/migrate_schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import _rename_legacy_columns, add_masters_fields


def migrate(path: Path) -> bool:
    if not path.exists():
        print(f"  skip (missing): {path}")
        return False
    df = pd.read_csv(path)
    before = list(df.columns)
    df = add_masters_fields(_rename_legacy_columns(df))
    df.to_csv(path, index=False)
    renamed = sorted(set(before) - set(df.columns))
    added = sorted(set(df.columns) - set(before))
    print(f"  updated {path.name}: renamed={renamed or '—'}, added={added or '—'}")
    return True


def main() -> None:
    print(f"Migrating dataset files to schema v{config.SCHEMA_VERSION}…")
    migrate(config.LEGACY_DATA_PATH)
    migrate(config.DATA_PATH)
    print("Done. Re-run training with --force-reload to rebuild the cache.")


if __name__ == "__main__":
    main()
