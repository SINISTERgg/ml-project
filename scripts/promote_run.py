"""promote_run.py — promote a registered run to champion.

Useful after a schema change: the old champion may have a higher CV score
while still using stale feature names. This script copies the chosen run's
bundle, metadata, explainer, reference features and plots into the live slots.

Usage:
    python scripts/promote_run.py <run_id>
    python scripts/promote_run.py --latest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, experiments


def resolve_run_id(arg: str) -> str:
    if arg != "--latest":
        return arg
    runs = sorted(p.name for p in config.RUNS_DIR.iterdir() if p.is_dir())
    if not runs:
        raise SystemExit("No runs found.")
    return runs[-1]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/promote_run.py <run_id> | --latest")
    run_id = resolve_run_id(sys.argv[1])
    run_root = config.RUNS_DIR / run_id
    meta_path = run_root / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Run not found or incomplete: {run_root}")
    meta = json.loads(meta_path.read_text())

    registry = experiments.load_registry()
    registry["champion"] = None
    experiments.save_registry(registry)

    result = experiments.select_champion(run_id, meta, run_root)
    print(f"Promotion: {result}")
    print(f"Champion bundle -> {config.BUNDLE_PATH}")


if __name__ == "__main__":
    main()
