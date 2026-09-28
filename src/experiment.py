"""Atomic JSON writes and clustering checkpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any



def atomic_json(path: str | Path, payload: Any) -> None:
    """Write JSON via a sibling temporary file, then atomically replace it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, target)
