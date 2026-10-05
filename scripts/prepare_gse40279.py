"""Download and prepare the public GSE40279 benchmark data.

The raw GEO archives are intentionally not tracked in Git.  The existing
preparation code downloads them, verifies their SHA-256 hashes from
data/gse40279_data/manifest.json, and writes the compact runner-ready data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gse40279.prepare import prepare_methylation_data


DEFAULT_CONFIG = ROOT / "config" / "gse40279_paper.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config

    import json

    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = prepare_methylation_data(config)
    print(f"Prepared GSE40279 data under {output}")


if __name__ == "__main__":
    main()
