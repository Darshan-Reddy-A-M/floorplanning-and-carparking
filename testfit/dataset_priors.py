"""Load compact, versioned priors extracted from the RPLAN and CubiCasa5K datasets.

Raw datasets are intentionally not bundled. The committed JSON files are small
aggregate summaries, while tools/dataset contains scripts for rebuilding them
from user-owned/local copies of the datasets.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RPLAN_PRIORS = ROOT / "data" / "priors" / "resplan_priors.json"
CUBICASA_INSPECTION = ROOT / "data" / "inspection" / "cubicasa_inspection.json"


@lru_cache(maxsize=1)
def get_dataset_prior_summary() -> dict:
    summary = {
        "rplan_plans": 0,
        "rplan_room_count_medians": {},
        "cubicasa_plans_inspected": 0,
        "cubicasa_adjacency_confidence": 0.5,
        "sources_loaded": [],
    }
    try:
        rplan = json.loads(RPLAN_PRIORS.read_text(encoding="utf-8"))
        summary["rplan_plans"] = int(rplan.get("source", {}).get("plans_processed", 0))
        summary["rplan_room_count_medians"] = {
            name: float(info.get("count_per_plan", {}).get("median", 0) or 0)
            for name, info in rplan.get("rooms", {}).items()
        }
        summary["sources_loaded"].append("RPLAN")
    except (OSError, ValueError, TypeError):
        pass
    try:
        cubicasa = json.loads(CUBICASA_INSPECTION.read_text(encoding="utf-8"))
        findings = cubicasa.get("findings", {})
        adjacency = findings.get("adjacency", {}).get("evidence", {})
        room_share = float(adjacency.get("plans_with_room_polygons_share", 0.5) or 0.5)
        near_share = float(adjacency.get("plans_with_near_touching_rooms_share", room_share) or room_share)
        summary["cubicasa_plans_inspected"] = int(cubicasa.get("geometry", {}).get("plans_analyzed", cubicasa.get("parameters", {}).get("geom_sample", 0)) or 0)
        summary["cubicasa_adjacency_confidence"] = max(0.25, min(1.0, 0.5 * room_share + 0.5 * near_share))
        summary["sources_loaded"].append("CubiCasa5K")
    except (OSError, ValueError, TypeError):
        pass
    return summary
