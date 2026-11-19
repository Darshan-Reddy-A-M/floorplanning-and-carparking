from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


INPUT_PATH = Path("data/training/cubicasa5k_candidates_100.jsonl")
OUTPUT_DIR = Path("data/training/ai_baseline")
SEED = 42

LABELS = [
    "bedroom",
    "bathroom",
    "kitchen",
    "living_area",
    "dining",
    "circulation_entry",
    "storage_closet",
    "utility",
    "outdoor",
    "garage",
    "carport",
    "office_study",
    "recreation",
    "other_known_room",
    "generic_room",
    "user_defined",
    "unknown",
]

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}

LABEL_MAP = {
    "bedroom": "bedroom",
    "bath": "bathroom",
    "bathroom": "bathroom",
    "toilet": "bathroom",
    "kitchen": "kitchen",
    "livingroom": "living_area",
    "living": "living_area",
    "lounge": "living_area",
    "dining": "dining",
    "entry": "circulation_entry",
    "lobby": "circulation_entry",
    "draughtlobby": "circulation_entry",
    "hall": "circulation_entry",
    "corridor": "circulation_entry",
    "hallway": "circulation_entry",
    "stair": "circulation_entry",
    "closet": "storage_closet",
    "storage": "storage_closet",
    "utility": "utility",
    "laundry": "utility",
    "outdoor": "outdoor",
    "terrace": "outdoor",
    "garage": "garage",
    "carport": "carport",
    "office": "office_study",
    "library": "office_study",
    "den": "office_study",
    "recreationroom": "recreation",
    "room": "generic_room",
    "userdefined": "user_defined",
    "undefined": "unknown",
    "unknown": "unknown",
    "elevated": "other_known_room",
    "technicalroom": "other_known_room",
    "dressingroom": "other_known_room",
    "attic": "other_known_room",
    "basement": "other_known_room",
    "opentobelow": "other_known_room",
}


def canonical_label(space: dict) -> str:
    raw_label = str(space.get("label", ""))
    key = raw_label.strip().casefold().replace(" ", "").replace("_", "")

    # Use the original room label first. Do not infer a room type
    # from ambiguous class tokens.
    return LABEL_MAP.get(key, "unknown")


def load_plans(path: Path) -> list[dict]:
    plans = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                source_plan = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                ) from error

            width = float(source_plan["svg_width"])
            height = float(source_plan["svg_height"])

            if (
                not np.isfinite(width)
                or not np.isfinite(height)
                or width <= 0
                or height <= 0
            ):
                raise ValueError(
                    f"Invalid SVG dimensions in plan "
                    f"{source_plan.get('plan_id')}."
                )

            rooms = []

            for space in source_plan.get("spaces", []):
                bbox = space.get("bbox_svg_units")

                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = map(float, bbox)

                if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
                    continue

                if x2 <= x1 or y2 <= y1:
                    continue

                rooms.append({
                    "original_label": str(space.get("label", "")),
                    "label": canonical_label(space),
                    "bbox": [
                        x1 / width,
                        y1 / height,
                        x2 / width,
                        y2 / height,
                    ],
                })

            if rooms:
                plans.append({
                    "plan_id": str(source_plan["plan_id"]),
                    "rooms": rooms,
                })

    ids = [plan["plan_id"] for plan in plans]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate plan IDs found in the input dataset.")

    return plans


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH.resolve()}"
        )

    plans = load_plans(INPUT_PATH)

    if len(plans) < 3:
        raise ValueError("At least three valid plans are required.")

    rng = random.Random(SEED)
    rng.shuffle(plans)

    count = len(plans)
    train_end = int(count * 0.8)
    validation_end = int(count * 0.9)

    splits = {
        "train": plans[:train_end],
        "validation": plans[train_end:validation_end],
        "test": plans[validation_end:],
    }

    all_ids = []

    for split_name, split_plans in splits.items():
        output_path = OUTPUT_DIR / f"{split_name}.jsonl"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            for plan in split_plans:
                all_ids.append(plan["plan_id"])
                handle.write(json.dumps(plan) + "\n")

        print(
            f"{split_name}: {len(split_plans)} plans, "
            f"{sum(len(p['rooms']) for p in split_plans)} room records"
        )

    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Plan leakage detected across data splits.")

    label_counts = Counter(
        room["label"]
        for plan in plans
        for room in plan["rooms"]
    )

    original_label_counts = Counter(
        room["original_label"]
        for plan in plans
        for room in plan["rooms"]
    )

    metadata = {
        "seed": SEED,
        "input": str(INPUT_PATH),
        "plan_count": len(plans),
        "label_to_id": LABEL_TO_ID,
        "canonical_label_counts": dict(label_counts),
        "original_label_counts": dict(original_label_counts),
        "coordinate_format": "normalized_bbox_xyxy",
        "coordinate_units": "normalized_svg_coordinates_not_physical_units",
        "warning": (
            "This is a pilot dataset. Normalized SVG coordinates are not "
            "verified physical measurements. Site dimensions, setbacks, "
            "and user requirements are not supplied for every plan."
        ),
    }

    metadata_path = OUTPUT_DIR / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nCanonical label counts:")
    for label, amount in sorted(label_counts.items()):
        print(f"  {label}: {amount}")

    print("\nUnique plan IDs:", len(set(all_ids)))
    print("Metadata:", metadata_path)
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
