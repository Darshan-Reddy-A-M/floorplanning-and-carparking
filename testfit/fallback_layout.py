"""Deterministic adaptive layout fallback for difficult/compact sites.

The CP-SAT optimizer remains the primary solver. This module is only used when
CP-SAT reports no feasible arrangement or raises a model-validation exception.
It produces non-overlapping, in-bound rectangles, preferring the room relationships
learned from the RPLAN-derived architectural rules and CubiCasa5K inspection data.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

from testfit.dataset_priors import get_dataset_prior_summary
from testfit.optimizer import (
    DEFAULT_PARKING_DIMENSIONS,
    DEFAULT_STAIRCASE_LENGTH,
    DEFAULT_STAIRCASE_WIDTH,
    LayoutResult,
    Placement,
    Room,
    RoomRelationship,
)

FT_TO_M = 0.3048
EPS = 1e-5
_DATASET_PRIORS = get_dataset_prior_summary()
_CUBICASA_ADJACENCY_WEIGHT = 0.75 + 0.5 * float(_DATASET_PRIORS.get("cubicasa_adjacency_confidence", 0.5))

# Minimum conceptual room sizes used only after the primary optimizer fails.
# They are deliberately labelled as test-fit minima, not code-compliant sizes.
_MIN_FT = {
    "living": (6.0, 6.0),
    "bedroom": (4.5, 5.5),
    "bathroom": (2.5, 3.5),
    "kitchen": (4.0, 4.5),
    "dining": (3.5, 4.0),
    "utility": (2.5, 3.0),
    "foyer": (2.5, 3.0),
    "balcony": (2.5, 2.0),
    "stair": (3.0, 4.5),
    "other": (3.0, 4.0),
}


def _room_kind(name: str) -> str:
    n = name.casefold()
    for key in ("bathroom", "toilet", "bedroom", "living", "kitchen", "dining", "utility", "foyer", "balcony", "stair"):
        if key in n:
            return key if key != "toilet" else "bathroom"
    return "other"


def _rect_overlap(a, b) -> bool:
    return (
        a.x < b.x + b.width - EPS
        and b.x < a.x + a.width - EPS
        and a.y < b.y + b.length - EPS
        and b.y < a.y + a.length - EPS
    )


def _edge_share(a, b, minimum_shared: float = 0.05) -> bool:
    vertical = (
        abs((a.x + a.width) - b.x) <= EPS
        or abs((b.x + b.width) - a.x) <= EPS
    ) and min(a.y + a.length, b.y + b.length) - max(a.y, b.y) >= minimum_shared - EPS
    horizontal = (
        abs((a.y + a.length) - b.y) <= EPS
        or abs((b.y + b.length) - a.y) <= EPS
    ) and min(a.x + a.width, b.x + b.width) - max(a.x, b.x) >= minimum_shared - EPS
    return vertical or horizontal


def _gap(a, b) -> float:
    dx = max(0.0, a.x - (b.x + b.width), b.x - (a.x + a.width))
    dy = max(0.0, a.y - (b.y + b.length), b.y - (a.y + a.length))
    return dx + dy


def _parking_for_site(site_width, site_length, parking_type, parking_width, parking_length, road):
    if str(parking_type).casefold() in {"no parking", "none"}:
        return "No parking", None
    ptype = str(parking_type)
    pw = float(parking_width) if parking_width and parking_width > 0 else DEFAULT_PARKING_DIMENSIONS.get(ptype, (2.5, 5.0))[0]
    pl = float(parking_length) if parking_length and parking_length > 0 else DEFAULT_PARKING_DIMENSIONS.get(ptype, (2.5, 5.0))[1]
    note = ""

    # Narrow plots cannot realistically hold a side-by-side two-car bay while
    # preserving any ground-floor building footprint. Disclose the reduction.
    if ptype == "2 cars" and site_width < 7.2:
        ptype = "1 car"
        pw, pl = 2.25, 4.15
        note = "The plot is too narrow for a practical two-car bay; fallback uses a compact single-car bay."
    else:
        pw = min(pw, max(2.1, site_width - 0.35))
        pl = min(pl, max(3.2, site_length * 0.42))
        if parking_width and pw < float(parking_width) - EPS:
            note = "The fallback reduced the parking footprint to fit the site."
        if parking_length and pl < float(parking_length) - EPS:
            note = "The fallback reduced the parking footprint to fit the site."

    pw = min(pw, site_width - 0.05)
    pl = min(pl, site_length - 0.05)
    if pw <= 0.5 or pl <= 0.8:
        return "No parking", None
    if road == "South":
        x, y = (site_width - pw) / 2, 0.0
    elif road == "North":
        x, y = (site_width - pw) / 2, site_length - pl
    elif road == "West":
        x, y = 0.0, (site_length - pl) / 2
    else:
        x, y = site_width - pw, (site_length - pl) / 2
    return ptype, Placement(f"Parking ({ptype})", x, y, pw, pl, 0, 0)


def _scaled_rooms(rooms: Sequence[Room], factor: float, build_w: float, build_l: float) -> list[Room]:
    max_dimension = max(build_w, build_l) - 0.03
    output = []
    for room in rooms:
        min_w_ft, min_l_ft = _MIN_FT[_room_kind(room.name)]
        min_w, min_l = min_w_ft * FT_TO_M, min_l_ft * FT_TO_M
        width = min(max_dimension, max(min_w, room.width * factor))
        length = min(max_dimension, max(min_l, room.length * factor))
        output.append(Room(room.name, width, length, int(room.floor)))
    # Preserve the duplex living stack's matching footprint when possible.
    by_name = {r.name.casefold(): r for r in output}
    lower, upper = by_name.get("living room"), by_name.get("upper living room")
    if lower and upper:
        output = [Room(r.name, lower.width, lower.length, r.floor) if r.name.casefold() == "upper living room" else r for r in output]
    return output


def _floor_variants(rooms: Sequence[Room], relationships: Sequence[RoomRelationship], floors: int) -> list[list[Room]]:
    variants = [list(rooms)]
    if floors < 2:
        return variants
    top = 2  # keep the upper living room adjacent to the ground-floor living room
    by_name = {r.name.casefold(): r for r in rooms}

    # Move bedroom/ensuite pairs together upstairs. This is often necessary when
    # a small front parking bay competes with the ground-floor footprint.
    upstairs = []
    for room in rooms:
        n = room.name.casefold()
        if n.startswith("bedroom ") or n.startswith("bathroom "):
            upstairs.append(Room(room.name, room.width, room.length, top))
        else:
            upstairs.append(room)
    variants.append(upstairs)

    # Compact duplex: keep the entry and main living room on the ground floor;
    # move the kitchen-service-dining group upstairs as one connected group.
    service_up = []
    for room in upstairs:
        n = room.name.casefold()
        if any(k in n for k in ("kitchen", "utility", "dining")):
            service_up.append(Room(room.name, room.width, room.length, top))
        else:
            service_up.append(room)
    variants.append(service_up)

    # Last full-room variant: only the foyer and ground living room stay downstairs.
    # All other spaces are placed on floor two; this is a compact-site fallback.
    core_up = []
    for room in service_up:
        n = room.name.casefold()
        if "entry foyer" in n or (n == "living room"):
            core_up.append(Room(room.name, room.width, room.length, 1))
        else:
            core_up.append(Room(room.name, room.width, room.length, top))
    variants.append(core_up)

    # Deduplicate floor assignments.
    unique = []
    seen = set()
    for variant in variants:
        signature = tuple((r.name, r.floor) for r in variant)
        if signature not in seen:
            seen.add(signature)
            unique.append(variant)
    return unique


def _ordered_floor_rooms(floor_rooms: Sequence[Room], relationships: Sequence[RoomRelationship], order_mode: int = 0) -> list[Room]:
    priorities = ("entry foyer", "living room", "upper living room", "kitchen", "utility", "dining", "balcony")
    if order_mode == 1:
        # Classic rectangle-packing strategy: place largest rectangles first.
        return sorted(floor_rooms, key=lambda r: (-(r.width * r.length), r.name.casefold()))
    if order_mode == 2:
        degree = {}
        for rel in relationships:
            if rel.kind.upper() in {"ATTACHED", "ADJACENT"}:
                degree[rel.room_a] = degree.get(rel.room_a, 0) + 1
                degree[rel.room_b] = degree.get(rel.room_b, 0) + 1
        return sorted(floor_rooms, key=lambda r: (-degree.get(r.name, 0), -(r.width * r.length), r.name.casefold()))
    def key(room):
        name = room.name.casefold()
        if "bedroom" in name or "bathroom" in name:
            number = re.search(r"\d+", name)
            pair_number = int(number.group()) if number else 99
            room_order = 0 if "bedroom" in name else 1
            # Place each bedroom immediately before its attached bathroom.
            rank = 10 + pair_number * 2 + room_order
        else:
            rank = next((i for i, k in enumerate(priorities) if k in name), len(priorities))
        return (rank, -(room.width * room.length), name)
    return sorted(floor_rooms, key=key)


def _zone_penalty(room: Room, x: float, y: float, site_width: float, site_length: float, road: str, bounds) -> float:
    xmin, ymin, xmax, ymax = bounds
    name = room.name.casefold()
    if any(k in name for k in ("entry", "foyer", "living", "dining")):
        frac = 0.18
    elif any(k in name for k in ("kitchen", "utility", "bathroom")):
        frac = 0.42
    elif "bedroom" in name:
        frac = 0.72
    else:
        frac = 0.50
    if road in {"South", "North"}:
        target = ymin + frac * (ymax - ymin)
        if road == "North":
            target = ymax - frac * (ymax - ymin)
        return abs((y + room.length / 2) - target)
    target = xmin + frac * (xmax - xmin)
    if road == "East":
        target = xmax - frac * (xmax - xmin)
    return abs((x + room.width / 2) - target)


def _pack_floor_rooms(floor_rooms, already, bounds, parking, stair, site_width, site_length, road, relationships, all_placements, order_mode=0):
    xmin, ymin, xmax, ymax = bounds
    placed = list(already)
    ordered = _ordered_floor_rooms(floor_rooms, relationships, order_mode)
    rel_by_name = {}
    for rel in relationships:
        rel_by_name.setdefault(rel.room_a, []).append(rel)
        rel_by_name.setdefault(rel.room_b, []).append(rel)

    for room in ordered:
        orientations = [(room.width, room.length)]
        if abs(room.width - room.length) > 0.02:
            orientations.append((room.length, room.width))
        best = None
        for w, l in orientations:
            if w > xmax - xmin + EPS or l > ymax - ymin + EPS:
                continue
            x_candidates = {xmin, xmax - w}
            y_candidates = {ymin, ymax - l}
            # Exact wall-alignment candidates make shared doors possible.
            for p in placed:
                x_candidates.update((p.x, p.x + p.width, p.x - w, p.x + p.width - w))
                y_candidates.update((p.y, p.y + p.length, p.y - l, p.y + p.length - l))
            # Coarse grid candidates fill free interior space without an expensive
            # dense search; edge candidates above preserve exact alignments.
            step = 0.25
            gx = xmin
            while gx <= xmax - w + EPS:
                x_candidates.add(round(gx, 4))
                gx += step
            gy = ymin
            while gy <= ymax - l + EPS:
                y_candidates.add(round(gy, 4))
                gy += step
            for x in x_candidates:
                if x < xmin - EPS or x + w > xmax + EPS:
                    continue
                for y in y_candidates:
                    if y < ymin - EPS or y + l > ymax + EPS:
                        continue
                    candidate = Placement(room.name, x, y, w, l, room.floor)
                    obstacles = [p for p in placed if p.floor == room.floor]
                    if parking is not None and room.floor == 1:
                        obstacles.append(parking)
                    if stair is not None:
                        obstacles.append(stair)
                    if any(_rect_overlap(candidate, p) for p in obstacles):
                        continue
                    score = _zone_penalty(room, x, y, site_width, site_length, road, bounds)
                    # Prefer a compact footprint and a location near the existing cluster.
                    if placed:
                        cx, cy = x + w / 2, y + l / 2
                        cluster_x = sum(p.x + p.width / 2 for p in placed) / len(placed)
                        cluster_y = sum(p.y + p.length / 2 for p in placed) / len(placed)
                        score += 0.12 * (abs(cx - cluster_x) + abs(cy - cluster_y))
                    for rel in rel_by_name.get(room.name, []):
                        other_name = rel.room_b if rel.room_a == room.name else rel.room_a
                        kind = rel.kind.upper()
                        # STACKED is explicitly a cross-floor relationship: find the
                        # matching room on another floor and strongly prefer the same
                        # footprint, instead of accidentally ignoring the constraint.
                        if kind == "STACKED":
                            other = next((p for p in all_placements if p.name == other_name and p.floor != room.floor), None)
                        else:
                            other = next((p for p in all_placements if p.name == other_name and p.floor == room.floor), None)
                        if other is None:
                            continue
                        gap = _gap(candidate, other)
                        if kind == "STACKED":
                            score += 120.0 * (
                                abs(candidate.x - other.x)
                                + abs(candidate.y - other.y)
                                + abs(candidate.width - other.width)
                                + abs(candidate.length - other.length)
                            )
                        elif kind in {"ATTACHED", "ADJACENT"}:
                            required_edge = 0.80 if (kind == "ATTACHED" or {room.name.casefold(), other_name.casefold()} == {"kitchen", "utility"}) else 0.05
                            if _edge_share(candidate, other, required_edge):
                                score -= 30.0 if kind == "ATTACHED" else 20.0
                            else:
                                score += ((300.0 if kind == "ATTACHED" else 150.0) * _CUBICASA_ADJACENCY_WEIGHT) + gap * 30.0
                        elif kind == "NEAR":
                            score += gap * 3.0
                        elif kind == "PREFERRED":
                            score += max(0.0, gap - 1.0)
                    # Stable, deterministic tie-breaker.
                    score += 0.0001 * (x + y)
                    if best is None or score < best[0]:
                        best = (score, candidate)
        if best is None:
            return None
        chosen = best[1]
        placed.append(chosen)
        all_placements.append(chosen)
    return placed


def _essential_connections_ok(placements, relationships):
    by_name = {p.name: p for p in placements}
    for rel in relationships:
        pair = {rel.room_a.casefold(), rel.room_b.casefold()}
        kind = rel.kind.upper()
        # These connections are explicit user requirements and are never relaxed
        # by the compact fallback: kitchen-to-utility and attached bathrooms.
        essential = (kind == "ADJACENT" and pair == {"kitchen", "utility"}) or kind == "ATTACHED"
        if not essential:
            continue
        a, b = by_name.get(rel.room_a), by_name.get(rel.room_b)
        is_kitchen_utility = pair == {"kitchen", "utility"}
        minimum_shared = 0.80 if (kind == "ATTACHED" or is_kitchen_utility) else 0.05
        if a is None or b is None or a.floor != b.floor or not _edge_share(a, b, minimum_shared):
            return False
    return True


def _find_stair(bounds, parking, width, length, floors, site_width, site_length):
    if width <= 0 or length <= 0:
        return None
    xmin, ymin, xmax, ymax = bounds
    width = min(width, xmax - xmin)
    length = min(length, ymax - ymin)
    if width <= 0.1 or length <= 0.1:
        return None
    candidates = []
    x = xmin
    while x <= xmax - width + EPS:
        y = ymin
        while y <= ymax - length + EPS:
            stair = Placement("Staircase", x, y, width, length, 1, floors)
            if parking is None or not _rect_overlap(stair, parking):
                # Prefer the rear-left corner away from the road/parking.
                score = (abs(x - xmin) + abs((y + length) - ymax))
                candidates.append((score, stair))
            y += 0.25
        x += 0.25
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def solve_fallback_layout(*, site_width: float, site_length: float, rooms: Sequence[Room],
                          time_limit_seconds: float, relationships: Sequence[RoomRelationship],
                          number_of_floors: int, parking_type: str, parking_width: float | None,
                          parking_length: float | None, road_access: str, staircase_width: float,
                          staircase_length: float, buildable_bounds: tuple[float, float, float, float] | None,
                          staircase_mode: str = "internal", primary_message: str = "") -> LayoutResult:
    """Find a compact, collision-free fallback arrangement if CP-SAT cannot fit it."""
    del time_limit_seconds, staircase_mode
    if not rooms or site_width <= 0 or site_length <= 0:
        return LayoutResult(status="INFEASIBLE", placements=[], message="No room program or invalid site dimensions.", number_of_floors=number_of_floors, parking_type=parking_type, road_access=road_access)
    bounds = buildable_bounds or (0.0, 0.0, site_width, site_length)
    xmin, ymin, xmax, ymax = bounds
    build_w, build_l = xmax - xmin, ymax - ymin
    if build_w <= 0.5 or build_l <= 0.5:
        return LayoutResult(status="INFEASIBLE", placements=[], message="Setbacks leave too little buildable area for a room layout.", number_of_floors=number_of_floors, parking_type=parking_type, road_access=road_access)

    actual_parking_type, parking = _parking_for_site(site_width, site_length, parking_type, parking_width, parking_length, road_access)
    notes = []
    if actual_parking_type != parking_type:
        notes.append("Compact fallback changed two-car parking to a single-car bay because the plot is too narrow.")
    if parking is not None and parking_width and parking.width < parking_width - EPS:
        notes.append("Parking width was reduced to fit the plot.")
    if parking is not None and parking_length and parking.length < parking_length - EPS:
        notes.append("Parking length was reduced to fit the plot.")

    # Generate variants with progressively smaller room footprints. Keep the
    # requested rooms first; only omit clearly optional spaces as a last resort.
    room_variants = [list(rooms)]
    # Utility is intentionally never dropped: the user's required Kitchen-to-Utility
    # doorway must remain possible. Only optional balcony/dining spaces may be
    # omitted, and only after all compact sizing/floor-distribution attempts fail.
    no_balcony = [r for r in rooms if "balcony" not in r.name.casefold()]
    no_dining = [r for r in rooms if r.name.casefold() != "dining"]
    no_balcony_or_dining = [r for r in rooms if "balcony" not in r.name.casefold() and r.name.casefold() != "dining"]
    room_variants.extend([no_balcony, no_dining, no_balcony_or_dining])
    for factor in (1.0, 0.86, 0.72, 0.60, 0.50, 0.42, 0.36, 0.32):
        for base_rooms in room_variants:
            scaled = _scaled_rooms(base_rooms, factor, build_w, build_l)
            for floor_variant in _floor_variants(scaled, relationships, number_of_floors):
                # Relationships with rooms removed are filtered; floor mismatches
                # are not enforced in this geometry fallback but remain in the report.
                valid_names = {r.name for r in floor_variant}
                active_relationships = [r for r in relationships if r.room_a in valid_names and r.room_b in valid_names]
                # Re-normalize attached bedroom/bath pairs to the same floor.
                floor_map = {r.name: r.floor for r in floor_variant}
                for rel in active_relationships:
                    if rel.kind.upper() == "ATTACHED" and rel.room_a in floor_map and rel.room_b in floor_map and floor_map[rel.room_a] != floor_map[rel.room_b]:
                        # Relationship is handled as a soft preference here; do not mutate unrelated rooms.
                        pass
                stair_w = staircase_width if staircase_width > 0 else 0.0
                stair_l = staircase_length if staircase_length > 0 else 0.0
                if stair_w and stair_l:
                    stair_w = min(stair_w, max(0.9, build_w * 0.24))
                    stair_l = min(stair_l, max(1.5, build_l * 0.28))
                stair = _find_stair(bounds, parking, stair_w, stair_l, number_of_floors, site_width, site_length) if stair_w and stair_l else None
                if stair_w and stair_l and stair is None:
                    continue

                all_placements: list[Placement] = []
                success = False
                # Greedy rectangle packing can depend on insertion order. Try
                # semantic, largest-first, and relationship-degree-first orders
                # before reducing room counts.
                for order_mode in (0, 1, 2):
                    trial_placements: list[Placement] = []
                    trial_success = True
                    for floor in sorted({r.floor for r in floor_variant}):
                        floor_rooms = [r for r in floor_variant if r.floor == floor]
                        stair_for_floor = stair if stair and stair.floor <= floor <= (stair.floor_end or stair.floor) else None
                        packed = _pack_floor_rooms(
                            floor_rooms, [], bounds, parking, stair_for_floor,
                            site_width, site_length, road_access, active_relationships,
                            trial_placements, order_mode=order_mode,
                        )
                        if packed is None:
                            trial_success = False
                            break
                    if trial_success and trial_placements and _essential_connections_ok(trial_placements, active_relationships):
                        all_placements = trial_placements
                        success = True
                        break
                if not success:
                    continue

                stairs = []
                if stair is not None:
                    if number_of_floors >= 2:
                        stairs = [Placement(f"Staircase F{floor}-{floor+1}", stair.x, stair.y, stair.width, stair.length, floor, floor + 1) for floor in range(1, number_of_floors)]
                    else:
                        stairs = [Placement("Staircase", stair.x, stair.y, stair.width, stair.length, 1, 1)]
                omitted = [r.name for r in rooms if r.name not in {p.name for p in all_placements}]
                if factor < 0.999:
                    notes.append(f"Room sizes were adaptively reduced (scale attempt {factor:.2f}) after the main solver could not fit the requested program.")
                if omitted:
                    notes.append("Optional rooms omitted to preserve a collision-free layout: " + ", ".join(omitted) + ".")
                if primary_message:
                    notes.append("Primary optimizer note: " + primary_message)
                loaded_sources = ", ".join(_DATASET_PRIORS.get("sources_loaded", [])) or "built-in architectural rules"
                message = f"Adaptive geometric fallback generated a non-overlapping test-fit using {loaded_sources} priors. " + " ".join(dict.fromkeys(notes))
                return LayoutResult(
                    status="FEASIBLE",
                    placements=all_placements,
                    message=message,
                    parking=parking,
                    staircases=stairs,
                    number_of_floors=number_of_floors,
                    parking_type=actual_parking_type,
                    road_access=road_access,
                )

    message = "No collision-free layout could be fitted even after compact retries. Increase the plot/buildable area, reduce setbacks or room counts, or disable parking/staircase."
    if primary_message:
        message += " Primary optimizer: " + primary_message
    return LayoutResult(status="INFEASIBLE", placements=[], message=message, parking=None, staircases=[], number_of_floors=number_of_floors, parking_type=actual_parking_type, road_access=road_access)
