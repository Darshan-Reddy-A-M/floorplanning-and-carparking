"""Floor-aware CP-SAT layout solving with parking and stair footprints."""

from __future__ import annotations

import math
import time
from typing import Sequence

from ortools.sat.python import cp_model

from testfit.optimizer import (
    DEFAULT_PARKING_DIMENSIONS,
    DEFAULT_STAIRCASE_LENGTH,
    DEFAULT_STAIRCASE_WIDTH,
    GRID_UNITS_PER_METER,
    HARD_RELATIONSHIP_KINDS,
    MIN_ATTACHED_EDGE_UNITS,
    MINIMUM_POSITIVE_SHARED_EDGE_UNITS,
    RELATIONSHIP_KINDS,
    SOFT_RELATIONSHIP_DISTANCE_THRESHOLD_UNITS,
    Room,
    RoomRelationship,
    Placement,
    LayoutResult,
    _add_hard_edge_touch_constraint,
    _solve_single_floor_layout,
    _to_grid_units,
)

ROAD_ACCESS_SIDES = {"North", "South", "East", "West"}
PARKING_TYPES = {"No parking", *DEFAULT_PARKING_DIMENSIONS}

def _room_zone_fraction(name: str) -> float:
    """Return a normalized front-to-back target for architectural zoning.

    0.0 is the road/front edge and 1.0 is the rear/private edge.  The
    target is intentionally broad: it is a soft planning prior, not a hard
    room location rule.
    """
    value = str(name).casefold()
    if any(k in value for k in ("entry foyer", "foyer", "entrance", "living", "dining")):
        return 0.20
    if any(k in value for k in ("kitchen", "utility")):
        return 0.32
    if any(k in value for k in ("bathroom", "stair")):
        return 0.50
    if any(k in value for k in ("bedroom", "master bedroom", "family lounge")):
        return 0.72
    return 0.50


def _room_zone_class(name: str) -> str:
    """Classify rooms for residential front-to-back planning.

    Public rooms belong near the road/entrance, service rooms in the middle,
    and private rooms toward the rear.  This is intentionally based on room
    function rather than fixed coordinates so the rule works for all road
    orientations.
    """
    value = str(name).casefold()
    if any(k in value for k in ("entry foyer", "foyer", "entrance", "living", "dining")):
        return "public"
    if any(k in value for k in ("kitchen", "utility", "bathroom", "toilet", "powder")):
        return "service"
    if any(k in value for k in ("bedroom", "master bedroom", "family lounge")):
        return "private"
    return "service"


def _add_zone_penalty(
    model: cp_model.CpModel,
    room_index: int,
    room_name: str,
    room_width: int,
    room_length: int,
    x_position: cp_model.IntVar,
    y_position: cp_model.IntVar,
    buildable_x_min: int,
    buildable_y_min: int,
    buildable_x_max: int,
    buildable_y_max: int,
    road_access: str,
) -> cp_model.IntVar:
    """Create a soft distance-to-zone-center penalty for one room."""
    fraction = _room_zone_fraction(room_name)
    if road_access in {"South", "North"}:
        depth = max(1, buildable_y_max - buildable_y_min)
        if road_access == "South":
            target_center = buildable_y_min + round(fraction * depth)
        else:
            target_center = buildable_y_max - round(fraction * depth)
        target_start = target_center - room_length // 2
        penalty = model.new_int_var(0, max(1, buildable_y_max - buildable_y_min), f"zone_{room_index}_penalty")
        model.add_abs_equality(penalty, y_position - target_start)
        return penalty

    depth = max(1, buildable_x_max - buildable_x_min)
    if road_access == "West":
        target_center = buildable_x_min + round(fraction * depth)
    else:
        target_center = buildable_x_max - round(fraction * depth)
    target_start = target_center - room_width // 2
    penalty = model.new_int_var(0, max(1, buildable_x_max - buildable_x_min), f"zone_{room_index}_penalty")
    model.add_abs_equality(penalty, x_position - target_start)
    return penalty



def _add_optional_edge_touch_constraint(
    model: cp_model.CpModel,
    name: str,
    first_index: int,
    second_index: int,
    first_width: int,
    first_length: int,
    second_width: int,
    second_length: int,
    x_positions,
    y_positions,
    minimum_shared_edge_units: int = MINIMUM_POSITIVE_SHARED_EDGE_UNITS,
):
    """Return a Boolean that is true when two rooms share a real wall edge."""
    a_left = model.new_bool_var(f"{name}_a_left")
    b_left = model.new_bool_var(f"{name}_b_left")
    a_below = model.new_bool_var(f"{name}_a_below")
    b_below = model.new_bool_var(f"{name}_b_below")
    sides = [a_left, b_left, a_below, b_below]
    model.add(sum(sides) <= 1)

    model.add(x_positions[first_index] + first_width == x_positions[second_index]).only_enforce_if(a_left)
    model.add(y_positions[first_index] <= y_positions[second_index] + second_length - minimum_shared_edge_units).only_enforce_if(a_left)
    model.add(y_positions[second_index] <= y_positions[first_index] + first_length - minimum_shared_edge_units).only_enforce_if(a_left)

    model.add(x_positions[second_index] + second_width == x_positions[first_index]).only_enforce_if(b_left)
    model.add(y_positions[first_index] <= y_positions[second_index] + second_length - minimum_shared_edge_units).only_enforce_if(b_left)
    model.add(y_positions[second_index] <= y_positions[first_index] + first_length - minimum_shared_edge_units).only_enforce_if(b_left)

    model.add(y_positions[first_index] + first_length == y_positions[second_index]).only_enforce_if(a_below)
    model.add(x_positions[first_index] <= x_positions[second_index] + second_width - minimum_shared_edge_units).only_enforce_if(a_below)
    model.add(x_positions[second_index] <= x_positions[first_index] + first_width - minimum_shared_edge_units).only_enforce_if(a_below)

    model.add(y_positions[second_index] + second_length == y_positions[first_index]).only_enforce_if(b_below)
    model.add(x_positions[first_index] <= x_positions[second_index] + second_width - minimum_shared_edge_units).only_enforce_if(b_below)
    model.add(x_positions[second_index] <= x_positions[first_index] + first_width - minimum_shared_edge_units).only_enforce_if(b_below)
    return sides

def _canonical_parking_type(value: str) -> str:
    labels = {parking_type.casefold(): parking_type for parking_type in PARKING_TYPES}
    labels["none"] = "No parking"
    labels["no parking"] = "No parking"
    try:
        return labels[str(value).strip().casefold()]
    except KeyError as error:
        raise ValueError("Parking must be No parking, 1 car, or 2 cars.") from error


def _select_architectural_anchor_indices(
    rooms: Sequence[Room],
) -> dict[int, int]:
    """Choose one circulation/core anchor room per floor.

    The anchor is used only for spatial-quality objectives: it is not a room
    resize or a replacement for user-defined relationships. Foyer/entry spaces
    are preferred, followed by living/family/common spaces, then the largest
    remaining room on the floor.
    """
    priority_keywords = (
        "entry foyer",
        "foyer",
        "entrance",
        "living",
        "family lounge",
        "dining",
        "kitchen",
    )
    anchors: dict[int, int] = {}
    for floor in sorted({room.floor for room in rooms}):
        candidates = [
            (index, room)
            for index, room in enumerate(rooms)
            if room.floor == floor and "bathroom" not in room.name.casefold()
        ]
        if not candidates:
            continue
        selected = None
        for keyword in priority_keywords:
            selected = next(
                (item for item in candidates if keyword in item[1].name.casefold()),
                None,
            )
            if selected is not None:
                break
        if selected is None:
            selected = max(
                candidates,
                key=lambda item: item[1].width * item[1].length,
            )
        anchors[floor] = selected[0]
    return anchors


def solve_building_layout(
    *,
    site_width: float,
    site_length: float,
    rooms: Sequence[Room],
    time_limit_seconds: float,
    relationships: Sequence[RoomRelationship],
    number_of_floors: int,
    parking_type: str,
    parking_width: float | None,
    parking_length: float | None,
    road_access: str,
    staircase_width: float,
    staircase_length: float,
    buildable_bounds: tuple[float, float, float, float] | None = None,
    staircase_mode: str = "internal",
) -> LayoutResult:
    """Validate building inputs and solve the corresponding floor-aware model."""
    if (
        isinstance(number_of_floors, bool)
        or not isinstance(number_of_floors, int)
        or number_of_floors not in (1, 2, 3, 4, 5)
    ):
        raise ValueError("Number of floors must be between 1 and 5.")
    if not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError("The solver time limit must be greater than zero.")

    site_width_units = _to_grid_units(site_width, "Site width")
    site_length_units = _to_grid_units(site_length, "Site length")
    if buildable_bounds is None:
        buildable_bounds_units = (0, 0, site_width_units, site_length_units)
    else:
        if len(buildable_bounds) != 4:
            raise ValueError("Buildable bounds must be (left, bottom, right, top).")
        bounds = tuple(float(value) for value in buildable_bounds)
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError("Buildable bounds must be finite.")
        x_min_m, y_min_m, x_max_m, y_max_m = bounds
        if (
            x_min_m < 0
            or y_min_m < 0
            or x_max_m > site_width
            or y_max_m > site_length
            or x_min_m >= x_max_m
            or y_min_m >= y_max_m
        ):
            raise ValueError("Setbacks must leave a positive buildable area inside the site.")

        buildable_bounds_units = (
            math.ceil(x_min_m * GRID_UNITS_PER_METER),
            math.ceil(y_min_m * GRID_UNITS_PER_METER),
            math.floor(x_max_m * GRID_UNITS_PER_METER),
            math.floor(y_max_m * GRID_UNITS_PER_METER),
        )
    parking_type = _canonical_parking_type(parking_type)
    normalized_road = str(road_access).strip().title()
    staircase_mode = str(staircase_mode).strip().lower()
    if staircase_mode not in {"internal", "external"}:
        raise ValueError("staircase_mode must be 'internal' or 'external'.")
    if normalized_road not in ROAD_ACCESS_SIDES:
        raise ValueError("Road / Site Access must be North, South, East, or West.")

    normalized_rooms: list[Room] = []
    room_indices: dict[str, int] = {}
    room_floors: dict[str, int] = {}
    dimensions: list[tuple[int, int]] = []
    for index, room in enumerate(rooms, start=1):
        if not room.name.strip():
            raise ValueError(f"Room {index} must have a name.")
        if room.name in room_indices:
            raise ValueError(f"Room names must be unique: {room.name!r}.")
        if isinstance(room.floor, bool):
            raise ValueError(f"{room.name} floor must be an integer from 1 to {number_of_floors}.")
        try:
            floor = int(room.floor)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"{room.name} floor must be an integer from 1 to {number_of_floors}."
            ) from error
        if floor != room.floor or floor < 1 or floor > number_of_floors:
            raise ValueError(
                f"{room.name} floor must be between 1 and {number_of_floors}."
            )
        room_indices[room.name] = index - 1
        room_floors[room.name] = floor
        normalized_rooms.append(Room(room.name, room.width, room.length, floor))
        dimensions.append(
            (
                _to_grid_units(room.width, f"{room.name} width"),
                _to_grid_units(room.length, f"{room.name} length"),
            )
        )

    validated_relationships: list[RoomRelationship] = []
    for index, relationship in enumerate(relationships, start=1):
        kind = relationship.kind.strip().upper()
        if kind not in RELATIONSHIP_KINDS:
            allowed = ", ".join(sorted(RELATIONSHIP_KINDS))
            raise ValueError(f"Relationship {index} must be one of: {allowed}.")
        if relationship.room_a == relationship.room_b:
            raise ValueError(f"Relationship {index} must refer to two different rooms.")
        if relationship.room_a not in room_indices or relationship.room_b not in room_indices:
            raise ValueError(
                f"Relationship {index} refers to a room that is not in the room list."
            )
        first_floor = room_floors[relationship.room_a]
        second_floor = room_floors[relationship.room_b]
        if kind == "STACKED":
            if abs(first_floor - second_floor) != 1:
                raise ValueError(
                    f"Relationship {index} STACKED must connect rooms on adjacent floors."
                )
        elif first_floor != second_floor:
            raise ValueError(
                f"Relationship {index} {kind} requires both rooms to be on the same floor."
            )
        validated_relationships.append(
            RoomRelationship(relationship.room_a, relationship.room_b, kind)
        )

    parking_dimensions: tuple[int, int] | None = None
    if parking_type != "No parking":
        default_width, default_length = DEFAULT_PARKING_DIMENSIONS[parking_type]
        parking_dimensions = (
            _to_grid_units(
                default_width if parking_width is None else parking_width,
                "Parking width",
            ),
            _to_grid_units(
                default_length if parking_length is None else parking_length,
                "Parking length",
            ),
        )

    staircase_dimensions: tuple[int, int] | None = None
    # A requested staircase is also placed on a single-floor plan so the
    # test-fit can represent roof/terrace/future-floor vertical circulation.
    if staircase_width > 0 and staircase_length > 0:
        staircase_dimensions = (
            _to_grid_units(staircase_width, "Staircase width"),
            _to_grid_units(staircase_length, "Staircase length"),
        )

    buildable_width_units = buildable_bounds_units[2] - buildable_bounds_units[0]
    buildable_length_units = buildable_bounds_units[3] - buildable_bounds_units[1]
    oversized_room = next(
        (
            normalized_rooms[index].name
            for index, (width, length) in enumerate(dimensions)
            if width > buildable_width_units or length > buildable_length_units
        ),
        None,
    )
    if oversized_room is not None:
        return LayoutResult(
            status="INFEASIBLE",
            placements=[],
            message=f"{oversized_room} is larger than the buildable area after setbacks.",
            number_of_floors=number_of_floors,
            parking_type=parking_type,
            road_access=normalized_road,
        )
    if staircase_dimensions is not None and (
        staircase_dimensions[0] > buildable_width_units
        or staircase_dimensions[1] > buildable_length_units
    ):
        return LayoutResult(
            status="INFEASIBLE",
            placements=[],
            message="The staircase is larger than the buildable area after setbacks.",
            number_of_floors=number_of_floors,
            parking_type=parking_type,
            road_access=normalized_road,
        )

    return _solve_floor_aware_model(
        site_width_units=site_width_units,
        site_length_units=site_length_units,
        rooms=normalized_rooms,
        dimensions=dimensions,
        room_indices=room_indices,
        time_limit_seconds=time_limit_seconds,
        relationships=validated_relationships,
        number_of_floors=number_of_floors,
        parking_type=parking_type,
        parking_dimensions=parking_dimensions,
        road_access=normalized_road,
        staircase_dimensions=staircase_dimensions,
        buildable_bounds_units=buildable_bounds_units,
        staircase_mode=staircase_mode,
    )


def _solve_floor_aware_model(
    *,
    site_width_units: int,
    site_length_units: int,
    rooms: Sequence[Room],
    dimensions: Sequence[tuple[int, int]],
    room_indices: dict[str, int],
    time_limit_seconds: float,
    relationships: Sequence[RoomRelationship],
    number_of_floors: int,
    parking_type: str,
    parking_dimensions: tuple[int, int] | None,
    road_access: str,
    staircase_dimensions: tuple[int, int] | None,
    buildable_bounds_units: tuple[int, int, int, int],
    staircase_mode: str = "internal",
) -> LayoutResult:
    # Keep floor lookup local to this solver function. The outer public
    # entry point has its own room_floors dictionary, which is not in scope here.
    room_floors = {room.name: room.floor for room in rooms}

    model = cp_model.CpModel()
    x_positions: list[cp_model.IntVar] = []
    y_positions: list[cp_model.IntVar] = []
    x_intervals = []
    y_intervals = []
    floor_room_indices = {floor: [] for floor in range(1, number_of_floors + 1)}
    buildable_x_min, buildable_y_min, buildable_x_max, buildable_y_max = (
        buildable_bounds_units
    )

    for index, room in enumerate(rooms):
        width_units, length_units = dimensions[index]
        x = model.new_int_var(
            buildable_x_min, buildable_x_max - width_units, f"room_{index}_x"
        )
        y = model.new_int_var(
            buildable_y_min, buildable_y_max - length_units, f"room_{index}_y"
        )
        model.add(x + width_units <= buildable_x_max)
        model.add(y + length_units <= buildable_y_max)
        x_positions.append(x)
        y_positions.append(y)
        x_intervals.append(
            model.new_fixed_size_interval_var(x, width_units, f"room_{index}_x_interval")
        )
        y_intervals.append(
            model.new_fixed_size_interval_var(y, length_units, f"room_{index}_y_interval")
        )
        floor_room_indices[room.floor].append(index)

    architectural_anchors = _select_architectural_anchor_indices(rooms)

    # ------------------------------------------------------------------
    # Residential front-to-back sequencing
    # ------------------------------------------------------------------
    # Keep front-to-back zoning SOFT.  A hard public->service->private
    # ordering can contradict legitimate architectural relationships such as
    # an attached master bathroom.  The soft zoning objective below still
    # prefers public spaces near the road, service spaces in the middle and
    # private spaces toward the rear, while allowing attached rooms to share
    # a wall wherever the plan topology requires it.

    vertical_alignment_distances: list[cp_model.IntVar] = []

    # Prefer a consistent vertical circulation stack by aligning the
    # architectural anchor on each floor with the anchor on the floor above.
    # This is a soft objective: different floor plans remain independent, but
    # matching XY positions are preferred when the geometry permits them.
    for lower_floor in range(1, number_of_floors):
        lower_index = architectural_anchors.get(lower_floor)
        upper_index = architectural_anchors.get(lower_floor + 1)
        if lower_index is None or upper_index is None:
            continue

        lower_width, lower_length = dimensions[lower_index]
        upper_width, upper_length = dimensions[upper_index]

        center_dx = model.new_int_var(
            0, 2 * site_width_units, f"floor_stack_{lower_floor}_dx"
        )
        center_dy = model.new_int_var(
            0, 2 * site_length_units, f"floor_stack_{lower_floor}_dy"
        )
        model.add_abs_equality(
            center_dx,
            2 * x_positions[lower_index]
            + lower_width
            - 2 * x_positions[upper_index]
            - upper_width,
        )
        model.add_abs_equality(
            center_dy,
            2 * y_positions[lower_index]
            + lower_length
            - 2 * y_positions[upper_index]
            - upper_length,
        )
        alignment_distance = model.new_int_var(
            0,
            2 * (site_width_units + site_length_units),
            f"floor_stack_{lower_floor}_distance",
        )
        model.add(alignment_distance == center_dx + center_dy)
        vertical_alignment_distances.append(alignment_distance)

    # Put the primary ground-floor entrance/core room on the road-facing
    # buildable edge. This gives the entrance a real facade relationship instead
    # of drawing a door on an arbitrary room after solving.
    ground_anchor_index = architectural_anchors.get(1)
    if ground_anchor_index is not None:
        anchor_width, anchor_length = dimensions[ground_anchor_index]
        anchor_x = x_positions[ground_anchor_index]
        anchor_y = y_positions[ground_anchor_index]
        if road_access == "South":
            model.add(anchor_y == buildable_y_min)
        elif road_access == "North":
            model.add(anchor_y + anchor_length == buildable_y_max)
        elif road_access == "West":
            model.add(anchor_x == buildable_x_min)
        else:
            model.add(anchor_x + anchor_width == buildable_x_max)

    staircase_positions: list[tuple[int, int, cp_model.IntVar, cp_model.IntVar, object, object]] = []
    floor_staircase_indices = {floor: [] for floor in range(1, number_of_floors + 1)}
    if staircase_dimensions is not None:
        stair_width, stair_length = staircase_dimensions
        x = model.new_int_var(
            buildable_x_min, buildable_x_max - stair_width, "staircase_x"
        )
        y = model.new_int_var(
            buildable_y_min, buildable_y_max - stair_length, "staircase_y"
        )
        model.add(x + stair_width <= buildable_x_max)
        model.add(y + stair_length <= buildable_y_max)
        x_interval = model.new_fixed_size_interval_var(
            x, stair_width, "staircase_x_interval"
        )
        y_interval = model.new_fixed_size_interval_var(
            y, stair_length, "staircase_y_interval"
        )
        staircase_positions.append(
            (1, number_of_floors, x, y, x_interval, y_interval)
        )
        for floor in range(1, number_of_floors + 1):
            floor_staircase_indices[floor].append(0)

    for floor in range(1, number_of_floors + 1):
        floor_x_intervals = [
            x_intervals[index] for index in floor_room_indices[floor]
        ]
        floor_y_intervals = [
            y_intervals[index] for index in floor_room_indices[floor]
        ]
        for stair_index in floor_staircase_indices[floor]:
            floor_x_intervals.append(staircase_positions[stair_index][4])
            floor_y_intervals.append(staircase_positions[stair_index][5])
        if len(floor_x_intervals) > 1:
            model.add_no_overlap_2d(floor_x_intervals, floor_y_intervals)

    # A second-floor balcony must be over an uncovered portion of the ground
    # floor, not directly over a room. It is still required to touch its linked
    # upstairs bedroom through the ATTACHED relationship below.
    balcony_index = next(
        (i for i, room in enumerate(rooms)
         if room.floor == 2 and room.name.casefold() == "balcony"),
        None,
    )
    if balcony_index is not None:
        for ground_index, ground_room in enumerate(rooms):
            if ground_room.floor != 1:
                continue
            model.add_no_overlap_2d(
                [x_intervals[balcony_index], x_intervals[ground_index]],
                [y_intervals[balcony_index], y_intervals[ground_index]],
            )

    # Build one connected room graph on every floor.
    # Every room must share a real wall with at least one other room, and every
    # room after the architectural anchor must touch an earlier room in the
    # ordered graph.  The second rule is what prevents two separate clusters
    # with a large empty strip between them.
    for floor in range(1, number_of_floors + 1):
        floor_indices = [i for i, room in enumerate(rooms) if room.floor == floor]
        if len(floor_indices) > 1:
            anchor = architectural_anchors.get(floor)
            if anchor in floor_indices:
                floor_indices = [anchor] + [i for i in floor_indices if i != anchor]

            touch_options = {i: [] for i in floor_indices}
            touch_pairs = {}
            for pos_a, first_index in enumerate(floor_indices):
                for second_index in floor_indices[pos_a + 1:]:
                    sides = _add_optional_edge_touch_constraint(
                        model,
                        f"connect_f{floor}_{first_index}_{second_index}",
                        first_index, second_index,
                        *dimensions[first_index], *dimensions[second_index],
                        x_positions, y_positions,
                    )
                    touch = model.new_bool_var(
                        f"connect_f{floor}_{first_index}_{second_index}_touch"
                    )
                    model.add_max_equality(touch, sides)
                    touch_options[first_index].append(touch)
                    touch_options[second_index].append(touch)
                    touch_pairs[(first_index, second_index)] = touch

            # No isolated room.
            for room_index, options in touch_options.items():
                if options:
                    model.add(sum(options) >= 1)

            # No disconnected sub-cluster: each room after the anchor must
            # touch at least one room that appeared earlier in the chain.
            for pos, room_index in enumerate(floor_indices[1:], start=1):
                previous = [
                    touch_pairs[(prior, room_index)]
                    if (prior, room_index) in touch_pairs
                    else touch_pairs[(room_index, prior)]
                    for prior in floor_indices[:pos]
                ]
                model.add(sum(previous) >= 1)

    # Staircase placement depends on building type. Internal stairs connect
    # to the living room and upper-floor family lounge; external stairs remain
    # outside the room cluster while staying inside the site/buildable area.
    if staircase_positions and staircase_dimensions:
        stair_width, stair_length = staircase_dimensions
        stair_x = staircase_positions[0][2]
        stair_y = staircase_positions[0][3]
        for floor in range(1, number_of_floors + 1):
            floor_room_indices_here = [
                i for i, room in enumerate(rooms) if room.floor == floor
            ]
            touch_vars = {}
            for room_index in floor_room_indices_here:
                sides = _add_optional_edge_touch_constraint(
                    model,
                    f"stair_connect_f{floor}_{room_index}",
                    room_index, len(rooms),
                    dimensions[room_index][0], dimensions[room_index][1],
                    stair_width, stair_length,
                    x_positions + [stair_x], y_positions + [stair_y],
                )
                touch = model.new_bool_var(f"stair_connect_f{floor}_{room_index}_touch")
                model.add_max_equality(touch, sides)
                touch_vars[room_index] = touch
            if staircase_mode == "external":
                if touch_vars:
                    model.add(sum(touch_vars.values()) == 0)
            else:
                preferred_name = "Living Room" if floor == 1 else "Upper Living Room"
                preferred_index = next(
                    (i for i in floor_room_indices_here
                     if rooms[i].name.casefold() == preferred_name.casefold()),
                    None,
                )
                if preferred_index is None and floor == 1:
                    preferred_index = next(
                        (i for i in floor_room_indices_here
                         if any(k in rooms[i].name.casefold() for k in ("living", "lounge", "family"))),
                        None,
                    )
                if preferred_index is not None:
                    model.add(touch_vars[preferred_index] == 1)
                if len(touch_vars) > 1:
                    model.add(sum(touch_vars.values()) >= 2)

    parking_vars: tuple[cp_model.IntVar, cp_model.IntVar, object, object] | None = None
    parking_access_distance: cp_model.IntVar | None = None
    if parking_dimensions is not None:
        parking_width_units, parking_length_units = parking_dimensions
        # Parking may occupy the site setback/front yard so vehicle access can
        # connect directly to the road-facing gate. Rooms and staircase remain
        # constrained to the buildable bounds.
        parking_x = model.new_int_var(
            0, site_width_units - parking_width_units, "parking_x"
        )
        parking_y = model.new_int_var(
            0, site_length_units - parking_length_units, "parking_y"
        )
        model.add(parking_x + parking_width_units <= site_width_units)
        model.add(parking_y + parking_length_units <= site_length_units)
        parking_x_interval = model.new_fixed_size_interval_var(
            parking_x, parking_width_units, "parking_x_interval"
        )
        parking_y_interval = model.new_fixed_size_interval_var(
            parking_y, parking_length_units, "parking_y_interval"
        )
        parking_vars = (
            parking_x,
            parking_y,
            parking_x_interval,
            parking_y_interval,
        )

        for index, room in enumerate(rooms):
            if room.floor != 1:
                continue
            model.add_no_overlap_2d(
                [parking_x_interval, x_intervals[index]],
                [parking_y_interval, y_intervals[index]],
            )
        for _, _, _, _, stair_x_interval, stair_y_interval in staircase_positions:
            model.add_no_overlap_2d(
                [parking_x_interval, stair_x_interval],
                [parking_y_interval, stair_y_interval],
            )

        # Put parking beside the entrance foyer as well as near the road-facing gate.
        foyer_index = next(
            (i for i, room in enumerate(rooms)
             if room.floor == 1 and "foyer" in room.name.casefold()),
            None,
        )
        if foyer_index is not None:
            parking_touch_sides = _add_optional_edge_touch_constraint(
                model,
                f"parking_entry_foyer_touch_{foyer_index}",
                foyer_index, len(rooms),
                dimensions[foyer_index][0], dimensions[foyer_index][1],
                parking_width_units, parking_length_units,
                x_positions + [parking_x], y_positions + [parking_y],
            )
            model.add(sum(parking_touch_sides) >= 1)

        if road_access in {"South", "North"}:
            parking_access_distance = model.new_int_var(
                0, site_length_units, "parking_access_distance"
            )
        else:
            parking_access_distance = model.new_int_var(
                0, site_width_units, "parking_access_distance"
            )
        if road_access == "South":
            model.add(parking_access_distance == parking_y)
        elif road_access == "North":
            model.add(
                parking_access_distance
                == site_length_units - parking_y - parking_length_units
            )
        elif road_access == "West":
            model.add(parking_access_distance == parking_x)
        else:
            model.add(
                parking_access_distance
                == site_width_units - parking_x - parking_width_units
            )

    near_distances: list[cp_model.IntVar] = []
    preferred_penalties: list[cp_model.IntVar] = []
    stacked_distances_half_units: list[cp_model.IntVar] = []
    internal_gap_distances: list[cp_model.IntVar] = []
    staircase_anchor_distances: list[cp_model.IntVar] = []
    architectural_zone_penalties: list[cp_model.IntVar] = []
    # Architectural zoning: public spaces stay toward the road/front, service
    # rooms occupy the middle, and bedrooms/family spaces move toward the
    # private rear. This fixes the old failure mode where the first bounding-box
    # objective simply packed every rectangle into one arbitrary cluster.
    for index, room in enumerate(rooms):
        architectural_zone_penalties.append(
            _add_zone_penalty(
                model,
                index,
                room.name,
                dimensions[index][0],
                dimensions[index][1],
                x_positions[index],
                y_positions[index],
                buildable_x_min,
                buildable_y_min,
                buildable_x_max,
                buildable_y_max,
                road_access,
            )
        )

    # Penalize unnecessary empty space between rooms on the same floor.
    # This is deliberately a soft objective: circulation remains possible, but
    # isolated rectangles are discouraged.
    for first_index in range(len(rooms)):
        for second_index in range(first_index + 1, len(rooms)):
            if rooms[first_index].floor != rooms[second_index].floor:
                continue
            first_width, first_length = dimensions[first_index]
            second_width, second_length = dimensions[second_index]
            horizontal_gap = model.new_int_var(
                0, site_width_units, f"cluster_{first_index}_{second_index}_dx"
            )
            vertical_gap = model.new_int_var(
                0, site_length_units, f"cluster_{first_index}_{second_index}_dy"
            )
            model.add_max_equality(
                horizontal_gap,
                [
                    0,
                    x_positions[first_index]
                    - (x_positions[second_index] + second_width),
                    x_positions[second_index]
                    - (x_positions[first_index] + first_width),
                ],
            )
            model.add_max_equality(
                vertical_gap,
                [
                    0,
                    y_positions[first_index]
                    - (y_positions[second_index] + second_length),
                    y_positions[second_index]
                    - (y_positions[first_index] + first_length),
                ],
            )
            pair_gap = model.new_int_var(
                0, site_width_units + site_length_units,
                f"cluster_{first_index}_{second_index}_gap",
            )
            model.add(pair_gap == horizontal_gap + vertical_gap)
            internal_gap_distances.append(pair_gap)

    # Keep the staircase close to the architectural circulation/core anchors on
    # every served floor. The same staircase shaft is used on all floors.
    if staircase_positions and architectural_anchors:
        for _, _, stair_x, stair_y, _, _ in staircase_positions:
            stair_width, stair_length = staircase_dimensions or (0, 0)
            for floor, anchor_index in architectural_anchors.items():
                if floor > number_of_floors:
                    continue
                anchor_width, anchor_length = dimensions[anchor_index]
                center_dx = model.new_int_var(
                    0, 2 * site_width_units, f"stair_anchor_{floor}_dx"
                )
                center_dy = model.new_int_var(
                    0, 2 * site_length_units, f"stair_anchor_{floor}_dy"
                )
                model.add_abs_equality(
                    center_dx,
                    2 * stair_x + stair_width
                    - 2 * x_positions[anchor_index] - anchor_width,
                )
                model.add_abs_equality(
                    center_dy,
                    2 * stair_y + stair_length
                    - 2 * y_positions[anchor_index] - anchor_length,
                )
                distance = model.new_int_var(
                    0, 2 * (site_width_units + site_length_units),
                    f"stair_anchor_{floor}_distance",
                )
                model.add(distance == center_dx + center_dy)
                staircase_anchor_distances.append(distance)

    # The ground-floor ensuite must not share a wall with Dining. This avoids
    # a bathroom opening directly into the dining area while still allowing it
    # to attach to Bedroom 1. A one-grid-unit separation is enough to prevent
    # accidental wall contact; the room-graph constraints keep the overall plan
    # connected through the bedroom/living circulation path.
    bedroom_bath_index = room_indices.get("Bathroom 1")
    dining_index = room_indices.get("Dining")
    if (
        bedroom_bath_index is not None
        and dining_index is not None
        and room_floors.get("Bathroom 1") == 1
        and room_floors.get("Dining") == 1
    ):
        bath_w, bath_h = dimensions[bedroom_bath_index]
        dining_w, dining_h = dimensions[dining_index]
        bath_left = model.new_bool_var("bath1_left_of_dining")
        dining_left = model.new_bool_var("dining_left_of_bath1")
        bath_below = model.new_bool_var("bath1_below_dining")
        dining_below = model.new_bool_var("dining_below_bath1")
        model.add(
            x_positions[bedroom_bath_index] + bath_w + 1
            <= x_positions[dining_index]
        ).only_enforce_if(bath_left)
        model.add(
            x_positions[dining_index] + dining_w + 1
            <= x_positions[bedroom_bath_index]
        ).only_enforce_if(dining_left)
        model.add(
            y_positions[bedroom_bath_index] + bath_h + 1
            <= y_positions[dining_index]
        ).only_enforce_if(bath_below)
        model.add(
            y_positions[dining_index] + dining_h + 1
            <= y_positions[bedroom_bath_index]
        ).only_enforce_if(dining_below)
        model.add_bool_or([bath_left, dining_left, bath_below, dining_below])

    for relationship_index, relationship in enumerate(relationships):
        first_index = room_indices[relationship.room_a]
        second_index = room_indices[relationship.room_b]
        first_width, first_length = dimensions[first_index]
        second_width, second_length = dimensions[second_index]

        if relationship.kind in HARD_RELATIONSHIP_KINDS:
            pair_names = {rooms[first_index].name.casefold(), rooms[second_index].name.casefold()}
            if relationship.kind == "ATTACHED" or (
                relationship.kind == "ADJACENT" and pair_names == {"kitchen", "utility"}
            ):
                # Reserve enough shared wall for a visible, usable conceptual door.
                minimum_shared_edge_units = math.ceil(0.80 * GRID_UNITS_PER_METER)
            else:
                minimum_shared_edge_units = MINIMUM_POSITIVE_SHARED_EDGE_UNITS
            _add_hard_edge_touch_constraint(
                model,
                relationship_index,
                first_index,
                second_index,
                first_width,
                first_length,
                second_width,
                second_length,
                x_positions,
                y_positions,
                minimum_shared_edge_units,
            )
            continue

        if relationship.kind == "STACKED":
            # For the duplex living-room pair, stacking means the upper living
            # room must occupy exactly the same XY footprint as the ground-floor
            # living room, rather than merely being nearby. Their dimensions
            # are intentionally identical in the room program.
            stacked_names = {
                rooms[first_index].name.casefold(),
                rooms[second_index].name.casefold(),
            }
            if stacked_names == {"living room", "upper living room"}:
                # Exact footprint stacking is preferred when the room sizes match.
                # For a compact/adaptive layout, allow different dimensions and let
                # the center-alignment objective below guide the upper room instead
                # of aborting the entire generation with a ValueError.
                if first_width == second_width and first_length == second_length:
                    model.add(x_positions[first_index] == x_positions[second_index])
                    model.add(y_positions[first_index] == y_positions[second_index])

            center_dx = model.new_int_var(
                0, 2 * site_width_units, f"rel_{relationship_index}_center_dx"
            )
            center_dy = model.new_int_var(
                0, 2 * site_length_units, f"rel_{relationship_index}_center_dy"
            )
            model.add_abs_equality(
                center_dx,
                2 * x_positions[first_index]
                + first_width
                - 2 * x_positions[second_index]
                - second_width,
            )
            model.add_abs_equality(
                center_dy,
                2 * y_positions[first_index]
                + first_length
                - 2 * y_positions[second_index]
                - second_length,
            )
            stacked_distance = model.new_int_var(
                0,
                2 * (site_width_units + site_length_units),
                f"rel_{relationship_index}_stacked_distance",
            )
            model.add(stacked_distance == center_dx + center_dy)
            stacked_distances_half_units.append(stacked_distance)
            continue

        max_distance = site_width_units + site_length_units
        horizontal_gap = model.new_int_var(
            0, site_width_units, f"rel_{relationship_index}_horizontal_gap"
        )
        vertical_gap = model.new_int_var(
            0, site_length_units, f"rel_{relationship_index}_vertical_gap"
        )
        model.add_max_equality(
            horizontal_gap,
            [
                0,
                x_positions[first_index]
                - (x_positions[second_index] + second_width),
                x_positions[second_index]
                - (x_positions[first_index] + first_width),
            ],
        )
        model.add_max_equality(
            vertical_gap,
            [
                0,
                y_positions[first_index]
                - (y_positions[second_index] + second_length),
                y_positions[second_index]
                - (y_positions[first_index] + first_length),
            ],
        )
        distance = model.new_int_var(
            0, max_distance, f"rel_{relationship_index}_distance"
        )
        model.add(distance == horizontal_gap + vertical_gap)
        if relationship.kind == "NEAR":
            near_distances.append(distance)
        else:
            preferred_penalty = model.new_int_var(
                0,
                max(0, max_distance - SOFT_RELATIONSHIP_DISTANCE_THRESHOLD_UNITS),
                f"rel_{relationship_index}_preferred_penalty",
            )
            model.add_max_equality(
                preferred_penalty,
                [0, distance - SOFT_RELATIONSHIP_DISTANCE_THRESHOLD_UNITS],
            )
            preferred_penalties.append(preferred_penalty)

    bbox_area_vars: list[cp_model.IntVar] = []
    bbox_dimension_terms: list[cp_model.IntVar] = []
    for floor in range(1, number_of_floors + 1):
        indices = floor_room_indices[floor]
        if not indices:
            continue
        left = model.new_int_var(0, site_width_units, f"floor_{floor}_left")
        right = model.new_int_var(0, site_width_units, f"floor_{floor}_right")
        bottom = model.new_int_var(0, site_length_units, f"floor_{floor}_bottom")
        top = model.new_int_var(0, site_length_units, f"floor_{floor}_top")
        width = model.new_int_var(0, site_width_units, f"floor_{floor}_bbox_width")
        length = model.new_int_var(0, site_length_units, f"floor_{floor}_bbox_length")
        model.add_min_equality(left, [x_positions[index] for index in indices])
        model.add_max_equality(
            right,
            [x_positions[index] + dimensions[index][0] for index in indices],
        )
        model.add_min_equality(bottom, [y_positions[index] for index in indices])
        model.add_max_equality(
            top,
            [y_positions[index] + dimensions[index][1] for index in indices],
        )
        model.add(width == right - left)
        model.add(length == top - bottom)
        area = model.new_int_var(
            0, site_width_units * site_length_units, f"floor_{floor}_bbox_area"
        )
        model.add_multiplication_equality(area, [width, length])
        bbox_area_vars.append(area)
        bbox_dimension_terms.extend([width, length])

    total_bbox_area = model.new_int_var(
        0,
        site_width_units * site_length_units * max(1, number_of_floors),
        "total_occupied_bbox_area",
    )
    model.add(total_bbox_area == sum(bbox_area_vars) if bbox_area_vars else total_bbox_area == 0)

    # Parking access is a primary requirement, not a late tie-breaker.
    # Optimize the parking location toward the road first, then toward a
    # side edge. This prevents earlier compactness objectives from freezing
    # a rear parking location before parking has been optimized.
    objective_stages = []
    if parking_access_distance is not None:
        objective_stages.append(("parking road-side distance", parking_access_distance))

    if parking_vars is not None:
        parking_x, parking_y = parking_vars[:2]
        # Prefer parking near the centre of the road-facing boundary.
        # This avoids pushing South/North parking into the extreme left corner.
        if road_access in {"South", "North"}:
            parking_side_distance = model.new_int_var(
                0, site_width_units, "parking_side_distance"
            )
            parking_center_twice = 2 * parking_x + parking_width_units
            model.add_abs_equality(
                parking_side_distance,
                parking_center_twice - site_width_units,
            )
        else:
            parking_side_distance = model.new_int_var(
                0, site_length_units, "parking_side_distance"
            )
            parking_center_twice = 2 * parking_y + parking_length_units
            model.add_abs_equality(
                parking_side_distance,
                parking_center_twice - site_length_units,
            )

        objective_stages.append(
            ("parking frontage centering", parking_side_distance)
        )

    # After road-side parking, reduce gaps between same-floor rooms before
    # optimizing the overall bounding box.
    if internal_gap_distances:
        objective_stages.append(
            ("residential cluster gaps", sum(internal_gap_distances))
        )
    objective_stages.append(("occupied bounding-box area", total_bbox_area))
    if bbox_dimension_terms:
        objective_stages.append(
            ("occupied bounding-box dimensions", sum(bbox_dimension_terms))
        )
    if architectural_zone_penalties:
        objective_stages.append(("architectural public-private zoning", sum(architectural_zone_penalties)))
    if stacked_distances_half_units:
        objective_stages.append(
            ("STACKED center distance", sum(stacked_distances_half_units))
        )
    if vertical_alignment_distances:
        objective_stages.append(
            ("vertical anchor alignment", sum(vertical_alignment_distances))
        )
    if staircase_anchor_distances:
        objective_stages.append(("staircase to circulation core", sum(staircase_anchor_distances)))
    if near_distances:
        objective_stages.append(("NEAR distance", sum(near_distances)))
    if preferred_penalties:
        objective_stages.append(("PREFERRED penalty", sum(preferred_penalties)))
    # Access-aware tie-breaker: place the entrance/core toward the road instead
    # of blindly minimizing x+y.
    if road_access == "South":
        access_terms = [y_positions[i] for i in range(len(rooms))]
    elif road_access == "North":
        access_terms = [site_length_units - y_positions[i] - dimensions[i][1] for i in range(len(rooms))]
    elif road_access == "West":
        access_terms = [x_positions[i] for i in range(len(rooms))]
    else:
        access_terms = [site_width_units - x_positions[i] - dimensions[i][0] for i in range(len(rooms))]
    objective_stages.append(("road-side access placement", sum(access_terms)))

    start_time = time.monotonic()
    last_solver: cp_model.CpSolver | None = None
    all_stages_optimal = True
    for stage_index, (_, objective) in enumerate(objective_stages):
        remaining_seconds = time_limit_seconds - (time.monotonic() - start_time)
        if remaining_seconds <= 0.01:
            all_stages_optimal = False
            break
        model.minimize(objective)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = remaining_seconds
        solver.parameters.num_search_workers = 1
        status_code = solver.solve(model)
        if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if last_solver is None:
                return LayoutResult(
                    status=solver.status_name(status_code),
                    placements=[],
                    message="CP-SAT did not find a feasible arrangement.",
                    number_of_floors=number_of_floors,
                    parking_type=parking_type,
                    road_access=road_access,
                )
            all_stages_optimal = False
            break
        last_solver = solver
        if status_code != cp_model.OPTIMAL:
            all_stages_optimal = False
            break
        if stage_index < len(objective_stages) - 1:
            model.add(objective == solver.value(objective))

    if last_solver is None:
        return LayoutResult(
            status="UNKNOWN",
            placements=[],
            message="CP-SAT did not return a feasible arrangement within the time limit.",
            number_of_floors=number_of_floors,
            parking_type=parking_type,
            road_access=road_access,
        )

    placements = [
        Placement(
            name=room.name,
            x=last_solver.value(x_positions[index]) / GRID_UNITS_PER_METER,
            y=last_solver.value(y_positions[index]) / GRID_UNITS_PER_METER,
            width=dimensions[index][0] / GRID_UNITS_PER_METER,
            length=dimensions[index][1] / GRID_UNITS_PER_METER,
            floor=room.floor,
        )
        for index, room in enumerate(rooms)
    ]
    staircases = []
    if staircase_dimensions is not None:
        stair_width, stair_length = staircase_dimensions
        for index, (lower_floor, upper_floor, x, y, _, _) in enumerate(
            staircase_positions
        ):
            staircases.append(
                Placement(
                    name=f"Staircase F{lower_floor}-F{upper_floor}",
                    x=last_solver.value(x) / GRID_UNITS_PER_METER,
                    y=last_solver.value(y) / GRID_UNITS_PER_METER,
                    width=stair_width / GRID_UNITS_PER_METER,
                    length=stair_length / GRID_UNITS_PER_METER,
                    floor=lower_floor,
                    floor_end=upper_floor,
                )
            )

    parking = None
    if parking_vars is not None and parking_dimensions is not None:
        parking_width_units, parking_length_units = parking_dimensions
        parking = Placement(
            name=f"Parking ({parking_type})",
            x=last_solver.value(parking_vars[0]) / GRID_UNITS_PER_METER,
            y=last_solver.value(parking_vars[1]) / GRID_UNITS_PER_METER,
            width=parking_width_units / GRID_UNITS_PER_METER,
            length=parking_length_units / GRID_UNITS_PER_METER,
            floor=0,
            floor_end=0,
        )

    status = "OPTIMAL" if all_stages_optimal else "FEASIBLE"
    message = (
        "All objective stages were optimized."
        if status == "OPTIMAL"
        else "A feasible layout was found; some objective stages were not proven optimal before the time limit."
    )
    return LayoutResult(
        status=status,
        placements=placements,
        message=message,
        parking=parking,
        staircases=staircases,
        number_of_floors=number_of_floors,
        parking_type=parking_type,
        road_access=road_access,
    )
