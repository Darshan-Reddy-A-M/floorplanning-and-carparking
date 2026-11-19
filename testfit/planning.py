"""Unit conversion and knowledge-guided preliminary room-program recommendations."""

from __future__ import annotations

import math
from collections.abc import Sequence

from testfit.optimizer import Placement, Room, RoomRelationship
from testfit.knowledge import load_architectural_knowledge


FEET_PER_METER = 1 / 0.3048
SQUARE_FEET_PER_SQUARE_METER = FEET_PER_METER**2

STAIRCASE_DEFAULT_FEET = (6.5, 13.0)

PARKING_DEFAULT_FEET = {
    "1 car": (9.0, 18.0),
    "2 cars": (18.0, 18.0),
}

BUILDING_TYPES = (
    "Single-Family House",
    "Duplex",
    "Rental / Multi-Unit Building",
    "Apartment / Multi-Unit Residential",
    "Custom",
)


def feet_to_meters(value: float) -> float:
    """Convert a finite distance in feet to meters."""
    value = float(value)

    if not math.isfinite(value):
        raise ValueError("Distance in feet must be finite.")

    return value * 0.3048


def meters_to_feet(value: float) -> float:
    """Convert a finite distance in meters to feet."""
    value = float(value)

    if not math.isfinite(value):
        raise ValueError("Distance in meters must be finite.")

    return value / 0.3048


def feet_area_to_square_meters(value: float) -> float:
    """Convert square feet to square meters."""
    return float(value) / SQUARE_FEET_PER_SQUARE_METER


def square_meters_to_feet(value: float) -> float:
    """Convert square meters to square feet."""
    return float(value) * SQUARE_FEET_PER_SQUARE_METER


def floor_label(floor: int) -> str:
    """Return a human-readable floor label."""
    return {
        1: "Ground Floor",
        2: "First Floor",
        3: "Second Floor",
        4: "Third Floor",
        5: "Fourth Floor",
    }.get(floor, f"Floor {floor}")


def select_entrance_room(
    placements: Sequence[Placement],
    side: str,
) -> Placement | None:
    """Choose the most appropriate ground-floor room for the main entrance.

    The requested exterior frontage is the primary criterion.

    For North/South frontage, rooms are ranked using their horizontal
    exterior edge. For East/West frontage, rooms are ranked using their
    vertical exterior edge.

    Room-name keywords are used only as tie-breakers.
    Bathrooms are excluded.
    """

    candidates = [
        room
        for room in placements
        if room.floor == 1
        and "bathroom" not in room.name.casefold()
    ]

    if not candidates:
        return None

    side = side.title()

    if side not in {"North", "South", "East", "West"}:
        raise ValueError(
            "Entrance side must be North, South, East, or West."
        )

    keyword_priority = (
        "entry foyer",
        "foyer",
        "entrance",
        "living",
        "dining",
        "kitchen",
    )

    north_edge = max(
        room.y + room.length
        for room in candidates
    )

    south_edge = min(
        room.y
        for room in candidates
    )

    east_edge = max(
        room.x + room.width
        for room in candidates
    )

    west_edge = min(
        room.x
        for room in candidates
    )

    def frontage_gap(room: Placement) -> float:
        """Distance between a room and the requested exterior edge."""

        if side == "North":
            return max(
                0.0,
                north_edge - (room.y + room.length),
            )

        if side == "South":
            return max(
                0.0,
                room.y - south_edge,
            )

        if side == "East":
            return max(
                0.0,
                east_edge - (room.x + room.width),
            )

        return max(
            0.0,
            room.x - west_edge,
        )

    def keyword_rank(room: Placement) -> int:
        """Use room name only as a secondary preference."""

        name = room.name.casefold()

        for rank, keyword in enumerate(keyword_priority):
            if keyword in name:
                return rank

        return len(keyword_priority)

    return min(
        candidates,
        key=lambda room: (
            frontage_gap(room),
            keyword_rank(room),
            room.name.casefold(),
        ),
    )


def setback_bounds_feet(
    site_width: float,
    site_length: float,
    road_access: str,
    front: float,
    rear: float,
    left: float,
    right: float,
) -> tuple[float, float, float, float]:
    """Return the buildable rectangle as x-min, y-min, x-max, y-max in feet.

    Left and right setbacks are interpreted while facing the site from the road.
    """

    front_side = road_access.title()

    if front_side not in {
        "North",
        "South",
        "East",
        "West",
    }:
        raise ValueError(
            "Road access must be North, South, East, or West."
        )

    side_setbacks = {
        "North": {
            "front": "North",
            "rear": "South",
            "left": "East",
            "right": "West",
        },
        "South": {
            "front": "South",
            "rear": "North",
            "left": "West",
            "right": "East",
        },
        "East": {
            "front": "East",
            "rear": "West",
            "left": "South",
            "right": "North",
        },
        "West": {
            "front": "West",
            "rear": "East",
            "left": "North",
            "right": "South",
        },
    }[front_side]

    values = {
        "front": front,
        "rear": rear,
        "left": left,
        "right": right,
    }

    side_values = {
        side: float(values[name])
        for name, side in side_setbacks.items()
    }

    if any(
        not math.isfinite(value) or value < 0
        for value in side_values.values()
    ):
        raise ValueError(
            "Setbacks must be finite values greater than or equal to zero."
        )

    x_min = side_values["West"]
    y_min = side_values["South"]
    x_max = float(site_width) - side_values["East"]
    y_max = float(site_length) - side_values["North"]

    if x_min >= x_max or y_min >= y_max:
        raise ValueError(
            "Setbacks must leave a positive buildable area inside the site."
        )

    return x_min, y_min, x_max, y_max


def recommend_room_program(
    *,
    buildable_bounds_ft: tuple[float, float, float, float],
    number_of_floors: int,
    building_type: str,
    units_per_floor: int = 1,
    parking_area_ft2: float = 0.0,
    staircase_size_ft: tuple[float, float] = STAIRCASE_DEFAULT_FEET,
) -> tuple[list[Room], list[RoomRelationship], float]:
    """Build a knowledge-guided residential room program.

    ResPlan-derived architectural knowledge is used as a planning prior.
    Site area and building type remain the primary feasibility controls.
    """

    knowledge = load_architectural_knowledge()

    x_min, y_min, x_max, y_max = buildable_bounds_ft

    floor_area_ft2 = (
        (x_max - x_min)
        * (y_max - y_min)
    )

    staircase_area_ft2 = (
        staircase_size_ft[0] * staircase_size_ft[1] * number_of_floors
        if staircase_size_ft[0] > 0 and staircase_size_ft[1] > 0
        else 0.0
    )

    usable_total_ft2 = max(
        0.0,
        (
            floor_area_ft2 * number_of_floors
            - max(0.0, parking_area_ft2)
            - staircase_area_ft2
        )
        * 0.82,
    )

    rooms_ft: list[tuple[str, int, float, float]] = []
    relationships: list[RoomRelationship] = []

    # ============================================================
    # RENTAL / APARTMENT
    # ============================================================
    if building_type in {
        "Rental / Multi-Unit Building",
        "Apartment / Multi-Unit Residential",
    }:
        for floor in range(1, number_of_floors + 1):
            for unit in range(1, units_per_floor + 1):
                prefix = f"Floor {floor} - Unit {unit}"

                living = f"{prefix} Living"
                bedroom = f"{prefix} Bedroom"
                kitchen = f"{prefix} Kitchen"
                bathroom = f"{prefix} Bathroom"

                rooms_ft.extend(
                    [
                        (living, floor, 10, 12),
                        (bedroom, floor, 10, 11),
                        (kitchen, floor, 7, 8),
                        (bathroom, floor, 5, 8),
                    ]
                )

                relationships.extend(
                    [
                        RoomRelationship(
                            bedroom,
                            bathroom,
                            "ATTACHED",
                        ),
                        RoomRelationship(
                            living,
                            kitchen,
                            "NEAR",
                        ),
                    ]
                )

        return (
            [
                Room(
                    name,
                    feet_to_meters(width),
                    feet_to_meters(length),
                    floor,
                )
                for name, floor, width, length in rooms_ft
            ],
            relationships,
            floor_area_ft2,
        )

    # ============================================================
    # DUPLEX (two-storey single-family home)
    # ============================================================
    # A duplex here means a two-storey family residence, not a rental
    # building with independent dwelling units.  Keep the program compact
    # enough to leave real circulation space for a 30–50 ft buildable depth.
    # The ResPlan statistics are still used as priors elsewhere, but they
    # should not force five bedrooms into a typical 40×60 ft family home.
    # The living room is widened to absorb the marked side strip; the upper
    # living room uses the same width so the two floors remain stacked.
    if building_type == "Duplex" and number_of_floors == 2:
        rooms_ft = [
            # Ground floor: public spaces form a connected chain, while the
            # bedroom ensuite is attached only to Bedroom 1 (not to Dining).
            ("Entry Foyer", 1, 6, 7),
            ("Living Room", 1, 17.5, 16),
            ("Dining", 1, 9, 10),
            ("Kitchen", 1, 10, 11),
            ("Utility", 1, 6, 7),
            ("Bedroom 1", 1, 12, 12),
            ("Bathroom 1", 1, 5, 8),
            # Upper living room has the same dimensions and exact XY footprint
            # as the ground-floor living room. The optimizer enforces stacking.
            ("Upper Living Room", 2, 17.5, 16),
            ("Bedroom 4", 2, 12, 13),
            ("Bedroom 2", 2, 11, 12),
            ("Bedroom 3", 2, 11, 12),
            ("Bathroom 2", 2, 5, 8),  # Bedroom 2 ensuite
            ("Bathroom 3", 2, 5, 8),  # Bedroom 3 ensuite
            ("Bathroom 4", 2, 5, 8),  # Bedroom 4 ensuite
            ("Balcony", 2, 6, 5),
        ]
        relationships = [
            # Ground-floor room connections.
            RoomRelationship("Entry Foyer", "Living Room", "ADJACENT"),
            RoomRelationship("Living Room", "Dining", "ADJACENT"),
            RoomRelationship("Dining", "Kitchen", "ADJACENT"),
            RoomRelationship("Kitchen", "Utility", "ADJACENT"),
            RoomRelationship("Dining", "Utility", "ADJACENT"),
            RoomRelationship("Living Room", "Bedroom 1", "ADJACENT"),
            RoomRelationship("Bedroom 1", "Bathroom 1", "ATTACHED"),
            # Put the upper living room directly over the ground-floor living
            # room and connect the upper bedrooms to it.
            RoomRelationship("Living Room", "Upper Living Room", "STACKED"),
            RoomRelationship("Upper Living Room", "Bedroom 4", "ADJACENT"),
            RoomRelationship("Upper Living Room", "Bedroom 2", "ADJACENT"),
            RoomRelationship("Upper Living Room", "Bedroom 3", "ADJACENT"),
            # Each bedroom has its own attached bathroom.
            RoomRelationship("Bedroom 2", "Bathroom 2", "ATTACHED"),
            RoomRelationship("Bedroom 3", "Bathroom 3", "ATTACHED"),
            RoomRelationship("Bedroom 4", "Bathroom 4", "ATTACHED"),
            RoomRelationship("Bedroom 4", "Balcony", "ATTACHED"),
        ]
        return (
            [
                Room(name, feet_to_meters(width), feet_to_meters(length), floor)
                for name, floor, width, length in rooms_ft
            ],
            relationships,
            floor_area_ft2,
        )

    # ============================================================
    # RESPLAN ROOM COUNT PRIORS
    # ============================================================
    preferred_bedrooms = knowledge.preferred_count("bedroom")
    minimum_bedrooms = knowledge.minimum_count("bedroom")
    maximum_bedrooms = knowledge.maximum_count("bedroom")

    preferred_bathrooms = knowledge.preferred_count("bathroom")
    minimum_bathrooms = knowledge.minimum_count("bathroom")
    maximum_bathrooms = knowledge.maximum_count("bathroom")

    # Calculate an area-based bedroom count first.
    #
    # This prevents the dataset preference from blindly forcing two
    # bedrooms onto an extremely small site.
    area_based_bedrooms = max(
        1,
        min(
            5,
            int(usable_total_ft2 // 450),
        ),
    )

    # For multi-floor buildings, allow total usable area to influence
    # bedroom count. For a normal single-floor plan, retain the
    # ResPlan-derived preferred count when the site can support it.
    if number_of_floors > 1:
        bedrooms = area_based_bedrooms
    elif area_based_bedrooms >= preferred_bedrooms:
        bedrooms = preferred_bedrooms
    else:
        bedrooms = area_based_bedrooms

    bedrooms = max(
        minimum_bedrooms,
        min(
            maximum_bedrooms,
            bedrooms,
        ),
    )

    # ============================================================
    # BATHROOM COUNT
    # ============================================================
    area_based_bathrooms = max(
        1,
        math.ceil(bedrooms / 2),
    )

    if bedrooms >= preferred_bedrooms:
        bathrooms = preferred_bathrooms
    else:
        bathrooms = area_based_bathrooms

    bathrooms = max(
        minimum_bathrooms,
        min(
            maximum_bathrooms,
            bathrooms,
        ),
    )

    bathrooms = min(
        bathrooms,
        bedrooms,
    )

    # ============================================================
    # CORE GROUND-FLOOR ROOMS
    # ============================================================
    rooms_ft.extend(
        [
            ("Entry Foyer", 1, 6, 7),
            ("Living Room", 1, 17.5, 16),
            ("Kitchen", 1, 10, 11),
            ("Dining", 1, 9, 10),
        ]
    )

    if usable_total_ft2 >= 850:
        rooms_ft.append(
            ("Utility", 1, 6, 7)
        )

    # ============================================================
    # BEDROOM DISTRIBUTION
    # ============================================================
    bedroom_floors: list[int] = []

    for index in range(bedrooms):
        if number_of_floors == 1:
            floor = 1

        elif building_type == "Duplex" and index == 0:
            floor = 1

        else:
            floor = 2 + (
                index
                - (1 if building_type == "Duplex" else 0)
            ) % (number_of_floors - 1)

        bedroom_floors.append(floor)

        size = (
            (12, 12)
            if index == 0
            else (11, 12)
        )

        rooms_ft.append(
            (
                f"Bedroom {index + 1}",
                floor,
                *size,
            )
        )

    # ============================================================
    # BATHROOMS
    # ============================================================
    for index in range(bathrooms):
        floor = bedroom_floors[
            min(
                index,
                len(bedroom_floors) - 1,
            )
        ]

        bathroom_name = f"Bathroom {index + 1}"

        rooms_ft.append(
            (
                bathroom_name,
                floor,
                5,
                8,
            )
        )

        relationships.append(
            RoomRelationship(
                f"Bedroom {index + 1}",
                bathroom_name,
                "ATTACHED",
            )
        )

    # ============================================================
    # MULTI-FLOOR FAMILY SPACE
    # ============================================================
    # Upper-floor bedrooms need a same-floor common space for soft
    # architectural relationships. Cross-floor PREFERRED/NEAR relationships
    # are invalid in the floor-aware optimizer.
    family_lounge_floor = None
    if number_of_floors > 1 and bedrooms >= 2:
        family_lounge_floor = 2
        rooms_ft.append(
            (
                "Family Lounge",
                family_lounge_floor,
                11,
                12,
            )
        )

    # ============================================================
    # CORE CONNECTIVITY
    # ============================================================
    relationships.extend(
        [
            RoomRelationship("Entry Foyer", "Living Room", "NEAR"),
            RoomRelationship("Kitchen", "Living Room", "NEAR"),
            RoomRelationship("Living Room", "Dining", "NEAR"),
            RoomRelationship("Kitchen", "Dining", "ADJACENT"),
            RoomRelationship("Kitchen", "Dining", "PREFERRED"),
        ]
    )

    # ============================================================
    # BEDROOM / BATHROOM SAME-FLOOR RELATIONSHIPS
    # ============================================================
    room_floor_lookup = {
        name: floor for name, floor, _, _ in rooms_ft
    }

    common_room_by_floor = {1: "Living Room"}
    if family_lounge_floor is not None:
        common_room_by_floor[family_lounge_floor] = "Family Lounge"

    for index in range(bedrooms):
        bedroom_name = f"Bedroom {index + 1}"
        bedroom_floor = room_floor_lookup[bedroom_name]
        common_room = common_room_by_floor.get(bedroom_floor)

        if common_room is not None:
            relationships.append(
                RoomRelationship(
                    bedroom_name,
                    common_room,
                    "PREFERRED",
                )
            )

    for index in range(bathrooms):
        bathroom_name = f"Bathroom {index + 1}"
        bathroom_floor = room_floor_lookup[bathroom_name]
        common_room = common_room_by_floor.get(bathroom_floor)

        if common_room is not None:
            relationships.append(
                RoomRelationship(
                    bathroom_name,
                    common_room,
                    "PREFERRED",
                )
            )

    # ============================================================
    # ADAPT AUTOMATIC PROGRAM TO THE ACTUAL SITE
    # ============================================================
    rooms_ft, relationships = _adapt_room_program_to_site(
        rooms_ft,
        relationships,
        buildable_bounds_ft=buildable_bounds_ft,
        number_of_floors=number_of_floors,
        building_type=building_type,
    )

    # ============================================================
    # CONVERT TO SOLVER ROOM OBJECTS
    # ============================================================
    rooms = [
        Room(
            name,
            feet_to_meters(width),
            feet_to_meters(length),
            floor,
        )
        for name, floor, width, length in rooms_ft
    ]

    return (
        rooms,
        relationships,
        floor_area_ft2,
    )



def _room_minimum_size(name: str) -> tuple[float, float]:
    """Return practical minimum room dimensions in FEET.

    Architectural reference values below are specified in metres, so convert
    them here because the adaptive room-program functions operate in feet.
    """
    value = name.casefold()
    if "bathroom" in value:
        dims_m = (1.45, 2.10)
    elif "entry foyer" in value or "foyer" in value:
        dims_m = (1.50, 1.80)
    elif "utility" in value:
        dims_m = (1.50, 1.80)
    elif "kitchen" in value:
        dims_m = (2.40, 2.70)
    elif "dining" in value:
        dims_m = (2.40, 2.70)
    elif "living" in value or "family lounge" in value:
        dims_m = (2.80, 3.20)
    elif "bedroom" in value:
        dims_m = (2.70, 3.00)
    else:
        dims_m = (2.00, 2.40)
    return meters_to_feet(dims_m[0]), meters_to_feet(dims_m[1])


def adapt_room_dimensions_to_site(
    rooms: list[Room],
    *,
    buildable_bounds_ft: tuple[float, float, float, float],
) -> list[Room]:
    """Resize individual rooms to fit within site dimensions without deleting rooms.

    Room objects use metres; site bounds and the reference room program use feet.
    This pass is intentionally dimension-only so user-requested rooms and their
    relationships are retained. The optimizer remains responsible for arranging
    the complete set of rooms and reporting genuine infeasibility.
    """
    bx0, by0, bx1, by1 = map(float, buildable_bounds_ft)
    max_w_ft = max(0.1, bx1 - bx0) * 0.92
    max_l_ft = max(0.1, by1 - by0) * 0.92
    adapted: list[Room] = []
    for room in rooms:
        w_ft = meters_to_feet(room.width)
        l_ft = meters_to_feet(room.length)
        # Rotate only when the alternate orientation fits better.
        if w_ft > max_w_ft and l_ft <= max_w_ft:
            w_ft, l_ft = l_ft, w_ft
        min_w_ft, min_l_ft = _room_minimum_size(room.name)
        # If the site is too small for a typical minimum, cap the minimum to
        # the available dimension rather than creating a room larger than site.
        min_w_ft = min(min_w_ft, max_w_ft)
        min_l_ft = min(min_l_ft, max_l_ft)
        if w_ft > max_w_ft:
            w_ft = max(min_w_ft, max_w_ft)
        if l_ft > max_l_ft:
            l_ft = max(min_l_ft, max_l_ft)
        # Guard against unusual input and guarantee each room footprint fits.
        w_ft = min(max(w_ft, min_w_ft), max_w_ft)
        l_ft = min(max(l_ft, min_l_ft), max_l_ft)
        adapted.append(Room(room.name, feet_to_meters(w_ft), feet_to_meters(l_ft), room.floor))
    return adapted


def _adapt_room_program_to_site(
    rooms_ft: list[tuple[str, int, float, float]],
    relationships: list[RoomRelationship],
    *,
    buildable_bounds_ft: tuple[float, float, float, float],
    number_of_floors: int,
    building_type: str,
) -> tuple[list[tuple[str, int, float, float]], list[RoomRelationship]]:
    """Make the automatically generated program fit a wider range of sites.

    The original templates use comfortable reference dimensions.  Those are
    useful on a 40x60 site but can make smaller or narrow sites infeasible.
    This pass therefore (1) rotates a room footprint when that is the only way
    it can fit the buildable width/depth, (2) scales oversized rooms toward a
    practical minimum, and (3) removes only optional rooms when a floor is
    still over-programmed.  User-entered room dimensions are not touched by
    this function because the current automatic mode is the only caller.
    """
    bx0, by0, bx1, by1 = map(float, buildable_bounds_ft)
    buildable_w = max(0.1, bx1 - bx0)
    buildable_l = max(0.1, by1 - by0)
    floor_capacity = buildable_w * buildable_l
    target_ratio = 0.72 if number_of_floors > 1 else 0.68

    adapted: list[tuple[str, int, float, float]] = []
    for name, floor, width, length in rooms_ft:
        w, l = float(width), float(length)
        # Rotate the footprint if its original orientation cannot fit.
        if w > buildable_w and l <= buildable_w:
            w, l = l, w
        if l > buildable_l and w <= buildable_l and w > buildable_w:
            w, l = l, w
        min_w, min_l = _room_minimum_size(name)
        min_w = min(min_w, buildable_w * 0.88)
        min_l = min(min_l, buildable_l * 0.88)
        # Keep the room above a practical minimum, but make the reference
        # footprint smaller when the site is narrow.
        if w > buildable_w:
            w = max(min_w, buildable_w * 0.92)
        if l > buildable_l:
            l = max(min_l, buildable_l * 0.92)
        # If both dimensions are too large, use the same aspect ratio while
        # respecting the minimum dimensions.
        if w > buildable_w or l > buildable_l:
            scale = min(buildable_w / max(w, 0.1), buildable_l / max(l, 0.1), 0.92)
            w = max(min_w, w * scale)
            l = max(min_l, l * scale)
        adapted.append((name, floor, w, l))

    # Adapt each floor independently.  Optional spaces are the first to be
    # removed; bedrooms/bathrooms/core rooms are retained whenever possible.
    optional_keywords = ("utility", "family lounge")
    by_floor: dict[int, list[tuple[str, int, float, float]]] = {f: [] for f in range(1, number_of_floors + 1)}
    for item in adapted:
        by_floor.setdefault(item[1], []).append(item)

    retained: list[tuple[str, int, float, float]] = []
    for floor in range(1, number_of_floors + 1):
        items = by_floor.get(floor, [])
        total = sum(w * l for _, _, w, l in items)
        if total > floor_capacity * target_ratio:
            # Scale comfortable rooms first; never force a room below its
            # practical minimum. This is intentionally deterministic.
            factor = math.sqrt((floor_capacity * target_ratio) / max(total, 0.1))
            scaled_items = []
            for name, fl, w, l in items:
                min_w, min_l = _room_minimum_size(name)
                scaled_items.append((name, fl, max(min_w, w * factor), max(min_l, l * factor)))
            items = scaled_items
            total = sum(w * l for _, _, w, l in items)

        # If the minimum-sized program is still too large, drop optional spaces
        # and then extra bedrooms beyond the first two on a floor.
        if total > floor_capacity * 0.84:
            removable = [
                item for item in items
                if any(k in item[0].casefold() for k in optional_keywords)
            ]
            for item in removable:
                if total <= floor_capacity * 0.80:
                    break
                items.remove(item)
                total -= item[2] * item[3]

        if total > floor_capacity * 0.84:
            extra_bedrooms = [
                item for item in items
                if "bedroom" in item[0].casefold()
                and not "master" in item[0].casefold()
            ]
            extra_bedrooms.sort(key=lambda item: int(item[0].split()[-1]) if item[0].split()[-1].isdigit() else 99, reverse=True)
            for item in extra_bedrooms:
                if total <= floor_capacity * 0.80:
                    break
                # Keep at least one bedroom on a residential floor.
                bedroom_count = sum("bedroom" in x[0].casefold() for x in items)
                if bedroom_count <= 1:
                    break
                items.remove(item)
                total -= item[2] * item[3]

        retained.extend(items)

    retained_names = {item[0] for item in retained}
    filtered_relationships = [
        rel for rel in relationships
        if rel.room_a in retained_names and rel.room_b in retained_names
    ]

    # Remove hard relationships that became inappropriate after an optional
    # room was dropped; retained relationships remain the architectural graph.
    return retained, filtered_relationships

def planning_suggestions(
    *,
    floor_area_ft2: float,
    number_of_floors: int,
    parking_type: str,
    bedroom_count: int,
) -> list[str]:
    """Return transparent planning suggestions."""
    suggestions = []

    if floor_area_ft2 < 700 and bedroom_count >= 3:
        suggestions.append(
            "With this buildable area, fewer bedrooms may allow a more comfortable layout."
        )

    if number_of_floors == 1 and floor_area_ft2 > 900:
        suggestions.append(
            "An additional floor could allow more bedrooms or larger rooms."
        )

    if parking_type == "2 cars":
        suggestions.append(
            "Two-car parking uses more site area and may reduce the available open area."
        )

    if number_of_floors > 1:
        suggestions.append(
            "Moving some bedrooms to upper floors can leave more room for ground-floor circulation."
        )

    return suggestions