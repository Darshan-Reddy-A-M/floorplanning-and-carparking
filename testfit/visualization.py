"""Architectural 2D and interactive 3D renderers for the AI test-fit."""
from __future__ import annotations

import math
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FEET_PER_METER = 3.280839895013123


def _ft(v: float) -> float:
    return float(v) * FEET_PER_METER


def _short_name(name: str) -> str:
    s = str(name)
    replacements = [
        (r"^Floor \d+ - Unit \d+ ", ""),
        (r"^U\d+ F\d+ ", ""),
        (r"^Staircase F\d+-F\d+$", "STAIR"),
        (r"^Parking \(.+\)$", "PARKING"),
        (r"Master Bedroom", "MASTER BED"),
        (r"Master Bathroom", "MASTER BATH"),
        (r"Entry Foyer", "FOYER"),
        (r"Upper Living Room", "UPPER LIVING"),
        (r"Living Room", "LIVING"),
        (r"Family Lounge", "FAMILY"),
        (r"Dining", "DINING"),
        (r"Kitchen", "KITCHEN"),
        (r"Utility", "UTILITY"),
        (r"Bedroom (\d+)", r"BED \1"),
        (r"Bathroom (\d+)", r"BATH \1"),
    ]
    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s, flags=re.I)
    return s.upper()


def _room_color(name: str) -> str:
    n = name.casefold()
    if "living" in n or "family" in n:
        return "#dce9f6"
    if "dining" in n:
        return "#f4e7d1"
    if "kitchen" in n or "utility" in n:
        return "#e7f0df"
    if "balcony" in n:
        return "#dff3e6"
    if "bedroom" in n:
        return "#eee4f3"
    if "bathroom" in n or "bath" in n:
        return "#e1eef0"
    if "foyer" in n or "entrance" in n:
        return "#f2ead9"
    return "#f3f1ec"


def _fit_font(text: str, width: float, height: float) -> float:
    chars = max(4, len(text))
    return max(5.0, min(11.0, 4.2 * max(width, 0.1) / chars, 8.0 * max(height, 0.1)))


def _dimension_text_ft(width_m: float, length_m: float) -> str:
    return f"{_ft(width_m):.1f} × {_ft(length_m):.1f} ft"


def _room_label_text(label: str, width: float, height: float, ax, fig, width_m: float, height_m: float):
    """Choose a label that fits inside the room rectangle in screen space."""
    px0, py0 = ax.transData.transform((0.0, 0.0))
    px1, py1 = ax.transData.transform((width, height))
    width_pt = abs(px1 - px0) * 72 / fig.dpi
    height_pt = abs(py1 - py0) * 72 / fig.dpi

    candidates = [label + "\n" + _dimension_text_ft(width_m, height_m), label]
    for candidate in candidates:
        lines = candidate.splitlines()
        longest = max(map(len, lines))
        n = len(lines)
        fs = min(9.5, width_pt / max(0.62 * longest, 1), height_pt / max(1.35 * n, 1))
        if fs >= 6.0:
            return candidate, fs

    # Compact room names are preferred to spilling text outside a small room.
    compact = {
        "ENTRY FOYER": "FOYER",
        "LIVING ROOM": "LIVING",
        "FAMILY LOUNGE": "FAMILY",
        "BEDROOM": "BED",
        "BATHROOM": "BATH",
    }
    short = label
    for long_name, short_name in compact.items():
        short = short.replace(long_name, short_name)
    max_chars = max(4, int(width_pt / (0.62 * 5.5)))
    wrapped = "\n".join(textwrap.wrap(short, width=max_chars, break_long_words=False))
    longest = max(map(len, wrapped.splitlines()))
    n = wrapped.count("\n") + 1
    fs = min(7.5, width_pt / max(0.62 * longest, 1), height_pt / max(1.35 * n, 1))
    return wrapped, max(4.5, fs)


def _same_floor_rooms(placements):
    return [
        p for p in placements
        if "parking" not in str(getattr(p, "name", "")).casefold()
        and "staircase" not in str(getattr(p, "name", "")).casefold()
        and not str(getattr(p, "name", "")).casefold().startswith("stair")
    ]


def _stair_placements(placements):
    return [
        p for p in placements
        if "staircase" in str(getattr(p, "name", "")).casefold()
        or str(getattr(p, "name", "")).casefold().startswith("stair")
    ]


def _shared_edges(rooms, eps=0.035):
    """Return real wall-to-wall shared edges; these are the only places for internal doors."""
    edges = []
    for i, a in enumerate(rooms):
        ax, ay, aw, ah = map(float, (a.x, a.y, a.width, a.length))
        for b in rooms[i + 1:]:
            bx, by, bw, bh = map(float, (b.x, b.y, b.width, b.length))
            if abs(ax + aw - bx) <= eps or abs(bx + bw - ax) <= eps:
                lo = max(ay, by)
                hi = min(ay + ah, by + bh)
                if hi - lo > 0.65:
                    x = (ax + aw + bx) / 2
                    edges.append(("v", x, lo, hi, a, b))
            if abs(ay + ah - by) <= eps or abs(by + bh - ay) <= eps:
                lo = max(ax, bx)
                hi = min(ax + aw, bx + bw)
                if hi - lo > 0.65:
                    y = (ay + ah + by) / 2
                    edges.append(("h", y, lo, hi, a, b))
    return edges


def _door_pair_kind(a, b):
    """Return the intended room connection for a pair, or None if no door belongs there."""
    def kind(room):
        name = str(getattr(room, "name", room)).casefold()
        if "stair" in name:
            return "staircase"
        if "balcony" in name:
            return "balcony"
        if "bathroom" in name or re.search(r"\bbath\s*\d*\b", name):
            return "bathroom"
        if "bedroom" in name or "master bed" in name:
            return "bedroom"
        if "upper living" in name or "living" in name or "family lounge" in name:
            return "living"
        if "kitchen" in name:
            return "kitchen"
        if "utility" in name:
            return "utility"
        if "dining" in name:
            return "dining"
        if "foyer" in name or "entrance" in name:
            return "foyer"
        return name.strip()

    pair = frozenset((kind(a), kind(b)))
    # Bathroom doors belong to their attached bedroom, not to every bedroom
    # that happens to share a wall. In particular, Bathroom 2 must not open
    # directly into Bedroom 3.
    if pair == frozenset(("bedroom", "bathroom")):
        bedroom_room = a if kind(a) == "bedroom" else b
        bathroom_room = a if kind(a) == "bathroom" else b
        bedroom_match = re.search(r"(?:bedroom|master\s*bed(?:room)?)\s*(\d+)", str(getattr(bedroom_room, "name", bedroom_room)).casefold())
        bathroom_match = re.search(r"(?:bathroom|bath)\s*(\d+)", str(getattr(bathroom_room, "name", bathroom_room)).casefold())
        if bedroom_match and bathroom_match and bedroom_match.group(1) != bathroom_match.group(1):
            return None
    allowed = {
        frozenset(("dining", "living")),
        frozenset(("dining", "kitchen")),
        frozenset(("living", "bedroom")),
        frozenset(("bedroom", "bathroom")),
        frozenset(("utility", "kitchen")),
        frozenset(("staircase", "living")),
        frozenset(("balcony", "bedroom")),
        frozenset(("foyer", "living")),
    }
    return pair if pair in allowed else None


def _exterior_edges(room, others, eps=0.04):
    """Edges exposed to open space, rather than to another room."""
    x, y, w, h = map(float, (room.x, room.y, room.width, room.length))
    sides = [("v", x, y, y + h, "W"), ("v", x + w, y, y + h, "E"),
             ("h", y, x, x + w, "S"), ("h", y + h, x, x + w, "N")]
    out = []
    for orientation, fixed, start, end, side in sides:
        blocked = False
        for other in others:
            if other is room:
                continue
            ox, oy, ow, oh = map(float, (other.x, other.y, other.width, other.length))
            if orientation == "v":
                if abs(fixed - ox) <= eps or abs(fixed - (ox + ow)) <= eps:
                    if min(end, oy + oh) - max(start, oy) > 0.65:
                        blocked = True
                        break
            else:
                if abs(fixed - oy) <= eps or abs(fixed - (oy + oh)) <= eps:
                    if min(end, ox + ow) - max(start, ox) > 0.65:
                        blocked = True
                        break
        if not blocked:
            out.append((orientation, fixed, start, end, side))
    return out


def _draw_door(ax, orientation, fixed, start, end, width=0.9, main=False):
    """Show a doorway as a clean break in the wall, without swing/leaf symbols."""
    center = (start + end) / 2
    half = min(width / 2, (end - start) * 0.32)
    if half < 0.25:
        return
    if orientation == "v":
        ax.plot(
            [fixed, fixed], [center - half, center + half],
            color="white", linewidth=5.5, solid_capstyle="butt", zorder=9
        )
    else:
        ax.plot(
            [center - half, center + half], [fixed, fixed],
            color="white", linewidth=5.5, solid_capstyle="butt", zorder=9
        )


def _draw_window(ax, orientation, fixed, start, end, length=None):
    if length is None:
        length = min(1.6, max(0.75, (end - start) * 0.40))
    length = min(length, max(0.5, end - start - 0.25))
    center = (start + end) / 2
    lo, hi = center - length / 2, center + length / 2
    if orientation == "v":
        ax.plot([fixed, fixed], [lo, hi], color="#247ea4", linewidth=4.5, zorder=10)
        ax.plot([fixed, fixed], [lo, hi], color="white", linewidth=1.2, zorder=11)
    else:
        ax.plot([lo, hi], [fixed, fixed], color="#247ea4", linewidth=4.5, zorder=10)
        ax.plot([lo, hi], [fixed, fixed], color="white", linewidth=1.2, zorder=11)


def _stairs_2d(ax, p):
    """Draw a U-shaped, two-flight stair with a half-turn landing in 2D."""
    x, y, w, h = map(float, (p.x, p.y, p.width, p.length))
    # Two parallel half-width flights; the landing occupies the far end.
    landing_h = h * 0.18
    flight_top = y + h - landing_h
    lane_gap = w * 0.045
    lane_w = (w - lane_gap) / 2
    left_x = x
    right_x = x + lane_w + lane_gap
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#e5ebf5",
                           edgecolor="#315da8", linewidth=1.5, zorder=3))
    ax.add_patch(Rectangle((x, flight_top), w, landing_h,
                           facecolor="#d4deef", edgecolor="#315da8",
                           linewidth=1.0, zorder=4))
    # Treads are confined to each flight instead of being drawn across one box.
    n = max(6, min(12, int((h - landing_h) / 0.42)))
    for i in range(1, n):
        yy = y + (flight_top - y) * i / n
        ax.plot([left_x + lane_w * .08, left_x + lane_w * .92], [yy, yy],
                color="#315da8", linewidth=.75, zorder=8)
        ax.plot([right_x + lane_w * .08, right_x + lane_w * .92], [yy, yy],
                color="#315da8", linewidth=.75, zorder=8)
    # First flight rises toward the landing; the second returns in the opposite direction.
    ax.annotate("", xy=(left_x + lane_w*.5, flight_top-lane_w*.04),
                xytext=(left_x + lane_w*.5, y + (flight_top-y)*.18),
                arrowprops=dict(arrowstyle="->", color="#214f9a", lw=1.2), zorder=9)
    ax.annotate("", xy=(right_x + lane_w*.5, y + (flight_top-y)*.18),
                xytext=(right_x + lane_w*.5, flight_top-lane_w*.04),
                arrowprops=dict(arrowstyle="->", color="#214f9a", lw=1.2), zorder=9)
    ax.text(x+w/2, y+h*.91, "LANDING", fontsize=5.5, color="#214f9a",
            fontweight="bold", ha="center", va="center", zorder=10)
    ax.text(x+w/2, y+h*.06, "UP", fontsize=6.2, color="#214f9a",
            fontweight="bold", ha="center", va="center", zorder=10,
            bbox=dict(facecolor="#e5ebf5", edgecolor="none", alpha=.9, pad=.25))


def _parking_from_site(site_width, site_length, buildable_bounds, road_access, parking):
    if parking is None:
        return None
    pw, pl = float(parking.width), float(parking.length)
    if hasattr(parking, "x") and hasattr(parking, "y"):
        return float(parking.x), float(parking.y), pw, pl
    bx0, by0, bx1, by1 = map(float, buildable_bounds)
    gap = 0.25
    if road_access == "South":
        x = max(0.15, min((site_width - pw) / 2, site_width - pw - 0.15)); y = max(0.15, by0 - pl - gap)
    elif road_access == "North":
        x = max(0.15, min((site_width - pw) / 2, site_width - pw - 0.15)); y = min(site_length - pl - 0.15, by1 + gap)
    elif road_access == "West":
        x = max(0.15, bx0 - pw - gap); y = max(0.15, min((site_length - pl) / 2, site_length - pl - 0.15))
    else:
        x = min(site_width - pw - 0.15, bx1 + gap); y = max(0.15, min((site_length - pl) / 2, site_length - pl - 0.15))
    return x, y, pw, pl


def create_floor_plan(
    placements, site_width=None, site_length=None, buildable_bounds=None, *,
    parking=None, staircases=None, road_access="South", floor=1,
    floor_number=None, entrance_side=None, entrance_room_name=None,
    unit_label=None
):
    # Support both:
    # create_floor_plan(placements, site_width, site_length)
    # create_floor_plan(site_width, site_length, placements)
    if isinstance(placements, (int, float)) and isinstance(site_width, (int, float)):
        if isinstance(site_length, (list, tuple)):
            placements, site_width, site_length = site_length, placements, site_width

    if floor_number is not None:
        floor = floor_number

    # Keep the existing function body below this point.

    """Render one floor at a time in feet; never overlay different floors."""
    if site_width is None or site_length is None:
        raise ValueError("site_width and site_length are required")
    site_width_m, site_length_m = float(site_width), float(site_length)
    if buildable_bounds is None:
        buildable_bounds = (0, 0, site_width_m, site_length_m)
    bx0, by0, bx1, by1 = map(float, buildable_bounds)

    # Convert every displayed dimension to feet; the solver remains in metres.
    site_w, site_l = _ft(site_width_m), _ft(site_length_m)
    build_w = (_ft(bx0), _ft(by0), _ft(bx1), _ft(by1))

    fig, ax = plt.subplots(figsize=(6.2, 7.8), dpi=110, facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-max(2.0, site_w * 0.035), site_w * 1.04)
    ax.set_ylim(-max(2.0, site_l * 0.035), site_l * 1.035)
    ax.set_xlabel("Site width (ft)", fontsize=10)
    ax.set_ylabel("Site length (ft)", fontsize=10)
    floor_title = "GROUND FLOOR" if int(floor) == 1 else ("FIRST FLOOR" if int(floor) == 2 else f"FLOOR {int(floor)}")
    ax.set_title(f"{floor_title} PLAN — ARCHITECTURAL LAYOUT", loc="left", fontsize=13, fontweight="bold", pad=12)

    # Site and road.
    ax.add_patch(Rectangle((0, 0), site_w, site_l, fill=False, linewidth=2.5, edgecolor="#111", label="Site boundary"))
    ax.add_patch(Rectangle((build_w[0], build_w[1]), build_w[2] - build_w[0], build_w[3] - build_w[1],
                           facecolor="#edf5ee", edgecolor="#62a77a", linewidth=1.2, alpha=.22, hatch="..", label="Buildable area"))
    if road_access == "South":
        ax.plot([0, site_w], [0, 0], linewidth=6, color="#555"); road_label=(site_w/2, -site_l*.025)
    elif road_access == "North":
        ax.plot([0, site_w], [site_l, site_l], linewidth=6, color="#555"); road_label=(site_w/2, site_l*1.025)
    elif road_access == "West":
        ax.plot([0, 0], [0, site_l], linewidth=6, color="#555"); road_label=(-site_w*.025, site_l/2)
    else:
        ax.plot([site_w, site_w], [0, site_l], linewidth=6, color="#555"); road_label=(site_w*1.025, site_l/2)
    ax.text(*road_label, "ROAD", ha="center", va="center", fontsize=8, fontweight="bold")

    floor = int(floor)
    rooms = [
        p for p in _same_floor_rooms(placements)
        if int(getattr(p, "floor", 1)) == floor
    ]
    stair_source = list(_stair_placements(placements))
    if staircases:
        stair_source.extend(list(staircases))
    # Deduplicate if callers already included stairs in placements.
    seen_stairs = set()
    stairs = []
    for p in stair_source:
        key = (str(getattr(p, "name", "")), round(float(p.x), 4), round(float(p.y), 4),
               round(float(p.width), 4), round(float(p.length), 4))
        if key in seen_stairs:
            continue
        seen_stairs.add(key)
        if int(getattr(p, "floor", 1)) <= floor <= int(getattr(p, "floor_end", getattr(p, "floor", 1))):
            stairs.append(p)
    all_arch = rooms + stairs

    # Rooms are deliberately drawn with zero spacing. Shared walls are the exact
    # same coordinate, which makes the plan read like one building rather than
    # disconnected floating rectangles.
    for p in rooms:
        x, y, w, h = map(float, (p.x, p.y, p.width, p.length))
        x, y, w, h = map(_ft, (x, y, w, h))
        label = _short_name(p.name)
        ax.add_patch(Rectangle((x, y), w, h, facecolor=_room_color(p.name), edgecolor="#222", linewidth=1.5, zorder=2))
        # Compute a conservative font size in screen space. Room name and
        # dimensions are separate so long dimension strings cannot dominate
        # small rooms or collide with neighbouring labels.
        px0, py0 = ax.transData.transform((0.0, 0.0))
        px1, py1 = ax.transData.transform((w, h))
        width_pt = abs(px1 - px0) * 72 / fig.dpi
        height_pt = abs(py1 - py0) * 72 / fig.dpi
        name_fs = max(5.0, min(8.0, width_pt / max(0.68 * len(label), 1), height_pt * 0.34))
        dim_text = _dimension_text_ft(p.width, p.length)
        if unit_label and str(unit_label).casefold() not in {"ft", "feet"}:
            dim_text = f"{_ft(p.width):.1f} × {_ft(p.length):.1f} {unit_label}"
        dim_fs = max(4.5, min(6.2, width_pt / max(0.60 * len(dim_text), 1), height_pt * 0.25))
        # Short labels are wrapped only when the room is too narrow.
        if width_pt < 42 and len(label) > 7:
            label = label.replace(" ", "\\n", 1)
        ax.text(x + w/2, y + h*0.56, label, ha="center", va="center",
                fontsize=name_fs, color="#111827", fontweight="bold", linespacing=0.9,
                clip_on=True, zorder=5)
        ax.text(x + w/2, y + h*0.34, dim_text, ha="center", va="center",
                fontsize=dim_fs, color="#333", clip_on=True, zorder=5)

    for p in stairs:
        x, y, w, h = map(float, (p.x, p.y, p.width, p.length))
        x, y, w, h = map(_ft, (x, y, w, h))
        # Render a proper two-flight U-turn stair rather than a single hatched box.
        class _StairRect:
            pass
        stair_shape = _StairRect()
        stair_shape.x, stair_shape.y, stair_shape.width, stair_shape.length = x, y, w, h
        _stairs_2d(ax, stair_shape)

    # Draw internal doors only on verified shared walls. Kitchen doors are
    # handled separately below so a kitchen-to-utility mark can never be drawn
    # at an unrelated edge or inside the kitchen rectangle.
    shared_edges = _shared_edges(all_arch)
    for orientation, fixed, start, end, a, b in shared_edges:
        pair = _door_pair_kind(a, b)
        if pair is None or "staircase" in pair:
            continue
        if pair in (frozenset(("kitchen", "utility")), frozenset(("kitchen", "dining"))):
            continue
        _draw_door(ax, orientation, _ft(fixed), _ft(start), _ft(end), width=_ft(2.6))

    # Explicit kitchen-to-utility door: draw ONLY on their common horizontal
    # wall (bottom of kitchen / top of utility), at the centre of the overlap.
    kitchen = next((r for r in rooms if "kitchen" in str(getattr(r, "name", "")).casefold()), None)
    utility = next((r for r in rooms if "utility" in str(getattr(r, "name", "")).casefold()), None)
    if kitchen is not None and utility is not None:
        kx, ky, kw, kh = map(float, (kitchen.x, kitchen.y, kitchen.width, kitchen.length))
        ux, uy, uw, uh = map(float, (utility.x, utility.y, utility.width, utility.length))
        overlap_lo, overlap_hi = max(kx, ux), min(kx + kw, ux + uw)
        # In this plan the utility is directly below the kitchen.
        if overlap_hi - overlap_lo > 0.65 and abs(ky - (uy + uh)) <= 0.08:
            _draw_door(ax, "h", _ft(ky), _ft(overlap_lo), _ft(overlap_hi), width=_ft(3.0))
        # Also support a side-by-side arrangement, but only when they truly
        # share a vertical wall; never draw a floating mark inside either room.
        else:
            overlap_lo_y, overlap_hi_y = max(ky, uy), min(ky + kh, uy + uh)
            if overlap_hi_y - overlap_lo_y > 0.65 and abs((kx + kw) - ux) <= 0.08:
                _draw_door(ax, "v", _ft(kx + kw), _ft(overlap_lo_y), _ft(overlap_hi_y), width=_ft(3.0))
            elif overlap_hi_y - overlap_lo_y > 0.65 and abs((ux + uw) - kx) <= 0.08:
                _draw_door(ax, "v", _ft(kx), _ft(overlap_lo_y), _ft(overlap_hi_y), width=_ft(3.0))

    # Preserve the kitchen-to-dining doorway, but draw it only at a real shared wall.
    dining = next((r for r in rooms if "dining" in str(getattr(r, "name", "")).casefold()), None)
    if kitchen is not None and dining is not None:
        kx, ky, kw, kh = map(float, (kitchen.x, kitchen.y, kitchen.width, kitchen.length))
        dx, dy, dw, dh = map(float, (dining.x, dining.y, dining.width, dining.length))
        overlap_lo, overlap_hi = max(ky, dy), min(ky + kh, dy + dh)
        if overlap_hi - overlap_lo > 0.65 and abs((kx + kw) - dx) <= 0.08:
            _draw_door(ax, "v", _ft(kx + kw), _ft(overlap_lo), _ft(overlap_hi), width=_ft(2.6))
        elif overlap_hi - overlap_lo > 0.65 and abs((dx + dw) - kx) <= 0.08:
            _draw_door(ax, "v", _ft(kx), _ft(overlap_lo), _ft(overlap_hi), width=_ft(2.6))
        else:
            overlap_lo_x, overlap_hi_x = max(kx, dx), min(kx + kw, dx + dw)
            if overlap_hi_x - overlap_lo_x > 0.65 and abs(ky - (dy + dh)) <= 0.08:
                _draw_door(ax, "h", _ft(ky), _ft(overlap_lo_x), _ft(overlap_hi_x), width=_ft(2.6))
            elif overlap_hi_x - overlap_lo_x > 0.65 and abs((ky + kh) - dy) <= 0.08:
                _draw_door(ax, "h", _ft(ky + kh), _ft(overlap_lo_x), _ft(overlap_hi_x), width=_ft(2.6))

    # Window symbols intentionally omitted; doors remain in the architectural plan.

    park = _parking_from_site(site_width_m, site_length_m, buildable_bounds, road_access, parking) if floor == 1 else None
    if park:
        px, py, pw, pl = map(_ft, park)
        ax.add_patch(Rectangle((px, py), pw, pl, facecolor="#fff1d8", edgecolor="#a45b08", linewidth=1.8, hatch="xx", zorder=1, label="Parking"))
        ax.text(px+pw/2, py+pl/2, "PARKING", ha="center", va="center", fontsize=max(6, min(9, _fit_font("PARKING", pw, pl))), fontweight="bold")
    # Always mark the vehicle gate on the road-facing site boundary, even when
    # a parking rectangle was not supplied to the renderer.
    if road_access == "South":
        gate = (site_w / 2, 0)
    elif road_access == "North":
        gate = (site_w / 2, site_l)
    elif road_access == "West":
        gate = (0, site_l / 2)
    else:
        gate = (site_w, site_l / 2)
    ax.scatter([gate[0]], [gate[1]], marker="s", s=70, color="#a45b08",
               zorder=10, label="Vehicle gate")
    ax.annotate("Vehicle gate", xy=gate, xytext=(6, 7),
                textcoords="offset points", fontsize=6.5, color="#8a4b08")

    # A separate pedestrian gate is shown on the same road-facing boundary.
    # Offset it from the vehicle gate so both symbols remain visible.
    pedestrian_offset = max(2.5, site_w * 0.12) if road_access in {"South", "North"} else max(2.5, site_l * 0.12)
    if road_access == "South":
        pedestrian_gate = (min(site_w * 0.22, site_w - 1), 0)
    elif road_access == "North":
        pedestrian_gate = (min(site_w * 0.22, site_w - 1), site_l)
    elif road_access == "West":
        pedestrian_gate = (0, min(site_l * 0.22, site_l - 1))
    else:
        pedestrian_gate = (site_w, min(site_l * 0.22, site_l - 1))
    ax.scatter([pedestrian_gate[0]], [pedestrian_gate[1]], marker="D", s=38,
               color="#2563eb", zorder=10, label="Pedestrian gate")
    ax.annotate("Pedestrian gate", xy=pedestrian_gate, xytext=(6, -10),
                textcoords="offset points", fontsize=6.2, color="#1d4ed8")

    # Main entrance: use the requested foyer when supplied; otherwise find it
    # by a conventional room name. Entrance side is independent of road access.
    foyer = next((
        p for p in rooms
        if entrance_room_name and str(getattr(p, "name", "")).casefold() == entrance_room_name.casefold()
    ), None)
    if foyer is None:
        foyer = next((
            p for p in rooms
            if any(k in str(getattr(p, "name", "")).casefold()
                   for k in ("entry foyer", "foyer", "entrance"))
        ), None)
    if foyer:
        candidates = _exterior_edges(foyer, rooms + stairs)
        wanted_side = entrance_side or road_access
        wanted = {"South":"S", "North":"N", "West":"W", "East":"E"}.get(wanted_side)
        chosen = next((e for e in candidates if e[4] == wanted), candidates[0] if candidates else None)
        if chosen:
            orientation, fixed, start, end, _ = chosen
            fixed_ft, start_ft, end_ft = _ft(fixed), _ft(start), _ft(end)
            _draw_door(ax, orientation, fixed_ft, start_ft, end_ft, width=3.0, main=True)
            mid = (start_ft + end_ft) / 2
            offset = max(0.8, site_w * 0.018)
            if orientation == "h":
                label_x, label_y = mid, fixed_ft - offset
            else:
                label_x, label_y = fixed_ft + offset, mid
            entry_label = f"MAIN ENTRY ({entrance_side})" if entrance_side else "MAIN ENTRY"
            ax.text(label_x, label_y, entry_label, fontsize=6.2, color="#111827",
                    fontweight="bold", ha="center", va="center", zorder=12,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.2))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.grid(True, alpha=.12)
    fig.subplots_adjust(left=.08, right=.88, bottom=.10, top=.88)
    return fig


def _cuboid_mesh(go, x, y, w, h, z0, z1, color, opacity=.42, name=None):
    """Return a solid cuboid mesh for an architectural room mass."""
    xs=[x,x+w,x+w,x,x,x+w,x+w,x]
    ys=[y,y,y+h,y+h,y,y,y+h,y+h]
    zs=[z0,z0,z0,z0,z1,z1,z1,z1]
    i=[0,0,0,4,4,4,0,0,1,1,2,2]
    j=[1,2,3,5,6,7,4,1,5,2,6,3]
    k=[2,3,1,6,7,5,5,4,1,6,3,7]
    return go.Mesh3d(x=xs,y=ys,z=zs,i=i,j=j,k=k,color=color,opacity=opacity,name=name or "Room",showlegend=False,flatshading=True)


def _wall_lines(go, x, y, w, h, z0, z1, color="#444", width=4):
    xs=[]; ys=[]; zs=[]
    edges=[((x,y,z0),(x+w,y,z0)),((x+w,y,z0),(x+w,y+h,z0)),((x+w,y+h,z0),(x,y+h,z0)),((x,y+h,z0),(x,y,z0)),
           ((x,y,z1),(x+w,y,z1)),((x+w,y,z1),(x+w,y+h,z1)),((x+w,y+h,z1),(x,y+h,z1)),((x,y+h,z1),(x,y,z1)),
           ((x,y,z0),(x,y,z1)),((x+w,y,z0),(x+w,y,z1)),((x+w,y+h,z0),(x+w,y+h,z1)),((x,y+h,z0),(x,y+h,z1))]
    for a,b in edges:
        xs += [a[0],b[0],None]; ys += [a[1],b[1],None]; zs += [a[2],b[2],None]
    return go.Scatter3d(x=xs,y=ys,z=zs,mode="lines",line=dict(color=color,width=width),showlegend=False,hoverinfo="skip")


def _camera_presets():
    return {
        "South-East": dict(eye=dict(x=1.55,y=-1.65,z=1.15)),
        "South-West": dict(eye=dict(x=-1.55,y=-1.65,z=1.15)),
        "North-East": dict(eye=dict(x=1.55,y=1.65,z=1.15)),
        "North-West": dict(eye=dict(x=-1.55,y=1.65,z=1.15)),
        "Street": dict(eye=dict(x=1.95,y=-2.35,z=.78)),
        "Top": dict(eye=dict(x=.02,y=.02,z=2.7)),
    }


def create_3d_floor_plan(placements, site_width, site_length, *, parking=None, staircases=None, road_access="South", floor_height=3.0, view="South-East"):
    """Interactive solid architectural massing with doors, windows, stairs and camera presets."""
    import plotly.graph_objects as go

    site_width=float(site_width); site_length=float(site_length); floor_height=float(floor_height)
    rooms=[p for p in placements if "parking" not in str(getattr(p,"name","")).casefold() and "staircase" not in str(getattr(p,"name","")).casefold()]
    stairs=list(staircases or [])
    fig=go.Figure()

    # Site plane.
    fig.add_trace(go.Mesh3d(x=[0,site_width,site_width,0],y=[0,0,site_length,site_length],z=[0,0,0,0],i=[0,0],j=[1,2],k=[2,3],color="#ecebe6",opacity=.18,name="Site",showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0,site_width,site_width,0,0],y=[0,0,site_length,site_length,0],z=[.02,.02,.02,.02,.02],mode="lines",line=dict(color="#333",width=4),showlegend=False))

    # Floor-by-floor solid room masses.
    floor_values=sorted({int(getattr(p,"floor",1)) for p in rooms}) or [1]
    for p in rooms:
        x,y,w,h=map(float,(p.x,p.y,p.width,p.length)); floor=int(getattr(p,"floor",1)); z0=(floor-1)*floor_height; z1=z0+floor_height*.86
        color=_room_color(p.name)
        fig.add_trace(_cuboid_mesh(go,x,y,w,h,z0,z1,color,opacity=.96,name=_short_name(p.name)))
        fig.add_trace(_wall_lines(go,x,y,w,h,z0,z1,color="#30343b",width=3))

    # Continuous floor slabs make the stacked building read as one structure.
    for floor in floor_values:
        fr=[p for p in rooms if int(getattr(p,"floor",1))==floor]
        if not fr: continue
        x0=min(p.x for p in fr); y0=min(p.y for p in fr); x1=max(p.x+p.width for p in fr); y1=max(p.y+p.length for p in fr)
        z=(floor-1)*floor_height
        fig.add_trace(go.Mesh3d(x=[x0,x1,x1,x0],y=[y0,y0,y1,y1],z=[z,z,z,z],i=[0,0],j=[1,2],k=[2,3],color="#b8b3a8",opacity=.82,showlegend=False))
        if floor == max(floor_values):
            fig.add_trace(go.Mesh3d(x=[x0,x1,x1,x0],y=[y0,y0,y1,y1],z=[z+floor_height*.86,z+floor_height*.86,z+floor_height*.86,z+floor_height*.86],i=[0,0],j=[1,2],k=[2,3],color="#c8c1b4",opacity=.72,showlegend=False))

    # Doors on actual shared walls and main entrance.
    for floor in floor_values:
        fr=[p for p in rooms if int(getattr(p,"floor",1))==floor]
        edges=_shared_edges(fr)
        z=(floor-1)*floor_height+1.0
        for orientation,fixed,start,end,a,b in edges:
            if _door_pair_kind(a, b) is None:
                continue

        # Keep the 3D massing clean: no window or door glyphs are drawn here.
        # The 2D plan remains the place for architectural door symbols.

    # Main entrance marker on the road side of the foyer.
    foyer=next((p for p in rooms if any(k in p.name.casefold() for k in ("entry foyer","foyer","entrance"))),None)
    if foyer:
        ext=_exterior_edges(foyer,rooms)
        wanted={"South":"S","North":"N","West":"W","East":"E"}.get(road_access)
        chosen=next((e for e in ext if e[4]==wanted),ext[0] if ext else None)

    # Switchback stair flights: two half-width runs with a shared landing.
    # This gives the 3D view the same zig-zag / U-turn stair character as the 2D plan.
    for p in stairs:
        x, y, w, h = map(float, (p.x, p.y, p.width, p.length))
        lower = int(getattr(p, "floor", 1))
        upper = int(getattr(p, "floor_end", lower))
        half_w = w * 0.46
        gap_x = w * 0.04
        run_h = h
        n_each = 6
        for level in range(lower, upper + 1):
            z0 = (level - 1) * floor_height
            rise = floor_height * .78 / (2 * n_each)
            # First run ascends from the front to the rear in the left lane.
            for i in range(n_each):
                y0 = y + run_h * i / n_each
                y1 = y + run_h * (i + 1) / n_each
                zz = z0 + rise * (i + 1)
                fig.add_trace(go.Mesh3d(
                    x=[x, x+half_w, x+half_w, x], y=[y0, y0, y1, y1],
                    z=[zz, zz, zz, zz], i=[0, 0], j=[1, 2], k=[2, 3],
                    color="#64748b", opacity=.98, showlegend=False
                ))
            # Second run returns toward the front in the right lane, completing the U-turn.
            for i in range(n_each):
                y0 = y + run_h * (n_each - i - 1) / n_each
                y1 = y + run_h * (n_each - i) / n_each
                zz = z0 + floor_height * .39 + rise * (i + 1)
                fig.add_trace(go.Mesh3d(
                    x=[x+half_w+gap_x, x+w, x+w, x+half_w+gap_x],
                    y=[y0, y0, y1, y1], z=[zz, zz, zz, zz],
                    i=[0, 0], j=[1, 2], k=[2, 3],
                    color="#64748b", opacity=.98, showlegend=False
                ))
            # Intermediate landing joins the two flights at half height.
            landing_z = z0 + floor_height * .39
            fig.add_trace(go.Mesh3d(
                x=[x, x+w, x+w, x], y=[y+h*.44, y+h*.44, y+h*.56, y+h*.56],
                z=[landing_z]*4, i=[0, 0], j=[1, 2], k=[2, 3],
                color="#94a3b8", opacity=.98, showlegend=False
            ))
        roof_z = upper * floor_height
        fig.add_trace(go.Mesh3d(
            x=[x, x+w, x+w, x], y=[y, y, y+h, y+h],
            z=[roof_z, roof_z, roof_z, roof_z],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color="#94a3b8", opacity=.70, name="Terrace stair landing",
            showlegend=False
        ))

    if parking is not None:
        p=parking; x,y,w,h=map(float,(p.x,p.y,p.width,p.length))
        fig.add_trace(go.Mesh3d(x=[x,x+w,x+w,x],y=[y,y,y+h,y+h],z=[.04,.04,.04,.04],i=[0,0],j=[1,2],k=[2,3],color="#d89a2b",opacity=.65,name="Parking",showlegend=False))

    # Plotly receives metre-based solver geometry above. Convert the displayed
    # coordinates to feet so the entire application uses the same user-facing unit.
    for trace in fig.data:
        for axis in ("x", "y", "z"):
            values = getattr(trace, axis, None)
            if values is not None:
                try:
                    setattr(trace, axis, [None if v is None else float(v) * FEET_PER_METER for v in values])
                except TypeError:
                    pass

    views=_camera_presets()
    max_z=max(1.0,len(floor_values)*_ft(floor_height))
    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(color="#111827"),
        title="3D ARCHITECTURAL VISUALIZATION",
        scene=dict(
            # Hide graph axes, tick labels and grid in the 3D view only.
            xaxis=dict(visible=False, showgrid=False, showticklabels=False, title="", zeroline=False, showbackground=False),
            yaxis=dict(visible=False, showgrid=False, showticklabels=False, title="", zeroline=False, showbackground=False),
            zaxis=dict(visible=False, showgrid=False, showticklabels=False, title="", zeroline=False, showbackground=False),
            aspectmode="data",
            camera=views.get(view, views["South-East"]),
            bgcolor="#ffffff",
        ),
        margin=dict(l=10,r=10,t=48,b=10), height=680, showlegend=False,
    )
    return fig
