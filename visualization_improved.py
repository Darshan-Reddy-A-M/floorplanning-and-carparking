"""Architectural 2D and interactive 3D renderers for the AI test-fit."""
from __future__ import annotations

import math
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc

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
    center = (start + end) / 2
    half = min(width / 2, (end - start) * 0.32)
    if half < 0.25:
        return
    if orientation == "v":
        lo, hi = center - half, center + half
        ax.plot([fixed, fixed], [lo, hi], color="white", linewidth=5.5, zorder=9)
        ax.plot([fixed, fixed], [lo, lo + half], color="#555", linewidth=1.2, zorder=10)
        ax.add_patch(Arc((fixed, lo), 2 * half, 2 * half, angle=90, theta1=0, theta2=90, color="#555", linewidth=1.0, zorder=10))
    else:
        lo, hi = center - half, center + half
        ax.plot([lo, hi], [fixed, fixed], color="white", linewidth=5.5, zorder=9)
        ax.plot([lo, lo + half], [fixed, fixed], color="#555", linewidth=1.2, zorder=10)
        ax.add_patch(Arc((lo, fixed), 2 * half, 2 * half, angle=0, theta1=0, theta2=90, color="#555", linewidth=1.0, zorder=10))
    if main:
        ax.text(fixed, center, "ENTRY", fontsize=5.5, color="#333", ha="center", va="center", zorder=11)


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
    x, y, w, h = map(float, (p.x, p.y, p.width, p.length))
    n = max(7, min(13, int(h / 0.30)))
    for i in range(1, n):
        yy = y + h * i / n
        ax.plot([x + 0.08 * w, x + 0.92 * w], [yy, yy], color="#315da8", linewidth=0.9, zorder=8)
    ax.text(x + w / 2, y + h * 0.10, "UP", fontsize=6.5, color="#315da8", fontweight="bold", ha="center", va="center", zorder=9)


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
    placements, site_width, site_length, buildable_bounds=None, *,
    parking=None, staircases=None, road_access="South", floor=1
):
    """Render one floor at a time in feet; never overlay different floors."""
    site_width_m, site_length_m = float(site_width), float(site_length)
    if buildable_bounds is None:
        buildable_bounds = (0, 0, site_width_m, site_length_m)
    bx0, by0, bx1, by1 = map(float, buildable_bounds)

    # Convert every displayed dimension to feet; the solver remains in metres.
    site_w, site_l = _ft(site_width_m), _ft(site_length_m)
    build_w = (_ft(bx0), _ft(by0), _ft(bx1), _ft(by1))

    fig, ax = plt.subplots(figsize=(8.2, 10.0), dpi=150)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-max(2.0, site_w * 0.035), site_w * 1.04)
    ax.set_ylim(-max(2.0, site_l * 0.035), site_l * 1.035)
    ax.set_xlabel("Site width (ft)", fontsize=10)
    ax.set_ylabel("Site length (ft)", fontsize=10)
    floor_title = "GROUND FLOOR" if int(floor) == 1 else f"FLOOR {int(floor)}"
    ax.set_title(f"{floor_title} PLAN — ARCHITECTURAL LAYOUT", fontsize=13, fontweight="bold", pad=12)

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
        dim_fs = max(4.5, min(6.2, width_pt / max(0.60 * len(dim_text), 1), height_pt * 0.25))
        # Short labels are wrapped only when the room is too narrow.
        if width_pt < 42 and len(label) > 7:
            label = label.replace(" ", "\\n", 1)
        ax.text(x + w/2, y + h*0.56, label, ha="center", va="center",
                fontsize=name_fs, fontweight="bold", linespacing=0.9,
                clip_on=True, zorder=5)
        ax.text(x + w/2, y + h*0.34, dim_text, ha="center", va="center",
                fontsize=dim_fs, color="#333", clip_on=True, zorder=5)

    for p in stairs:
        x, y, w, h = map(float, (p.x, p.y, p.width, p.length))
        x, y, w, h = map(_ft, (x, y, w, h))
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#e5ebf5", edgecolor="#315da8", linewidth=1.8, hatch="///", zorder=3))
        # stairs helper expects metres, so draw directly in feet here.
        n = max(7, min(13, int(h / 1.0)))
        for i in range(1, n):
            yy = y + h * i / n
            ax.plot([x + .08*w, x + .92*w], [yy, yy], color="#315da8", linewidth=.8, zorder=8)
        ax.text(x+w/2, y+h*.10, "UP", fontsize=7, color="#315da8", fontweight="bold", ha="center", zorder=9)

    # Internal doors only occur on true shared walls. Keep them sparse and useful.
    shared = _shared_edges(all_arch)
    for orientation, fixed, start, end, a, b in shared:
        # Avoid drawing a door on every structural partition. Bathrooms, bedrooms,
        # foyer, kitchen and living spaces get priority; the remaining shared walls
        # are left as clean wall lines.
        names = f"{a.name} {b.name}".casefold()
        priority = any(k in names for k in ("bathroom", "bedroom", "foyer", "entrance", "kitchen", "living", "family", "dining"))
        if priority:
            _draw_door(ax, orientation, _ft(fixed), _ft(start), _ft(end), width=min(3.0, _ft(.9)))

    # Windows: choose exposed exterior walls, not only site/setback edges. This
    # gives rooms on the outside of a compact cluster natural openings.
    for p in rooms:
        exposed = _exterior_edges(p, rooms + stairs)
        if not exposed:
            continue
        # Prefer a side not facing the internal entry cluster; otherwise longest wall.
        exposed.sort(key=lambda e: (0 if e[4] in {"N", "S"} else 1, -(e[3]-e[2])))
        orientation, fixed, start, end, _ = exposed[0]
        win_len_m = 0.95 if "bath" in p.name.casefold() else min(1.8, max(.85, (end-start)*.42))
        _draw_window(ax, orientation, _ft(fixed), _ft(start), _ft(end), length=_ft(win_len_m))

    park = _parking_from_site(site_width_m, site_length_m, buildable_bounds, road_access, parking) if floor == 1 else None
    if park:
        px, py, pw, pl = map(_ft, park)
        ax.add_patch(Rectangle((px, py), pw, pl, facecolor="#fff1d8", edgecolor="#a45b08", linewidth=1.8, hatch="xx", zorder=1, label="Parking"))
        ax.text(px+pw/2, py+pl/2, "PARKING", ha="center", va="center", fontsize=max(6, min(9, _fit_font("PARKING", pw, pl))), fontweight="bold")
        if road_access == "South": gate=(px+pw/2,0)
        elif road_access == "North": gate=(px+pw/2,site_l)
        elif road_access == "West": gate=(0,py+pl/2)
        else: gate=(site_w,py+pl/2)
        ax.scatter([gate[0]], [gate[1]], marker="s", s=70, color="#a45b08", zorder=10, label="Vehicle gate")

    # Main entrance: use the foyer/entry room's road-facing exposed edge.
    foyer = next((p for p in rooms if any(k in p.name.casefold() for k in ("entry foyer", "foyer", "entrance"))), None)
    if foyer:
        candidates = _exterior_edges(foyer, rooms + stairs)
        wanted = {"South":"S", "North":"N", "West":"W", "East":"E"}.get(road_access)
        chosen = next((e for e in candidates if e[4] == wanted), candidates[0] if candidates else None)
        if chosen:
            orientation, fixed, start, end, _ = chosen
            _draw_door(ax, orientation, _ft(fixed), _ft(start), _ft(end), width=3.0, main=True)

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
        fig.add_trace(go.Mesh3d(x=[x0,x1,x1,x0],y=[y0,y0,y1,y1],z=[z,z,z,z],i=[0,0],j=[1,2],k=[2,3],color="#b8b3a8",opacity=.35,showlegend=False))
        if floor == max(floor_values):
            fig.add_trace(go.Mesh3d(x=[x0,x1,x1,x0],y=[y0,y0,y1,y1],z=[z+floor_height*.86,z+floor_height*.86,z+floor_height*.86,z+floor_height*.86],i=[0,0],j=[1,2],k=[2,3],color="#c8c1b4",opacity=.16,showlegend=False))

    # Doors on actual shared walls and main entrance.
    for floor in floor_values:
        fr=[p for p in rooms if int(getattr(p,"floor",1))==floor]
        edges=_shared_edges(fr)
        z=(floor-1)*floor_height+1.0
        for orientation,fixed,start,end,a,b in edges:
            names=f"{a.name} {b.name}".casefold()
            if not any(k in names for k in ("bathroom","bedroom","foyer","entrance","kitchen","living","family","dining")):
                continue
            c=(start+end)/2; half=min(.45,(end-start)*.20)
            if orientation=="v": xs=[fixed,fixed]; ys=[c-half,c+half]
            else: xs=[c-half,c+half]; ys=[fixed,fixed]
            fig.add_trace(go.Scatter3d(x=xs,y=ys,z=[z,z],mode="lines",line=dict(color="#6d472b",width=9),showlegend=False))

        # Windows are omitted from this massing view so room volumes and the
        # two-floor proportions are easier to inspect.

    # Main entrance marker on the road side of the foyer.
    foyer=next((p for p in rooms if any(k in p.name.casefold() for k in ("entry foyer","foyer","entrance"))),None)
    if foyer:
        ext=_exterior_edges(foyer,rooms)
        wanted={"South":"S","North":"N","West":"W","East":"E"}.get(road_access)
        chosen=next((e for e in ext if e[4]==wanted),ext[0] if ext else None)
        if chosen:
            orientation,fixed,start,end,_=chosen; c=(start+end)/2; half=min(.55,(end-start)*.22); z=1.0
            if orientation=="v": xs=[fixed,fixed]; ys=[c-half,c+half]
            else: xs=[c-half,c+half]; ys=[fixed,fixed]
            fig.add_trace(go.Scatter3d(x=xs,y=ys,z=[z,z],mode="lines",line=dict(color="#704020",width=12),showlegend=False))

    # Realistic stair flights: solid treads rising through the stair shaft.
    for p in stairs:
        x,y,w,h=map(float,(p.x,p.y,p.width,p.length)); z0=(int(getattr(p,"floor",1))-1)*floor_height; n=12
        for i in range(n):
            yy=y+h*(i+1)/n; zz=z0+floor_height*.78*(i+1)/n; tread=h/n
            fig.add_trace(go.Mesh3d(x=[x,x+w,x+w,x],y=[yy-tread,yy-tread,yy,yy],z=[zz,zz,zz,zz],i=[0,0],j=[1,2],k=[2,3],color="#64748b",opacity=.95,showlegend=False))

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
        title="3D ARCHITECTURAL VISUALIZATION",
        scene=dict(
    		xaxis_title="Width (ft)",
    		yaxis_title="Length (ft)",
    		zaxis_title="Height (ft)",
    		aspectmode="data",
    		camera=views.get(view, views["South-East"]),
   	 	bgcolor="rgba(0,0,0,0)",
	),
        margin=dict(l=10,r=10,t=48,b=10), height=680, showlegend=False,
    )
    return fig
