"""AI Test-Fit Planner — Streamlit dashboard backed by the existing optimizer."""
from __future__ import annotations

import math
import re

# OR-Tools must load before Streamlit/Pandas on some Windows environments.
from ortools.sat.python import cp_model  # noqa: F401
import streamlit as st

from testfit.optimizer import Room, RoomRelationship, calculate_metrics, solve_layout
from testfit.planning import (
    BUILDING_TYPES,
    PARKING_DEFAULT_FEET,
    STAIRCASE_DEFAULT_FEET,
    feet_to_meters,
    floor_label,
    planning_suggestions,
    recommend_room_program,
    adapt_room_dimensions_to_site,
    setback_bounds_feet,
    select_entrance_room,
    square_meters_to_feet,
)
from testfit.visualization import create_3d_floor_plan, create_floor_plan

st.set_page_config(page_title="AI Test-Fit Planner", page_icon="🏙️", layout="wide")

st.markdown("""
<style>
:root { --blue:#0879ed; --navy:#142b59; --muted:#607493; --line:#dce7f3; --green:#079b70; }
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] { background:#f5f9fd; color:var(--navy); }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1800px; padding:1rem 1.2rem 2rem; }
h1,h2,h3,h4,p,label,span,div { color:inherit; }
[data-testid="stMetric"] { background:#fff; border:1px solid #e1eaf4; border-radius:14px; padding:12px 15px; box-shadow:0 2px 8px rgba(25,64,115,.035); }
[data-testid="stMetricLabel"] { color:#5d7190 !important; font-weight:600; }
[data-testid="stMetricValue"] { color:#102858 !important; font-weight:750; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:#dce7f3 !important; border-radius:15px; }
input, textarea, [data-baseweb="select"] > div { background:#fff !important; }
.stButton > button[kind="primary"] { background:#0879ed; border:0; border-radius:10px; min-height:45px; font-weight:700; }
.stButton > button { border-radius:9px; }
section[data-testid="stSidebar"] { background:#f2f7fc; }
.app-brand { display:flex; align-items:center; gap:13px; padding:0 0 13px; border-bottom:1px solid #dce7f3; margin-bottom:16px; }
.brand-icon { background:#e7f2ff; color:#0879ed; border-radius:13px; padding:10px 13px; font-size:26px; }
.brand-title { color:#142b59; font-size:27px; line-height:1.15; font-weight:800; }
.brand-sub { color:#6a7f9e; font-size:13px; }
.section-card { background:#fff; border:1px solid #e1eaf4; border-radius:14px; padding:15px; margin-bottom:12px; }
.section-title { font-size:17px; font-weight:800; color:#142b59; margin-bottom:8px; }
.muted { color:#71839e; font-size:12px; }
.check-row { display:flex; align-items:center; gap:8px; padding:6px 0; border-bottom:1px solid #edf2f7; }
.pass { color:#079b70; font-weight:750; }
</style>
""", unsafe_allow_html=True)


def _feet_area(m2: float) -> float:
    return square_meters_to_feet(m2)


def _customize_program(rooms, relationships, *, bedroom_count, bathroom_count,
                       living_count, kitchen_count, dining_count, utility_count,
                       number_of_floors, building_type):
    """Apply the room-count controls to the actual solver program.

    For the residential single-family/duplex templates this edits the room list
    and relationships before optimization, so the controls are not decorative.
    Multi-unit templates retain their per-unit program to avoid breaking unit IDs.
    """
    if building_type in {"Rental / Multi-Unit Building", "Apartment / Multi-Unit Residential"}:
        return rooms, relationships

    original = list(rooms)
    by_name = {r.name: r for r in original}
    kept = []

    # Keep and resize the requested number of bedroom rooms. Existing numbered
    # bedrooms are retained first; additional bedrooms are added to upper floors.
    bedroom_rooms = sorted(
        [r for r in original if re.fullmatch(r"Bedroom \d+", r.name)],
        key=lambda r: int(re.search(r"\d+", r.name).group()),
    )
    bath_rooms = sorted(
        [r for r in original if re.fullmatch(r"Bathroom \d+", r.name)],
        key=lambda r: int(re.search(r"\d+", r.name).group()),
    )
    bedroom_rooms = bedroom_rooms[:bedroom_count]
    existing_bed_names = {r.name for r in bedroom_rooms}
    for i in range(1, bedroom_count + 1):
        name = f"Bedroom {i}"
        if name not in existing_bed_names:
            floor = 1 if number_of_floors == 1 or i == 1 else min(number_of_floors, 2 + (i - 2) % max(1, number_of_floors - 1))
            bedroom_rooms.append(Room(name, feet_to_meters(11 if i > 1 else 12), feet_to_meters(12), floor))
    bedroom_rooms = sorted(bedroom_rooms, key=lambda r: int(re.search(r"\d+", r.name).group()))
    bedroom_names = {r.name for r in bedroom_rooms}

    # Bathrooms are named sequentially and placed on the same floor as the
    # corresponding bedroom where possible, so each requested connection is real.
    bath_rooms = bath_rooms[:bathroom_count]
    existing_bath_names = {r.name for r in bath_rooms}
    bed_by_num = {int(re.search(r"\d+", r.name).group()): r for r in bedroom_rooms}
    for i in range(1, bathroom_count + 1):
        name = f"Bathroom {i}"
        target_floor = bed_by_num.get(i, bedroom_rooms[0]).floor
        if name not in existing_bath_names:
            bath_rooms.append(Room(name, feet_to_meters(5), feet_to_meters(8), target_floor))
        else:
            bath_rooms = [Room(r.name, r.width, r.length, target_floor if r.name == name else r.floor) for r in bath_rooms]
    bath_rooms = sorted(bath_rooms, key=lambda r: int(re.search(r"\d+", r.name).group()))
    bath_names = {r.name for r in bath_rooms}

    core_aliases = {
        "Living Room": (living_count, 1, 17.5, 16),
        "Kitchen": (kitchen_count, 1, 10, 11),
        "Dining": (dining_count, 1, 9, 10),
        "Utility": (utility_count, 1, 6, 7),
    }
    core_names = set(core_aliases)
    for r in original:
        if re.fullmatch(r"Bedroom \d+", r.name) or re.fullmatch(r"Bathroom \d+", r.name):
            continue
        if r.name in core_names:
            count, floor, w_ft, l_ft = core_aliases[r.name]
            if count:
                kept.append(Room(r.name, feet_to_meters(w_ft), feet_to_meters(l_ft), floor))
            continue
        kept.append(r)
    kept.extend(bedroom_rooms)
    kept.extend(bath_rooms)

    valid_names = {r.name for r in kept}
    # Remove stale links and ensure every bathroom is attached to its numbered
    # bedroom if that bedroom exists. Existing core/stacking relationships remain.
    rels = [r for r in relationships if r.room_a in valid_names and r.room_b in valid_names]
    rels = [r for r in rels if not (re.fullmatch(r"Bedroom \d+", r.room_a) and re.fullmatch(r"Bathroom \d+", r.room_b))]
    rels = [r for r in rels if not (re.fullmatch(r"Bathroom \d+", r.room_a) and re.fullmatch(r"Bedroom \d+", r.room_b))]
    existing_pairs = {(r.room_a, r.room_b, r.kind) for r in rels}
    for i in range(1, bathroom_count + 1):
        bed, bath = f"Bedroom {i}", f"Bathroom {i}"
        if bed in valid_names and bath in valid_names and (bed, bath, "ATTACHED") not in existing_pairs:
            rels.append(RoomRelationship(bed, bath, "ATTACHED"))
    # Remove relationships to optional rooms if the user selected zero of them.
    rels = [r for r in rels if r.room_a in valid_names and r.room_b in valid_names]
    return kept, rels


# Header / top navigation
st.markdown('''<div class="app-brand"><div class="brand-icon">▥</div><div><div class="brand-title">AI Test-Fit Planner</div><div class="brand-sub">Generate floor plans and parking layouts from your site requirements</div></div></div>''', unsafe_allow_html=True)
nav = st.radio("Workspace", ["⌂  Floor Plan", "▣  Parking Layout", "⬡  3D View", "▤  Reports"], horizontal=True, label_visibility="collapsed", key="workspace_nav")
st.markdown("---")

left, center, right = st.columns([1.02, 1.95, 1.08], gap="medium")

# Inputs panel
with left:
    with st.container(border=True):
        st.markdown("### 📍 Site & Requirements")
        st.markdown("**Site dimensions**")
        presets = {
            "20 × 30 ft — compact": (20.0, 30.0),
            "20 × 40 ft — narrow": (20.0, 40.0),
            "25 × 40 ft": (25.0, 40.0),
            "30 × 40 ft": (30.0, 40.0),
            "30 × 50 ft": (30.0, 50.0),
            "30 × 60 ft": (30.0, 60.0),
            "40 × 50 ft": (40.0, 50.0),
            "40 × 60 ft": (40.0, 60.0),
            "50 × 60 ft": (50.0, 60.0),
            "60 × 80 ft": (60.0, 80.0),
            "Custom dimensions": None,
        }
        selected_site = st.selectbox("Recommended site size", list(presets), index=7)
        a, b = st.columns(2)
        if presets[selected_site] is None:
            with a:
                site_width_ft = st.number_input("Site width (ft)", min_value=10.0, max_value=500.0, value=40.0, step=1.0, key="custom_site_width")
            with b:
                site_length_ft = st.number_input("Site length (ft)", min_value=10.0, max_value=500.0, value=60.0, step=1.0, key="custom_site_length")
        else:
            site_width_ft, site_length_ft = presets[selected_site]
            st.caption(f"Selected plot: {site_width_ft:g} × {site_length_ft:g} ft ({site_width_ft * site_length_ft:,.0f} sq ft)")
        # Keep the current residential/duplex optimizer template internally; building type
        # is intentionally not exposed as a user control, as requested.
        building_type = "Duplex"
        road_access = st.selectbox("Road side / main entrance", ["South", "North", "East", "West"], index=0)
        number_of_floors = st.number_input("Number of floors", min_value=1, max_value=5, value=2, step=1)
        st.markdown("**Setbacks (ft)**")
        s1, s2 = st.columns(2)
        with s1:
            setback_front_ft = st.number_input("Front", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
            setback_left_ft = st.number_input("Left", min_value=0.0, max_value=100.0, value=3.0, step=1.0)
        with s2:
            setback_rear_ft = st.number_input("Rear", min_value=0.0, max_value=100.0, value=3.0, step=1.0)
            setback_right_ft = st.number_input("Right", min_value=0.0, max_value=100.0, value=3.0, step=1.0)
        st.markdown("**Room requirements**")
        default_beds = 3
        r1, r2 = st.columns(2)
        with r1:
            bedroom_count = st.number_input("Bedrooms", min_value=1, max_value=8, value=default_beds, step=1)
            living_count = st.number_input("Living rooms", min_value=1, max_value=1, value=1, step=1)
            dining_count = st.number_input("Dining rooms", min_value=0, max_value=1, value=1, step=1)
        with r2:
            bathroom_count = int(bedroom_count)
            st.metric("Attached bathrooms", bathroom_count, "One private bathroom per bedroom")
            kitchen_count = st.number_input("Kitchens", min_value=1, max_value=1, value=1, step=1)
            utility_count = st.number_input("Utility rooms", min_value=0, max_value=1, value=1, step=1)
        st.markdown("**Parking**")
        parking_type = st.selectbox("Number of cars", ["No parking", "1 car", "2 cars"], index=2)
        keep_driveway = st.checkbox("Keep driveway access", value=True)
        if parking_type == "No parking":
            parking_width_ft = parking_length_ft = 0.0
        else:
            pdefault = PARKING_DEFAULT_FEET[parking_type]
            p1, p2 = st.columns(2)
            with p1:
                parking_width_ft = st.number_input("Parking width (ft)", min_value=5.0, max_value=40.0, value=float(pdefault[0]), step=0.5)
            with p2:
                parking_length_ft = st.number_input("Parking length (ft)", min_value=10.0, max_value=50.0, value=float(pdefault[1]), step=0.5)
        st.markdown("**Staircase**")
        staircase_enabled = st.checkbox("Include staircase", value=(number_of_floors > 1))
        if staircase_enabled:
            q1, q2 = st.columns(2)
            with q1:
                staircase_width_ft = st.number_input("Stair width (ft)", min_value=3.0, max_value=20.0, value=float(STAIRCASE_DEFAULT_FEET[0]), step=0.5)
            with q2:
                staircase_length_ft = st.number_input("Stair length (ft)", min_value=5.0, max_value=30.0, value=float(STAIRCASE_DEFAULT_FEET[1]), step=0.5)
        else:
            staircase_width_ft = staircase_length_ft = 0.0
        generate = st.button("⚙️  Generate Layout  →", type="primary", use_container_width=True)

# Convert dimensions and derive buildable area
site_width_m, site_length_m = feet_to_meters(site_width_ft), feet_to_meters(site_length_ft)
bounds_ft = setback_bounds_feet(site_width_ft, site_length_ft, road_access, setback_front_ft, setback_rear_ft, setback_left_ft, setback_right_ft)
bounds_m = tuple(feet_to_meters(v) for v in bounds_ft)
buildable_area_ft2 = max(0.0, (bounds_ft[2]-bounds_ft[0])*(bounds_ft[3]-bounds_ft[1]))
site_area_ft2 = site_width_ft * site_length_ft
parking_area_ft2 = parking_width_ft * parking_length_ft
stair_size_ft = (staircase_width_ft, staircase_length_ft) if staircase_enabled else (0.0, 0.0)

# Create the default room program then apply the user controls before solving.
try:
    rooms, relationships, floor_area_ft2 = recommend_room_program(
        buildable_bounds_ft=bounds_ft,
        number_of_floors=int(number_of_floors),
        building_type=building_type,
        units_per_floor=1,
        parking_area_ft2=parking_area_ft2,
        staircase_size_ft=stair_size_ft,
    )
    rooms, relationships = _customize_program(
        rooms, relationships,
        bedroom_count=int(bedroom_count), bathroom_count=int(bathroom_count),
        living_count=int(living_count), kitchen_count=int(kitchen_count),
        dining_count=int(dining_count), utility_count=int(utility_count),
        number_of_floors=int(number_of_floors), building_type=building_type,
    )
    # Apply site-aware dimensions AFTER customization, since customization
    # deliberately starts from comfortable reference room sizes. Keep all
    # requested rooms and relationships; let CP-SAT decide if arrangement fits.
    rooms = adapt_room_dimensions_to_site(rooms, buildable_bounds_ft=bounds_ft)
except Exception as exc:
    rooms, relationships, floor_area_ft2 = [], [], 0.0
    st.error(f"Could not build the room requirements: {exc}")

if generate and rooms:
    try:
        with st.spinner("Generating a constraint-checked layout..."):
            result = solve_layout(
                site_width=site_width_m, site_length=site_length_m,
                time_limit_seconds=45.0, rooms=rooms, relationships=relationships,
                number_of_floors=int(number_of_floors), parking_type=parking_type,
                parking_width=feet_to_meters(parking_width_ft) if parking_width_ft else None,
                parking_length=feet_to_meters(parking_length_ft) if parking_length_ft else None,
                road_access=road_access,
                staircase_width=feet_to_meters(staircase_width_ft) if staircase_enabled else 0.0,
                staircase_length=feet_to_meters(staircase_length_ft) if staircase_enabled else 0.0,
                buildable_bounds=bounds_m,
                staircase_mode="internal" if int(number_of_floors) > 1 else "external",
            )
            metrics = calculate_metrics(
                site_width=site_width_m, site_length=site_length_m,
                placements=result.placements, relationships=relationships,
                number_of_floors=int(number_of_floors), parking=result.parking,
                staircases=result.staircases,
            )
            st.session_state["layout_result"] = result
            st.session_state["layout_metrics"] = metrics
            st.session_state["layout_relationships"] = relationships
            st.session_state["layout_inputs"] = {
                "site_width_m": site_width_m, "site_length_m": site_length_m,
                "bounds_m": bounds_m, "road_access": road_access,
                "floor_area_ft2": floor_area_ft2, "parking_type": result.parking_type,
                "bedroom_count": int(bedroom_count), "bathroom_count": int(bathroom_count),
                "building_type": building_type, "number_of_floors": int(number_of_floors),
            }
            st.session_state["layout_program"] = rooms
    except Exception as exc:
        st.error(f"Layout generation failed: {exc}")
        st.exception(exc)

# Summary cards
with center:
    c1, c2, c3 = st.columns(3)
    c1.metric("Buildable area", f"{buildable_area_ft2:,.0f} sq ft", f"{bounds_ft[2]-bounds_ft[0]:.0f} ft × {bounds_ft[3]-bounds_ft[1]:.0f} ft")
    c2.metric("Rooms", str(len(rooms)), f"{bedroom_count} bedrooms · {bathroom_count} bathrooms")
    current_result = st.session_state.get("layout_result")
    actual_parking_type = current_result.parking_type if current_result is not None else parking_type
    c3.metric("Parking", "No parking" if actual_parking_type == "No parking" else actual_parking_type, f"At {road_access} side" if actual_parking_type != "No parking" else "Not requested")

    result = current_result
    if result is None:
        st.markdown('<div class="section-card"><div class="section-title">🏡 Floor-plan preview</div><p class="muted">Set your requirements on the left, then select <b>Generate Layout</b> to create a plan.</p></div>', unsafe_allow_html=True)
        st.info("Your existing constraint-based optimizer will generate the layout using these inputs.")
    else:
        metrics = st.session_state["layout_metrics"]
        saved = st.session_state["layout_inputs"]
        # A new input combination should not silently display a stale plan.
        current_signature = (site_width_m, site_length_m, tuple(bounds_m), road_access, int(number_of_floors), building_type, int(bedroom_count), int(bathroom_count), parking_type, int(living_count), int(kitchen_count), int(dining_count), int(utility_count))
        saved_signature = st.session_state.get("layout_signature")
        if generate:
            st.session_state["layout_signature"] = current_signature
            saved_signature = current_signature
        if saved_signature != current_signature:
            st.warning("Inputs have changed. Select **Generate Layout** to refresh the plan for these requirements.")
        if str(result.status).lower() in {"optimal", "feasible", "success", "ok"}:
            st.success(f"Layout generated · Solver status: {result.status}")
        else:
            st.warning(f"Solver status: {result.status}")
        if result.message:
            st.caption(result.message)
        view_key = "workspace_nav"
        if nav.startswith("▣"):
            view_mode = "Parking Layout"
        elif nav.startswith("⬡"):
            view_mode = "3D View"
        elif nav.startswith("▤"):
            view_mode = "Reports"
        else:
            view_mode = "Floor Plan"
        floors = sorted({p.floor for p in result.placements}) or [1]
        selected_floor = st.selectbox("Floor to view", floors, format_func=floor_label, key="dashboard_floor")
        floor_placements = [p for p in result.placements if p.floor == selected_floor]
        special = []
        if result.parking is not None and selected_floor == 1 and view_mode in {"Parking Layout", "Floor Plan"}:
            special.append(result.parking)
        for stair in result.staircases:
            if stair.floor <= selected_floor <= getattr(stair, "floor_end", stair.floor):
                special.append(stair)
        if view_mode in {"Floor Plan", "Parking Layout"}:
            fig = create_floor_plan(
                placements=floor_placements + special,
                site_width=saved["site_width_m"], site_length=saved["site_length_m"],
                buildable_bounds=saved["bounds_m"], parking=result.parking if selected_floor == 1 else None,
                road_access=saved["road_access"], floor=selected_floor,
            )
            fig.set_size_inches(7.4, 8.8)
            fig.set_dpi(110)
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        elif view_mode == "3D View":
            camera = st.selectbox("Camera angle", ["South-East", "South-West", "North-East", "North-West", "Top", "Street"], key="dashboard_camera")
            fig3 = create_3d_floor_plan(
                placements=result.placements, site_width=saved["site_width_m"], site_length=saved["site_length_m"],
                parking=result.parking, staircases=result.staircases, road_access=saved["road_access"], view=camera,
            )
            fig3.update_layout(height=700, autosize=True, paper_bgcolor="#fff", plot_bgcolor="#fff", font=dict(color="#142b59"),
                               dragmode="orbit", scene_dragmode="orbit")
            st.plotly_chart(fig3, use_container_width=True, config={"displaylogo": False, "responsive": True, "scrollZoom": True})
            st.caption("3D controls: drag to orbit 360°, scroll to zoom, and use the camera presets to jump to a view.")
        else:
            st.markdown("### 📄 Layout report")
            st.write(f"**Site:** {site_width_ft:g} × {site_length_ft:g} ft · **Floors:** {saved['number_of_floors']}")
            st.write(f"**Rooms placed:** {len(result.placements)} · **Room area:** {_feet_area(metrics.total_room_area):,.0f} sq ft")
            st.write(f"**Overlap violations:** {metrics.overlap_count} · **Boundary violations:** {metrics.boundary_violation_count}")
            st.markdown("| Room | Floor | Width | Length |\n|---|---:|---:|---:|\n" + "\n".join(f"| {p.name} | {p.floor} | {p.width/0.3048:.1f} ft | {p.length/0.3048:.1f} ft |" for p in result.placements))

with right:
    st.markdown("### ✅ Layout checks")
    result = st.session_state.get("layout_result")
    metrics = st.session_state.get("layout_metrics")
    if result is None:
        for label, detail in [("Site boundary", "Rooms remain inside the plot"), ("Buildable area", "Setbacks are respected"), ("Room overlaps", "No rooms occupy the same area"), ("Room dimensions", "Minimum dimensions checked"), ("Door placement", "Based on room connections"), ("Parking fit", "Driveway access checked"), ("Room connectivity", "Required room relationships"), ("Circulation", "Staircase and access")]:
            st.markdown(f"<div class='check-row'>🟦 <div><b>{label}</b><div class='muted'>{detail}</div></div></div>", unsafe_allow_html=True)
        st.caption("Checks will update after generation; they are not marked Pass before a plan is tested.")
    else:
        checks = [
            ("Site boundary", "All placed rooms stay inside the site", metrics.boundary_violation_count == 0),
            ("Room overlaps", "No overlapping room footprints", metrics.overlap_count == 0),
            ("Room dimensions", "All generated rooms have positive usable dimensions", all(p.width > 0 and p.length > 0 for p in result.placements)),
            ("Parking fit", "Parking footprint is included when requested", parking_type == "No parking" or result.parking is not None),
            ("Room connectivity", "Required attached/adjacent relationships", (metrics.attached_satisfied + metrics.adjacent_satisfied) == (metrics.attached_total + metrics.adjacent_total)),
            ("Circulation & access", "Staircase included when requested", not staircase_enabled or bool(result.staircases)),
        ]
        for label, detail, ok in checks:
            icon = "🟢" if ok else "🔴"
            status = "Pass" if ok else "Review"
            st.markdown(f"<div class='check-row'>{icon} <div style='flex:1'><b>{label}</b><div class='muted'>{detail}</div></div><span class='{'pass' if ok else ''}'>{status}</span></div>", unsafe_allow_html=True)
        st.markdown("### 🏆 Layout score")
        satisfaction = max(0, min(100, int(getattr(metrics, "relationship_satisfaction_percent", 0))))
        st.metric("Relationship satisfaction", f"{satisfaction}/100")
        st.progress(satisfaction / 100)
        enclosed_names = ("bedroom", "bathroom", "living", "dining", "kitchen", "utility", "foyer", "family")
        usable_room_area_m2 = sum(p.width * p.length for p in result.placements
                                  if any(tag in str(p.name).casefold() for tag in enclosed_names)
                                  and "balcony" not in str(p.name).casefold())
        total_enclosed_floor_area_m2 = max(0.0, buildable_area_ft2 * 0.09290304 * int(number_of_floors))
        enclosed_utilization = (100.0 * usable_room_area_m2 / total_enclosed_floor_area_m2
                                if total_enclosed_floor_area_m2 else 0.0)
        st.metric("Space utilization", f"{enclosed_utilization:.0f}%", "Usable enclosed room area ÷ total enclosed floor area")
        if enclosed_utilization < 85:
            st.caption("Below the 85% target. This is a planning target, not a guarantee; circulation, walls, stairs, and code requirements can reduce the achievable value.")
        if metrics.overlap_count == 0 and metrics.boundary_violation_count == 0:
            st.success("Basic geometry checks passed. Please verify door locations and local building rules before construction.")
        else:
            st.warning("The generated layout needs review; one or more geometry checks failed.")

    if result is not None:
        with st.expander("Room placement details"):
            st.markdown("| Room | Floor | X | Y | Width | Length |\n|---|---:|---:|---:|---:|---:|\n" + "\n".join(f"| {p.name} | {p.floor} | {p.x/0.3048:.1f} | {p.y/0.3048:.1f} | {p.width/0.3048:.1f} ft | {p.length/0.3048:.1f} ft |" for p in result.placements))
        with st.expander("Architectural relationships"):
            st.markdown("| Room A | Room B | Type |\n|---|---|---|\n" + "\n".join(f"| {r.room_a} | {r.room_b} | {r.kind} |" for r in st.session_state.get("layout_relationships", [])))
        try:
            entry = select_entrance_room(result.placements, side=road_access)
            if entry:
                st.caption(f"Suggested entrance: {entry.name} ({floor_label(entry.floor)})")
            suggestions = planning_suggestions(floor_area_ft2=floor_area_ft2, number_of_floors=int(number_of_floors), parking_type=parking_type, bedroom_count=int(bedroom_count))
            if suggestions:
                with st.expander("Planning suggestions"):
                    for item in suggestions:
                        st.write(f"• {item}")
        except Exception:
            pass

st.caption("Conceptual test-fit only. Confirm local bylaws, structure, ventilation, fire safety and door positions with a licensed professional.")
