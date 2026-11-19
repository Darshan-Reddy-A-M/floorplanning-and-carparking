
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from testfit.optimizer import calculate_metrics
from testfit.v4_optimizer import solve_building_layout
from testfit.planning import (
    feet_to_meters,
    recommend_room_program,
    setback_bounds_feet,
)

app = FastAPI(
    title="Planify AI API",
    description="Backend for Planify AI floor-plan generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    width: float = Field(default=40, gt=0, le=300)
    depth: float = Field(default=60, gt=0, le=300)
    unit: Literal["ft", "m"] = "ft"

    floors: int = Field(default=2, ge=1, le=4)
    building_type: Literal[
        "Single-Family House",
        "Duplex",
        "Rental / Multi-Unit Building",
        "Apartment / Multi-Unit Residential",
        "Custom",
    ] = "Duplex"

    road_access: Literal["North", "South", "East", "West"] = "South"

    front_setback: float = Field(default=5, ge=0)
    rear_setback: float = Field(default=5, ge=0)
    left_setback: float = Field(default=3, ge=0)
    right_setback: float = Field(default=3, ge=0)

    parking_spaces: int = Field(default=1, ge=0, le=2)
    time_limit_seconds: float = Field(default=15, gt=0, le=60)


@app.get("/")
def home():
    return {
        "message": "Planify AI backend is running",
        "status": "ok",
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "application": "Planify AI",
        "backend": "FastAPI",
    }


@app.post("/api/generate")
def generate_plan(request: GenerateRequest):
    try:
        # Convert site dimensions and setbacks to feet first.
        factor = 1.0 if request.unit == "ft" else 1.0 / 0.3048

        width_ft = request.width * factor
        depth_ft = request.depth * factor

        front_ft = request.front_setback
        rear_ft = request.rear_setback
        left_ft = request.left_setback
        right_ft = request.right_setback

        if request.unit == "m":
            front_ft *= factor
            rear_ft *= factor
            left_ft *= factor
            right_ft *= factor

        bounds_ft = setback_bounds_feet(
            width_ft,
            depth_ft,
            request.road_access,
            front_ft,
            rear_ft,
            left_ft,
            right_ft,
        )

        if bounds_ft[2] <= bounds_ft[0] or bounds_ft[3] <= bounds_ft[1]:
            raise HTTPException(
                status_code=422,
                detail="Setbacks leave no buildable area.",
            )

        # Use the project's existing room-program generator.
        parking_type = (
            "1 car" if request.parking_spaces == 1
            else "2 cars" if request.parking_spaces == 2
            else "No parking"
        )

        parking_dimensions_ft = {
            "1 car": (9.0, 18.0),
            "2 cars": (18.0, 18.0),
        }

        parking_width_m = None
        parking_length_m = None
        parking_area_ft2 = 0.0

        if parking_type != "No parking":
            parking_width_ft, parking_length_ft = parking_dimensions_ft[parking_type]
            parking_width_m = feet_to_meters(parking_width_ft)
            parking_length_m = feet_to_meters(parking_length_ft)
            parking_area_ft2 = parking_width_ft * parking_length_ft

        staircase_width_ft = 6.5
        staircase_length_ft = 13.0

        rooms, relationships, _ = recommend_room_program(
            buildable_bounds_ft=bounds_ft,
            number_of_floors=request.floors,
            building_type=request.building_type,
            units_per_floor=1,
            parking_area_ft2=parking_area_ft2,
            staircase_size_ft=(
                staircase_width_ft,
                staircase_length_ft,
            ),
        )

        if not rooms:
            raise HTTPException(
                status_code=422,
                detail="The room planner did not create any rooms.",
            )

        width_m = feet_to_meters(width_ft)
        depth_m = feet_to_meters(depth_ft)
        bounds_m = tuple(feet_to_meters(v) for v in bounds_ft)

        result = solve_building_layout(
            site_width=width_m,
            site_length=depth_m,
            rooms=rooms,
            relationships=relationships,
            time_limit_seconds=request.time_limit_seconds,
            number_of_floors=request.floors,
            parking_type=parking_type,
            parking_width=parking_width_m,
            parking_length=parking_length_m,
            road_access=request.road_access,
            staircase_width=feet_to_meters(staircase_width_ft),
            staircase_length=feet_to_meters(staircase_length_ft),
            buildable_bounds=bounds_m,
            staircase_mode=(
                "internal"
                if request.building_type in {
                    "Duplex",
                    "Apartment / Multi-Unit Residential",
                }
                else "external"
            ),
        )

        if not result.placements:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "No feasible layout was found.",
                    "optimizer_status": result.status,
                    "optimizer_message": result.message,
                },
            )

        metrics = calculate_metrics(
            site_width=width_m,
            site_length=depth_m,
            placements=result.placements,
            relationships=relationships,
            number_of_floors=request.floors,
            parking=result.parking,
            staircases=result.staircases,
        )

        # Return display-friendly coordinates in feet.
        placements = [
            {
                "name": p.name,
                "x": round(p.x / 0.3048, 2),
                "y": round(p.y / 0.3048, 2),
                "width": round(p.width / 0.3048, 2),
                "depth": round(p.length / 0.3048, 2),
                "floor": p.floor,
            }
            for p in result.placements
        ]

        def serialize_special(p):
            if p is None:
                return None
            return {
                "name": p.name,
                "x": round(p.x / 0.3048, 2),
                "y": round(p.y / 0.3048, 2),
                "width": round(p.width / 0.3048, 2),
                "depth": round(p.length / 0.3048, 2),
                "floor": p.floor,
            }

        return {
            "status": result.status,
            "message": result.message,
            "site": {
                "width": width_ft,
                "depth": depth_ft,
                "unit": "ft",
                "buildable_bounds": [
                    round(v, 2) for v in bounds_ft
                ],
            },
            "floors": request.floors,
            "rooms": placements,
            "parking": serialize_special(result.parking),
            "staircases": [
                serialize_special(p) for p in result.staircases
            ],
            "metrics": {
    		"site_area_sqft": round(width_ft * depth_ft, 2),
    		"room_area_sqm": round(metrics.total_room_area, 2),
    		"overlap_count": metrics.overlap_count,
    		"boundary_violation_count": metrics.boundary_violation_count,
    		"relationship_satisfaction_percent": round(
        		metrics.relationship_satisfaction_percent, 2
    		),
    		"attached_satisfied": metrics.attached_satisfied,
    		"attached_total": metrics.attached_total,
    		"adjacent_satisfied": metrics.adjacent_satisfied,
    		"adjacent_total": metrics.adjacent_total,
    		"near_satisfied": metrics.near_satisfied,
    		"near_total": metrics.near_total,
    		"preferred_satisfied": metrics.preferred_satisfied,
    		"preferred_total": metrics.preferred_total,
    		"stacked_satisfied": metrics.stacked_satisfied,
    		"stacked_total": metrics.stacked_total,
},
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Floor-plan generation failed: {exc}",
        ) from exc
