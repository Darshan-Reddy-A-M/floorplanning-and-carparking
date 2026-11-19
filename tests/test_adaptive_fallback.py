"""Regression checks for compact plot generation and requested door rules."""
import unittest
from types import SimpleNamespace

from testfit.fallback_layout import _edge_share, solve_fallback_layout
from testfit.optimizer import Room, RoomRelationship, calculate_metrics
from testfit.visualization import _door_pair_kind

FT = 0.3048


def sample_program():
    rooms_ft = [
        ("Entry Foyer", 6, 7, 1), ("Living Room", 17.5, 16, 1),
        ("Dining", 9, 10, 1), ("Kitchen", 10, 11, 1), ("Utility", 6, 7, 1),
        ("Bedroom 1", 12, 12, 1), ("Bathroom 1", 5, 8, 1),
        ("Upper Living Room", 17.5, 16, 2), ("Bedroom 2", 11, 12, 2),
        ("Bedroom 3", 11, 12, 2), ("Bathroom 2", 5, 8, 2),
        ("Bathroom 3", 5, 8, 2), ("Balcony", 6, 5, 2),
    ]
    rooms = [Room(name, w * FT, l * FT, floor) for name, w, l, floor in rooms_ft]
    relationships = [
        RoomRelationship("Entry Foyer", "Living Room", "ADJACENT"),
        RoomRelationship("Living Room", "Dining", "ADJACENT"),
        RoomRelationship("Dining", "Kitchen", "ADJACENT"),
        RoomRelationship("Kitchen", "Utility", "ADJACENT"),
        RoomRelationship("Bedroom 1", "Bathroom 1", "ATTACHED"),
        RoomRelationship("Living Room", "Upper Living Room", "STACKED"),
        RoomRelationship("Bedroom 2", "Bathroom 2", "ATTACHED"),
        RoomRelationship("Bedroom 3", "Bathroom 3", "ATTACHED"),
    ]
    return rooms, relationships


class AdaptiveFallbackTests(unittest.TestCase):
    def test_compact_and_standard_plots_generate_non_overlapping_layouts(self):
        rooms, relationships = sample_program()
        for width_ft, length_ft in ((20, 30), (30, 50), (40, 60)):
            with self.subTest(site=(width_ft, length_ft)):
                bounds = (3 * FT, 5 * FT, (width_ft - 3) * FT, (length_ft - 3) * FT)
                result = solve_fallback_layout(
                    site_width=width_ft * FT,
                    site_length=length_ft * FT,
                    rooms=rooms,
                    time_limit_seconds=1,
                    relationships=relationships,
                    number_of_floors=2,
                    parking_type="2 cars",
                    parking_width=18 * FT,
                    parking_length=18 * FT,
                    road_access="South",
                    staircase_width=3 * FT,
                    staircase_length=10 * FT,
                    buildable_bounds=bounds,
                )
                self.assertTrue(result.placements, result.message)
                metrics = calculate_metrics(
                    width_ft * FT, length_ft * FT, result.placements, relationships,
                    number_of_floors=2, parking=result.parking, staircases=result.staircases,
                )
                self.assertEqual(metrics.overlap_count, 0)
                self.assertEqual(metrics.boundary_violation_count, 0)
                self.assertEqual(metrics.attached_satisfied, metrics.attached_total)
                placed = {p.name: p for p in result.placements}
                self.assertIn("Utility", placed)
                kitchen, utility = placed["Kitchen"], placed["Utility"]
                self.assertEqual(kitchen.floor, utility.floor)
                self.assertTrue(_edge_share(kitchen, utility, 0.80), "Kitchen and Utility need a door-width shared wall")
                self.assertTrue(_door_pair_kind(SimpleNamespace(name="Kitchen"), SimpleNamespace(name="Utility")))

    def test_stacked_living_rooms_are_given_a_cross_floor_alignment_preference(self):
        # The fallback must consider STACKED relationships across floors; previously
        # it searched only same-floor rooms and silently ignored this relationship.
        from testfit.fallback_layout import _pack_floor_rooms
        lower = Room("Living Room", 4.0, 4.0, 1)
        upper = Room("Upper Living Room", 4.0, 4.0, 2)
        relationships = [RoomRelationship("Living Room", "Upper Living Room", "STACKED")]
        all_placements = []
        bounds = (0.0, 0.0, 12.0, 12.0)
        self.assertIsNotNone(_pack_floor_rooms([lower], [], bounds, None, None, 12.0, 12.0, "South", relationships, all_placements))
        packed = _pack_floor_rooms([upper], [], bounds, None, None, 12.0, 12.0, "South", relationships, all_placements)
        self.assertIsNotNone(packed)
        placed_upper = next(p for p in all_placements if p.name == "Upper Living Room")
        placed_lower = next(p for p in all_placements if p.name == "Living Room")
        self.assertEqual((placed_upper.x, placed_upper.y), (placed_lower.x, placed_lower.y))

    def test_bathroom_two_does_not_open_into_bedroom_three(self):
        self.assertIsNone(_door_pair_kind(SimpleNamespace(name="Bathroom 2"), SimpleNamespace(name="Bedroom 3")))
        self.assertIsNotNone(_door_pair_kind(SimpleNamespace(name="Bathroom 2"), SimpleNamespace(name="Bedroom 2")))


if __name__ == "__main__":
    unittest.main()
