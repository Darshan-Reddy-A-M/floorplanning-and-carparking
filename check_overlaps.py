from itertools import combinations

from testfit.optimizer import Placement, _rectangles_overlap

# Coordinates from your API response, in feet.
rooms = [
    Placement("Entry Foyer", 3.02, 5.02, 6, 6.99, 1),
    Placement("Living Room", 3.02, 12.01, 14.01, 16.01, 1),
    Placement("Dining", 3.02, 28.02, 8.99, 10.01, 1),
    Placement("Kitchen", 12.01, 28.02, 10.01, 10.99, 1),
    Placement("Utility", 15.03, 39.01, 6, 6.99, 1),
    Placement("Bedroom 1", 3.02, 39.01, 12.01, 12.01, 1),
    Placement("Bathroom 1", 17.03, 20.01, 4.99, 8.01, 1),
    Placement("Family Lounge", 14.01, 5.05, 10.99, 12.01, 2),
    Placement("Master Bedroom", 14.63, 17.06, 12.01, 12.99, 2),
    Placement("Bedroom 2", 3.02, 5.05, 10.99, 12.01, 2),
    Placement("Bedroom 3", 3.02, 17.06, 10.99, 12.01, 2),
    Placement("Bathroom 2", 25, 9.06, 4.99, 8.01, 2),
]

staircases = [
    Placement("Staircase F1-F2", 22.01, 30.05, 6.5, 12.99, 1),
]

parking = Placement("Parking", 17.03, 0, 8.99, 18.01, 0)

building_shapes = rooms + staircases

print("\nROOM / STAIRCASE OVERLAPS")
count = 0

for first, second in combinations(building_shapes, 2):
    if first.floor == second.floor and _rectangles_overlap(first, second):
        count += 1
        print(f"{first.name} overlaps {second.name}")

print("\nPARKING OVERLAPS")
for shape in building_shapes:
    if _rectangles_overlap(parking, shape):
        print(f"Parking overlaps {shape.name}")

print(f"\nTotal room/staircase overlaps: {count}")