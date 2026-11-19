"""
Dataset-derived architectural knowledge layer.

This module reads the compact ResPlan-derived rules from:

    data/knowledge/architectural_rules.json

It deliberately stays separate from the OR-Tools solver.

The solver should use this module for:
- room-program recommendations
- learned room relationships
- architectural preferences

It should NOT treat statistical dataset priors as absolute
architectural requirements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_RULES_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "knowledge"
    / "architectural_rules.json"
)


class ArchitecturalKnowledge:
    """Access ResPlan-derived architectural knowledge."""

    def __init__(
        self,
        rules_path: str | Path | None = None,
    ) -> None:

        self.rules_path = (
            Path(rules_path)
            if rules_path is not None
            else DEFAULT_RULES_PATH
        )

        self._rules = self._load_rules()

    def _load_rules(self) -> dict[str, Any]:
        """Load and validate the architectural rules JSON."""

        if not self.rules_path.exists():
            raise FileNotFoundError(
                "Architectural rules file not found: "
                f"{self.rules_path}"
            )

        with self.rules_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            rules = json.load(file)

        if not isinstance(rules, dict):
            raise ValueError(
                "Architectural rules must contain a JSON object."
            )

        required_sections = {
            "room_program",
            "connectivity",
            "architectural_presence",
            "hard_constraints",
            "soft_constraints",
        }

        missing = sorted(
            required_sections - set(rules.keys())
        )

        if missing:
            raise ValueError(
                "Architectural rules are missing required "
                f"sections: {missing}"
            )

        return rules

    @property
    def rules(self) -> dict[str, Any]:
        """Return the complete rules dictionary."""
        return self._rules

    # ------------------------------------------------------------
    # Room program
    # ------------------------------------------------------------

    def room_program(
        self,
        room_type: str,
    ) -> dict[str, Any]:
        """Return the learned program information for a room."""

        room_type = str(room_type).casefold()

        return self._rules.get(
            "room_program",
            {},
        ).get(
            room_type,
            {},
        )

    def preferred_count(
        self,
        room_type: str,
    ) -> int | None:
        """Return the dataset-derived preferred room count."""

        program = self.room_program(room_type)

        expected = program.get(
            "expected_count",
            {},
        )

        value = expected.get("preferred")

        if value is None:
            return None

        return int(value)

    def minimum_count(
        self,
        room_type: str,
    ) -> int | None:
        """Return the soft/default minimum count."""

        program = self.room_program(room_type)

        expected = program.get(
            "expected_count",
            {},
        )

        value = expected.get("minimum")

        if value is None:
            return None

        return int(value)

    def maximum_count(
        self,
        room_type: str,
    ) -> int | None:
        """Return the soft/default maximum count."""

        program = self.room_program(room_type)

        expected = program.get(
            "expected_count",
            {},
        )

        value = expected.get("maximum")

        if value is None:
            return None

        return int(value)

    def presence_rate(
        self,
        room_type: str,
    ) -> float | None:
        """Return observed ResPlan presence rate."""

        program = self.room_program(room_type)

        value = program.get(
            "presence_rate"
        )

        if value is None:
            return None

        return float(value)

    # ------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------

    def strong_relationships(
        self,
    ) -> list[dict[str, Any]]:
        """Return strong dataset-observed room relationships."""

        return list(
            self._rules.get(
                "connectivity",
                {},
            ).get(
                "strong_relationships",
                [],
            )
        )

    def common_adjacencies(
        self,
    ) -> list[dict[str, Any]]:
        """Return common adjacency relationships."""

        return list(
            self._rules.get(
                "connectivity",
                {},
            ).get(
                "common_adjacencies",
                [],
            )
        )

    def front_door_relationships(
        self,
    ) -> list[dict[str, Any]]:
        """Return observed front-door relationships."""

        return list(
            self._rules.get(
                "connectivity",
                {},
            ).get(
                "front_door_relationships",
                [],
            )
        )

    def relationship_exists(
        self,
        room_a: str,
        room_b: str,
        relation: str | None = None,
    ) -> bool:
        """
        Check whether a relationship was observed in ResPlan.

        Room order is treated as irrelevant.
        """

        a = str(room_a).casefold()
        b = str(room_b).casefold()

        relation_value = (
            str(relation).casefold()
            if relation is not None
            else None
        )

        for item in self.strong_relationships():

            rooms = item.get(
                "rooms",
                [],
            )

            if len(rooms) != 2:
                continue

            first = str(
                rooms[0]
            ).casefold()

            second = str(
                rooms[1]
            ).casefold()

            same_pair = (
                {a, b}
                == {first, second}
            )

            if not same_pair:
                continue

            if relation_value is None:
                return True

            observed_relation = str(
                item.get(
                    "relation",
                    "",
                )
            ).casefold()

            if observed_relation == relation_value:
                return True

        return False

    # ------------------------------------------------------------
    # Architectural preferences
    # ------------------------------------------------------------

    def design_preference(
        self,
        room_type: str,
    ) -> dict[str, Any]:
        """Return learned design preferences for a room."""

        room_type = str(
            room_type
        ).casefold()

        return self._rules.get(
            "design_preferences",
            {},
        ).get(
            room_type,
            {},
        )

    def preferred_connections(
        self,
        room_type: str,
    ) -> list[str]:
        """Return preferred connected room types."""

        preference = self.design_preference(
            room_type
        )

        return list(
            preference.get(
                "preferred_connections",
                [],
            )
        )

    # ------------------------------------------------------------
    # Hard / soft constraints
    # ------------------------------------------------------------

    def hard_constraints(
        self,
    ) -> dict[str, Any]:
        """Return hard program requirements."""

        return dict(
            self._rules.get(
                "hard_constraints",
                {},
            )
        )

    def soft_constraints(
        self,
    ) -> dict[str, Any]:
        """Return soft architectural preferences."""

        return dict(
            self._rules.get(
                "soft_constraints",
                {},
            )
        )

    # ------------------------------------------------------------
    # Architectural element presence
    # ------------------------------------------------------------

    def element_presence_rate(
        self,
        element: str,
    ) -> float | None:
        """Return observed presence rate of an element."""

        element = str(
            element
        ).casefold()

        data = self._rules.get(
            "architectural_presence",
            {},
        ).get(
            element,
            {},
        )

        value = data.get(
            "presence_rate"
        )

        if value is None:
            return None

        return float(value)

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """
        Return a small summary useful for debugging and tests.
        """

        room_program = self._rules.get(
            "room_program",
            {},
        )

        connectivity = self._rules.get(
            "connectivity",
            {},
        )

        return {
            "rules_path": str(
                self.rules_path
            ),
            "room_types": sorted(
                room_program.keys()
            ),
            "strong_relationship_count": len(
                connectivity.get(
                    "strong_relationships",
                    [],
                )
            ),
            "common_adjacency_count": len(
                connectivity.get(
                    "common_adjacencies",
                    [],
                )
            ),
            "front_door_relationship_count": len(
                connectivity.get(
                    "front_door_relationships",
                    [],
                )
            ),
        }


def load_architectural_knowledge(
    rules_path: str | Path | None = None,
) -> ArchitecturalKnowledge:
    """
    Convenience function for loading the knowledge layer.
    """

    return ArchitecturalKnowledge(
        rules_path=rules_path
    )