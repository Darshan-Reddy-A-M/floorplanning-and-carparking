"""Tests for the offline dataset inspection tools.

All tests use tiny synthetic fixtures where practical. The RPLAN tests are
aligned with the current vector/graph inspector and do not assume the old
PNG/raster representation.
"""

import json
import tempfile
import unittest
from pathlib import Path

from tools.dataset import (
    check_repo_clean,
    common,
    inspect_cubicasa,
    inspect_rplan,
    summarize_inspections,
)


SVG_TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg"
width="1000" height="800" viewBox="0 0 1000 800">
<g id="Wall" class="Wall"><polygon points="0,0 10,0 10,10"/></g>
<g id="Space" class="Space Kitchen">
    <polygon points="100,100 300,100 300,300 100,300"/>
</g>
<g id="Space2" class="Space Living">
    <polygon points="304,100 604,100 604,300 304,300"/>
</g>
<g class="Space Bed">
    <path d="M 100 400 L 300 400 L 300 700 L 100 700 Z"/>
</g>
<g class="Door"><rect x="296" y="150" width="8" height="40"/></g>
<g class="Window"><rect x="100" y="95" width="60" height="10"/></g>
{extra}
<metadata>scale 1:100</metadata>
</svg>"""


def make_cubicasa(root: Path, extra_svg: str = "") -> None:
    svg = SVG_TEMPLATE.format(extra=extra_svg)

    for plan_id, floors in (
        (1001, ("F1",)),
        (1002, ("F1", "F2")),
        (1003, ("F1",)),
    ):
        folder = root / "high_quality_fixture" / str(plan_id)
        folder.mkdir(parents=True)

        (folder / "model.svg").write_text(
            svg,
            encoding="utf-8",
        )

        for floor in floors:
            (folder / f"{floor}_original.png").write_bytes(b"x")

    (root / "train.txt").write_text(
        "high_quality_fixture/1001/\n"
        "high_quality_fixture/1002/\n"
        "high_quality_fixture/9999/\n",
        encoding="utf-8",
    )


class RplanInspectorTests(unittest.TestCase):
    """Tests for the current ResPlan vector/graph inspector."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "fixture_rplan"

        report_path = Path("data/inspection/rplan_inspection.json")

        if report_path.exists():
            self.report = json.loads(
                report_path.read_text(encoding="utf-8")
            )
        else:
            self.report = None

    def tearDown(self):
        self._tmp.cleanup()

    def require_report(self):
        if self.report is None:
            self.skipTest(
                "data/inspection/rplan_inspection.json is not available"
            )

    def test_report_has_current_schema(self):
        self.require_report()

        self.assertEqual(
            self.report["schema_version"],
            1,
        )

        self.assertEqual(
            self.report["dataset"],
            "ResPlan",
        )

        self.assertEqual(
            self.report["tool"],
            "resplan_vector_graph_inspector_v1",
        )

    def test_pickle_is_detected(self):
        self.require_report()

        self.assertTrue(
            self.report["files"]["ResPlan.pkl"]["exists"]
        )

        self.assertGreater(
            self.report["files"]["ResPlan.pkl"]["bytes"],
            0,
        )

    def test_dataset_has_expected_plan_count(self):
        self.require_report()

        evidence = self.report["findings"]["sample_count"]["evidence"]

        self.assertEqual(
            evidence["pickle_dataset_length"],
            17000,
        )

    def test_split_counts_are_reported(self):
        self.require_report()

        counts = (
            self.report["findings"]
            ["sample_count"]
            ["evidence"]
            ["split"]
            ["counts"]
        )

        self.assertEqual(
            counts["train"],
            13053,
        )

        self.assertEqual(
            counts["val"],
            1632,
        )

        self.assertEqual(
            counts["test"],
            1632,
        )

        self.assertEqual(
            counts["augmented"],
            683,
        )

    def test_vector_geometry_is_detected(self):
        self.require_report()

        geometry = self.report["aggregate"]["geometry_types"]

        self.assertIn(
            "Polygon",
            geometry,
        )

        self.assertIn(
            "MultiPolygon",
            geometry,
        )

        self.assertGreater(
            geometry["Polygon"],
            0,
        )

        self.assertGreater(
            geometry["MultiPolygon"],
            0,
        )

    def test_room_labels_are_detected(self):
        self.require_report()

        labels = self.report["aggregate"]["room_labels"]

        expected = {
            "living",
            "kitchen",
            "bedroom",
            "bathroom",
            "balcony",
            "front_door",
            "stair",
            "storage",
        }

        self.assertTrue(
            expected.issubset(labels.keys())
        )

    def test_graph_edge_types_are_detected(self):
        self.require_report()

        edges = self.report["aggregate"]["edge_types"]

        for edge_type in (
            "via_door",
            "direct",
            "adjacency",
            "via_window",
        ):
            self.assertIn(
                edge_type,
                edges,
            )

            self.assertGreater(
                edges[edge_type],
                0,
            )

    def test_graph_statistics_are_reported(self):
        self.require_report()

        evidence = self.report["findings"]["adjacency"]["evidence"]

        nodes = evidence["graph_nodes"]
        edges = evidence["graph_edges"]

        self.assertGreater(
            nodes["min"],
            0,
        )

        self.assertGreater(
            nodes["average"],
            0,
        )

        self.assertGreater(
            edges["min"],
            0,
        )

        self.assertGreater(
            edges["average"],
            0,
        )

    def test_architectural_elements_are_reported(self):
        self.require_report()

        findings = self.report["findings"]

        for key in (
            "annotations",
            "doors_windows",
            "entrances",
            "stairs",
            "parking_garage",
            "multi_floor",
        ):
            self.assertIn(
                key,
                findings,
            )

            self.assertIn(
                "status",
                findings[key],
            )

    def test_adjacency_is_reported(self):
        self.require_report()

        self.assertIn(
            self.report["findings"]["adjacency"]["status"],
            {
                common.STATUS_CONFIRMED,
                common.STATUS_PARTIAL,
            },
        )

    def test_room_areas_are_reported_with_scale_caveat(self):
        self.require_report()

        finding = self.report["findings"]["room_areas"]

        self.assertIn(
            finding["status"],
            {
                common.STATUS_CONFIRMED,
                common.STATUS_PARTIAL,
            },
        )

        self.assertTrue(
            finding["caveat"]
        )

    def test_room_aspect_ratios_are_reported(self):
        self.require_report()

        finding = self.report["findings"]["room_aspect_ratios"]

        self.assertIn(
            finding["status"],
            {
                common.STATUS_CONFIRMED,
                common.STATUS_PARTIAL,
            },
        )

    def test_doors_and_windows_are_reported(self):
        self.require_report()

        self.assertIn(
            self.report["findings"]["doors_windows"]["status"],
            {
                common.STATUS_CONFIRMED,
                common.STATUS_PARTIAL,
            },
        )

    def test_entrances_are_reported(self):
        self.require_report()

        self.assertIn(
            self.report["findings"]["entrances"]["status"],
            {
                common.STATUS_CONFIRMED,
                common.STATUS_PARTIAL,
            },
        )

    def test_stairs_are_reported(self):
        self.require_report()

        self.assertIn(
            self.report["findings"]["stairs"]["status"],
            {
                common.STATUS_CONFIRMED,
                common.STATUS_PARTIAL,
                common.STATUS_NOT_FOUND,
                common.STATUS_CANNOT,
            },
        )

    def test_multi_floor_status_is_explicit(self):
        self.require_report()

        finding = self.report["findings"]["multi_floor"]

        self.assertIn(
            finding["status"],
            {
                common.STATUS_CONFIRMED,
                common.STATUS_PARTIAL,
                common.STATUS_NOT_FOUND,
                common.STATUS_CANNOT,
            },
        )

    def test_reliable_priors_are_reported(self):
        self.require_report()

        finding = self.report["findings"]["reliable_priors"]

        self.assertIn(
            "evidence",
            finding,
        )

        self.assertIn(
            "candidate_priors",
            finding["evidence"],
        )

    def test_sample_plans_are_present(self):
        self.require_report()

        self.assertIn(
            "sample_plans",
            self.report,
        )

        self.assertGreater(
            len(self.report["sample_plans"]),
            0,
        )

    def test_report_does_not_expose_absolute_dataset_path(self):
        self.require_report()

        text = json.dumps(
            self.report
        )

        self.assertNotIn(
            "C:\\Darshan Reddy\\FloorPlanning_Project\\Datasets",
            text,
        )


class CubiCasaInspectorTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

        self.root = (
            Path(self._tmp.name)
            / "fixture_cubicasa"
        )

        make_cubicasa(
            self.root
        )

        self.report = inspect_cubicasa.inspect(
            self.root,
            sample=10,
            geom_sample=10,
        )

        self.findings = self.report["findings"]
        self.geometry = self.report["geometry"]["per_category"]

    def tearDown(self):
        self._tmp.cleanup()

    def test_plan_structure_and_floor_tokens(self):
        structure = self.report["plan_structure"]

        self.assertEqual(
            structure["plan_directories"],
            3,
        )

        self.assertEqual(
            structure["floor_tokens_per_plan_histogram"],
            {1: 2, 2: 1},
        )

        self.assertEqual(
            self.findings["multi_floor"]["status"],
            common.STATUS_PARTIAL,
        )

    def test_split_list_coverage(self):
        split = self.report["split_lists"][0]

        self.assertEqual(
            (
                split["lines"],
                split["lines_matching_plan_dirs"],
            ),
            (3, 2),
        )

    def test_polygon_rect_and_path_geometry_are_measured_exactly(self):
        self.assertEqual(
            self.geometry["kitchen"]["area"]["median"],
            200 * 200,
        )

        self.assertAlmostEqual(
            self.geometry["living"]["aspect"]["median"],
            1.5,
        )

        self.assertEqual(
            self.geometry["bed"]["area"]["median"],
            200 * 300,
        )

        self.assertEqual(
            self.geometry["kitchen"]["vertices"]["median"],
            4,
        )

    def test_adjacency_only_between_near_touching_rooms(self):
        pairs = (
            self.report["geometry"]
            ["most_common_near_pairs_plans"]
        )

        self.assertEqual(
            [p[0] for p in pairs],
            [["kitchen", "living"]],
        )

        self.assertEqual(
            self.findings["adjacency"]["status"],
            common.STATUS_CONFIRMED,
        )

    def test_present_features_confirmed_absent_features_not_found(self):
        self.assertEqual(
            self.findings["doors_windows"]["status"],
            common.STATUS_CONFIRMED,
        )

        for key in (
            "entrances",
            "stairs",
            "parking_garage",
        ):
            self.assertEqual(
                self.findings[key]["status"],
                common.STATUS_NOT_FOUND,
                key,
            )

    def test_stairs_are_detected_when_they_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "with_stairs"

            make_cubicasa(
                root,
                extra_svg=(
                    '<g class="Stairs">'
                    '<rect x="1" y="1" '
                    'width="50" height="100"/>'
                    "</g>"
                    '<g class="Garage">'
                    '<rect x="0" y="0" '
                    'width="300" height="600"/>'
                    "</g>"
                ),
            )

            report = inspect_cubicasa.inspect(
                root,
                sample=10,
                geom_sample=10,
            )

        self.assertEqual(
            report["findings"]["stairs"]["status"],
            common.STATUS_CONFIRMED,
        )

        self.assertEqual(
            report["findings"]["parking_garage"]["status"],
            common.STATUS_CONFIRMED,
        )

    def test_scale_hints_are_recorded_but_do_not_assert_a_scale(self):
        hints = [
            h[0]
            for h in self.report["svg"]["scale_hints"]
        ]

        self.assertTrue(
            any(
                "scale 1:100" in h
                for h in hints
            )
        )

        self.assertEqual(
            self.findings["room_areas"]["status"],
            common.STATUS_PARTIAL,
        )

    def test_malformed_svg_is_counted_not_fatal(self):
        (
            self.root
            / "high_quality_fixture"
            / "1001"
            / "model.svg"
        ).write_text(
            "<svg><oops>",
            encoding="utf-8",
        )

        report = inspect_cubicasa.inspect(
            self.root,
            sample=10,
            geom_sample=10,
        )

        self.assertEqual(
            len(report["svg"]["parse_failures"]),
            1,
        )

        self.assertEqual(
            report["svg"]["files_parsed"],
            2,
        )

    def test_output_never_contains_the_absolute_dataset_path(self):
        self.assertNotIn(
            str(self.root),
            json.dumps(self.report),
        )

    def test_path_parser(self):
        pts, multi = inspect_cubicasa.parse_path(
            "M 0 0 H 10 V 5 H 0 Z"
        )

        self.assertEqual(
            pts,
            [
                (0, 0),
                (10, 0),
                (10, 5),
                (0, 5),
            ],
        )

        rel, _ = inspect_cubicasa.parse_path(
            "m 1 1 l 4 0 l 0 3 z"
        )

        self.assertEqual(
            rel,
            [
                (1, 1),
                (5, 1),
                (5, 4),
            ],
        )

        self.assertIsNone(
            inspect_cubicasa.parse_path(
                "M 0 0 C 1 1 2 2 3 3 Z"
            )[0]
        )

        _, multi = inspect_cubicasa.parse_path(
            "M 0 0 L 1 0 L 1 1 Z "
            "M 5 5 L 6 5 L 6 6 Z"
        )

        self.assertTrue(multi)

    def test_near_pairs_ignores_interior_overlap_and_far_rooms(self):
        def room(x0, y0, x1, y1):
            return (
                "room",
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                },
            )

        a = (
            "a",
            room(0, 0, 100, 100)[1],
        )

        self.assertEqual(
            inspect_cubicasa.near_pairs(
                [
                    a,
                    (
                        "b",
                        room(
                            100,
                            0,
                            200,
                            100,
                        )[1],
                    ),
                ]
            ),
            {("a", "b")},
        )

        self.assertEqual(
            inspect_cubicasa.near_pairs(
                [
                    a,
                    (
                        "b",
                        room(
                            50,
                            50,
                            150,
                            150,
                        )[1],
                    ),
                ]
            ),
            set(),
        )

        self.assertEqual(
            inspect_cubicasa.near_pairs(
                [
                    a,
                    (
                        "b",
                        room(
                            400,
                            0,
                            500,
                            100,
                        )[1],
                    ),
                ]
            ),
            set(),
        )


class SafetyAndSummaryTests(unittest.TestCase):

    def test_guard_refuses_datasets_inside_the_repo(self):
        inside = common.repo_root() / "tests"

        with self.assertRaises(SystemExit):
            common.guard_dataset_root(inside)

        self.assertEqual(
            common.guard_dataset_root(
                inside,
                allow_inside_repo=True,
            ),
            inside.resolve(),
        )

        with self.assertRaises(SystemExit):
            common.guard_dataset_root(
                common.repo_root()
                / "does_not_exist"
            )

    def test_repo_scan_flags_raw_dataset_lookalikes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            (root / "reports").mkdir()

            (
                root / "reports" / "ok.png"
            ).write_bytes(
                b"x" * 10
            )

            (
                root / "big.bin"
            ).write_bytes(
                b"x" * 2_000_000
            )

            (root / "plan").mkdir()

            (
                root / "plan" / "model.svg"
            ).write_text(
                "<svg/>",
                encoding="utf-8",
            )

            (root / "imgs").mkdir()

            for i in range(45):
                (
                    root / "imgs" / f"{i}.png"
                ).write_bytes(b"x")

            (
                root / "data.h5"
            ).write_bytes(b"x")

            problems = check_repo_clean.scan(
                [
                    p
                    for p in root.rglob("*")
                    if p.is_file()
                ],
                root,
            )

        text = "\n".join(problems)

        for expected in (
            "big.bin",
            "model.svg",
            "imgs/",
            "data.h5",
        ):
            self.assertIn(
                expected,
                text,
            )

        self.assertNotIn(
            "ok.png",
            text,
        )

    def test_this_repository_is_clean(self):
        root = common.repo_root()

        self.assertEqual(
            check_repo_clean.scan(
                check_repo_clean.candidate_files(root),
                root,
            ),
            [],
        )

    def test_gitignore_blocks_raw_datasets(self):
        text = (
            common.repo_root()
            / ".gitignore"
        ).read_text(
            encoding="utf-8"
        )

        for entry in (
            "data/raw/",
            "RPLAN/",
            "cubicasa*/",
            "*.zip",
        ):
            self.assertIn(
                entry,
                text,
            )

    def test_summary_report_uses_current_inspection_schema(self):
        rplan_path = Path(
            "data/inspection/rplan_inspection.json"
        )

        cubi_path = Path(
            "data/inspection/cubicasa_inspection.json"
        )

        if not rplan_path.exists():
            self.skipTest(
                "RPLAN inspection report is not available"
            )

        if not cubi_path.exists():
            self.skipTest(
                "CubiCasa inspection report is not available"
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            out = tmp / "out.md"

            import sys

            argv = sys.argv

            sys.argv = [
                "x",
                "--rplan",
                str(rplan_path),
                "--cubicasa",
                str(cubi_path),
                "--out",
                str(out),
            ]

            try:
                summarize_inspections.main()
            finally:
                sys.argv = argv

            text = out.read_text(
                encoding="utf-8"
            )

        self.assertIn(
            "# Dataset inspection findings",
            text,
        )

        self.assertIn(
            "## RPLAN",
            text,
        )

        self.assertIn(
            "## CubiCasa5K",
            text,
        )

        self.assertIn(
            "15. Which features are reliable enough to use as priors?",
            text,
        )

        self.assertIn(
            "### CP-SAT remains responsible",
            text,
        )

    def test_helpers(self):
        self.assertEqual(
            common.normalize_name(
                "Wall12-3"
            ),
            "Wall#-#",
        )

        s = common.percentile_summary(
            [1, 2, 3, 4, 5]
        )

        self.assertEqual(
            (
                s["min"],
                s["median"],
                s["max"],
            ),
            (1, 3, 5),
        )

        self.assertIsNone(
            common.percentile_summary([])
        )


if __name__ == "__main__":
    unittest.main()