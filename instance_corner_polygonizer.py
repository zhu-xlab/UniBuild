#!/usr/bin/env python3
"""Direction-aware building-instance corner polygonization.

Pipeline:
    binary mask
      -> connected component instances
      -> contour extraction
      -> one-pass approxPolyDP
      -> dominant-direction-aware pruning
      -> optional dominant-axis snapping
      -> raster mask / JSON vertices / visualization / summary.csv

This script does not perform expensive epsilon/IoU search and does not depend on
geopandas, shapely, or buildingregulariser. It only requires OpenCV and NumPy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class ContourResult:
    contour_index: int
    parent_index: int
    depth: int
    is_hole: bool
    original_points: int
    vertex_count: int
    dominant_angle_deg: float
    contour_local: np.ndarray
    contour_global: np.ndarray


@dataclass
class InstanceResult:
    instance_id: int
    bbox_xywh: tuple[int, int, int, int]
    area: int
    original_points: int
    vertex_count: int
    iou: float
    rendered_crop: np.ndarray
    original_contours_global: list[np.ndarray]
    polygon_contours_global: list[np.ndarray]
    contour_records: list[ContourResult]


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast direction-aware polygonization for binary building labels. "
            "It extracts connected-component instances, simplifies each contour once, "
            "removes redundant vertices according to local and dominant directions, "
            "and optionally snaps edges to the dominant building axes."
        )
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "massachusetts_labels",
        help="Directory containing label masks.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "massachusetts_images",
        help="Optional directory containing matching source images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "massachusetts_contour_directional",
        help="Directory for masks, JSON vertices, visualizations, and summary.csv.",
    )
    parser.add_argument(
        "--pattern",
        default="*.tif*",
        help="Glob used to select label files.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Foreground threshold used when reading labels.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=4,
        help="Connected-component connectivity.",
    )
    parser.add_argument(
        "--min-instance-area",
        type=int,
        default=5,
        help="Ignore connected components smaller than this area in pixels.",
    )
    parser.add_argument(
        "--bbox-padding",
        type=int,
        default=1,
        help="Extra padding around each instance bbox before contour extraction.",
    )
    parser.add_argument(
        "--chain-approx",
        choices=("none", "simple"),
        default="simple",
        help="Contour extraction mode. 'simple' is much faster and usually enough.",
    )
    parser.add_argument(
        "--epsilon-ratio",
        type=float,
        default=0.0015,
        help="One-pass approxPolyDP epsilon divided by contour perimeter.",
    )
    parser.add_argument(
        "--min-epsilon-px",
        type=float,
        default=1.5,
        help="Minimum epsilon in pixels for one-pass contour simplification.",
    )
    parser.add_argument(
        "--max-epsilon-px",
        type=float,
        default=4.5,
        help="Maximum epsilon in pixels. Lower values preserve large-building details.",
    )
    parser.add_argument(
        "--angle-threshold-deg",
        type=float,
        default=12.0,
        help=(
            "Remove a vertex only when the local turn is below this threshold "
            "and the point is also close to the line between its neighbors."
        ),
    )
    parser.add_argument(
        "--no-preserve-curves",
        action="store_false",
        dest="preserve_curves",
        help=(
            "Disable curve-transition protection. By default, points in a smooth "
            "transition are preserved when the cumulative turning angle is large."
        ),
    )
    parser.add_argument(
        "--curve-turn-threshold-deg",
        type=float,
        default=30.0,
        help=(
            "If the cumulative turning angle within a local window exceeds this "
            "value, the point is treated as part of a smooth transition and kept."
        ),
    )
    parser.add_argument(
        "--curve-window",
        type=int,
        default=3,
        help="Number of neighboring vertices on each side used for cumulative-turn protection.",
    )
    parser.add_argument(
        "--max-collinear-distance-px",
        type=float,
        default=1.0,
        help=(
            "A low-turn point is removed only if its perpendicular distance to the "
            "line connecting neighboring vertices is below this value."
        ),
    )
    parser.add_argument(
        "--direction-threshold-deg",
        type=float,
        default=12.0,
        help=(
            "Threshold for deciding whether an edge follows the dominant direction "
            "or its orthogonal direction."
        ),
    )
    parser.add_argument(
        "--no-snap",
        action="store_false",
        dest="snap_to_dominant",
        help="Disable dominant-axis snapping. By default snapping is enabled.",
    )
    parser.add_argument(
        "--max-snap-shift-px",
        type=float,
        default=3,
        help="Maximum allowed vertex displacement during dominant-axis snapping.",
    )
    parser.add_argument(
        "--direction-prune-iters",
        type=int,
        default=3,
        help="Number of iterations for direction-aware pruning.",
    )
    parser.add_argument(
        "--no-jagged-collapse",
        action="store_false",
        dest="jagged_collapse",
        help=(
            "Disable post-processing that collapses repeated short zigzag vertices. "
            "By default, it is enabled for contours with more than "
            "--jagged-min-contour-vertices vertices."
        ),
    )
    parser.add_argument(
        "--jagged-min-contour-vertices",
        type=int,
        default=10,
        help="Apply zigzag collapse only when a simplified contour has more vertices than this value.",
    )
    parser.add_argument(
        "--jagged-min-run-vertices",
        type=int,
        default=5,
        help=(
            "Minimum number of consecutive vertices in a candidate zigzag run, including "
            "the first and last support vertices."
        ),
    )
    parser.add_argument(
        "--jagged-max-run-vertices",
        type=int,
        default=12,
        help="Maximum number of consecutive vertices considered for one zigzag collapse run.",
    )
    parser.add_argument(
        "--jagged-short-edge-px",
        type=float,
        default=4.0,
        help="An edge shorter than this value is treated as a short zigzag edge.",
    )
    parser.add_argument(
        "--jagged-min-short-edge-ratio",
        type=float,
        default=0.70,
        help="Minimum ratio of short edges required in a candidate zigzag run.",
    )
    parser.add_argument(
        "--jagged-max-mean-line-distance-px",
        type=float,
        default=1.5,
        help=(
            "Maximum mean perpendicular distance from intermediate vertices to the "
            "straight support segment of a candidate zigzag run."
        ),
    )
    parser.add_argument(
        "--jagged-min-direction-switches",
        type=int,
        default=2,
        help=(
            "Minimum number of sign changes in local direction/offset required to treat "
            "a run as repeated back-and-forth zigzag."
        ),
    )
    parser.add_argument(
        "--jagged-collapse-iters",
        type=int,
        default=20,
        help="Maximum number of zigzag runs to collapse within each contour.",
    )
    parser.set_defaults(snap_to_dominant=True, preserve_curves=True, jagged_collapse=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.label_dir.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {args.label_dir}")
    if not 0 <= args.threshold <= 255:
        raise ValueError("--threshold must be between 0 and 255.")
    if args.min_instance_area < 1:
        raise ValueError("--min-instance-area must be positive.")
    if args.bbox_padding < 0:
        raise ValueError("--bbox-padding must be non-negative.")
    if args.epsilon_ratio < 0:
        raise ValueError("--epsilon-ratio must be non-negative.")
    if args.min_epsilon_px < 0 or args.max_epsilon_px < 0:
        raise ValueError("epsilon pixel values must be non-negative.")
    if args.max_epsilon_px and args.max_epsilon_px < args.min_epsilon_px:
        raise ValueError("--max-epsilon-px must be >= --min-epsilon-px.")
    if args.direction_prune_iters < 0:
        raise ValueError("--direction-prune-iters must be non-negative.")
    if args.curve_window < 1:
        raise ValueError("--curve-window must be positive.")
    if args.curve_turn_threshold_deg < 0:
        raise ValueError("--curve-turn-threshold-deg must be non-negative.")
    if args.max_collinear_distance_px < 0:
        raise ValueError("--max-collinear-distance-px must be non-negative.")
    if args.jagged_min_contour_vertices < 3:
        raise ValueError("--jagged-min-contour-vertices must be >= 3.")
    if args.jagged_min_run_vertices < 4:
        raise ValueError("--jagged-min-run-vertices must be >= 4.")
    if args.jagged_max_run_vertices < args.jagged_min_run_vertices:
        raise ValueError("--jagged-max-run-vertices must be >= --jagged-min-run-vertices.")
    if args.jagged_short_edge_px <= 0:
        raise ValueError("--jagged-short-edge-px must be positive.")
    if not 0.0 <= args.jagged_min_short_edge_ratio <= 1.0:
        raise ValueError("--jagged-min-short-edge-ratio must be in [0, 1].")
    if args.jagged_max_mean_line_distance_px < 0:
        raise ValueError("--jagged-max-mean-line-distance-px must be non-negative.")
    if args.jagged_min_direction_switches < 0:
        raise ValueError("--jagged-min-direction-switches must be non-negative.")
    if args.jagged_collapse_iters < 0:
        raise ValueError("--jagged-collapse-iters must be non-negative.")


# -----------------------------------------------------------------------------
# Basic mask / contour utilities
# -----------------------------------------------------------------------------


def load_binary_mask(path: Path, threshold: int) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise ValueError(f"OpenCV could not read label: {path}")
    if label.ndim == 3:
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    return np.where(label > threshold, 255, 0).astype(np.uint8)


def mask_iou(source: np.ndarray, candidate: np.ndarray) -> float:
    source_fg = source > 0
    candidate_fg = candidate > 0
    union = np.count_nonzero(source_fg | candidate_fg)
    if union == 0:
        return 1.0
    intersection = np.count_nonzero(source_fg & candidate_fg)
    return float(intersection / union)


def contour_depths(hierarchy: np.ndarray) -> list[int]:
    depths: list[int] = []
    for index in range(len(hierarchy)):
        depth = 0
        parent = int(hierarchy[index][3])
        while parent >= 0:
            depth += 1
            parent = int(hierarchy[parent][3])
        depths.append(depth)
    return depths


def clean_contour(contour: np.ndarray) -> np.ndarray | None:
    """Remove consecutive duplicate vertices and return an OpenCV contour."""
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return None

    cleaned = [pts[0]]
    for pt in pts[1:]:
        if not np.allclose(pt, cleaned[-1]):
            cleaned.append(pt)
    pts = np.asarray(cleaned, dtype=np.float64)

    if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return None

    pts = np.rint(pts).astype(np.int32)

    # A second duplicate-removal pass after rounding.
    cleaned_int = [pts[0]]
    for pt in pts[1:]:
        if not np.array_equal(pt, cleaned_int[-1]):
            cleaned_int.append(pt)
    pts = np.asarray(cleaned_int, dtype=np.int32)
    if len(pts) >= 2 and np.array_equal(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return None

    return pts.reshape(-1, 1, 2)


def clip_contour_to_shape(contour: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    height, width = shape
    pts = contour.reshape(-1, 2).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    return clean_contour(pts)


def contour_area_abs(contour: np.ndarray) -> float:
    return abs(float(cv2.contourArea(contour)))


def shift_contour(contour: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = contour.copy()
    shifted[:, 0, 0] += dx
    shifted[:, 0, 1] += dy
    return shifted


# -----------------------------------------------------------------------------
# Direction-aware polygonization
# -----------------------------------------------------------------------------


def angle_mod_180(angle_deg: float) -> float:
    angle = angle_deg % 180.0
    if angle < 0:
        angle += 180.0
    return angle


def angle_distance_180(a_deg: float, b_deg: float) -> float:
    diff = abs(angle_mod_180(a_deg) - angle_mod_180(b_deg))
    return min(diff, 180.0 - diff)


def edge_angle_deg(p0: np.ndarray, p1: np.ndarray) -> float:
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    return angle_mod_180(math.degrees(math.atan2(dy, dx)))


def turn_angle_deg(p_prev: np.ndarray, p_curr: np.ndarray, p_next: np.ndarray) -> float:
    """Return direction-change angle in [0, 180]. 0 means almost straight."""
    a1 = edge_angle_deg(p_prev, p_curr)
    a2 = edge_angle_deg(p_curr, p_next)
    return angle_distance_180(a1, a2)


def point_to_segment_distance(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
    """Euclidean distance from a point to a finite line segment."""
    p = point.astype(np.float64)
    a = seg_start.astype(np.float64)
    b = seg_end.astype(np.float64)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    projection = a + t * ab
    return float(np.linalg.norm(p - projection))



def signed_angle_diff_deg(angle_deg: float, reference_deg: float) -> float:
    """Signed angle difference in [-90, 90] for undirected line orientations."""
    return float((angle_mod_180(angle_deg) - angle_mod_180(reference_deg) + 90.0) % 180.0 - 90.0)


def count_sign_changes(values: np.ndarray, eps: float = 1e-6) -> int:
    """Count sign alternations after removing near-zero values."""
    signs: list[int] = []
    for value in values:
        value = float(value)
        if abs(value) <= eps:
            continue
        sign = 1 if value > 0 else -1
        if signs and sign == signs[-1]:
            continue
        signs.append(sign)
    if len(signs) < 2:
        return 0
    return int(sum(1 for a, b in zip(signs[:-1], signs[1:]) if a != b))


def candidate_zigzag_run_score(
    segment: np.ndarray,
    short_edge_px: float,
    min_short_edge_ratio: float,
    max_mean_line_distance_px: float,
    min_direction_switches: int,
) -> tuple[bool, float]:
    """Check whether a consecutive segment is a repeated short zigzag run.

    The first and last vertices are treated as support vertices. Intermediate
    vertices are removable only when most edges are short, the entire segment is
    close to the support line, and local directions/offsets switch signs several
    times. This targets the common case where adjacent close boundaries create
    short back-and-forth stair steps along an otherwise stable direction.
    """
    pts = segment.astype(np.float64).reshape(-1, 2)
    if len(pts) < 4:
        return False, 0.0

    support = pts[-1] - pts[0]
    support_len = float(np.linalg.norm(support))
    if support_len <= 1e-6:
        return False, 0.0

    edges = np.diff(pts, axis=0)
    edge_lengths = np.linalg.norm(edges, axis=1)
    if len(edge_lengths) == 0:
        return False, 0.0

    short_ratio = float(np.mean(edge_lengths <= short_edge_px))
    if short_ratio < min_short_edge_ratio:
        return False, 0.0

    internal = pts[1:-1]
    if len(internal) == 0:
        return False, 0.0
    distances = np.array([point_to_segment_distance(p, pts[0], pts[-1]) for p in internal], dtype=np.float64)
    mean_distance = float(np.mean(distances))
    if mean_distance > max_mean_line_distance_px:
        return False, 0.0

    base_angle = edge_angle_deg(pts[0], pts[-1])
    edge_angles = np.array([edge_angle_deg(pts[k], pts[k + 1]) for k in range(len(pts) - 1)], dtype=np.float64)
    angle_deviation = np.array([signed_angle_diff_deg(a, base_angle) for a in edge_angles], dtype=np.float64)
    angle_switches = count_sign_changes(angle_deviation, eps=5.0)

    unit = support / support_len
    normal = np.array([-unit[1], unit[0]], dtype=np.float64)
    signed_offsets = (internal - pts[0]) @ normal
    offset_switches = count_sign_changes(signed_offsets, eps=0.25)

    direction_switches = max(angle_switches, offset_switches)
    if direction_switches < min_direction_switches:
        return False, 0.0

    score = len(pts) + 0.5 * direction_switches + 2.0 * short_ratio - mean_distance / max(max_mean_line_distance_px, 1e-6)
    return True, float(score)


def collapse_repeated_jagged_points(
    contour: np.ndarray,
    min_contour_vertices: int,
    min_run_vertices: int,
    max_run_vertices: int,
    short_edge_px: float,
    min_short_edge_ratio: float,
    max_mean_line_distance_px: float,
    min_direction_switches: int,
    max_iterations: int,
) -> np.ndarray | None:
    """Collapse repeated jagged vertices and keep the last support vertex.

    For a detected run [p_i, ..., p_j], the intermediate vertices are removed and
    p_j is preserved. This means continuous zigzag points are represented by the
    final support vertex rather than many short alternating vertices.
    """
    current = clean_contour(contour)
    if current is None:
        return None
    pts = current.reshape(-1, 2).astype(np.float64)
    if len(pts) <= min_contour_vertices or max_iterations <= 0:
        return current

    min_run_vertices = max(4, min_run_vertices)
    max_run_vertices = max(min_run_vertices, max_run_vertices)

    for _ in range(max_iterations):
        n = len(pts)
        if n <= max(3, min_contour_vertices):
            break

        best_remove: set[int] | None = None
        best_score = -1e18

        for start in range(n):
            max_w = min(max_run_vertices, n - 1)
            for w in range(max_w, min_run_vertices - 1, -1):
                indices = [(start + k) % n for k in range(w)]
                if len(set(indices)) < w or w >= n:
                    continue
                segment = np.asarray([pts[idx] for idx in indices], dtype=np.float64)
                ok, score = candidate_zigzag_run_score(
                    segment,
                    short_edge_px=short_edge_px,
                    min_short_edge_ratio=min_short_edge_ratio,
                    max_mean_line_distance_px=max_mean_line_distance_px,
                    min_direction_switches=min_direction_switches,
                )
                if not ok:
                    continue
                remove = set(indices[1:-1])
                if not remove or n - len(remove) < 3:
                    continue
                if score > best_score:
                    best_score = score
                    best_remove = remove

        if not best_remove:
            break

        pts = np.asarray([pt for idx, pt in enumerate(pts) if idx not in best_remove], dtype=np.float64)
        cleaned = clean_contour(pts)
        if cleaned is None or len(cleaned) < 3:
            break
        pts = cleaned.reshape(-1, 2).astype(np.float64)

    return clean_contour(pts)


def cumulative_turn_angle_deg(pts: np.ndarray, center_index: int, window: int) -> float:
    """Sum local direction changes around one vertex on a closed contour."""
    n = len(pts)
    if n < 4:
        return 0.0
    total = 0.0
    for offset in range(-window, window + 1):
        i = (center_index + offset) % n
        total += turn_angle_deg(pts[(i - 1) % n], pts[i], pts[(i + 1) % n])
    return float(total)


def is_curve_transition_vertex(
    pts: np.ndarray,
    index: int,
    preserve_curves: bool,
    curve_turn_threshold_deg: float,
    curve_window: int,
) -> bool:
    """Return True if a vertex belongs to a smooth but cumulatively curved transition."""
    if not preserve_curves:
        return False
    return cumulative_turn_angle_deg(pts, index, curve_window) >= curve_turn_threshold_deg


def is_safe_to_remove_as_collinear(
    pts: np.ndarray,
    index: int,
    angle_threshold_deg: float,
    max_collinear_distance_px: float,
    preserve_curves: bool,
    curve_turn_threshold_deg: float,
    curve_window: int,
) -> bool:
    """A point is removable only if it is both low-turn and geometrically close to a straight segment."""
    n = len(pts)
    p_prev = pts[(index - 1) % n]
    p_curr = pts[index]
    p_next = pts[(index + 1) % n]
    turn = turn_angle_deg(p_prev, p_curr, p_next)
    if turn > angle_threshold_deg:
        return False
    if is_curve_transition_vertex(
        pts, index, preserve_curves, curve_turn_threshold_deg, curve_window
    ):
        return False
    distance = point_to_segment_distance(p_curr, p_prev, p_next)
    return distance <= max_collinear_distance_px


def estimate_dominant_angle(contour: np.ndarray) -> float:
    """Use the longest polygon edge as the dominant building direction."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 2:
        return 0.0
    next_pts = np.roll(pts, -1, axis=0)
    vecs = next_pts - pts
    lengths = np.linalg.norm(vecs, axis=1)
    if not np.any(lengths > 1e-6):
        return 0.0
    index = int(np.argmax(lengths))
    return edge_angle_deg(pts[index], next_pts[index])


def edge_axis_class(
    p0: np.ndarray,
    p1: np.ndarray,
    dominant_angle_deg: float,
    direction_threshold_deg: float,
) -> int:
    """Return 0 for dominant axis, 1 for orthogonal axis, -1 for other."""
    angle = edge_angle_deg(p0, p1)
    d0 = angle_distance_180(angle, dominant_angle_deg)
    d1 = angle_distance_180(angle, dominant_angle_deg + 90.0)
    if d0 <= direction_threshold_deg and d0 <= d1:
        return 0
    if d1 <= direction_threshold_deg:
        return 1
    return -1


def one_pass_approx_poly(
    contour: np.ndarray,
    epsilon_ratio: float,
    min_epsilon_px: float,
    max_epsilon_px: float,
) -> np.ndarray | None:
    if len(contour) < 3:
        return None
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    if epsilon_ratio <= 0:
        polygon = contour.copy()
    else:
        epsilon = epsilon_ratio * perimeter
        epsilon = max(epsilon, min_epsilon_px)
        if max_epsilon_px > 0:
            epsilon = min(epsilon, max_epsilon_px)
        polygon = cv2.approxPolyDP(contour, epsilon, True)

    polygon = clean_contour(polygon)
    if polygon is None or len(polygon) < 3:
        return None
    return polygon


def remove_nearly_collinear_points(
    contour: np.ndarray,
    angle_threshold_deg: float,
    max_collinear_distance_px: float,
    preserve_curves: bool,
    curve_turn_threshold_deg: float,
    curve_window: int,
) -> np.ndarray | None:
    """Remove only truly redundant straight-line points, while preserving smooth transitions.

    The previous version removed a point whenever the local turn was small. That is
    aggressive for rounded or smoothly changing boundaries, because every local
    turn can be small while the accumulated turn is meaningful. This version
    removes a point only when all conditions are met:
      1) local turn is small;
      2) the point is close to the segment between its two neighbors;
      3) the cumulative turn in its local window is not large.
    """
    pts = contour.reshape(-1, 2)
    if len(pts) <= 3:
        return clean_contour(contour)

    keep: list[np.ndarray] = []
    n = len(pts)
    for i in range(n):
        if is_safe_to_remove_as_collinear(
            pts,
            i,
            angle_threshold_deg=angle_threshold_deg,
            max_collinear_distance_px=max_collinear_distance_px,
            preserve_curves=preserve_curves,
            curve_turn_threshold_deg=curve_turn_threshold_deg,
            curve_window=curve_window,
        ):
            continue
        keep.append(pts[i])

    if len(keep) < 3:
        return clean_contour(contour)
    return clean_contour(np.asarray(keep, dtype=np.int32))


def dominant_direction_prune(
    contour: np.ndarray,
    dominant_angle_deg: float,
    angle_threshold_deg: float,
    direction_threshold_deg: float,
    iterations: int,
    max_collinear_distance_px: float,
    preserve_curves: bool,
    curve_turn_threshold_deg: float,
    curve_window: int,
) -> np.ndarray | None:
    """Delete redundant points along the same dominant axis, but protect smooth transitions."""
    current = clean_contour(contour)
    if current is None:
        return None

    for _ in range(max(0, iterations)):
        pts = current.reshape(-1, 2)
        if len(pts) <= 3:
            break

        keep: list[np.ndarray] = []
        n = len(pts)
        changed = False

        for i in range(n):
            p_prev = pts[(i - 1) % n]
            p_curr = pts[i]
            p_next = pts[(i + 1) % n]

            prev_cls = edge_axis_class(
                p_prev,
                p_curr,
                dominant_angle_deg,
                direction_threshold_deg,
            )
            next_cls = edge_axis_class(
                p_curr,
                p_next,
                dominant_angle_deg,
                direction_threshold_deg,
            )
            turn = turn_angle_deg(p_prev, p_curr, p_next)
            distance = point_to_segment_distance(p_curr, p_prev, p_next)
            protected_curve = is_curve_transition_vertex(
                pts,
                i,
                preserve_curves=preserve_curves,
                curve_turn_threshold_deg=curve_turn_threshold_deg,
                curve_window=curve_window,
            )

            same_dominant_axis = prev_cls == next_cls and prev_cls in (0, 1)
            safe_collinear = (
                turn <= angle_threshold_deg
                and distance <= max_collinear_distance_px
                and not protected_curve
            )

            # Direction pruning should not erase gradual bends. Therefore, even if
            # both adjacent edges are classified as the same dominant axis, we only
            # remove the point when it is also geometrically close to a straight line.
            if same_dominant_axis and safe_collinear:
                changed = True
                continue

            if safe_collinear:
                changed = True
                continue

            keep.append(p_curr)

        if len(keep) < 3:
            break
        next_contour = clean_contour(np.asarray(keep, dtype=np.int32))
        if next_contour is None:
            break
        current = next_contour
        if not changed:
            break

    return current


def line_from_edge(
    p0: np.ndarray,
    p1: np.ndarray,
    axis_class: int,
    dominant_angle_deg: float,
) -> tuple[np.ndarray, float] | None:
    """Return line in normal form n dot x = c."""
    if axis_class not in (0, 1):
        return None

    theta = math.radians(dominant_angle_deg)
    dominant_dir = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    orthogonal_dir = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float64)

    # If the edge direction is dominant, its normal is orthogonal.
    # If the edge direction is orthogonal, its normal is dominant.
    normal = orthogonal_dir if axis_class == 0 else dominant_dir
    midpoint = (p0.astype(np.float64) + p1.astype(np.float64)) * 0.5
    c = float(np.dot(normal, midpoint))
    return normal, c


def intersect_lines(
    line1: tuple[np.ndarray, float],
    line2: tuple[np.ndarray, float],
) -> np.ndarray | None:
    n1, c1 = line1
    n2, c2 = line2
    mat = np.vstack([n1, n2])
    det = float(np.linalg.det(mat))
    if abs(det) < 1e-6:
        return None
    rhs = np.array([c1, c2], dtype=np.float64)
    return np.linalg.solve(mat, rhs)


def snap_to_dominant_axes(
    contour: np.ndarray,
    dominant_angle_deg: float,
    direction_threshold_deg: float,
    max_shift_px: float,
    shape: tuple[int, int],
) -> np.ndarray | None:
    """Snap edge-supported vertices to intersections of dominant-axis lines."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 4:
        return clip_contour_to_shape(contour, shape)

    edge_lines: list[tuple[np.ndarray, float] | None] = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        cls = edge_axis_class(p0, p1, dominant_angle_deg, direction_threshold_deg)
        edge_lines.append(line_from_edge(p0, p1, cls, dominant_angle_deg))

    snapped = pts.copy()
    for i in range(n):
        prev_line = edge_lines[(i - 1) % n]
        next_line = edge_lines[i]
        if prev_line is None or next_line is None:
            continue
        intersection = intersect_lines(prev_line, next_line)
        if intersection is None:
            continue
        shift = float(np.linalg.norm(intersection - pts[i]))
        if shift <= max_shift_px:
            snapped[i] = intersection

    snapped_contour = clip_contour_to_shape(np.rint(snapped).astype(np.int32), shape)
    if snapped_contour is None or len(snapped_contour) < 3:
        return clip_contour_to_shape(contour, shape)

    # Avoid invalid collapse after snapping.
    if contour_area_abs(snapped_contour) < 1.0:
        return clip_contour_to_shape(contour, shape)
    return snapped_contour


def simplify_contour_directional(
    contour: np.ndarray,
    shape: tuple[int, int],
    epsilon_ratio: float,
    min_epsilon_px: float,
    max_epsilon_px: float,
    angle_threshold_deg: float,
    direction_threshold_deg: float,
    snap_to_dominant: bool,
    max_snap_shift_px: float,
    direction_prune_iters: int,
    max_collinear_distance_px: float,
    preserve_curves: bool,
    curve_turn_threshold_deg: float,
    curve_window: int,
    jagged_collapse: bool,
    jagged_min_contour_vertices: int,
    jagged_min_run_vertices: int,
    jagged_max_run_vertices: int,
    jagged_short_edge_px: float,
    jagged_min_short_edge_ratio: float,
    jagged_max_mean_line_distance_px: float,
    jagged_min_direction_switches: int,
    jagged_collapse_iters: int,
) -> tuple[np.ndarray | None, float]:
    """Extract instance corners with direction pruning and line intersections."""
    from instance_corner_extractor import (
        InstanceCornerConfig,
        recover_line_intersection_corners,
    )

    config = InstanceCornerConfig(
        epsilon_ratio=epsilon_ratio,
        min_epsilon_px=min_epsilon_px,
        max_epsilon_px=max_epsilon_px,
        angle_threshold_deg=angle_threshold_deg,
        direction_threshold_deg=direction_threshold_deg,
        max_collinear_distance_px=max_collinear_distance_px,
        preserve_curves=preserve_curves,
        curve_turn_threshold_deg=curve_turn_threshold_deg,
        curve_window=curve_window,
        direction_prune_iters=direction_prune_iters,
        snap_to_dominant=snap_to_dominant,
        max_snap_shift_px=max_snap_shift_px,
    )
    polygon, dominant_angle = recover_line_intersection_corners(
        contour,
        shape,
        config,
    )
    if polygon is None or len(polygon) < 3:
        polygon = clean_contour(contour)
    if polygon is None or len(polygon) < 3:
        return None, dominant_angle
    polygon = clip_contour_to_shape(
        np.rint(polygon).astype(np.int32).reshape(-1, 1, 2),
        shape,
    )
    return polygon, dominant_angle


def simplify_all_contours_in_mask(
    mask: np.ndarray,
    args: argparse.Namespace,
    scale_for_params: float = 1.0,
) -> tuple[list[np.ndarray], np.ndarray | None, list[float], list[np.ndarray]]:
    """Extract and simplify all contours in a mask.

    scale_for_params rescales pixel-distance parameters when the mask itself has
    been upsampled. The ratio-based epsilon is kept unchanged because the contour
    perimeter is already scaled by the same factor.
    """
    chain_mode = cv2.CHAIN_APPROX_SIMPLE if args.chain_approx == "simple" else cv2.CHAIN_APPROX_NONE
    contours, hierarchy_raw = cv2.findContours(mask, cv2.RETR_TREE, chain_mode)
    if hierarchy_raw is None or not contours:
        return [], None, [], []

    hierarchy = hierarchy_raw[0].astype(np.int32)
    simplified_contours: list[np.ndarray] = []
    dominant_angles: list[float] = []
    valid_original_contours: list[np.ndarray] = []

    for contour in contours:
        simplified, dominant_angle = simplify_contour_directional(
            contour,
            shape=mask.shape,
            epsilon_ratio=args.epsilon_ratio,
            min_epsilon_px=args.min_epsilon_px * scale_for_params,
            max_epsilon_px=args.max_epsilon_px * scale_for_params,
            angle_threshold_deg=args.angle_threshold_deg,
            direction_threshold_deg=args.direction_threshold_deg,
            snap_to_dominant=args.snap_to_dominant,
            max_snap_shift_px=args.max_snap_shift_px * scale_for_params,
            direction_prune_iters=args.direction_prune_iters,
            max_collinear_distance_px=args.max_collinear_distance_px * scale_for_params,
            preserve_curves=args.preserve_curves,
            curve_turn_threshold_deg=args.curve_turn_threshold_deg,
            curve_window=args.curve_window,
            jagged_collapse=args.jagged_collapse,
            jagged_min_contour_vertices=args.jagged_min_contour_vertices,
            jagged_min_run_vertices=args.jagged_min_run_vertices,
            jagged_max_run_vertices=args.jagged_max_run_vertices,
            jagged_short_edge_px=args.jagged_short_edge_px * scale_for_params,
            jagged_min_short_edge_ratio=args.jagged_min_short_edge_ratio,
            jagged_max_mean_line_distance_px=args.jagged_max_mean_line_distance_px * scale_for_params,
            jagged_min_direction_switches=args.jagged_min_direction_switches,
            jagged_collapse_iters=args.jagged_collapse_iters,
        )
        if simplified is None or len(simplified) < 3:
            simplified = clean_contour(contour)
        if simplified is None or len(simplified) < 3:
            simplified = contour
        simplified = clip_contour_to_shape(simplified, mask.shape)
        if simplified is None or len(simplified) < 3:
            simplified = contour

        simplified_contours.append(simplified.astype(np.int32))
        dominant_angles.append(float(dominant_angle))
        valid_original_contours.append(contour.astype(np.int32))

    return simplified_contours, hierarchy, dominant_angles, valid_original_contours


def render_contours_with_hierarchy(
    shape: tuple[int, int],
    contours: list[np.ndarray],
    hierarchy: np.ndarray | None,
) -> np.ndarray:
    rendered = np.zeros(shape, dtype=np.uint8)
    if not contours:
        return rendered
    if hierarchy is not None and len(hierarchy) == len(contours):
        cv2.drawContours(rendered, contours, -1, 255, cv2.FILLED, hierarchy=hierarchy[np.newaxis, ...])
    else:
        cv2.drawContours(rendered, contours, -1, 255, cv2.FILLED)
    return rendered


def build_contour_records(
    contours_local: list[np.ndarray],
    hierarchy: np.ndarray | None,
    dominant_angles: list[float],
    x: int,
    y: int,
) -> tuple[list[ContourResult], list[np.ndarray]]:
    if hierarchy is not None and len(hierarchy) == len(contours_local):
        depths = contour_depths(hierarchy)
    else:
        depths = [0 for _ in contours_local]

    records: list[ContourResult] = []
    contours_global: list[np.ndarray] = []
    for idx, contour in enumerate(contours_local):
        contour_global = shift_contour(contour, x, y)
        contours_global.append(contour_global)
        parent_index = int(hierarchy[idx][3]) if hierarchy is not None and idx < len(hierarchy) else -1
        angle = dominant_angles[idx] if idx < len(dominant_angles) else estimate_dominant_angle(contour)
        records.append(
            ContourResult(
                contour_index=idx,
                parent_index=parent_index,
                depth=int(depths[idx]),
                is_hole=bool(depths[idx] % 2),
                original_points=int(len(contour)),
                vertex_count=int(len(contour)),
                dominant_angle_deg=float(angle),
                contour_local=contour,
                contour_global=contour_global,
            )
        )
    return records, contours_global



# -----------------------------------------------------------------------------
# Instance extraction and rendering
# -----------------------------------------------------------------------------


def extract_instances(source_mask: np.ndarray, args: argparse.Namespace) -> Iterable[tuple[int, tuple[int, int, int, int], int, np.ndarray]]:
    """Yield instance_id, padded bbox, area, crop mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (source_mask > 0).astype(np.uint8),
        connectivity=args.connectivity,
    )
    height, width = source_mask.shape
    pad = args.bbox_padding

    for instance_id in range(1, num_labels):
        area = int(stats[instance_id, cv2.CC_STAT_AREA])
        if area < args.min_instance_area:
            continue

        x = int(stats[instance_id, cv2.CC_STAT_LEFT])
        y = int(stats[instance_id, cv2.CC_STAT_TOP])
        w = int(stats[instance_id, cv2.CC_STAT_WIDTH])
        h = int(stats[instance_id, cv2.CC_STAT_HEIGHT])

        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(width, x + w + pad)
        y1 = min(height, y + h + pad)
        crop_labels = labels[y0:y1, x0:x1]
        crop_mask = np.where(crop_labels == instance_id, 255, 0).astype(np.uint8)
        yield instance_id, (x0, y0, x1 - x0, y1 - y0), area, crop_mask


def process_instance(
    instance_id: int,
    bbox_xywh: tuple[int, int, int, int],
    area: int,
    crop_mask: np.ndarray,
    args: argparse.Namespace,
) -> InstanceResult | None:
    """Process one connected-component instance at the original label resolution only."""
    x, y, _, _ = bbox_xywh

    simplified_contours, hierarchy, dominant_angles, original_contours_local = simplify_all_contours_in_mask(
        crop_mask,
        args,
        scale_for_params=1.0,
    )
    if not simplified_contours:
        return None

    rendered_crop = render_contours_with_hierarchy(crop_mask.shape, simplified_contours, hierarchy)
    contour_records, polygon_contours_global = build_contour_records(
        simplified_contours,
        hierarchy,
        dominant_angles,
        x,
        y,
    )
    original_contours_global = [shift_contour(contour, x, y) for contour in original_contours_local]

    return InstanceResult(
        instance_id=instance_id,
        bbox_xywh=bbox_xywh,
        area=area,
        original_points=int(sum(len(c) for c in original_contours_local)),
        vertex_count=int(sum(len(c) for c in simplified_contours)),
        iou=mask_iou(crop_mask, rendered_crop),
        rendered_crop=rendered_crop,
        original_contours_global=original_contours_global,
        polygon_contours_global=polygon_contours_global,
        contour_records=contour_records,
    )

def compose_full_mask(shape: tuple[int, int], instances: list[InstanceResult]) -> np.ndarray:
    full = np.zeros(shape, dtype=np.uint8)
    for inst in instances:
        x, y, w, h = inst.bbox_xywh
        roi = full[y : y + h, x : x + w]
        # Instances from connected components do not overlap. OR composition is safe.
        roi[inst.rendered_crop > 0] = 255
    return full


# -----------------------------------------------------------------------------
# IO and visualization
# -----------------------------------------------------------------------------


def to_display_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.dtype == np.uint8:
        return image.copy()

    values = image.astype(np.float32)
    low, high = np.percentile(values, (2.0, 98.0))
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    values = np.clip((values - low) * (255.0 / (high - low)), 0, 255)
    return values.astype(np.uint8)


def find_source_image(image_dir: Path, stem: str) -> Path | None:
    if not image_dir.is_dir():
        return None
    for suffix in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def build_visualization(
    source_mask: np.ndarray,
    instances: list[InstanceResult],
    source_image_path: Path | None,
    global_iou: float,
) -> np.ndarray:
    """Visualize original-resolution polygonization at the original image resolution."""
    image = None
    if source_image_path is not None:
        image = cv2.imread(str(source_image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        image = cv2.cvtColor(source_mask, cv2.COLOR_GRAY2BGR)

    canvas = to_display_image(image)
    if canvas.shape[:2] != source_mask.shape:
        canvas = cv2.resize(
            canvas,
            (source_mask.shape[1], source_mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # Lightly tint the original mask region.
    tint = canvas.copy()
    tint[source_mask > 0] = (255, 120, 20)
    canvas = cv2.addWeighted(canvas, 0.78, tint, 0.22, 0.0)

    original_contours: list[np.ndarray] = []
    polygon_contours: list[np.ndarray] = []
    for inst in instances:
        original_contours.extend(inst.original_contours_global)
        polygon_contours.extend(inst.polygon_contours_global)

    # Draw at original resolution. Colors follow OpenCV BGR convention.
    if original_contours:
        cv2.drawContours(canvas, original_contours, -1, (0, 180, 0), 1, cv2.LINE_AA)
    if polygon_contours:
        cv2.drawContours(canvas, polygon_contours, -1, (0, 255, 255), 1, cv2.LINE_AA)

    radius = max(1, round(min(canvas.shape[:2]) / 2400))
    for contour in polygon_contours:
        for point in contour.reshape(-1, 2):
            cv2.circle(
                canvas,
                (int(point[0]), int(point[1])),
                radius,
                (0, 0, 255),
                cv2.FILLED,
                cv2.LINE_AA,
            )

    vertex_count = sum(inst.vertex_count for inst in instances)
    original_points = sum(inst.original_points for inst in instances)
    mean_iou = float(np.mean([inst.iou for inst in instances])) if instances else 1.0
    text = (
        f"original-resolution extraction | IoU={global_iou:.4f} "
        f"mean_inst_IoU={mean_iou:.4f} vertices={vertex_count} "
        f"orig_pts={original_points} inst={len(instances)}"
    )
    legend = "green=original contour  yellow=polygon  red=vertices"

    cv2.rectangle(canvas, (8, 8), (min(canvas.shape[1] - 8, 1500), 68), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        legend,
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def write_vertices_json(
    path: Path,
    label_path: Path,
    source_mask: np.ndarray,
    polygon_mask: np.ndarray,
    instances: list[InstanceResult],
    args: argparse.Namespace,
) -> None:
    instance_records = []
    for inst in instances:
        contours = []
        for rec in inst.contour_records:
            contours.append(
                {
                    "contour_index": int(rec.contour_index),
                    "parent_index": int(rec.parent_index),
                    "depth": int(rec.depth),
                    "is_hole": bool(rec.is_hole),
                    "original_points": int(rec.original_points),
                    "vertex_count": int(rec.vertex_count),
                    "dominant_angle_deg": float(rec.dominant_angle_deg),
                    "vertices_xy": rec.contour_global.reshape(-1, 2).astype(int).tolist(),
                }
            )
        instance_records.append(
            {
                "instance_id": int(inst.instance_id),
                "bbox_xywh": [int(v) for v in inst.bbox_xywh],
                "area": int(inst.area),
                "instance_iou": float(inst.iou),
                "original_points": int(inst.original_points),
                "vertex_count": int(inst.vertex_count),
                "contours": contours,
            }
        )

    payload = {
        "source_label": str(label_path),
        "width": int(source_mask.shape[1]),
        "height": int(source_mask.shape[0]),
        "global_iou": float(mask_iou(source_mask, polygon_mask)),
        "instance_count": int(len(instances)),
        "total_original_points": int(sum(inst.original_points for inst in instances)),
        "total_vertex_count": int(sum(inst.vertex_count for inst in instances)),
        "args": {
            "threshold": int(args.threshold),
            "connectivity": int(args.connectivity),
            "min_instance_area": int(args.min_instance_area),
            "bbox_padding": int(args.bbox_padding),
            "chain_approx": args.chain_approx,
            "epsilon_ratio": float(args.epsilon_ratio),
            "min_epsilon_px": float(args.min_epsilon_px),
            "max_epsilon_px": float(args.max_epsilon_px),
            "angle_threshold_deg": float(args.angle_threshold_deg),
            "preserve_curves": bool(args.preserve_curves),
            "curve_turn_threshold_deg": float(args.curve_turn_threshold_deg),
            "curve_window": int(args.curve_window),
            "max_collinear_distance_px": float(args.max_collinear_distance_px),
            "direction_threshold_deg": float(args.direction_threshold_deg),
            "snap_to_dominant": bool(args.snap_to_dominant),
            "max_snap_shift_px": float(args.max_snap_shift_px),
            "direction_prune_iters": int(args.direction_prune_iters),
            "jagged_collapse": bool(args.jagged_collapse),
            "jagged_min_contour_vertices": int(args.jagged_min_contour_vertices),
            "jagged_min_run_vertices": int(args.jagged_min_run_vertices),
            "jagged_max_run_vertices": int(args.jagged_max_run_vertices),
            "jagged_short_edge_px": float(args.jagged_short_edge_px),
            "jagged_min_short_edge_ratio": float(args.jagged_min_short_edge_ratio),
            "jagged_max_mean_line_distance_px": float(args.jagged_max_mean_line_distance_px),
            "jagged_min_direction_switches": int(args.jagged_min_direction_switches),
            "jagged_collapse_iters": int(args.jagged_collapse_iters),
        },
        "instances": instance_records,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def process_label(
    label_path: Path,
    image_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    source_mask = load_binary_mask(label_path, args.threshold)

    instances: list[InstanceResult] = []
    for instance_id, bbox_xywh, area, crop_mask in extract_instances(source_mask, args):
        result = process_instance(instance_id, bbox_xywh, area, crop_mask, args)
        if result is not None:
            instances.append(result)

    polygon_mask = compose_full_mask(source_mask.shape, instances)
    global_iou = mask_iou(source_mask, polygon_mask)
    mean_instance_iou = float(np.mean([inst.iou for inst in instances])) if instances else 1.0
    min_instance_iou = float(np.min([inst.iou for inst in instances])) if instances else 1.0

    mask_dir = output_dir / "masks"
    vertex_dir = output_dir / "vertices"
    visualization_dir = output_dir / "visualizations"
    for directory in (mask_dir, vertex_dir, visualization_dir):
        directory.mkdir(parents=True, exist_ok=True)

    stem = label_path.stem
    mask_path = mask_dir / f"{stem}_polygon.tif"
    json_path = vertex_dir / f"{stem}_vertices.json"
    visualization_path = visualization_dir / f"{stem}_vertices.png"

    if not cv2.imwrite(str(mask_path), polygon_mask):
        raise OSError(f"Failed to write mask: {mask_path}")

    write_vertices_json(json_path, label_path, source_mask, polygon_mask, instances, args)

    visualization = build_visualization(
        source_mask=source_mask,
        instances=instances,
        source_image_path=find_source_image(image_dir, stem),
        global_iou=global_iou,
    )
    if not cv2.imwrite(str(visualization_path), visualization):
        raise OSError(f"Failed to write visualization: {visualization_path}")

    return {
        "label": label_path.name,
        "global_iou": float(global_iou),
        "mean_instance_iou": float(mean_instance_iou),
        "min_instance_iou": float(min_instance_iou),
        "instances": int(len(instances)),
        "original_contour_points": int(sum(inst.original_points for inst in instances)),
        "simplified_vertices": int(sum(inst.vertex_count for inst in instances)),
        "mask_path": str(mask_path),
        "vertices_path": str(json_path),
        "visualization_path": str(visualization_path),
    }


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "label",
        "global_iou",
        "mean_instance_iou",
        "min_instance_iou",
        "instances",
        "original_contour_points",
        "simplified_vertices",
        "mask_path",
        "vertices_path",
        "visualization_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)

    label_paths = sorted(args.label_dir.glob(args.pattern))
    if not label_paths:
        raise FileNotFoundError(f"No labels matching {args.pattern!r} in {args.label_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for label_path in label_paths:
        row = process_label(
            label_path=label_path,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            args=args,
        )
        rows.append(row)
        print(
            f"{row['label']}: global_IoU={row['global_iou']:.6f}, "
            f"mean_inst_IoU={row['mean_instance_iou']:.6f}, "
            f"vertices={row['simplified_vertices']} "
            f"(from {row['original_contour_points']}), "
            f"instances={row['instances']}"
        )

    summary_path = args.output_dir / "summary.csv"
    write_summary(summary_path, rows)
    print(f"Wrote {len(rows)} results to {args.output_dir}")


if __name__ == "__main__":
    main()
