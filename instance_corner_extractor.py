"""Direction-aware building-instance corner extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class InstanceCornerConfig:
    epsilon_ratio: float = 0.0015
    min_epsilon_px: float = 1.5
    max_epsilon_px: float = 4.5
    angle_threshold_deg: float = 12.0
    direction_threshold_deg: float = 12.0
    max_collinear_distance_px: float = 1.0
    preserve_curves: bool = True
    curve_turn_threshold_deg: float = 30.0
    curve_window: int = 3
    direction_prune_iters: int = 3
    snap_to_dominant: bool = True
    max_snap_shift_px: float = 3.0
    corner_merge_edge_px: float = 8.0
    corner_max_shift_px: float = 8.0
    corner_min_line_angle_deg: float = 10.0
    corner_merge_iterations: int = 4


def clean_polygon(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        return None
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.any(np.abs(points[1:] - points[:-1]) > 1e-6, axis=1)
    points = points[keep]
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    return points if len(points) >= 3 else None


def angle_mod_180(angle):
    return float(angle % 180.0)


def edge_angle(first, second):
    delta = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    return angle_mod_180(math.degrees(math.atan2(float(delta[1]), float(delta[0]))))


def angle_distance(first, second):
    difference = abs(angle_mod_180(first) - angle_mod_180(second))
    return min(difference, 180.0 - difference)


def point_segment_distance(point, start, end):
    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    edge = end - start
    denominator = float(np.dot(edge, edge))
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - start))
    amount = float(np.dot(point - start, edge) / denominator)
    projection = start + np.clip(amount, 0.0, 1.0) * edge
    return float(np.linalg.norm(point - projection))


def safe_collinear_mask(points, config):
    count = len(points)
    edge_angles = [
        edge_angle(points[index], points[(index + 1) % count])
        for index in range(count)
    ]
    turns = [
        angle_distance(edge_angles[index - 1], edge_angles[index])
        for index in range(count)
    ]
    window = max(int(config.curve_window), 0)
    safe = np.zeros(count, dtype=bool)
    for index in range(count):
        if turns[index] > config.angle_threshold_deg:
            continue
        if (
            point_segment_distance(
                points[index],
                points[index - 1],
                points[(index + 1) % count],
            )
            > config.max_collinear_distance_px
        ):
            continue
        if config.preserve_curves:
            cumulative_turn = sum(
                turns[(index + offset) % count]
                for offset in range(-window, window + 1)
            )
            if cumulative_turn >= config.curve_turn_threshold_deg:
                continue
        safe[index] = True
    return safe


def remove_collinear(points, config):
    current = clean_polygon(points)
    if current is None or len(current) <= 3:
        return current
    result = clean_polygon(current[~safe_collinear_mask(current, config)])
    return current if result is None else result


def dominant_angle(points):
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    if not np.any(lengths > 1e-6):
        return 0.0
    index = int(np.argmax(lengths))
    return edge_angle(points[index], following[index])


def axis_class_from_angle(angle, dominant, threshold):
    primary = angle_distance(angle, dominant)
    orthogonal = angle_distance(angle, dominant + 90.0)
    if primary <= threshold and primary <= orthogonal:
        return 0
    if orthogonal <= threshold:
        return 1
    return -1


def direction_prune(points, dominant, config):
    current = clean_polygon(points)
    if current is None:
        return None
    for _ in range(max(int(config.direction_prune_iters), 0)):
        if len(current) <= 3:
            break
        remove = safe_collinear_mask(current, config)
        candidate = clean_polygon(current[~remove])
        if candidate is None:
            break
        current = candidate
        if not np.any(remove):
            break
    return current


def line_intersection(point_a, direction_a, point_b, direction_b):
    point_a = np.asarray(point_a, dtype=np.float64)
    point_b = np.asarray(point_b, dtype=np.float64)
    direction_a = np.asarray(direction_a, dtype=np.float64)
    direction_b = np.asarray(direction_b, dtype=np.float64)
    cross = float(
        direction_a[0] * direction_b[1]
        - direction_a[1] * direction_b[0]
    )
    if abs(cross) <= 1e-8:
        return None
    difference = point_b - point_a
    amount = float(
        (
            difference[0] * direction_b[1]
            - difference[1] * direction_b[0]
        )
        / cross
    )
    return point_a + amount * direction_a


def cross_2d(first, second):
    return float(
        first[0] * second[1]
        - first[1] * second[0]
    )


def same_turn_direction(incoming, short_edge, outgoing):
    first_turn = cross_2d(incoming, short_edge)
    second_turn = cross_2d(short_edge, outgoing)
    return first_turn * second_turn > 1e-8


def snap_to_axes(points, dominant, config, shape):
    if not config.snap_to_dominant or len(points) < 3:
        return points
    angle = math.radians(dominant)
    axes = (
        np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64),
        np.asarray([-math.sin(angle), math.cos(angle)], dtype=np.float64),
    )
    snapped = points.astype(np.float64).copy()
    count = len(points)
    edge_angles = [
        edge_angle(points[index], points[(index + 1) % count])
        for index in range(count)
    ]
    edge_axes = [
        axis_class_from_angle(
            edge_angles[index],
            dominant,
            config.direction_threshold_deg,
        )
        for index in range(count)
    ]
    for index in range(count):
        previous_axis = edge_axes[(index - 1) % count]
        following_axis = edge_axes[index]
        if previous_axis not in (0, 1) or following_axis not in (0, 1):
            continue
        if previous_axis == following_axis:
            continue
        intersection = line_intersection(
            points[index - 1],
            axes[previous_axis],
            points[index],
            axes[following_axis],
        )
        if intersection is None:
            continue
        if np.linalg.norm(intersection - points[index]) <= config.max_snap_shift_px:
            snapped[index] = intersection
    height, width = shape
    snapped[:, 0] = np.clip(snapped[:, 0], 0, width - 1)
    snapped[:, 1] = np.clip(snapped[:, 1], 0, height - 1)
    result = clean_polygon(snapped)
    if result is None or abs(cv2.contourArea(result.reshape(-1, 1, 2))) < 1.0:
        return points
    return result


def simplify_contour(contour, shape, config):
    contour = np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0:
        return None, 0.0
    epsilon = max(
        float(config.epsilon_ratio) * perimeter,
        float(config.min_epsilon_px),
    )
    if config.max_epsilon_px > 0:
        epsilon = min(epsilon, float(config.max_epsilon_px))
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    polygon = remove_collinear(polygon, config)
    if polygon is None:
        return None, 0.0
    dominant = dominant_angle(polygon)
    polygon = direction_prune(polygon, dominant, config)
    if polygon is None:
        return None, dominant
    polygon = snap_to_axes(polygon, dominant, config, shape)
    polygon = remove_collinear(polygon, config)
    return polygon, dominant


def merge_short_edge_once(points, config):
    points = clean_polygon(points)
    if points is None or len(points) <= 3:
        return points, False
    count = len(points)
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    candidates = np.argsort(lengths)
    for edge_index in candidates:
        if lengths[edge_index] > config.corner_merge_edge_px:
            break
        rotated = np.roll(points, -int(edge_index), axis=0)
        previous = rotated[-1]
        first = rotated[0]
        second = rotated[1]
        following = rotated[2]
        incoming = first - previous
        short_edge = second - first
        outgoing = following - second
        if np.linalg.norm(incoming) <= 1e-6 or np.linalg.norm(outgoing) <= 1e-6:
            continue
        if not same_turn_direction(incoming, short_edge, outgoing):
            continue
        if (
            angle_distance(
                edge_angle(previous, first),
                edge_angle(second, following),
            )
            < config.corner_min_line_angle_deg
        ):
            continue
        intersection = line_intersection(previous, incoming, second, outgoing)
        if intersection is None:
            continue
        if max(
            float(np.linalg.norm(intersection - first)),
            float(np.linalg.norm(intersection - second)),
        ) > config.corner_max_shift_px:
            continue
        merged = np.concatenate(
            [
                intersection.reshape(1, 2),
                rotated[2:],
            ],
            axis=0,
        )
        merged = clean_polygon(merged)
        if merged is not None and len(merged) >= 3:
            return merged, True
    return points, False


def recover_line_intersection_corners(contour, shape, config):
    polygon, dominant = simplify_contour(contour, shape, config)
    if polygon is None or len(polygon) < 3:
        return polygon, dominant
    for _ in range(max(int(config.corner_merge_iterations), 0)):
        polygon, changed = merge_short_edge_once(polygon, config)
        if not changed:
            break
    polygon = remove_collinear(polygon, config)
    polygon = snap_to_axes(polygon, dominant, config, shape)
    polygon = remove_collinear(polygon, config)
    return polygon, dominant
