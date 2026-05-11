from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


Point = Tuple[int, int]


@dataclass
class Segment:
    label: str
    points: List[Point]
    length: float
    score: float

    def to_dict(self) -> Dict:
        return asdict(self)


def clean_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    out = np.zeros_like(mask, dtype=np.uint8)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            out[labels == idx] = 255
    return out


def _neighbors(y: int, x: int, skel: np.ndarray) -> List[Point]:
    h, w = skel.shape
    pts: List[Point] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                pts.append((ny, nx))
    return pts


def _trace_segments(skel: np.ndarray) -> List[List[Point]]:
    points = np.argwhere(skel > 0)
    if len(points) == 0:
        return []

    degree_map: Dict[Point, int] = {}
    for y, x in points:
        degree_map[(int(y), int(x))] = len(_neighbors(int(y), int(x), skel))

    nodes = {p for p, d in degree_map.items() if d != 2}
    visited_edges = set()
    segments: List[List[Point]] = []

    def edge_key(a: Point, b: Point):
        return tuple(sorted((a, b)))

    for node in nodes:
        for n in _neighbors(node[0], node[1], skel):
            key = edge_key(node, n)
            if key in visited_edges:
                continue

            path = [node]
            prev = node
            cur = n
            visited_edges.add(key)

            while True:
                path.append(cur)
                nbrs = _neighbors(cur[0], cur[1], skel)
                next_candidates = [pt for pt in nbrs if pt != prev]
                if len(next_candidates) != 1 or cur in nodes:
                    break
                nxt = next_candidates[0]
                visited_edges.add(edge_key(cur, nxt))
                prev, cur = cur, nxt

            if len(path) >= 2:
                segments.append(path)

    if not segments:
        return [[(int(y), int(x)) for y, x in points]]
    return segments


def _polyline_length(points: Sequence[Point]) -> float:
    if len(points) < 2:
        return 0.0
    arr = np.array(points, dtype=np.float32)
    diffs = np.diff(arr, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def mask_to_segments(
    mask: np.ndarray, label: str, min_area: int, min_length: float, score: float
) -> List[Segment]:
    cleaned = clean_mask(mask, min_area)
    skel = skeletonize(cleaned > 0)
    chains = _trace_segments(skel.astype(np.uint8))

    segments: List[Segment] = []
    for chain in chains:
        # convert (row, col) to (x, y)
        xy_chain = [(int(c), int(r)) for r, c in chain]
        length = _polyline_length(chain)
        if length < min_length:
            continue
        segments.append(Segment(label=label, points=xy_chain, length=length, score=score))
    return segments


def collect_segments(
    lead_mask: np.ndarray,
    crack_mask: np.ndarray,
    min_area: int,
    min_length: float,
    lead_score: float,
    crack_score: float,
) -> List[Segment]:
    lead_segments = mask_to_segments(lead_mask, "lead", min_area, min_length, lead_score)
    crack_segments = mask_to_segments(crack_mask, "crack", min_area, min_length, crack_score)
    return lead_segments + crack_segments

