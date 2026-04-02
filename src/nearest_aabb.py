"""Nearest-AABB BIM mapping (Eq. 4).

Maps defect 3D coordinates to nearest BIM element.
"""
import numpy as np

def point_to_aabb_distance(point, aabb_min, aabb_max):
    """Min distance from point to AABB. Returns 0 if inside."""
    return float(np.linalg.norm(point - np.clip(point, aabb_min, aabb_max)))

def nearest_aabb(defect_point, elements, k=5):
    """Optimised nearest AABB with k-centre pre-filtering."""
    centres = np.array([(e["aabb_min"]+e["aabb_max"])/2 for e in elements])
    dists = np.linalg.norm(centres - defect_point, axis=1)
    k_actual = min(k, len(elements))
    top_k = np.argpartition(dists, k_actual)[:k_actual]
    best_idx, best_dist = -1, float("inf")
    for i in top_k:
        d = point_to_aabb_distance(defect_point, elements[i]["aabb_min"], elements[i]["aabb_max"])
        if d < best_dist: best_idx, best_dist = i, d
    return best_idx, best_dist
