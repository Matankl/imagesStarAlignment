"""
triangle_match.py
=================
Triangle‑based star‑field matcher

Given   L1  – query frame
        L2  – reference / catalogue frame

1.  Keep the *top_k* brightest stars in each image.
2.  Pre‑compute **all** triples in L2, store their three
    internal angles (sorted) in a hash‑table.
3.  Iterate over triples in L1:
      • compute (sorted) internal angles
      • use the hash‑table to retrieve candidate L2 triples whose
        quantised angles are within *eps_angle_deg* of L1’s
      • for every candidate triple
            – estimate the 3‑DoF similarity transform with
              ``cv2.estimateAffinePartial2D``
            – count how many L1 stars land within *eps_xy_px*
              of some L2 star
            – keep the transform with the best overlap

Author: ChatGPT (OpenAI) – May 2025
"""

from __future__ import annotations
import itertools
import math
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple
import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Bring in the user‑supplied helpers
# ──────────────────────────────────────────────────────────────────────────────
from StarsCords import load_image, detect_stars       # noqa: E402
from imagesStarAlignment import MatchResult, visualise_matches  # noqa: E402



# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────
def _internal_angles(triplet: np.ndarray) -> Tuple[float, float, float]:
    """
    Return the three *internal* angles of the triangle (degrees, ascending).

    Parameters
    ----------
    triplet : (3, 2) ndarray
        Row‑vector (x, y) coordinates of the three vertices.

    Notes
    -----
    Uses the law of cosines with a small epsilon to guard against
    floating‑point noise in near‑collinear triples.
    """
    # Edge lengths – opposite to vertex order (a ↔ p0, b ↔ p1, c ↔ p2)
    a = np.linalg.norm(triplet[1] - triplet[2])
    b = np.linalg.norm(triplet[0] - triplet[2])
    c = np.linalg.norm(triplet[0] - triplet[1])

    eps = 1e-9
    cos_A = np.clip((b**2 + c**2 - a**2) / (2 * b * c + eps), -1.0, 1.0)
    cos_B = np.clip((a**2 + c**2 - b**2) / (2 * a * c + eps), -1.0, 1.0)
    cos_C = np.clip((a**2 + b**2 - c**2) / (2 * a * b + eps), -1.0, 1.0)

    A = math.degrees(math.acos(cos_A))
    B = math.degrees(math.acos(cos_B))
    C = math.degrees(math.acos(cos_C))

    return tuple(sorted((A, B, C)))


def _quantise_angles(angles: Tuple[float, float, float],
                     q_deg: float) -> Tuple[int, int, int]:
    """
    Quantise *angles* to an integer grid with spacing *q_deg*.

    Example
    -------
    q_deg = 0.5 →   23.4° ↦ round(23.4 / 0.5) = 47
    """
    return tuple(int(round(a / q_deg)) for a in angles)


class _TripletEntry(NamedTuple):
    """Helper container for a pre‑computed L2 triplet."""
    idx: Tuple[int, int, int]          # indices into coords2
    angles: Tuple[float, float, float] # exact (sorted) angles (degrees)


def _build_angle_index(coords: np.ndarray,
                       q_deg: float,
                       min_pair_dist: float = 1.0
                       ) -> Dict[Tuple[int, int, int], List[_TripletEntry]]:
    """
    Pre‑compute all 3‑star combinations in *coords* and bucket them
    by quantised angle triple.

    Returns
    -------
    dict[quantised_key → list of _TripletEntry]
    """
    index: Dict[Tuple[int, int, int], List[_TripletEntry]] = {}
    for i, j, k in itertools.combinations(range(len(coords)), 3):
        p = coords[[i, j, k]]

        # Reject degenerate / almost‑collinear triangles
        if (np.linalg.norm(p[0] - p[1]) < min_pair_dist or
                np.linalg.norm(p[0] - p[2]) < min_pair_dist or
                np.linalg.norm(p[1] - p[2]) < min_pair_dist):
            continue

        ang = _internal_angles(p)
        key = _quantise_angles(ang, q_deg)
        index.setdefault(key, []).append(_TripletEntry((i, j, k), ang))
    return index


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def match_images_by_triangles(
    l1_path: str,
    l2_path: str,
    *,
    eps_angle_deg: float = 2.0,
    eps_xy_px: float = 15.0,
    top_k: int = 70,
    min_matches: int = 3,
    verbose: bool = True,
    mindist: float = 700.0,
) -> Optional[MatchResult]:
    """
    Align two star‑field images using *triangle* (3‑star) voting.

    Returns
    -------
    MatchResult  on success
    None         if no satisfactory alignment found
    """
    # ────────────────────────────────────────────────────────────── 1 ──
    # Load & detect
    img1 = load_image(l1_path)
    img2 = load_image(l2_path)

    stars1 = detect_stars(img1)
    stars2 = detect_stars(img2)

    if verbose:
        print(f"[triangle] detected {len(stars1)} in L1  /  {len(stars2)} in L2")

    if len(stars1) < 3 or len(stars2) < 3:
        if verbose:
            print("[triangle] <3 stars – aborting")
        return None

    # Keep the brightest *top_k*
    stars1 = sorted(stars1, key=lambda s: -s[3])[:top_k]
    stars2 = sorted(stars2, key=lambda s: -s[3])[:top_k]

    coords1 = np.array([(x, y) for x, y, *_ in stars1], dtype=np.float32)
    coords2 = np.array([(x, y) for x, y, *_ in stars2], dtype=np.float32)

    # ────────────────────────────────────────────────────────────── 2 ──
    # Build L2 angle index
    q = eps_angle_deg / 2.0          # finer than tolerance ⇒ won’t miss matches
    angle_index = _build_angle_index(coords2, q, mindist)
    if verbose:
        total_triplets = sum(len(v) for v in angle_index.values())
        print(f"[triangle] pre‑indexed {total_triplets} L2 triplets")

    # ────────────────────────────────────────────────────────────── 3 ──
    best_overlap = 0
    best_transform = None
    best_matches = None
    best_pivots = None               # (i,j,k,  u,v,w)

    # Iterate over *all* L1 triplets
    for i, j, k in itertools.combinations(range(len(coords1)), 3):
        p = coords1[[i, j, k]]

        # Skip tiny triangles
        if (np.linalg.norm(p[0] - p[1]) < mindist or
                np.linalg.norm(p[0] - p[2]) < mindist or
                np.linalg.norm(p[1] - p[2]) < mindist):
            continue

        ang1 = _internal_angles(p)
        key = _quantise_angles(ang1, q)

        # Candidate L2 triples with the same quantised key
        candidates = angle_index.get(key, [])
        if not candidates:
            continue

        # Check each candidate (could be >1 due to hash collisions)
        for entry in candidates:
            idx2 = entry.idx
            q_triplet = coords2[list(idx2)]

            # Full precision angle test
            if any(abs(a - b) > eps_angle_deg for a, b in zip(ang1, entry.angles)):
                continue

            # Estimate similarity transform (3‑point → 3‑point)
            M, _ = cv2.estimateAffinePartial2D(
                p.astype(np.float32),
                q_triplet.astype(np.float32),
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=2000,
            )
            if M is None:
                continue

            # Apply to *all* L1 stars
            ones = np.ones((coords1.shape[0], 1), dtype=np.float32)
            transformed = np.hstack([coords1, ones]) @ M.T

            # Count close pairs (vectorised)
            d = np.linalg.norm(
                transformed[:, None, :] - coords2[None, :, :], axis=2
            )
            matches = np.where(d <= eps_xy_px)
            unique_l1 = set(matches[0])
            overlap = len(unique_l1)

            if overlap > best_overlap:
                best_overlap = overlap
                best_transform = M
                best_matches = list(zip(*matches))
                best_pivots = (i, j, k, *idx2)

                if verbose:
                    print(f"[triangle] new best – {overlap} matches "
                          f"(pivots L1:{i,j,k} ↔ L2:{idx2})")

                # Early exit – perfect fit
                if overlap == len(coords1):
                    break

    # ────────────────────────────────────────────────────────────── 4 ──
    if best_transform is None or best_overlap < min_matches:
        if verbose:
            print("[triangle] no satisfactory alignment")
        return None

    # One‑to‑one match pruning (keep 1st L1 occurrence)
    seen = set()
    filtered = []
    for i1, i2 in best_matches:
        if i1 not in seen:
            filtered.append((i1, i2))
            seen.add(i1)

    l1_matched = coords1[[i for i, _ in filtered]]
    l2_matched = coords2[[j for _, j in filtered]]

    # Extract sim‑transform params
    dx, dy = best_transform[0, 2], best_transform[1, 2]
    scale = np.linalg.norm(best_transform[0, :2])
    theta = math.atan2(best_transform[1, 0], best_transform[0, 0])

    return MatchResult(
        l1_points=l1_matched,
        l2_points=l2_matched,
        transform=(dx, dy, theta, scale),
        pivot_indices=best_pivots,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI demo
# ──────────────────────────────────────────────────────────────────────────────
def _demo():
    """
    Quick test over *all* permutations of images in ./images/.

    Adjust paths as needed.
    """
    image_dir_in = Path("images")
    image_dir_out = Path("images/out")
    image_files = sorted([f for f in image_dir_in.glob("*") if f.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    for a, b in itertools.permutations(image_files, 2):
        print(f"\n[triangle] {a.name}  vs  {b.name}")
        res = match_images_by_triangles(str(a), str(b), verbose=True)
        if res is None:
            print("  → failed")
        else:
            print(f"  → {len(res.l1_points)} matches | "
                  f"dx={res.transform[0]:.1f}  dy={res.transform[1]:.1f}  "
                  f"θ={math.degrees(res.transform[2]):.2f}°  s={res.transform[3]:.4f}"
                  )

            output_name = f"match_{a.stem}_vs_{b.stem}.png"
            visualise_matches(
                a,  # same as passed to the matcher
                b,
                res,
                output_path=str(image_dir_out / output_name)
            )

if __name__ == "__main__":
    _demo()
