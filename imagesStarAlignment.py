import math
import itertools
from typing import List, Optional, NamedTuple
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
from itertools import permutations


# Import the user‑supplied detection utilities ----------------------------------
from StarsCords import load_image, detect_stars


class MatchResult(NamedTuple):
    """Container for the result returned by :func:`match_images`."""

    # (x, y) pixel coordinates in L1
    l1_points: np.ndarray  # shape (M, 2)
    # (x, y) pixel coordinates in L2 that correspond to *l1_points*
    l2_points: np.ndarray  # shape (M, 2)
    # 4‑tuple (dx, dy, d_alpha_rad, d_scale)
    transform: Tuple[float, float, float, float]
    # indices **within the unsorted original detection list** of the two pivot stars in L1 & L2
    pivot_indices: Tuple[int, int, int, int]

    def as_pairs(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Return the mapping as a list of ``((x1, y1), (x2, y2))`` tuples."""
        return [((float(x1), float(y1)), (float(x2), float(y2))) for (x1, y1), (x2, y2) in zip(self.l1_points, self.l2_points)]


# -------------------------------------------------------------------------------
# Internal maths helpers
# -------------------------------------------------------------------------------

def _build_rotation_matrix(theta: float) -> np.ndarray:
    """Return a 2×2 rotation matrix for angle *theta* (radians)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _compute_two_point_transform(a1: np.ndarray,
                                 b1: np.ndarray,
                                 a2: np.ndarray,
                                 b2: np.ndarray
                                 ) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Return the full 3‑DoF similarity transform that maps *a1*→*a2*
    and *b1*→*b2*.

    We treat every star coordinate as a **row vector**
    p  = [x  y].  The forward model is

        p' = s · (p @ R) + t ,

    where R is the usual counter‑clockwise 2×2 rotation matrix and **t**
    is also a row vector (dx, dy).  Working with row‑vectors lets us
    apply the transform to many points at once with a single
        points @ R
    call.
    """
    # Vector from first → second pivot in each image ---------------------------
    v1 = b1 - a1
    v2 = b2 - a2

    # Isotropic scale ----------------------------------------------------------
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6:
        raise ValueError("Pivot stars in L1 are too close – cannot compute scale.")
    s = n2 / n1

    # Rotation (bearing difference) -------------------------------------------
    theta = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
    R = _build_rotation_matrix(theta)

    # Translation – **row‑vector** formulation ---------------------------------
    # Want:  a1  ─→  a2  ⇒  s·(a1@R) + t = a2   ⟹   t = a2 - s·(a1@R)
    t = a2 - s * (a1 @ R)
    dx, dy = float(t[0]), float(t[1])

    return R, dx, dy, theta, s


def _apply_similarity_transform(points: np.ndarray,
                                R: np.ndarray,
                                dx: float,
                                dy: float,
                                s: float
                                ) -> np.ndarray:
    """
    Vectorised application of the similarity transform.

    Parameters
    ----------
    points : (N, 2) array of row‑vector coordinates.

    Returns
    -------
    (N, 2) array with the transform applied.
    """
    return s * (points @ R) + np.array([dx, dy])


# ---------------------------------------------------------------------------
# Utility – pick a contrasting text colour for the background pixel under (x, y)
# ---------------------------------------------------------------------------
def _text_colour(img: np.ndarray, x: int, y: int) -> Tuple[int, int, int]:
    """
    Return (B, G, R) that contrasts with the local background so the digits
    remain readable on both light and dark skies.
    """
    # Sample a 3×3 neighbourhood, clamp indices to image bounds
    h, w = img.shape[:2]
    y0, y1 = max(0, y - 1), min(h, y + 2)
    x0, x1 = max(0, x - 1), min(w, x + 2)
    patch_mean = img[y0:y1, x0:x1].mean()
    # If the patch is bright use black, else white
    return (0, 0, 0) if patch_mean > 128 else (255, 255, 255)


# -------------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------------

def match_images(
    l1_path: str,
    l2_path: str,
    *,
    epsilon: float = 15.0,
    min_matches: int = 6,
    top_k: int = 30,
    verbose: bool = True,
    mindist: int = 500
) -> Optional[MatchResult]:
    """
    Align two star‑field images and return the best mapping of stars between them.
    """

    # Step 1: Load and detect stars
    img1 = load_image(l1_path)
    img2 = load_image(l2_path)

    stars1 = detect_stars(img1)
    stars2 = detect_stars(img2)

    if verbose:
        print(f"Detected {len(stars1)} candidate stars in L1 and {len(stars2)} in L2.")

    if len(stars1) < 2 or len(stars2) < 2:
        if verbose:
            print("Not enough stars in one of the images – aborting alignment.")
        return None

    # Step 2: Keep only the top-k brightest stars
    stars1_sorted = sorted(stars1, key=lambda s: -s[3])[:top_k]
    stars2_sorted = sorted(stars2, key=lambda s: -s[3])[:top_k]

    coords1 = np.array([(x, y) for x, y, *_ in stars1_sorted], dtype=np.float32)
    coords2 = np.array([(x, y) for x, y, *_ in stars2_sorted], dtype=np.float32)

    # Step 3: Brute-force all pairs and estimate transforms using OpenCV
    best_overlap_count = 0
    best_transform = None
    best_matches = None

    for (i1, j1) in itertools.permutations(range(len(coords1)), 2):
        pivot1 = np.array([coords1[i1], coords1[j1]])

        for (i2, j2) in itertools.permutations(range(len(coords2)), 2):
            pivot2 = np.array([coords2[i2], coords2[j2]])

            if np.linalg.norm(coords1[i1] - coords1[j1]) < mindist:  # or another threshold
                continue
            if np.linalg.norm(coords2[i2] - coords2[j2]) < mindist:
                continue

            # print(f"Trying pivot pair (L1: {i1},{j1}) and (L2: {i2},{j2})")
            # print(f"  Pivot distance L1: {np.linalg.norm(pivot1[0] - pivot1[1]):.2f}")
            # print(f"  Pivot distance L2: {np.linalg.norm(pivot2[0] - pivot2[1]):.2f}")

            # Use OpenCV to estimate similarity transform (affine without shear)
            matrix, inliers = cv2.estimateAffinePartial2D(pivot1, pivot2, method=cv2.RANSAC)
            if matrix is None:
                continue

            # Apply transform to all coords1
            ones = np.ones((coords1.shape[0], 1), dtype=np.float32)
            coords1_hom = np.hstack([coords1, ones])
            transformed = coords1_hom @ matrix.T

            # Count matches within epsilon
            dists = np.linalg.norm(transformed[:, None, :] - coords2[None, :, :], axis=2)
            matches = np.where(dists <= epsilon)

            unique_l1 = set(matches[0])
            overlap_count = len(unique_l1)

            if overlap_count > best_overlap_count:
                if verbose:
                    print(f"Trying pivot pair (L1: {i1},{j1}) and (L2: {i2},{j2})")
                    print(f"  Pivot distance L1: {np.linalg.norm(pivot1[0] - pivot1[1]):.2f}")
                    print(f"  Pivot distance L2: {np.linalg.norm(pivot2[0] - pivot2[1]):.2f}")
                best_overlap_count = overlap_count
                best_transform = matrix
                best_matches = list(zip(*matches))

    if verbose:
        print(f"Best match has {best_overlap_count} overlaps (min required = {min_matches})")

    if best_transform is None or best_overlap_count < min_matches:
        if verbose:
            print("No satisfactory alignment found.")
        return None

    # Step 4: Filter matches (1-to-1 from L1)
    seen_l1 = set()
    filtered_matches = []
    for i1, i2 in best_matches:
        if i1 not in seen_l1:
            filtered_matches.append((i1, i2))
            seen_l1.add(i1)

    l1_matched = coords1[[i1 for i1, _ in filtered_matches]]
    l2_matched = coords2[[i2 for _, i2 in filtered_matches]]

    # Extract similarity parameters: rotation, scale, translation
    dx, dy = best_transform[0, 2], best_transform[1, 2]
    scale = np.linalg.norm(best_transform[0, :2])  # ||a,b||
    theta = np.arctan2(best_transform[1, 0], best_transform[0, 0])
    transform_tuple = (dx, dy, theta, scale)

    return MatchResult(
        l1_points=l1_matched,
        l2_points=l2_matched,
        transform=transform_tuple,
        pivot_indices=None,  # Not tracked anymore
    )



# ---------------------------------------------------------------------------
# Utility – pick a contrasting text colour for the background pixel under (x, y)
# ---------------------------------------------------------------------------
def _text_colour(img: np.ndarray, x: int, y: int) -> Tuple[int, int, int]:
    """
    Return (B, G, R) that contrasts with the local background so the digits
    remain readable on both light and dark skies.
    """
    # Sample a 3×3 neighbourhood, clamp indices to image bounds
    h, w = img.shape[:2]
    y0, y1 = max(0, y - 1), min(h, y + 2)
    x0, x1 = max(0, x - 1), min(w, x + 2)
    patch_mean = img[y0:y1, x0:x1].mean()
    # If the patch is bright use black, else white
    return (0, 0, 0) if patch_mean > 128 else (255, 255, 255)


def visualise_matches(
    l1_path: str,
    l2_path: str,
    result: MatchResult,
    output_path: str = "match_visualisation.png",
    *,
    point_radius: int = 4,
    point_thickness: int = -1,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    spacer_px: int = 12,
) -> None:
    """
    Visualize original L1, transformed L1, and L2 side-by-side with matching labels.
    """

    # Load grayscale images
    img1_gray = load_image(l1_path)
    img2_gray = load_image(l2_path)

    # Convert to BGR for color drawing
    img1 = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2_gray, cv2.COLOR_GRAY2BGR)

    # Apply the transform to the image using warpAffine
    dx, dy, theta_deg, s = result.transform
    theta = np.radians(theta_deg)

    # Construct the similarity transform matrix (rotation + scale + translation)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ]) * s

    M_affine = np.hstack([R, np.array([[dx], [dy]])])  # Shape (2, 3)

    # Warp the original L1 image
    h1, w1 = img1.shape[:2]
    img1_transformed = cv2.warpAffine(img1, M_affine, (w1, h1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Resize all images to the same height
    def _resize_keep_aspect(img, h_target):
        h, w = img.shape[:2]
        if h == h_target:
            return img
        scale = h_target / h
        return cv2.resize(img, (int(round(w * scale)), h_target), interpolation=cv2.INTER_AREA)

    h_max = max(img1.shape[0], img2.shape[0])
    img1 = _resize_keep_aspect(img1, h_max)
    img1_transformed = _resize_keep_aspect(img1_transformed, h_max)
    img2 = _resize_keep_aspect(img2, h_max)

    # Prepare offsets
    spacer = np.zeros((h_max, spacer_px, 3), dtype=np.uint8)
    composite = np.hstack([img1, spacer, img1_transformed, spacer.copy(), img2])

    offset_l1 = np.array([0, 0])
    offset_transformed = np.array([img1.shape[1] + spacer.shape[1], 0])
    offset_l2 = np.array([img1.shape[1]*2 + spacer.shape[1]*2, 0])

    # Transform L1 star coordinates
    transformed_l1_coords = (result.l1_points @ R.T) + np.array([dx, dy])

    # Draw dots, labels, and connecting lines
    for idx, (p1, p1t, p2) in enumerate(zip(result.l1_points, transformed_l1_coords, result.l2_points), start=1):
        x1, y1 = map(int, map(round, p1))
        x1t, y1t = map(int, map(round, p1t))
        x2, y2 = map(int, map(round, p2))

        x1_off = x1 + offset_l1[0]
        y1_off = y1 + offset_l1[1]
        x1t_off = x1t + offset_transformed[0]
        y1t_off = y1t + offset_transformed[1]
        x2_off = x2 + offset_l2[0]
        y2_off = y2 + offset_l2[1]

        # Draw green dots
        cv2.circle(composite, (x1_off, y1_off), point_radius, (0, 255, 0), point_thickness)
        cv2.circle(composite, (x1t_off, y1t_off), point_radius, (0, 255, 0), point_thickness)
        cv2.circle(composite, (x2_off, y2_off), point_radius, (0, 255, 0), point_thickness)

        # Draw blue lines connecting the matches
        cv2.line(composite, (x1t_off, y1t_off), (x2_off, y2_off), color=(255, 0, 0), thickness=1)

        # Draw index labels with contrast-aware colors
        for (x, y, color_offset) in [(x1_off, y1_off, _text_colour(composite, x1_off, y1_off)),
                                     (x1t_off, y1t_off, _text_colour(composite, x1t_off, y1t_off)),
                                     (x2_off, y2_off, _text_colour(composite, x2_off, y2_off))]:
            cv2.putText(composite, str(idx), (x + point_radius + 2, y - 2),
                        font, font_scale, color_offset, font_thickness, cv2.LINE_AA)


    # Save output
    cv2.imwrite(output_path, composite)
    print(f"[visualise_matches] Saved visualization → '{Path(output_path).resolve()}'")



def main():
#     # Example usage (adjust file paths as needed)
#     # l1_file = "images/ST_db1.png"
#     # l2_file = "images/ST_db2.png"
#
#     # l2_file = "images/ST_db1.png"
#     # l1_file = "images/ST_db2.png"
#     #
#     l1_file = "images/fr1.jpg"
#     l2_file = "images/fr2.jpg"
#     # l2_file = "images/fr1.jpg"
#     # l1_file = "images/fr2.jpg"
    image_dir_in = Path("images")
    image_dir_out = Path("images/out")
    image_files = sorted([f for f in image_dir_in.glob("*") if f.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    for img1_path, img2_path in permutations(image_files, 2):
        print(f"Processing: {img1_path.name} vs {img2_path.name}")
        result = match_images(str(img1_path), str(img2_path), epsilon=15.0, min_matches=6, verbose=True)

        if result is None:
            print(f"No satisfactory alignment for {img1_path.name} vs {img2_path.name}.")
        else:
            dx, dy, theta, scale = result.transform
            print("Alignment found:")
            print(f"Translation: dx={dx:.2f}, dy={dy:.2f}")
            print(f"Rotation: {math.degrees(theta):.2f} degrees")
            print(f"Scale: {scale:.4f}")
            print(f"Matched pairs: {len(result.l1_points)}")

            output_name = f"match_{img1_path.stem}_vs_{img2_path.stem}.png"
            visualise_matches(
                str(img1_path),
                str(img2_path),
                result,
                output_path=str(image_dir_out / output_name)
            )


if __name__ == "__main__":
    main()
