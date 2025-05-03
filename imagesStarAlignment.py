import math
import itertools
from typing import List, Optional, NamedTuple
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
from itertools import permutations


# Import the detection utilities ----------------------------------
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
    mindist: int = 1000
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
                # if verbose:
                    # print(f"Trying pivot pair (L1: {i1},{j1}) and (L2: {i2},{j2})")
                    # print(f"  Pivot distance L1: {np.linalg.norm(pivot1[0] - pivot1[1]):.2f}")
                    # print(f"  Pivot distance L2: {np.linalg.norm(pivot2[0] - pivot2[1]):.2f}")
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


def visualise_matches(
    l1_path: str,
    l2_path: str,
    match_result: MatchResult,
    *,
    output_path: str,
    circle_radius: int = 4,
    circle_thickness: int = -1,        # -1 ➜ filled
    line_thickness: int = 1,
    colour_orig: Tuple[int, int, int] = (0, 255, 255),   # yellow
    colour_warp: Tuple[int, int, int] = (0, 0, 255),     # red
    colour_tgt: Tuple[int, int, int]  = (0, 255, 0),     # green
    colour_line: Tuple[int, int, int] = (255, 0, 0),     # blue
    font=cv2.FONT_HERSHEY_SIMPLEX
) -> None:
    """
    Write a composite PNG with three panels and the matched‑star overlay.

    Parameters
    ----------
    l1_path, l2_path   : str
        Filenames of the original images.
    match_result       : MatchResult
        Output of `match_images`.
    output_path        : str
        Where to save the PNG.
    """

    # ---------------------------------------------------------------------
    # 1. Load images (force 3‑channel BGR so we can draw colour)
    # ---------------------------------------------------------------------
    img1 = cv2.imread(l1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(l2_path, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load one of the input images.")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H   = max(h1, h2)                     # canvas height
    pad = 20                              # gap between panels

    # ---------------------------------------------------------------------
    # 2. Warp L1 → L2 frame with the 2×3 similarity matrix
    # ---------------------------------------------------------------------
    M = np.asarray([[match_result.transform[3] * math.cos(match_result.transform[2]),
                     -match_result.transform[3] * math.sin(match_result.transform[2]),
                     match_result.transform[0]],
                    [match_result.transform[3] * math.sin(match_result.transform[2]),
                      match_result.transform[3] * math.cos(match_result.transform[2]),
                      match_result.transform[1]]],
                   dtype=np.float32)       # build it back so it’s explicit

    img1_warp = cv2.warpAffine(img1, M, dsize=(w2, h2),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)

    # ---------------------------------------------------------------------
    # 3. Assemble the big canvas: [orig | warped | target]
    # ---------------------------------------------------------------------
    W = w1 + pad + w2 + pad + w2          # 3rd panel same width as L2
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Offsets of the three panels on the canvas
    off_orig = (0,             0)
    off_warp = (w1 + pad,      0)
    off_tgt  = (w1 + pad + w2 + pad, 0)

    # Paste images
    canvas[off_orig[1]:off_orig[1]+h1, off_orig[0]:off_orig[0]+w1] = img1
    canvas[off_warp[1]:off_warp[1]+h2, off_warp[0]:off_warp[0]+w2] = img1_warp
    canvas[off_tgt [1]:off_tgt [1]+h2, off_tgt [0]:off_tgt [0]+w2] = img2

    # ---------------------------------------------------------------------
    # 4. Draw matched stars
    # ---------------------------------------------------------------------
    # Convenience offsets for later
    ox, oy = off_orig
    wx, wy = off_warp
    tx, ty = off_tgt

    # Pre‑append the 1 for homogeneous coordinates once
    ones = np.ones((match_result.l1_points.shape[0], 1), dtype=np.float32)
    l1_h = np.hstack([match_result.l1_points.astype(np.float32), ones])
    l1_warped = (l1_h @ M.T)              # (N,2) array in warped frame

    for (p_orig, p_warp, p_tgt) in zip(match_result.l1_points,
                                       l1_warped,
                                       match_result.l2_points):

        # Integer pixel coords
        x_o, y_o = map(int, p_orig)
        x_w, y_w = map(int, p_warp)
        x_t, y_t = map(int, p_tgt)

        # Draw circles
        cv2.circle(canvas, (x_o + ox, y_o + oy), circle_radius,
                   colour_orig, circle_thickness, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x_w + wx, y_w + wy), circle_radius,
                   colour_warp, circle_thickness, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x_t + tx, y_t + ty), circle_radius,
                   colour_tgt, circle_thickness, lineType=cv2.LINE_AA)

        # Draw connecting line (from warped → target)
        cv2.line(canvas,
                 (x_w + wx, y_w + wy),
                 (x_t + tx, y_t + ty),
                 colour_line, line_thickness, cv2.LINE_AA)

    # ---------------------------------------------------------------------
    # 5. Panel titles
    # ---------------------------------------------------------------------
    fh = 0.6                              # font height scale
    th = 2                                # thickness
    cv2.putText(canvas, "Original",
                (ox + 10,  30), font, fh, colour_orig, th, cv2.LINE_AA)
    cv2.putText(canvas, "Warped to L2",
                (wx + 10,  30), font, fh, colour_warp, th, cv2.LINE_AA)
    cv2.putText(canvas, "Target",
                (tx + 10,  30), font, fh, colour_tgt,  th, cv2.LINE_AA)

    # ---------------------------------------------------------------------
    # 6. Save
    # ---------------------------------------------------------------------
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), canvas)
    print(f"[visualise_matches] wrote → {out_p}")




def main():
    # Example usage (adjust file paths as needed)
    # image_dir_in = Path("images")
    # image_dir_out = Path("images/out")
    image_dir_in = Path("images/dataset")
    image_dir_out = Path("images/dataset/datasetOut")
    image_files = sorted([f for f in image_dir_in.glob("*") if f.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    for img1_path, img2_path in permutations(image_files, 2):
        print(f"\nProcessing: {img1_path.name} vs {img2_path.name}")
        result = match_images(str(img1_path), str(img2_path), epsilon=15.0, min_matches=4, verbose=True)

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
