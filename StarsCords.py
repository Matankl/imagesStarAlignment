from pathlib import Path
import cv2
import numpy as np
import csv
from typing import List, Tuple
import os


def load_image(path: str) -> np.ndarray:
    """Load an image in grayscale."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def apply_tophat(image: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def detect_stars(image: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """
    Detect stars using OpenCV's SimpleBlobDetector.
    Returns a list of (x, y, radius, brightness) for each detected star.
    """
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.filterByArea = True
    params.minCircularity = 0.8  # More circular = more likely a star
    params.minConvexity = 0.8  # Closer to 1 = more star-like
    params.minArea = 5
    params.maxArea = 500
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    image = cv2.medianBlur(image, 5)
    image = apply_tophat(image)
    keypoints = detector.detect(image)

    stars = []
    for kp in keypoints:
        x, y = kp.pt
        radius = kp.size / 2
        x_int, y_int = int(round(x)), int(round(y))
        if 0 <= x_int < image.shape[1] and 0 <= y_int < image.shape[0]:
            brightness = float(image[y_int, x_int])
        else:
            brightness = 0.0
        stars.append((x, y, radius, brightness))

    mean_brightness = np.mean(image)
    stars = [(x, y, r, b) for (x, y, r, b) in stars if b > mean_brightness + 10]

    return stars


def save_to_csv(stars: List[Tuple[float, float, float, float]], output_path: str) -> None:
    """Save the list of star coordinates to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'r', 'b'])
        writer.writerows(stars)


def show_detected_stars(image: np.ndarray, stars: List[Tuple[float, float, float, float]], output_image_path: str) -> None:
    """
    Display the image with detected stars marked and save it to disk.

    Args:
        image: Grayscale input image.
        stars: List of detected stars (x, y, r, brightness).
        original_image_path: Path to the original image (used to determine output path).
    """
    # Convert grayscale to BGR for visualization
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw stars
    for x, y, r, _ in stars:
        center = (int(round(x)), int(round(y)))
        radius = int(round(r))
        cv2.circle(image_bgr, center, radius, (0, 255, 0), 2)

    # Resize for display (e.g., scale to 50% size)
    scale = 0.5
    resized_image = cv2.resize(image_bgr, (0, 0), fx=scale, fy=scale)

    # Show the image
    # cv2.imshow("Detected Stars", resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save the image with "_plot.png" suffix

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, image_bgr)
    print(f"Saved detected star plot to: {output_image_path}")


def process_star_image(image_path: str, output_dir: str, show_image: bool = False) -> None:
    """Full pipeline to process a star image and optionally show results."""
    #image prossing to pass better image to the blob detation
    image = load_image(image_path)
    # image = enhance_contrast(image)       # i want to reduce not to hence
    stars = detect_stars(image)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    csv_output = os.path.join(output_dir, f"{base_name}_StarCord.csv")
    image_output = os.path.join(output_dir, f"{base_name}_plot.png")

    save_to_csv(stars, csv_output)
    print(f"Processed {len(stars)} stars and saved to '{csv_output}'.")

    if show_image:
        show_detected_stars(image, stars, image_output)


if __name__ == "__main__":
    # image_dir_in = Path("images")
    # image_dir_out = r"images/SratCordsOut"

    image_dir_in = Path("images/dataset")
    image_dir_out = Path("images/dataset/plot")

    image_files = sorted([f for f in image_dir_in.glob("*") if f.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    for img1_path in image_files:
        print(f"Processing: {img1_path.name}")
        process_star_image(img1_path, image_dir_out, show_image=True)
