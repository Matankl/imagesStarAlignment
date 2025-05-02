import cv2
import numpy as np
import csv
from typing import List, Tuple


def load_image(path: str) -> np.ndarray:
    """Load an image in grayscale."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def detect_stars(image: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """
    Detect stars using OpenCV's SimpleBlobDetector.
    Returns a list of (x, y, radius, brightness) for each detected star.
    """
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 10000
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
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

    return stars


def save_to_csv(stars: List[Tuple[float, float, float, float]], output_path: str) -> None:
    """Save the list of star coordinates to a CSV file."""
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'r', 'b'])
        writer.writerows(stars)


def show_detected_stars(image: np.ndarray, stars: List[Tuple[float, float, float, float]]) -> None:
    """Display the image with stars marked."""
    # Convert grayscale to BGR for colored overlay
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y, r, _ in stars:
        center = (int(round(x)), int(round(y)))
        radius = int(round(r))
        cv2.circle(image_bgr, center, radius, (0, 255, 0), 1)
    cv2.imshow("Detected Stars", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_star_image(image_path: str, output_csv: str, show_image: bool = False) -> None:
    """Full pipeline to process a star image and optionally show results."""
    image = load_image(image_path)
    stars = detect_stars(image)
    save_to_csv(stars, output_csv)
    print(f"Processed {len(stars)} stars and saved to '{output_csv}'.")

    if show_image:
        show_detected_stars(image, stars)


if __name__ == "__main__":
    import sys
    # if len(sys.argv) not in [3, 4]:
    #     print("Usage: python star_extractor.py <image_path> <output_csv> [--show]")
    # else:
    #     img_path = sys.argv[1]
    #     csv_path = sys.argv[2]
    #     show_flag = len(sys.argv) == 4 and sys.argv[3] == "--show"
    #     process_star_image(img_path, csv_path, show_image=show_flag)
    img_path =
    csv_path =
    show_flag = len(sys.argv) == 4 and sys.argv[3] == "--show"
    process_star_image(img_path, csv_path, show_image=show_flag)
