import cv2
import numpy as np
import csv
from typing import List, Tuple
import os


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
    # height, width = image.shape[:2]
    # total_pixels = width * height
    # params.minArea = total_pixels * 0.00001  # attempt to make the sensitivity scale
    params.minArea = 30
    params.maxArea = 500
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    image = cv2.GaussianBlur(image, (5, 5), 0)

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




def show_detected_stars(image: np.ndarray, stars: List[Tuple[float, float, float, float]], original_image_path: str) -> None:
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
        cv2.circle(image_bgr, center, radius, (0, 255, 0), 1)

    # Resize for display (e.g., scale to 50% size)
    scale = 0.5
    resized_image = cv2.resize(image_bgr, (0, 0), fx=scale, fy=scale)

    # Show the image
    cv2.imshow("Detected Stars", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the image with "_plot.png" suffix
    base, ext = os.path.splitext(original_image_path)
    output_path = f"plot/{base}_plot.png"
    cv2.imwrite(output_path, image_bgr)
    print(f"Saved detected star plot to: {output_path}")



def process_star_image(image_path: str, output_csv: str, show_image: bool = False) -> None:
    """Full pipeline to process a star image and optionally show results."""
    image = load_image(image_path)
    stars = detect_stars(image)
    save_to_csv(stars, output_csv)
    print(f"Processed {len(stars)} stars and saved to '{output_csv}'.")

    if show_image:
        show_detected_stars(image, stars, image_path)


if __name__ == "__main__":
    import sys
    # if len(sys.argv) not in [3, 4]:
    #     print("Usage: python star_extractor.py <image_path> <output_csv> [--show]")
    # else:
    #     img_path = sys.argv[1]
    #     csv_path = sys.argv[2]
    #     show_flag = len(sys.argv) == 4 and sys.argv[3] == "--show"
    #     process_star_image(img_path, csv_path, show_image=show_flag)
    # img_path =r"images/fr1.jpg"
    # img_path =r"images/ST_db1.png"
    img_path =r"images/ST_db2.png"
    # img_path =r"images/ST_db1.png"
    csv_path =rf"{img_path} StarCord.csv"
    show_flag = len(sys.argv) == 4 and sys.argv[3] == "--show"
    process_star_image(img_path, csv_path, show_image=True)
