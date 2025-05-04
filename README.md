# Star Field Image Alignment

This project aligns two astronomical images by detecting stars and estimating the transformation (translation, rotation, and scale) needed to match them. It supports two algorithms: **pair-based matching** and **triangle-based matching**.

---

Part One: Algorithm – As Simple and Efficient as Possible
Step 1: Create a list of stars detected in the image, each with an ID, X, Y coordinates, radius, and brightness.
 This is implemented using a blob detector in OpenCV, after applying pre-processing filters such as TopHat and medianBlur.

Step 2: Implemented in the imagesStarAlignment
Apply a transformation between image L1 and image L2. The goal is to align one image on top of the other. To do this, we need to estimate the following transformation parameters: translation, rotation, and scale:
(deltaX, deltaY, deltaAlpha, deltaScale) = T
How we do it:
Select 2 stars from L1 that are neither too close nor too far apart. In our case, we limited the minimum distance between them to 1000 pixels. We assume these two stars also appear in image L2.

For all possible pairs in L2, compute the transformation T that maps the chosen L1 stars onto them.
Apply T to all stars in L1 to get L1T.
Count how many overlapping stars exist between L1T and L2 (within an EPSILON of 15 pixels).

To reduce false positives (which are minimal thanks to the filters that prevent the blob detector from detecting too many fake stars), we repeat the algorithm for every pair among the top K brightest stars in L1.
 The transformation T with the most overlapping stars is considered the correct match.
To prevent choosing stars in L1 that don’t appear in L2 (or satellites), we define a threshold for the number of matching stars for T. For example, if a transformation T leads to only 5 matching stars, it is likely incorrect and will be discarded. If no satisfactory T is found, we declare a mismatch.
Output:
A mapping of matching stars from the L1 list to L2.

An image containing both the original photos and the transformed image L1T.

A visual overlay showing the matched stars and their locations in L1, L2, and L1T.

Alternative Approach: Implemented in a triangle match file
 2. Use 3 stars instead of 2, starting from a certain minimum distance (e.g., 700 pixels). Compute the triangle angles between them and search for triplets in L2 with matching angles (within some epsilon tolerance).
To make this efficient, precompute all triangle angles for every triplet of stars in L2 and store them in a hash table.

Then find a matching triplet and compute the corresponding T.

Use the T with the best matching result.

It’s worth noting that this algorithm is significantly faster than the first one, but based on our results, it is generally less accurate.
Notes:
 There are many ways to reduce the search space of triplets in L2. For example, filter by distance (not too large or small), know the camera's focal length, and search only within the field of view defined by the focal length.




## Project Structure

```
.
├── StarsCords.py            # Detects stars and extracts coordinates
├── imagesStarAlignment.py   # Aligns star fields using 2-star matching
├── triangle_match.py        # Aligns star fields using 3-star triangle matching
├── images/                  # Input star field images
├── images/out/              # Output visualizations
└── README.md
```

---

## Features

- **Star Detection**:
  - Uses OpenCV’s `SimpleBlobDetector` with TopHat and median filtering.
  - Each star: `(x, y, radius, brightness)`.

- **Alignment Algorithms**:
  1. **Two-Star Matching (`imagesStarAlignment.py`)**:
     - Loops over top brightest star pairs.
     - Estimates a similarity transform.
     - Pick the one with the most overlaps.

  2. **Triangle Matching (`triangle_match.py`)**:
     - Uses internal angles of 3-star triangles.
     - Faster than pair matching but slightly less accurate.

- **Visualization**:
  - Side-by-side panels:
    - Original Image (L1)
    - Warped Image (L1 transformed)
    - Target Image (L2)
  - Stars and connections are color-coded.

---

## Output Example uplouded in the folders

- **Yellow**: Stars in original L1  
- **Red**: Warped stars (L1 → L2 frame)  
- **Green**: Stars in L2  
- **Blue lines**: Matched star pairs

---

## How to Run

### 1. Star Detection Only

```bash
python StarsCords.py
```

- Reads images from `images/`
- Saves results to `images/SratCordsOut/`

### 2. Two-Star Alignment

```bash
python imagesStarAlignment.py
```

- Aligns all image pairs in `images/`
- Saves visual results to `images/out/`

### 3. Triangle-Based Alignment

```bash
python triangle_match.py
```

- Aligns all image pairs using triangle voting
- Saves visual results to `images/out/`

---

## Configuration

| Parameter       | Description                            | Files Affected          |
|----------------|----------------------------------------|-------------------------|
| `epsilon`       | Match radius tolerance (pixels)        | 
| `mindist`       | Min star pair/triangle size            | 
| `top_k`         | Top brightest stars to consider        | 
| `min_matches`   | Minimum matching stars to accept T     | 



---

## License
 free to use and modify.
