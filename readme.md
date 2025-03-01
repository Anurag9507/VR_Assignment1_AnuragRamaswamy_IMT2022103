# VR_Assignment1_AnuragRamaswamy_IMT2022103

## Part 1: Use Computer Vision techniques to detect, segment, and count coins from an image containing scattered Indian coins.

### Implementation steps

- **Preprocessing** (`preprocess()`) - Converts images to grayscale, resizes them according to a fixed for consistency, applies Gaussian blur to reduce noise and uses adaptive thresholding to create binary images.
- **Coin Detection** (`detect_coin_like_shapes()`) - Detects coin-like shapes using contour detection and circularity measures to distinguish coin-like shapes from other shapes.
- **Segmentation** (`segment_coins_from_background()`) - Segmented coins from background and created a mask to make the output image contain only coins in a black background.
- **Process Coins** (`process_coins()`) - Images with coin contours are saved and the number of coins detected in each image is printed out on the terminal.

### Results and Observations

The code performed perfectly with different camera zooms.
However, when I tried it with overlapping / extremely closely spaced coins and dark backgrounds it gave a few errors.

### Dependencies:

- OpenCV
- NumPy

```
pip install numpy opencv-python
```

### Steps to Run

To run the part 1 code:

```
cd part1
python3 part1.py
```

- Ensure the `input` directory contains images of coins.
- The number of detected coins is printed in each image of `input` is printed out on the terminal.
- Images with detected coins are saved in the `output/` directory as `<image_name>_segmented.jpg`.

## Part 2: Create a stitched panorama from multiple overlapping images.

### Implementation steps

- **Scale Invariant Feature Transform(SIFT)** (`extract_features()`) was applied to detect keypoints and descriptors in images.
- Applied **BFMatcher** with **Lowe's ratio test** (`find_matches()`) to find corresponding features between images and filter out poor matches respectively.
- Calculated **homography transformation** using **RANSAC** (inside `find_matches()`) to warp one image onto another.
- The images were stitched together followed by **blending**, **cropping** (`crop_image()`) to have only the region of interest and **resizing** (`resize_image()`) for maintaining same image sizes.
- Output - (`process_images()`) - A **panorama** image was formed by stitching together all the overlapping input images in a folder. The **matching keypoints\*\* (`visualize_matches()`) between any two consecutive images was also saved as an image.

### Results and Observations:
The panorama image is better formed when there is more overlap between 2 consecutive images. When there is little to no overlap or an alignment issue, the panoramic image is not formed properly. 

### Dependencies:

- OpenCV
- NumPy

```
 pip install numpy opencv-python
```

### Steps to Run

To run the part 2 code:

```
cd part2
python3 part2.py
```

- Ensure the input directory is structured correctly. Each subfolder inside input/ should contain images to be stitched in sequential order (e.g., `0.jpg`, `1.jpg`, etc.) for better results.
- The panorama for each folder is saved in `output/<input_subfolder_name>/<input_subfolder_name>_panorama.jpg`.
- The `output/<input_subfolder_name>/` directory also contains images showing matching keypoints between images as `<input_subfolder_name>_matches_<image_number>.jpg`.

## Repository

Github: [VR_Assignment1_AnuragRamaswamy_IMT2022103](https://github.com/Anurag9507/VR_Assignment1_AnuragRamaswamy_IMT2022103)
