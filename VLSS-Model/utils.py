import numpy as np

from PIL import Image


def smooth(input, width, iter):
    # Ensure width is odd to have a symmetric window
    if width % 2 == 0:
        raise ValueError("Width must be an odd number for symmetry.")

    half_w = width // 2
    input = np.array(input, dtype=float)
    length = len(input)

    for _ in range(iter):
        smoothed = np.zeros_like(input)
        for i in range(length):
            start = max(0, i - half_w)
            end = min(length, i + half_w + 1)
            smoothed[i] = np.mean(input[start:end])
        input = smoothed

    return input.tolist()

def overlay_sense(image_path, ray_lengths, sense):
    """
    Overlay the raycast on the original image using a vectorized polar approach.
    For each pixel, we compute its polar coordinates relative to the center,
    interpolate the ray length and opacity, and blend the polygon color if the pixel
    falls within the raycast.

    Parameters:
      image_path (str): Path to the image file.
      ray_lengths (iterable): Iterable of 180 ray lengths.
      sense (iterable): Iterable of 180 binary (or [0,1]) opacities corresponding to each ray.

    Returns:
      np.ndarray: The resulting image with the overlaid raycast.
    """
    if len(ray_lengths) != 180 or len(sense) != 180:
        raise ValueError("Both sense and ray_lengths must be lists of 180 values.")

    # Open the image and convert to a numpy array.
    image = Image.open(image_path).convert("RGBA")
    image_array = np.array(image)
    height, width = image_array.shape[:2]
    center_x, center_y = width // 2, height - 1

    # Create a grid of pixel coordinates.
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dx = xx - center_x
    dy = center_y - yy  # note: image y increases downward, so invert for polar coords
    r = np.sqrt(dx**2 + dy**2)

    # Compute angle for each pixel (in radians).
    # Here, 0 rad is straight up; angles vary between -pi/2 and pi/2 for points in front of the sensor.
    theta = np.arctan2(dx, dy)

    # Define the angles corresponding to each ray (180 rays evenly spaced from -90 to 90 degrees).
    ray_angles = np.deg2rad(np.linspace(-90, 90, 180))

    # Interpolate ray_lengths and sense for every pixel based on its theta.
    # For angles outside [-pi/2, pi/2], we assume no overlay (i.e. set to 0).
    interpolated_ray_length = np.interp(theta, ray_angles, ray_lengths, left=0, right=0)
    interpolated_sense = np.interp(theta, ray_angles, sense, left=0, right=0)

    # Create a mask for pixels that fall inside the overlay region.
    overlay_mask = (
        (theta >= -np.pi / 2) & (theta <= np.pi / 2) & (r <= interpolated_ray_length)
    )

    # Updated polygon color to #e95d21
    polygon_color = np.array([233, 93, 33], dtype=np.float32)

    # Prepare a float copy of the image for blending.
    composite = image_array.astype(np.float32)

    # Compute per-pixel opacity only for the overlay region.
    alpha = np.zeros_like(r, dtype=np.float32)
    alpha[overlay_mask] = interpolated_sense[overlay_mask]

    # Blend only the RGB channels for pixels in the overlay region.
    for c in range(3):
        composite[overlay_mask, c] = (
            alpha[overlay_mask] * polygon_color[c]
            + (1 - alpha[overlay_mask]) * composite[overlay_mask, c]
        )

    # Ensure the alpha channel remains 255.
    composite[..., 3] = 255

    # Optionally, cast back to uint8.
    return composite.astype(np.uint8)