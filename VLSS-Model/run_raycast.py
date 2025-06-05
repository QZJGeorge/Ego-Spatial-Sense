import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation


class CamToRay:
    def __init__(self):
        # Check if GPU is available and set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load the model and processor
        self.processor = OneFormerProcessor.from_pretrained(
            "shi-labs/oneformer_cityscapes_swin_large"
        )
        self.model = OneFormerForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_cityscapes_swin_large"
        ).to(self.device) 

    def segment_image(self, image):
        """Perform semantic segmentation on the input image and return the segmentation map."""
        semantic_inputs = self.processor(
            images=image, task_inputs=["semantic"], return_tensors="pt"
        )

        # Move inputs to the same device as the model
        semantic_inputs = {k: v.to(self.device) for k, v in semantic_inputs.items()}

        with torch.no_grad():
            semantic_outputs = self.model(**semantic_inputs)

        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            semantic_outputs, target_sizes=[(image.height, image.width)]
        )[0]

        return (
            predicted_semantic_map.cpu().numpy()
        )  # Move back to CPU for further processing

    def get_raycast_from_semantic(self, processed_map_np):
        """
        Computes raycasting from the bottom center of the image in 1-degree increments spanning -90° to 90°.

        - 0° is straight up (forward).
        - -90° is to the left.
        - 90° is to the right.

        A ray advances one pixel at a time until it hits a 'solid' pixel (value 0).
        """

        h, w = processed_map_np.shape
        center_x, center_y = w // 2, h - 1  # Bottom center

        ray_lengths = []
        for angle_deg in range(-90, 90):
            rad = np.deg2rad(angle_deg)
            dx, dy = np.sin(rad), -np.cos(rad)
            x, y, step = center_x, center_y, 0

            while (
                0 <= int(y) < h and 0 <= int(x) < w and processed_map_np[int(y), int(x)]
            ):
                x += dx
                y += dy
                step += 1

            ray_lengths.append(step)

        return ray_lengths

    def overlay_raycast_on_image(self, image_array, ray_lengths):
        """Overlay the raycast on the original image and return the modified image."""
        height, width = image_array.shape[:2]
        center_x, center_y = width // 2, height - 1

        # Use a DPI that works for you.
        dpi = 100

        # Create a figure with no frame and an axes that fills the entire figure.
        fig = plt.figure(frameon=False, figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])

        # Display the image without any additional space.
        ax.imshow(image_array, aspect="auto")
        for i, length in enumerate(ray_lengths):
            angle_deg = -90 + i
            rad = np.deg2rad(angle_deg)
            x_end = center_x + length * np.sin(rad)
            y_end = center_y - length * np.cos(rad)
            ax.plot([center_x, x_end], [center_y, y_end], color="red", linewidth=2)

        ax.axis("off")
        fig.canvas.draw()

        # Capture the RGBA buffer from the canvas.
        overlay_image = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        # Check if the captured overlay image matches the original dimensions.
        if overlay_image.shape[0] != height or overlay_image.shape[1] != width:
            # Resize to match the original image dimensions.
            overlay_image = np.array(
                Image.fromarray(overlay_image).resize((width, height))
            )
        return overlay_image

    def get_raycast(self, image_path):
        # """Process an image and display a 2x2 grid of visualizations."""
        image = Image.open(image_path).convert("RGB")

        # Perform segmentation.
        semantic_map_np = self.segment_image(image)
        processed_map_np = (semantic_map_np == 0).astype(int)

        # Compute ray lengths using raycasting.
        ray_cast = self.get_raycast_from_semantic(processed_map_np)

        return ray_cast


def main():
    cam_to_ray = CamToRay()

    current_dir = os.getcwd()
    frame_dir = os.path.join(current_dir, "frames")
    save_dir = os.path.join(current_dir, "raycast")

    os.makedirs(save_dir, exist_ok=True)

    for filename in os.listdir(frame_dir):
        image_path = os.path.join(frame_dir, filename)
        base_name = os.path.splitext(filename)[0]
        json_name = f"{base_name}.json"
        save_path = os.path.join(save_dir, json_name)

        if os.path.exists(save_path):
            print(f"skipped {save_path} (already exists)")
            continue

        ray_cast = cam_to_ray.get_raycast(image_path)

        with open(save_path, "w") as f:
            json.dump(ray_cast, f, indent=4)
            print(f"processed {save_path}")


if __name__ == "__main__":
    main()