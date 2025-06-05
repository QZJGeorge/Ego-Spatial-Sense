import os
import cv2
import json
import torch

from utils import *
from model import SpatialSenseModel


current_dir = os.getcwd()

frame_dir = os.path.join(current_dir, "frames")
raycast_dir = os.path.join(current_dir, "raycast")
save_dir = os.path.join(current_dir, "sense")

os.makedirs(save_dir, exist_ok=True)

model = SpatialSenseModel(sensing_range=900.0)
model.load_state_dict(torch.load("model.pth", map_location='cuda'))

for filename in os.listdir(frame_dir):
    # Extract scene ID and frame index
    scene_id, index = filename.split("_")

    index = int(index.replace(".png", ""))

    # Build the filename for the previous frame
    prev_path = os.path.join(raycast_dir, f"{scene_id}_{index-1}.json")
    curr_path = os.path.join(raycast_dir, f"{scene_id}_{index}.json")

    # Skip this iteration if either file is missing
    if not os.path.exists(prev_path) or not os.path.exists(curr_path):
        continue

    # Load JSON data (lists of 180 numbers)
    with open(prev_path, 'r') as f:
        ray_cast_pre = json.load(f)
    with open(curr_path, 'r') as f:
        ray_cast = json.load(f)

    spatial_sense = model(torch.tensor(ray_cast_pre + ray_cast, dtype=torch.float32))

    spatial_sense = torch.clamp(
        -spatial_sense.squeeze(0) * 5.0, min=0.0, max=0.8
    ).cpu().tolist()

    spatial_sense = smooth(spatial_sense, width=9, iter=1)

    ray_cast = [max(300, x) for x in ray_cast]
    ray_cast = smooth(ray_cast, width=15, iter=20)

    img = overlay_sense(f"{frame_dir}/{filename}", ray_cast, spatial_sense)

    save_path = f'{save_dir}/{filename}'

    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"Processed {save_path}")




