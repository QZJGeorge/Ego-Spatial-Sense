import csv
import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from pyglet.gl import GL_POINTS
from f110_gym.envs.base_classes import Integrator


MAX_STEER = 0.52  # rad
MAX_SPEED = 8.0  # m/s
MIN_SPEED = 1.0  # m/s


def find_largest_gap(ranges, min_dist=2.0, target_gap_size=180):
    """
    Find the largest contiguous set of indices in 'ranges'
    whose values are >= min_dist. Then, prune the gap from both ends
    based on the distance values until the gap size shrinks to 'target_gap_size'.
    Returns a tuple:
        (start_index, end_index)
    of the pruned gap.
    """
    # Step 1: Find the largest gap
    above_threshold = ranges >= min_dist
    max_len = 0
    max_start = 0
    max_end = 0
    curr_start = None

    for i, val in enumerate(above_threshold):
        if val:
            if curr_start is None:
                curr_start = i
        else:
            if curr_start is not None:
                length = i - curr_start
                if length > max_len:
                    max_len = length
                    max_start = curr_start
                    max_end = i - 1
                curr_start = None

    if curr_start is not None:
        length = len(above_threshold) - curr_start
        if length > max_len:
            max_len = length
            max_start = curr_start
            max_end = len(above_threshold) - 1

    # Step 2: Prune the gap from both ends until the gap size is reduced to target_gap_size
    while (max_end - max_start + 1) > target_gap_size:
        if ranges[max_start] < ranges[max_end]:
            max_start += 1
        else:
            max_end -= 1

    return max_start, max_end


def compute_steer_angle_for_gap(start_idx, end_idx):
    """
    Compute steering angle toward the midpoint of the largest gap.
    """
    mid_idx = (start_idx + end_idx) // 2
    resolution_deg = 0.25
    angle_deg = resolution_deg * mid_idx - 90.0
    angle_rad = np.deg2rad(angle_deg)
    STEERING_GAIN = 1.0
    desired_steer = STEERING_GAIN * angle_rad
    # Clamp the steering angle
    return max(min(desired_steer, MAX_STEER), -MAX_STEER)


def main():
    # Load config and create environment
    with open("config.yaml", "r") as file:
        conf_dict = yaml.safe_load(file)
    conf = Namespace(**conf_dict)

    env = gym.make(
        "f110-v0",
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
    )

    # This dictionary will store the current speed & steering values
    render_info = {"speed": 0.0, "steer": 0.0}

    visited_points = []
    drawn_points = []

    def render_callback(env_renderer):
        e = env_renderer
        # Adjust camera based on car's position
        x_coords = e.cars[0].vertices[::2]
        y_coords = e.cars[0].vertices[1::2]
        top, bottom = max(y_coords), min(y_coords)
        left, right = min(x_coords), max(x_coords)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        # Overwrite the default score_label to show speed & steering
        current_speed = render_info["speed"]
        current_steer = render_info["steer"]
        steer_degs = np.rad2deg(current_steer)
        e.score_label.text = (
            f"Speed: {current_speed:.2f} m/s  |  "
            f"Steer: {current_steer:.2f} rad ({steer_degs:.1f} deg)"
        )

        # Plot the path of the car (visited points)
        for i, pt in enumerate(visited_points):
            scaled_x = 50.0 * pt[0]
            scaled_y = 50.0 * pt[1]
            if i < len(drawn_points):
                drawn_points[i].vertices = [scaled_x, scaled_y, 0.0]
            else:
                b = e.batch.add(
                    1,
                    GL_POINTS,
                    None,
                    ("v3f/stream", [scaled_x, scaled_y, 0.0]),
                    ("c3B/stream", [255, 0, 0]),
                )
                drawn_points.append(b)

    env.add_render_callback(render_callback)
    obs, step_reward, done, info = env.reset(
        poses=np.array([[conf.sx, conf.sy, conf.stheta]])
    )
    env.render()

    print("Gap-Follow Enabled with Constant Speed.")
    print("Press CTRL+C in the console to terminate early.")

    collected_data = []
    experiment_start_time = time.time()
    last_save_time = experiment_start_time

    try:
        while not done:
            lidar_ranges_ori = np.array(obs["scans"]).flatten()
            current_speed = obs["linear_vels_x"][0]
            # Take only the front 180 degrees or any region you want
            lidar_ranges = lidar_ranges_ori[360:1081]

            MIN_RANGE_THRESHOLD = 2.0

            MIN_RANGE_THRESHOLD = (
                2.0 + (current_speed - MIN_SPEED) * 0.1
            )  # Example scaling

            start_idx, end_idx = find_largest_gap(lidar_ranges, MIN_RANGE_THRESHOLD)
            steer = compute_steer_angle_for_gap(start_idx, end_idx)

            # Update speed based on steering angle
            de_accel_ratio = max(0.0, (1.0 - abs(steer) / MAX_STEER * 2))
            desired_speed = de_accel_ratio * (MAX_SPEED - MIN_SPEED) + MIN_SPEED

            # Store them in render_info so that render_callback can see them
            render_info["speed"] = desired_speed
            render_info["steer"] = steer

            # For logging
            current_time = time.time()
            if current_time - last_save_time >= 0.1:
                last_save_time = current_time
                elapsed_time = current_time - experiment_start_time
                # Append time, steer, desired_speed, and all lidar ranges
                collected_data.append(
                    [elapsed_time, steer, desired_speed] + lidar_ranges_ori.tolist()
                )

            # Save position for red dot drawing
            current_point = [obs["poses_x"][0], obs["poses_y"][0]]
            visited_points.append(current_point)

            # Take an action in the environment
            action = np.array([[steer, desired_speed]])
            obs, step_reward, done, info = env.step(action)

            # Render the environment with updated callback
            env.render(mode="human")

            # time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")
    finally:
        # Save CSV data
        with open("test_data.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["time", "steer", "desired_speed"]
                + [f"lidar_{i}" for i in range(len(lidar_ranges_ori))]
            )
            writer.writerows(collected_data)

        env.close()
        print("Data saved to 'test_data.csv'.")


if __name__ == "__main__":
    main()
