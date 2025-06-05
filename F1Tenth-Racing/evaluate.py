import argparse
import sys
import warnings

import gym
import numpy as np
import torch
import yaml
from pyglet.gl import GL_POINTS

from argparse import Namespace
from f110_gym.envs.base_classes import Integrator

from model.es2 import Es2Model
from model.mlp import MLPModel
from model.transformer import TransformerModel

# Configuration
np.random.seed(42)
warnings.filterwarnings("ignore")


TIME_RES = 0.01  # seconds
MAX_STEER = 0.52  # rad
MIN_SPEED = 1.0  # m/s
MAX_SPEED = 8.0  # m/s


def evaluate_racing(
    model=None,
    render=True,
    noise=0.0,
):
    # Load config and create environment
    with open("config.yaml", "r") as file:
        conf_dict = yaml.safe_load(file)
    conf = Namespace(**conf_dict)

    env = gym.make(
        "f110-v0",
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=TIME_RES,
        integrator=Integrator.RK4,
    )

    render_info = {"speed": 0.0, "steer": 0.0}
    visited_points, drawn_points, speed_list = [], [], []
    dist_travelled = 0.0

    def render_callback(env_renderer):
        e = env_renderer
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

        current_speed = render_info["speed"]
        current_steer = render_info["steer"]
        steer_degs = np.rad2deg(current_steer)
        e.score_label.text = (
            f"Speed: {current_speed:.2f} m/s  |  "
            f"Steer: {current_steer:.2f} rad ({steer_degs:.1f} deg)"
        )

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
    obs, _, done, _ = env.reset(poses=np.array([[conf.sx, conf.sy, conf.stheta]]))

    lidar_prev = np.array(obs["scans"]).flatten()[::4]
    current_speed = 0.0

    while not done:
        lidar_curr = np.array(obs["scans"]).flatten()[::4]

        # number of scans to zero out based on noise level (occulsion)
        # if noise > 0.0:
        #     indices_to_zero = np.random.choice(
        #         len(lidar_curr), int(len(lidar_curr) * noise), replace=False
        #     )
        #     lidar_curr[indices_to_zero] = 0.0

        # Apply noise to the lidar data (fluctuation)
        # if noise > 0.0:
        #     # Generate random multiplicative factors for each scan point
        #     factors = np.random.uniform(1.0 - noise, 1.0 + noise, size=len(lidar_curr))
        #     # Apply multiplicative noise
        #     lidar_curr *= factors

        current_speed = obs["linear_vels_x"][0]
        speed_list.append(current_speed)

        dist_travelled += current_speed * TIME_RES

        state = torch.cat([torch.tensor(lidar_prev), torch.tensor(lidar_curr)])
        state = state.float().unsqueeze(0).to(args.device)

        action = model(state).detach().cpu().numpy()
        steer = np.clip(action[0][0].item(), -MAX_STEER, MAX_STEER)
        speed = action[0][1].item()

        render_info["speed"] = speed
        render_info["steer"] = steer
        lidar_prev = lidar_curr

        action = np.array([[steer, speed]])
        obs, _, done, _ = env.step(action)

        current_point = [obs["poses_x"][0], obs["poses_y"][0]]
        visited_points.append(current_point)

        if render:
            env.render(mode="human")

    env.close()

    # Save to CSV
    avg_speed = sum(speed_list) / len(speed_list)

    print(
        f"Noise level: {noise:.2f}, Distance travelled: {dist_travelled:.2f} m, Avg speed: {avg_speed:.2f} m/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=0.0)
    # Model selection
    parser.add_argument("--model_path", type=str, default="pretrained/es2_0.pth")
    parser.add_argument("--device", type=str, default="cuda")
    # Evaluation parameters
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--sensing_range", type=float, default=10.0)
    # Transformer parameters
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=64)

    args = parser.parse_args()

    if "es2" in args.model_path:
        model = Es2Model(
            num_features=args.num_features,
            num_actions=args.num_actions,
            sensing_range=args.sensing_range,
        ).to(args.device)

        # Load the model
        print("Loading ES2 model...")
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

    elif "mlp" in args.model_path:
        model = MLPModel(
            num_features=args.num_features,
            num_actions=args.num_actions,
            sensing_range=args.sensing_range,
        ).to(args.device)

        # Load the model
        print("Loading MLP model...")
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

    elif "transformer" in args.model_path:
        model = TransformerModel(
            num_features=args.num_features,
            num_actions=args.num_actions,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            sensing_range=args.sensing_range,
        ).to(args.device)

        # Load the model
        print("Loading Transformer model...")
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

    else:
        print("Error: undefined model type.")
        sys.exit()

    # Evaluate with parsed arguments
    evaluate_racing(
        model=model,
        render=args.render,
        noise=args.noise,
    )
