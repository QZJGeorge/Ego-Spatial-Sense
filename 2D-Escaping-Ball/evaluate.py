import sys
import pygame
import random
import math
import colorsys
import torch
import argparse

from model.es2 import Es2Model
from model.mlp import MLPModel
from model.transformer import TransformerModel


# ---------------- Utility: Draw an Arrow ----------------
def draw_arrow(
    surface,
    color,
    start,
    end,
    arrow_width=3,
    arrow_head_length=10,
    arrow_head_angle=math.pi / 6,
):
    pygame.draw.line(surface, color, start, end, arrow_width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    left_angle = angle + math.pi - arrow_head_angle
    right_angle = angle + math.pi + arrow_head_angle
    left_point = (
        end[0] + arrow_head_length * math.cos(left_angle),
        end[1] + arrow_head_length * math.sin(left_angle),
    )
    right_point = (
        end[0] + arrow_head_length * math.cos(right_angle),
        end[1] + arrow_head_length * math.sin(right_angle),
    )
    pygame.draw.polygon(surface, color, [end, left_point, right_point])


# ---------------- Ball Class ----------------
class Ball:
    def __init__(self, x, y, radius, color, speed):
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y
        self.radius = radius
        self.color = color
        self.speed = speed
        # For moving balls, choose a random initial direction.
        self.dx = random.choice([-1, 1]) * self.speed
        self.dy = random.choice([-1, 1]) * self.speed

    def move(self):
        self.prev_x, self.prev_y = self.x, self.y
        self.x += self.dx
        self.y += self.dy

        # Bounce off the walls.
        if self.x - self.radius < 0 or self.x + self.radius > args.width:
            self.dx *= -1
        if self.y - self.radius < 0 or self.y + self.radius > args.height:
            self.dy *= -1

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)

    def collides_with(self, other):
        distance = math.hypot(self.x - other.x, self.y - other.y)
        return distance < self.radius + other.radius


# ---------------- Evaluate Function ----------------
def evaluate(model, args):
    """
    Run the simulation. The model now receives two LiDAR scans concatenated:
      - previous scan (length 360)
      - current scan (length 360)
    For the very first iteration, the previous scan is set equal to the current scan.
    The resulting input has 720 features.
    """

    random.seed(args.random_seed)

    pygame.init()
    pygame.font.init()

    if args.render:
        font = pygame.font.SysFont("Arial", 24)
        screen = pygame.display.set_mode((args.width, args.height))
        pygame.display.set_caption("Avoid the Balls with Model Movement")
    else:
        screen = None

    # Colors
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)

    clock = pygame.time.Clock()

    if args.render:
        screen = pygame.display.set_mode((args.width, args.height))
        pygame.display.set_caption("Avoid the Balls with Model Movement")
    else:
        screen = None

    clock = pygame.time.Clock()

    # Create the player (red ball) and blue balls.
    player = Ball(
        args.width // 2, args.height // 2, 20, RED, 0
    )  # Player does not move on its own.
    balls = []
    margin = 10
    min_distance = (
        player.radius + 15 + margin
    )  # Ensure blue balls don't start too near the player.
    for _ in range(args.num_balls):
        while True:
            x = random.randint(30, args.width - 30)
            y = random.randint(30, args.height - 30)
            if math.hypot(x - player.x, y - player.y) >= min_distance:
                break
        radius = 20
        speed = random.randint(1, 3)
        balls.append(Ball(x, y, radius, BLUE, speed))
    balls.append(player)

    # ---------------- LiDAR Scan Function ----------------
    def lidar_scan():
        distances = []
        intersections = []
        for i in range(args.num_features):
            rad = math.radians(i)
            dx = math.cos(rad)
            dy = math.sin(rad)

            min_distance_val = float("inf")
            intersection_point = None

            # Check intersections with walls.
            if dx != 0:
                t = (0 - player.x) / dx
                if t > 0:
                    y_int = player.y + t * dy
                    if 0 <= y_int <= args.height and t < min_distance_val:
                        min_distance_val = t
                        intersection_point = (0, y_int)
            if dx != 0:
                t = (args.width - player.x) / dx
                if t > 0:
                    y_int = player.y + t * dy
                    if 0 <= y_int <= args.height and t < min_distance_val:
                        min_distance_val = t
                        intersection_point = (args.width, y_int)
            if dy != 0:
                t = (0 - player.y) / dy
                if t > 0:
                    x_int = player.x + t * dx
                    if 0 <= x_int <= args.width and t < min_distance_val:
                        min_distance_val = t
                        intersection_point = (x_int, 0)
            if dy != 0:
                t = (args.height - player.y) / dy
                if t > 0:
                    x_int = player.x + t * dx
                    if 0 <= x_int <= args.width and t < min_distance_val:
                        min_distance_val = t
                        intersection_point = (x_int, args.height)

            # Check intersections with blue balls.
            for ball in balls:
                if ball == player:
                    continue
                bx = ball.x - player.x
                by = ball.y - player.y
                A = dx**2 + dy**2
                B = -2 * (bx * dx + by * dy)
                C = bx**2 + by**2 - ball.radius**2
                discriminant = B**2 - 4 * A * C
                if discriminant >= 0:
                    sqrt_disc = math.sqrt(discriminant)
                    t1 = (-B - sqrt_disc) / (2 * A)
                    t2 = (-B + sqrt_disc) / (2 * A)
                    if t1 > 0 and t1 < min_distance_val:
                        min_distance_val = t1
                        intersection_point = (player.x + t1 * dx, player.y + t1 * dy)
                    if t2 > 0 and t2 < min_distance_val:
                        min_distance_val = t2
                        intersection_point = (player.x + t2 * dx, player.y + t2 * dy)
            distances.append(min_distance_val)
            intersections.append(intersection_point)

        return distances, intersections

    # Variable to hold the previous scan.
    # For the first iteration, previous_scan will be set equal to the current scan.
    previous_scan = None

    # ---------------- Main Simulation Loop ----------------
    steps = 0

    while True:
        steps += 1

        if steps >= args.max_steps:
            print(f"Maximum steps reached: {args.max_steps}")

            return steps

        if args.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return steps

        # Update blue balls (the player ball does not move on its own)
        for ball in balls:
            if ball != player:
                ball.move()

        # Perform LiDAR scan.
        distances, intersections = lidar_scan()

        # Initialize previous_scan on the very first iteration.
        if previous_scan is None:
            previous_scan = distances.copy()

        # Create model input by concatenating the previous scan and current scan.
        # The resulting tensor has shape (1, 720): the first 360 values are from the previous scan,
        # and the next 360 are from the current scan.
        scan_input = (
            torch.tensor(previous_scan + distances, dtype=torch.float32)
            .unsqueeze(0)
            .to(args.device)
        )

        with torch.no_grad():
            action = model(scan_input)

        net_fx, net_fy = action[0][0].item(), action[0][1].item()

        # bound fx and fy to max speed
        fx = min(max(net_fx, -args.max_speed), args.max_speed)
        fy = min(max(net_fy, -args.max_speed), args.max_speed)

        net_force_magnitude = math.hypot(net_fx, net_fy)

        player.x += int(fx)
        player.y += int(fy)

        # Keep the player within bounds.
        player.x = max(player.radius, min(args.width - player.radius, player.x))
        player.y = max(player.radius, min(args.height - player.radius, player.y))

        # Update previous_scan for the next iteration.
        previous_scan = distances.copy()

        # Check for collisions between the player and blue balls.
        for ball in balls:
            if ball != player and player.collides_with(ball):
                print("Collision! Total steps", steps)

                return steps

        if args.render:
            screen.fill(WHITE)

            # Draw all balls.
            for ball in balls:
                ball.draw(screen)

            max_distance_threshold = math.hypot(args.width, args.height)
            for i, (d, intersection) in enumerate(zip(distances, intersections)):
                if intersection:
                    rad_angle_line = math.radians(i)
                    start_x = player.x + player.radius * math.cos(rad_angle_line)
                    start_y = player.y + player.radius * math.sin(rad_angle_line)
                    start_point = (start_x, start_y)
                    norm = min(d / max_distance_threshold, 1)
                    hue = math.sqrt(norm) * 0.33  # Adjust for color variation.
                    r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
                    color = (int(r * 255), int(g * 255), int(b * 255))
                    pygame.draw.line(screen, color, start_point, intersection, 1)

            if net_force_magnitude > 0:
                arrow_angle = math.degrees(math.atan2(net_fy, net_fx))
            else:
                arrow_angle = 0
            arrow_length = 40
            start_pos = (player.x, player.y)
            end_pos = (
                player.x + arrow_length * math.cos(math.radians(arrow_angle)),
                player.y + arrow_length * math.sin(math.radians(arrow_angle)),
            )
            draw_arrow(screen, RED, start_pos, end_pos)

            # Display net force information.
            if net_force_magnitude > 0:
                arrow_angle = math.degrees(math.atan2(net_fy, net_fx))
            else:
                arrow_angle = 0
            text_surface = font.render(
                f"Force Angle: {arrow_angle:.2f}°, Magnitude: {net_force_magnitude:.2f}",
                True,
                (0, 0, 0),
            )
            screen.blit(text_surface, (10, 10))
            pygame.display.flip()
            clock.tick(args.frequency)
        else:
            # In headless mode, just control simulation speed.
            clock.tick(args.frequency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Environment and simulation settings
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--frequency", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_balls", type=int, default=10)
    parser.add_argument("--max_speed", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=15000)

    # Visualization settings
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sensing_range", type=float, default=800.0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)

    # Model settings
    parser.add_argument("--model_path", type=str, default="pretrained/es2.pth")
    parser.add_argument("--device", type=str, default="cuda")

    # Transformer-specific parameters
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
        # Initialize Transformer model
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

    steps = evaluate(model, args)

    print(steps)
