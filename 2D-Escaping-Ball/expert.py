import pygame
import random
import math
import colorsys
import csv

random.seed(42)  # For reproducibility

# ---------------- Parameters (adjust these as needed) ----------------
NUM_RAYS = 360  # Total number of LiDAR rays (angular resolution = 360/NUM_RAYS degrees)
ANGLE_STEP = 360 / NUM_RAYS  # Angular separation between rays

FREQUENCY = 50  # Frequency of the simulation loop in Hz

# Potential field parameters:
POTENTIAL_FIELD_CONSTANT = 10000  # Scales the repulsive force magnitude
MIN_DISTANCE_FOR_FORCE = (
    10  # Minimum distance used in force calculation (to avoid singularities)
)
MAX_SPEED = 10  # Maximum net force (i.e. maximum movement per frame)

NUM_BACKGROUND_BALLS = 10  # Number of blue balls in the environment

MAX_POINTS = 15000

# ---------------- Pygame and Display Setup ----------------
pygame.init()
pygame.font.init()  # Ensure fonts are initialized


font = pygame.font.SysFont("Arial", 24)
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Avoid the Balls with Potential Fields")


# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)  # (For potential future use.)

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# ---------------- CSV Recording Setup ----------------
csv_file = open("simulation_data.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
header = ["fx", "fy"] + [f"scan_{i}" for i in range(NUM_RAYS)]
csv_writer.writerow(header)

# ---------------- Flags for Optional Drawings ----------------
show_lidar_rays = True  # Set to False to hide LiDAR rays.
show_direction_arrow = True  # Set to False to hide the net force arrow.


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
        if self.x - self.radius < 0 or self.x + self.radius > WIDTH:
            self.dx *= -1
        if self.y - self.radius < 0 or self.y + self.radius > HEIGHT:
            self.dy *= -1

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)

    def collides_with(self, other):
        distance = math.hypot(self.x - other.x, self.y - other.y)
        return distance < self.radius + other.radius


# ---------------- Create the Player and Blue Balls ----------------
player = Ball(WIDTH // 2, HEIGHT // 2, 20, RED, 0)  # Player-controlled (speed = 0)
balls = []
margin = 10
min_distance = player.radius + 15 + margin  # (15 is the blue ball's radius)
for _ in range(NUM_BACKGROUND_BALLS):
    while True:
        x = random.randint(30, WIDTH - 30)
        y = random.randint(30, HEIGHT - 30)
        if math.hypot(x - player.x, y - player.y) >= min_distance:
            break
    radius = 20
    speed = random.randint(1, 3)
    balls.append(Ball(x, y, radius, BLUE, speed))

# Append the player ball to the list
balls.append(player)


# ---------------- LiDAR Scan Function ----------------
def lidar_scan():
    """
    Simulate a 360-degree LiDAR scan.
    Returns a list of distances and a list of intersection points.
    """
    distances = []
    intersections = []
    for i in range(NUM_RAYS):
        angle = i * ANGLE_STEP
        rad = math.radians(angle)
        dx = math.cos(rad)
        dy = math.sin(rad)

        # Initialize with a large distance.
        min_distance_val = float("inf")
        intersection_point = None

        # Check intersections with walls.
        if dx != 0:
            t = (0 - player.x) / dx
            if t > 0:
                y_int = player.y + t * dy
                if 0 <= y_int <= HEIGHT and t < min_distance_val:
                    min_distance_val = t
                    intersection_point = (0, y_int)
        if dx != 0:
            t = (WIDTH - player.x) / dx
            if t > 0:
                y_int = player.y + t * dy
                if 0 <= y_int <= HEIGHT and t < min_distance_val:
                    min_distance_val = t
                    intersection_point = (WIDTH, y_int)
        if dy != 0:
            t = (0 - player.y) / dy
            if t > 0:
                x_int = player.x + t * dx
                if 0 <= x_int <= WIDTH and t < min_distance_val:
                    min_distance_val = t
                    intersection_point = (x_int, 0)
        if dy != 0:
            t = (HEIGHT - player.y) / dy
            if t > 0:
                x_int = player.x + t * dx
                if 0 <= x_int <= WIDTH and t < min_distance_val:
                    min_distance_val = t
                    intersection_point = (x_int, HEIGHT)

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
                sqrt_discriminant = math.sqrt(discriminant)
                t1 = (-B - sqrt_discriminant) / (2 * A)
                t2 = (-B + sqrt_discriminant) / (2 * A)
                if t1 > 0 and t1 < min_distance_val:
                    min_distance_val = t1
                    intersection_point = (player.x + t1 * dx, player.y + t1 * dy)
                if t2 > 0 and t2 < min_distance_val:
                    min_distance_val = t2
                    intersection_point = (player.x + t2 * dx, player.y + t2 * dy)

        distances.append(min_distance_val)
        intersections.append(intersection_point)
    return distances, intersections


# ---------------- Main Game Loop with Exception Handling ----------------
try:
    running = True
    sample_count = 0

    while running:
        if sample_count >= MAX_POINTS:
            print("Maximum number of points reached. Ending simulation.")
            running = False

     
        # Process Pygame events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update blue balls (the player ball does not move automatically).
        for ball in balls:
            if ball != player:
                ball.move()

        # Perform LiDAR scan.
        distances, intersections = lidar_scan()

        # ---------------- Potential Field Based Movement ----------------
        net_fx = 0
        net_fy = 0
        for i, d in enumerate(distances):
            effective_d = max(d, MIN_DISTANCE_FOR_FORCE)
            force_magnitude = POTENTIAL_FIELD_CONSTANT / (effective_d**2)
            ray_angle = math.radians(i * ANGLE_STEP)
            net_fx += -math.cos(ray_angle) * force_magnitude
            net_fy += -math.sin(ray_angle) * force_magnitude

        # bound fx and fy to max speed
        fx = min(max(net_fx, -MAX_SPEED), MAX_SPEED)
        fy = min(max(net_fy, -MAX_SPEED), MAX_SPEED)

        net_force_magnitude = math.hypot(fx, fy)

        # Update player's position.
        player.x += int(fx)
        player.y += int(fy)
        player.x = max(player.radius, min(WIDTH - player.radius, player.x))
        player.y = max(player.radius, min(HEIGHT - player.radius, player.y))

        # ---------------- Record CSV Data ----------------
        sample_count += 1

        if sample_count % 5 == 0:
            row = [fx, fy] + distances
            csv_writer.writerow(row)

        if sample_count % 1000 == 0:
            print(f"Sample count: {sample_count}")

        # ---------------- Visualization ----------------
        screen.fill(WHITE)

        # Draw all balls.
        for ball in balls:
            ball.draw(screen)

        # Check for collisions between the player and any blue ball.
        for ball in balls:
            if ball != player and player.collides_with(ball):
                print("Game Over!")
                running = False

        # Optionally, draw LiDAR rays.
        if show_lidar_rays:
            max_distance_threshold = math.hypot(WIDTH, HEIGHT)
            for i, (distance, intersection) in enumerate(
                zip(distances, intersections)
            ):
                if intersection:
                    actual_angle = i * ANGLE_STEP
                    rad_angle_line = math.radians(actual_angle)
                    start_x = player.x + player.radius * math.cos(rad_angle_line)
                    start_y = player.y + player.radius * math.sin(rad_angle_line)
                    start_point = (start_x, start_y)
                    norm = min(distance / max_distance_threshold, 1)
                    hue = math.sqrt(norm) * 0.33  # Adjust for color variation.
                    r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
                    color = (int(r * 255), int(g * 255), int(b * 255))
                    pygame.draw.line(screen, color, start_point, intersection, 1)

        # Optionally, draw an arrow indicating the net force.
        if show_direction_arrow:
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
        clock.tick(FREQUENCY)

except KeyboardInterrupt:
    print("KeyboardInterrupt received. Exiting and saving CSV data.")

finally:
    csv_file.close()
    pygame.quit()
