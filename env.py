# All libraries used for the environment
# Imports
import os
import random
import time
from collections import deque

# FLAPPY_HEADLESS=1 runs the game without a window and without sound, for
# training and evaluating on a machine with no screen. It has to be set
# before pygame starts, so it is read here and not in the class.
HEADLESS = os.environ.get('FLAPPY_HEADLESS') == '1'
if HEADLESS:
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

####################
# Global Variables
####################

# The assets sit next to this file, so the scripts run from any folder
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')


def sprite(name):
    return os.path.join(ASSETS, 'sprites', name)


# Initialize Pygame
pygame.init()

# Frame name
pygame.display.set_caption('Flappy Bird')

# Screen width and height
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Velocity and physics
# SPEED is the upward speed a flap gives. The models in models/ were trained
# with 20; it was set to 18 afterwards. FlappyBird(flap_speed=...) overrides it.
SPEED = 18
GRAVITY = 2.5
GAME_SPEED = 20

# Ground
GROUND_WIDTH = SCREEN_WIDTH
GROUND_HEIGHT = 100

# Pipes
PIPE_WIDTH = 80
PIPE_HEIGHT = 500
PIPE_GAP = 150

# Observation Shape
N_CHANNELS = 3
HEIGHT = SCREEN_HEIGHT
WIDTH = SCREEN_WIDTH

# Background image
BACKGROUND = pygame.image.load(sprite('background-day.png'))
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDTH, SCREEN_HEIGHT))
BEGIN_IMAGE = pygame.image.load(sprite('message.png')).convert_alpha()

# Audio for wing, dead, and point
wing = os.path.join(ASSETS, 'audio', 'wing.wav')
hit = os.path.join(ASSETS, 'audio', 'hit.wav')
point = os.path.join(ASSETS, 'audio', 'point.wav')


class _Silent:
    """Stands in for a sound when sound is off or there is no audio device."""

    def play(self):
        pass

    def set_volume(self, volume):
        pass

##########################
# Flappy Bird Environment
# Includes:
# The flappy bird game,
# Step,
# Observations,
# Rewards,
# And reset,
##########################

class FlappyBird(gym.Env):
    metadata = {'render_modes': ['human']}

    # Initialize flappy bird game, make sure all self values exist
    # Here we also define our observation space and action space
    #
    # render:     draw every step in the window (True, as before) or only when
    #             render() is called (False: much faster, for training)
    # sound:      play the wing, point and hit sounds
    # fps:        the most steps a second when drawing
    # flap_speed: the upward speed a flap gives (see SPEED above)

    def __init__(self, render=True, sound=True, fps=100, flap_speed=SPEED):
        super(FlappyBird, self).__init__()

        self.render_every_step = render and not HEADLESS
        self.fps = fps
        self.flap_speed = flap_speed

        # Define action space
        self.action_space = spaces.Discrete(2)  # Flap [1], Not flap [0]

        # Previous actions
        self.prev_actions = deque(maxlen=10)

        # Initialize time
        self.start_time = time.time()

        # Initialize mixer and load sounds
        # Without sound, or on a machine with no audio device, the game runs silent
        self.wing_sound = self.hit_sound = self.point_sound = _Silent()
        if sound and not HEADLESS:
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                pygame.mixer.set_num_channels(8)  # Adjust based on your needs
                self.wing_sound = pygame.mixer.Sound(wing)
                self.hit_sound = pygame.mixer.Sound(hit)
                self.point_sound = pygame.mixer.Sound(point)
            except pygame.error as error:
                print(f"No sound: {error}")

        # Optional: Set volume for each sound
        self.wing_sound.set_volume(0.5)  # Range: 0.0 to 1.0
        self.hit_sound.set_volume(0.5)
        self.point_sound.set_volume(0.5)

        # Set screen, clock, bird, and pipe with configurable FPS
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.bird = Bird(self.wing_sound, self.flap_speed)  # Pass wing_sound to Bird
        self.pipe_group = pygame.sprite.Group()

        # Pipes positions
        self.pipe_lower_y = None
        self.pipe_top_y = None

        # Distances
        self.v_distance_lower_y = None
        self.x_distance_to_gap = None

        # Pipes passed
        self.pipes_passed = 0

        # Ground group
        self.ground_group = pygame.sprite.Group()

        # Observation space
        # The five numbers are not really whole numbers between 0 and 255 (a
        # distance can be negative or larger). It stays declared like this
        # because the saved models were trained against exactly this space and
        # stable-baselines3 refuses to load them into any other.
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(5,), dtype=np.int32)

        # Load number images 0.png to 9.png
        self.number_images = [pygame.image.load(sprite(f'{i}.png')).convert_alpha() for i in range(10)]

        # Load background and begin images
        screen.blit(BACKGROUND, (0, 0))
        screen.blit(BEGIN_IMAGE, (120, 150))

    ##################################################
    # Get observation
    # Add more observations!!
    # Add bird velocity, distances to pipes,
    # Gap between pipes, position of the bird,
    # Previous actions
    ##################################################
    def _get_observation(self):

        # Current bird position
        bird_x = self.bird.rect[0]

        # Current distance to pipe
        # Using the first pipe of the group
        # Calculating with distance to pipe - bird_x
        min_distance_to_pipe = float("inf")
        for pipe in self.pipe_group:
            distance_to_pipe = pipe.rect[0] - bird_x
            if 0 < distance_to_pipe < min_distance_to_pipe:
                min_distance_to_pipe = distance_to_pipe

        # Calculate top of bottom pipe
        # Calculate bottom of top pipe
        if self.pipe_group:
            upper_pipe = self.pipe_group.sprites()[1]
            lower_pipe = self.pipe_group.sprites()[0]
            self.pipe_top_y = upper_pipe.rect[1] + upper_pipe.rect[3]
            self.pipe_lower_y = lower_pipe.rect[1]

        # Observations
        # Velocity of the bird
        velocity = self.bird.speed
        distance_to_ceiling = self.bird.rect.top
        distance_to_floor = SCREEN_HEIGHT - GROUND_HEIGHT - self.bird.rect.bottom
        # Vertical distance to top of lower pipe
        self.v_distance_lower_y = self.pipe_lower_y - self.bird.rect.bottom

        # Calculate y distance to center gap
        # First calculate center y of 2 pipes
        # Then pipe y - bird_y
        lower_pipe_bottom_y = self.pipe_group.sprites()[0].rect.bottom
        upper_pipe_top_y = self.pipe_group.sprites()[1].rect.top
        pipe_gap_middle_y = (lower_pipe_bottom_y + upper_pipe_top_y) / 2

        # Calculate x distance to center gap
        # First calculate center x of pipe
        # Then pipe x - bird_x
        left_pipe_right_x = self.pipe_group.sprites()[0].rect.right
        right_pipe_left_x = self.pipe_group.sprites()[1].rect.left
        pipe_gap_middle_x = (left_pipe_right_x + right_pipe_left_x) / 2

        self.x_distance_to_gap = pipe_gap_middle_x - bird_x

        # Combine all observations in one array
        observation = np.array([self.v_distance_lower_y, self.x_distance_to_gap,
                                velocity, distance_to_floor, distance_to_ceiling])

        return observation

    ############################################
    # Step function
    # Return observation, reward, done, info
    # Step equals every action happened in game
    # Frame by frame
    ############################################
    def step(self, action):
        if self.render_every_step:
            self.render()
        obs = self._get_observation()

        # Get everything from the reward value
        reward, terminated, truncated = self.reward_value()
        info = {'score': self.pipes_passed}

        # Store every action
        self.prev_actions.append(action)

        # Load in ground, pipes, and bird
        self.bird.begin()
        self.bird.update()
        self.ground_group.update()
        self.pipe_group.update()

        # If agent chooses 1 instead of 0
        if action == 1:
            self.bird.bump()  # This plays the wing sound

        # Keep adding pipes on screen
        if self.pipe_group and self._is_off_screen(self.pipe_group.sprites()[0]):
            self.pipe_group.remove(self.pipe_group.sprites()[0])
            self.pipe_group.remove(self.pipe_group.sprites()[0])
            pipes = get_random_pipes(SCREEN_WIDTH * 2)
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])

        if self.ground_group and self._is_off_screen(self.ground_group.sprites()[0]):
            self.ground_group.remove(self.ground_group.sprites()[0])
            new_ground = Ground(self.ground_group.sprites()[-1].rect.right)
            self.ground_group.add(new_ground)

        return obs, reward, terminated, truncated, info

    ####################################################
    # Reset function
    # Makes sure the game starts with all observations
    # Resets everything
    ####################################################

    def reset(self, seed=None, options=None):
        # Set every variable to its default value
        # When reset is called, everything will be reset
        # Reset is called every truncated or terminated
        # Or at start
        super().reset(seed=seed)

        # With a seed, the pipes (and the bird's colour) of this run and the
        # runs after it are the same every time. Without one, nothing changes.
        if seed is not None:
            random.seed(seed)

        self.info = {'score': 0}
        self.prev_actions.clear()
        self.bird = Bird(self.wing_sound, self.flap_speed)  # Pass wing_sound to Bird
        self.pipe_group = pygame.sprite.Group()
        self.ground_group = pygame.sprite.Group()
        self.pipes_passed = 0  # Reset the score

        for i in range(2):
            ground = Ground(i * GROUND_WIDTH)
            self.ground_group.add(ground)

        for i in range(2):
            pipes = get_random_pipes(SCREEN_WIDTH * i + 800)
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])

        return self._get_observation(), self.info

    ##################################
    # Render in game objects
    # User interface User experience
    ##################################

    def render(self, mode='human'):
        # Keep the window alive: without this the window manager marks it
        # "not responding" and it cannot be closed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.draw()

        # Set game ticks per second
        pygame.display.update()
        if not HEADLESS:
            self.clock.tick(self.fps)

    # Draw one frame on self.screen without showing it
    # (render() shows it; evaluate.py saves it as a picture)
    def draw(self):
        # Render background
        self.screen.fill((255, 255, 255))
        self.screen.blit(BACKGROUND, (0, 0))

        # Load in begin screen
        # Begin screen only when 0 actions have been taken yet
        if len(self.prev_actions) == 0:
            self.screen.blit(BEGIN_IMAGE, (120, 150))

        # Load in bird, pipes, and ground
        self.screen.blit(self.bird.image, self.bird.rect.topleft)  # Blit bird image directly onto the screen
        self.pipe_group.draw(self.screen)
        self.ground_group.draw(self.screen)

        # Draw the score
        self.draw_score()

    def draw_score(self):
        # Convert score to string
        score_str = str(self.pipes_passed)
        total_width = 0
        digit_images = []

        # Load digit images and calculate total width
        for digit_char in score_str:
            digit = int(digit_char)
            digit_image = self.number_images[digit]
            digit_images.append(digit_image)
            total_width += digit_image.get_width()

        # Calculate starting position for centering
        x_offset = (SCREEN_WIDTH - total_width) // 2
        y_offset = 50  # Adjust as needed for vertical position

        # Blit each digit image onto the screen
        for digit_image in digit_images:
            self.screen.blit(digit_image, (x_offset, y_offset))
            x_offset += digit_image.get_width()

    #########################################
    # Defining rewards
    # Using function inside the step function
    # Keeps checking for rewards
    #########################################

    def reward_value(self):
        reward = 0
        terminated = False
        truncated = False

        # Check for collision (bird dies)
        if pygame.sprite.spritecollideany(self.bird, self.pipe_group, pygame.sprite.collide_mask) \
                or self.bird.rect.top <= 0 \
                or self.bird.rect.bottom >= SCREEN_HEIGHT - GROUND_HEIGHT:
            reward -= 100  # Large negative reward upon death
            # The bird died: that is "terminated". "truncated" is for a run cut
            # off from outside (a time limit), which this game does not have.
            terminated = True
            self.hit_sound.play()  # Play hit sound
            return reward, terminated, truncated

        # Small positive reward for each timestep the bird is alive
        reward += 1

        # Reward for passing a pipe pair
        for pipe in self.pipe_group:
            if not pipe.inverted and not hasattr(pipe, 'passed') and pipe.rect.right < self.bird.rect.left:
                setattr(pipe, 'passed', True)
                reward += 10  # Reward for passing a pipe pair
                self.point_sound.play()  # Play point sound
                self.pipes_passed += 1  # Increment the score

                # Increase rewards for passing pipes after a certain number
                if self.pipes_passed > 100:
                    reward += 20  # Higher reward after 100 pipes passed
                elif self.pipes_passed > 50:
                    reward += 15  # Increased reward after 50 pipes passed
                else:
                    reward += 10  # Normal reward for passing pipes early on

        # Penalize unnecessary flapping
        if len(self.prev_actions) > 0 and self.prev_actions[-1] == 1:
            reward -= 0.05  # Small penalty for flapping

        return reward, terminated, truncated

    def _is_off_screen(self, sprite):
        return sprite.rect[0] < -(sprite.rect[2])

##############################
# In-game classes
# Bird(), Ground(), Pipes()
##############################

class Bird(pygame.sprite.Sprite):

    def __init__(self, wing_sound, flap_speed=SPEED):
        pygame.sprite.Sprite.__init__(self)

        # Store the wing sound
        self.wing_sound = wing_sound

        # Load images for each color
        self.blue_images = [
            pygame.image.load(sprite('bluebird-upflap.png')).convert_alpha(),
            pygame.image.load(sprite('bluebird-midflap.png')).convert_alpha(),
            pygame.image.load(sprite('bluebird-downflap.png')).convert_alpha()
        ]

        self.yellow_images = [
            pygame.image.load(sprite('yellowbird-upflap.png')).convert_alpha(),
            pygame.image.load(sprite('yellowbird-midflap.png')).convert_alpha(),
            pygame.image.load(sprite('yellowbird-downflap.png')).convert_alpha()
        ]

        self.red_images = [
            pygame.image.load(sprite('redbird-upflap.png')).convert_alpha(),
            pygame.image.load(sprite('redbird-midflap.png')).convert_alpha(),
            pygame.image.load(sprite('redbird-downflap.png')).convert_alpha()
        ]

        # Randomly choose a color
        self.color = random.choice(['blue', 'yellow', 'red'])

        if self.color == 'blue':
            self.original_images = self.blue_images
        elif self.color == 'yellow':
            self.original_images = self.yellow_images
        elif self.color == 'red':
            self.original_images = self.red_images

        self.flap_speed = flap_speed
        self.speed = flap_speed

        self.current_image = 0
        self.image = self.original_images[self.current_image]
        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDTH / 6
        self.rect[1] = SCREEN_HEIGHT / 2

        # Rotation angle
        self.angle = 0
        self.max_up_angle = -25  # Max tilt up
        self.max_down_angle = 25  # Max tilt down

    def update(self):
        # Update animation
        self.current_image = (self.current_image + 1) % len(self.original_images)
        self.image = self.original_images[self.current_image]

        # Update speed with gravity
        self.speed += GRAVITY

        # Update bird's vertical position
        self.rect[1] += self.speed

        # Calculate tilt angle
        self.calculate_tilt()

    def calculate_tilt(self):
        """
        Calculate the tilt angle based on the bird's vertical speed.
        """
        if self.speed < 0:
            # Flapping upwards
            self.angle = self.max_down_angle
        elif self.speed > 0:
            # Falling downwards
            # The faster the bird is falling, the more it tilts down
            self.angle = min(self.max_up_angle, self.speed * 2.5)  # Adjust the multiplier as needed
        else:
            self.angle = 0

        # Rotate the image
        rotated_image = pygame.transform.rotate(self.original_images[self.current_image], self.angle)
        old_center = self.rect.center
        self.image = rotated_image
        self.rect = self.image.get_rect()
        self.rect.center = old_center

        # Update the mask after rotation
        self.mask = pygame.mask.from_surface(self.image)

    def bump(self):
        self.speed = -self.flap_speed
        # Play wing sound
        self.wing_sound.play()
        # Optionally, set a specific tilt angle when flapping
        self.angle = self.max_up_angle
        rotated_image = pygame.transform.rotate(self.original_images[self.current_image], self.angle)
        old_center = self.rect.center
        self.image = rotated_image
        self.rect = self.image.get_rect()
        self.rect.center = old_center
        self.mask = pygame.mask.from_surface(self.image)

    def begin(self):
        self.current_image = (self.current_image + 1) % len(self.original_images)
        self.image = self.original_images[self.current_image]
        self.mask = pygame.mask.from_surface(self.image)


class Pipe(pygame.sprite.Sprite):

    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)

        self.inverted = inverted
        self.image = pygame.image.load(sprite('pipe-green.png')).convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDTH, PIPE_HEIGHT))

        self.rect = self.image.get_rect()
        self.rect[0] = xpos

        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = - (self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize

        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect[0] -= GAME_SPEED


class Ground(pygame.sprite.Sprite):

    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(sprite('base.png')).convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDTH, GROUND_HEIGHT))

        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED

# Pipe randomization
def get_random_pipes(xpos):
    size = random.randint(100, 300)
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted
