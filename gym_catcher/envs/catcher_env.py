import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering
import time
import math

# screen parameters
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 700

# cart parameters
CART_WIDTH = 50.0
CART_HEIGHT = 30.0
CART_MOVING_SPEED = 50

N_SENSORS = 7
VISION_ANGLE = 45.

# falling balls parameters
BALL_RADIUS = 25.0
FREQUENCY = 5  # new ball creates every FREQUENCY steps
MAX_BALLS = 10  # maximum number of ball on screen in parallel

BALL_MAX_HORIZONTAL_SPEED = 5
BALL_MAX_VERTICAL_SPEED = 49
BALL_MIN_VERTICAL_SPEED = 30

MAX_STEPS = 100  # number of steps, after which environment returns  done=True


class CatcherEnv(gym.Env):
    """

    Actions: Discrete(2)
    Num      Action
    0        move left
    1        move right

    Observations: Box(1 + N_SENSORS)
    Num             Meaning
    0               position of cart on the screen in range [-2.4; 2.4]
    1-N_SENSORS     discrete value for each sensor. 1 if sensor sees an object, 0 otherwise

    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # vertical threshold
        self.x_threshold = 2.4
        world_width = self.x_threshold * 2
        self.scale = SCREEN_WIDTH / world_width
        self.step_size = CART_MOVING_SPEED / (self.scale * 2)  # moves half of it's width per action

        self.action_space = spaces.Discrete(2)

        # 0 elem: position
        # 1 -> N_SENSORS: sensors
        min_arr = np.append([-self.x_threshold], np.zeros(N_SENSORS))
        max_arr = np.append([self.x_threshold], np.ones(N_SENSORS))
        self.observation_space = spaces.Box(low=min_arr, high=max_arr)

        self.viewer = None
        self.position = None
        self.clock = None
        self.balls = None
        self.sensors = None
        self.not_rendered_balls = None  # balls that not yet added to rendering view
        self.balls_to_remove = None  # balls that fell or being cached, so need to be deleted

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.clock += 1

        # create a ball every n iteration:
        if self.clock % FREQUENCY == 1 and len(self.balls) < MAX_BALLS:
            # create a ball
            vertical_speed = np.random.uniform(BALL_MIN_VERTICAL_SPEED, BALL_MAX_VERTICAL_SPEED)
            horizontal_speed = np.random.uniform(-BALL_MAX_HORIZONTAL_SPEED, BALL_MAX_HORIZONTAL_SPEED)

            speed_vector = (horizontal_speed, vertical_speed)
            init_x = np.random.uniform(-(SCREEN_WIDTH / 3), (SCREEN_WIDTH / 3)) + SCREEN_WIDTH / 2
            new_ball = FallingBall(BALL_RADIUS, speed_vector, SCREEN_HEIGHT, SCREEN_WIDTH, init_x=init_x)
            self.balls.append(new_ball)
            self.not_rendered_balls.append(new_ball)

        # fall for all ball
        for ball in self.balls:
            ball.fall()

        direction = -1 if action == 0 else 1

        self.position += direction * self.step_size
        # don't allow moving outside the space
        if self.position > self.x_threshold: self.position = self.x_threshold
        if self.position < -self.x_threshold: self.position = -self.x_threshold

        # delete ball if collision with ground or collision with cart
        reward = 0
        new_balls = []
        cart_x = self.position * self.scale + SCREEN_WIDTH / 2.0
        for ball in self.balls:
            if ball.collides_cart(cart_x):
                reward += 1
                self.balls_to_remove.append(ball)
            elif ball.ground_touched():
                self.balls_to_remove.append(ball)
            else:
                new_balls.append(ball)
        self.balls = new_balls

        sensor_observations = self.get_sensors_observations()
        return np.append(self.position, sensor_observations), reward, self.clock > MAX_STEPS, None

    def get_sensors_observations(self):
        activations = []
        start_x = self.position * self.scale + SCREEN_WIDTH / 2.0
        start_y = CART_HEIGHT / 2
        for sensor in self.sensors:
            activations.append(sensor.is_activated((start_x, start_y), self.balls))
        return np.array(list(reversed(activations)))

    def reset(self):
        self.position = 0
        self.clock = 0
        self.balls = []
        self.not_rendered_balls = []
        self.balls_to_remove = []
        self.sensors = []
        self.viewer = None

        # init sensors:
        angle_step = float(VISION_ANGLE) / (N_SENSORS - 1)
        for i in range(0, N_SENSORS):
            relative_angle = 90 - (VISION_ANGLE / 2) + angle_step * i
            self.sensors.append(Sensor(relative_angle, SCREEN_HEIGHT))

        sensor_observations = self.get_sensors_observations()
        return np.append(self.position, sensor_observations)

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(SCREEN_WIDTH, SCREEN_HEIGHT)
            l, r, t, b = -CART_WIDTH / 2, CART_WIDTH / 2, CART_HEIGHT / 2, -CART_HEIGHT / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            cart.set_color(.8, .6, .4)
            self.viewer.add_geom(cart)

            for sensor in self.sensors:
                points = sensor.get_points()
                line = rendering.Line(*points)
                line.add_attr(self.carttrans)
                self.viewer.add_geom(line)
                sensor.set_geom(line)

        # render not rendered balls
        for not_rendered_ball in self.not_rendered_balls:
            ball = rendering.make_circle(radius=not_rendered_ball.radius)
            ball.set_color(.7, .3, np.random.random())
            ball_trans = rendering.Transform()
            ball.add_attr(ball_trans)
            self.viewer.add_geom(ball)

            not_rendered_ball.set_geom_obj(ball)
            not_rendered_ball.set_transformer(ball_trans)

        # render balls
        for ball in self.balls:
            ball.draw()

        # remove fallen balls
        for fallen_ball in self.balls_to_remove:
            geom = fallen_ball.get_geom_obj()
            self.viewer.geoms.remove(geom)

        # highlight sensors if activated:
        for sensor in self.sensors:
            sensor.highlight_if_needed()

        cart_x = self.position * self.scale + SCREEN_WIDTH / 2.0
        cart_y = 0 + CART_HEIGHT / 2  # TOP OF CART
        self.carttrans.set_translation(cart_x, cart_y)

        self.not_rendered_balls = []
        self.balls_to_remove = []

        time.sleep(0.09)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class Sensor:
    def __init__(self, relative_angle, height):
        self.angle = relative_angle
        self.height = height
        self.delta_x = self.height / np.tan(np.deg2rad(relative_angle))
        self.geom = None
        self.geom_attrs = None
        self.highlight_needed = False

    def get_points(self):
        return (0, 0), (self.delta_x, self.height)

    def is_activated(self, anchor_position, balls):
        p_x = anchor_position[0]
        p_y = anchor_position[1]

        _, (q_x, q_y) = self.get_points()
        q_x += p_x
        q_y += p_y

        a = p_y - q_y
        b = q_x - p_x
        c = -a * p_x - b * p_y

        for ball in balls:
            x = ball.x
            y = ball.y
            r = ball.radius
            if Sensor._check_collision(a, b, c, x, y, r):
                self.highlight_needed = True
                return True
        return False

    @staticmethod
    def _check_collision(a, b, c, x, y, radius):
        # Finding the distance of line
        # from center.
        dist = ((abs(a * x + b * y + c)) /
                math.sqrt(a * a + b * b))

        if radius >= dist:
            return True  # collision
        else:
            return False  # no collision

    def set_geom(self, geom):
        self.geom = geom
        self.geom_attrs = self.geom.attrs

    def highlight_if_needed(self):
        if self.highlight_needed:
            self.geom.attrs = [rendering.LineWidth(5)] + self.geom_attrs
            self.highlight_needed = False
        else:
            self.geom.attrs = [rendering.LineWidth(1)] + self.geom_attrs
            self.highlight_needed = False


class FallingBall:
    def __init__(self, radius, speed_vector, height, width, init_x=0):
        self.radius = radius

        self.horizontal_speed = speed_vector[0]
        self.vertical_speed = speed_vector[1]

        self.h = height
        self.y = height
        self.width = width

        self.x = init_x
        self.transformer = None
        self.geom = None

    def fall(self):
        self.y -= self.vertical_speed
        self.x += self.horizontal_speed

        # walls collisions
        if self.x <= self.radius or self.x >= self.width - self.radius:
            self.horizontal_speed *= -1

    def set_transformer(self, transformer):
        self.transformer = transformer

    def set_geom_obj(self, geom):
        self.geom = geom

    def get_geom_obj(self):
        return self.geom

    def draw(self):
        if self.transformer is None:
            return
        self.transformer.set_translation(self.x, self.y)

    def ground_touched(self):
        return self.y < 0

    def collides_cart(self, cart_position):
        if self.y - self.radius > CART_HEIGHT:
            return False
        elif np.abs(cart_position - self.x) <= CART_WIDTH / 2 + self.radius:
            return True
        return False
