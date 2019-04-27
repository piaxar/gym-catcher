# gym-catcher

An environment for OpenAi gym.

Objective is to catch as many falling balls as possible, having only discrete (True/False) sensors in limited view angle.

This description also contains discussion about trained agent. For more, see Demo and Discussion chapters.

## Properties

### Action space:
There are two available actions: move left (0) and move right (1)
```
Actions: Discrete(2)
Num      Action
0        move left
1        move right
```

### Observation space:
Observation space is an array, where first value is cart position and
remaining elements are binary values, representing actuation of the
corresponding sensor

```
Observations: Box(1 + N_SENSORS)
Num             Meaning
0               position of cart on the screen in range [-2.4; 2.4]
1-N_SENSORS     discrete value for each sensor. 1 if sensor sees an object, 0 otherwise
```

### Reward:
For each caught ball gain 1 reward point.

### Max steps:
For training purposes, number of steps is limited to 100. After this value, environment returns True in 'done'.

### Environment settings
Environment is highly customizable. Following parameters are defined as global constants in ./gym_catcher/envs/catcher_env.py and could be changed:
```python
# screen parameters
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 700

# cart parameters
CART_WIDTH = 50.0
CART_HEIGHT = 30.0
CART_MOVING_SPEED = 50 # pixels per 1 move

N_SENSORS = 7 # sensors will be distributed uniformly and symmetrically
VISION_ANGLE = 45.

# falling balls parameters
BALL_RADIUS = 25.0
FREQUENCY = 5  # new ball creates every FREQUENCY steps
MAX_BALLS = 2  # maximum number of ball on screen in parallel

BALL_MAX_HORIZONTAL_SPEED = 5
BALL_MAX_VERTICAL_SPEED = 49
BALL_MIN_VERTICAL_SPEED = 30

MAX_STEPS = 100  # number of steps, after which environment returns  done=True
```

## Demo
### Random agent action
Agents samples random action from action space and apply it.

<img src="https://github.com/piaxar/gym-catcher/blob/master/demo/random_action.gif" alt="Random action" width="300" height="350">

### Trained agent action
Agent was trained, using [Sallimans et al.][1] algorithm.

<img src="https://github.com/piaxar/gym-catcher/blob/master/demo/trained.gif" alt="Trained agent" width="300" height="350">

### Trained agent action in hard setting
The same trained agent but with changes in balls generation. Now balls are generated in a way, that agent should select one of the falling objects in order to gain score.

<img src="https://github.com/piaxar/gym-catcher/blob/master/demo/trained_hard.gif" alt="Trained agent in hard environment" width="300" height="350">

## Discussion
An agent was trained in a given environment in order to assure that one can
come up with selective selection strategy. The agent was trained, using [Sallimans et al.][1] algorithm.
As the third demo shows, the trained bot can select one of
the falling objects. Probably, such a strategy is powered by choosing
the ball that activates more sensors together, when approaching. Therefore
an attempt to train selective attention strategy can be considered successful.

<!--References-->
[1]: https://arxiv.org/abs/1703.03864

