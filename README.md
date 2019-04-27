# gym-catcher

An environment for OpenAi gym.

Objective is to catch as many falling balls as possible, having only discrete (True/False) sensors in limited view angle.

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

## Demo
### Random agent action
Agents samples random action from action space and apply it.
<img src="https://github.com/piaxar/gym-catcher/blob/master/demo/random_action.gif" alt="Random action" width="300" height="350">

### Trained agent action
Agent trained, using [Sallimans et al.][1] algorithm.
<img src="https://github.com/piaxar/gym-catcher/blob/master/demo/trained.gif" alt="Trained agent" width="300" height="350">

### Trained agent action in hard setting
The same trained agent but with changes in balls generation. Now balls are generated in a way, that agent should select one of the falling objects in order to gain score.
<img src="https://github.com/piaxar/gym-catcher/blob/master/demo/trained_hard.gif" alt="Trained agent in hard environment" width="300" height="350">

<!--References-->
[1]: https://arxiv.org/abs/1703.03864

