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
**Note:** agent in demo acts randomly

<p align="center">
  <img src="https://github.com/piaxar/gym-catcher/blob/master/demo/demo.gif" width="300" height="350">
</p>




