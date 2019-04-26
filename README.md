# gym-catcher

An environment for OpenAi gym.

Objective is to catch as many falling balls as possible, having only discrete (True/False) sensors in limited view angle.

## Properties

### Action space:
```python
action_space = gym.spaces.Discrete(3)
```

### Observation space:
```python
observation_space = gym.spaces.Dict({'sensors': spaces.MultiBinary(N_SENSORS),
                                     'position': spaces.Box(low=-np.array([min_x]), high=np.array([max_x]))})

```

### Reward:
For each caught ball gain 1 reward point


## Demo

![Demo](https://github.com/piaxar/gym-catcher/demo/demo.gif)
