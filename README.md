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
                                     'position': spaces.Box(low=-np.array([MIN_X_COORD]), high=np.array([MAX_X_COORD]))})

```

### Reward:
For each caught ball gain 1 reward point.


## Demo
**Note:** agent in demo acts randomly


<img src="ttps://github.com/piaxar/gym-catcher/blob/master/demo/demo.gif" width="100">
