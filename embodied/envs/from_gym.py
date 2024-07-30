import functools
import embodied
import gymnasium as gym
import numpy as np

highway_config = {
    "observation": {
        "type": "RGBImageObservation",
        "observation_shape": (64, 64),
        "stack_size": 4,
        "scaling": 1.75
    },
    "policy_frequency": 2,
    "action": {
        "type": "ContinuousAction"
    }
}


class FromGym(embodied.Env):
# ENC
    def __init__(self, env, obs_key='image', act_key='action', **kwargs):
        if isinstance(env, str):
            self._env = gym.make(env, **kwargs)
            if env == 'highway-v0':
                self._env.configure(highway_config)
                self._env.reset()
        else:
            assert not kwargs, kwargs
            self._env = env
        self._obs_dict = hasattr(self._env.observation_space, 'spaces')
        self._act_dict = hasattr(self._env.action_space, 'spaces')
        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True
        self._info = None

        print(f"Initial observation space: {self._env.observation_space}")
        print(f"Initial action space: {self._env.action_space}")

    @property
    def env(self):
        return self._env

    @property
    def info(self):
        return self._info

    @functools.cached_property
    def obs_space(self):
        if self._obs_dict:
            spaces = self._flatten(self._env.observation_space.spaces)
        else:
            spaces = {self._obs_key: self._env.observation_space}

        print(f"Original observation spaces: {spaces}")
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        print(f"Converted observation spaces: {spaces}")
        return {
            **spaces,
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        }

    @functools.cached_property
    def act_space(self):
        if self._act_dict:
            spaces = self._flatten(self._env.action_space.spaces)
        else:
            spaces = {self._act_key: self._env.action_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        spaces['reset'] = embodied.Space(bool)
        print(f"Action spaces: {spaces}")
        return spaces

    def step(self, action):
        if action['reset'] or self._done:
            self._done = False
            obs, info = self._env.reset()
            return self._obs(obs, 0.0, is_first=True)
        if self._act_dict:
            action = self._unflatten(action)
        else:
            action = action[self._act_key]
        obs, reward, self._done, trunc, self._info = self._env.step(action)
        return self._obs(
            obs, reward,
            is_last=bool(self._done),
            is_terminal=bool(self._info.get('is_terminal', self._done)))

    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        if not self._obs_dict:
            obs = {self._obs_key: obs}
        obs = self._flatten(obs)

        # Convert observations to match the expected data type and shape
        for k, v in obs.items():
            if k == self._obs_key:
                obs[k] = v.astype(np.float32) / 255.0  # Normalize if it's image data
            else:
                obs[k] = np.asarray(v, dtype=np.float32)

        obs.update(
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal)
        return obs

    def render(self):
        image = self._env.render('rgb_array')
        assert image is not None
        return image

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def _flatten(self, nest, prefix=None):
        result = {}
        for key, value in nest.items():
            key = prefix + '/' + key if prefix else key
            if isinstance(value, gym.spaces.Dict) or isinstance(value, dict):
                value = value.spaces if isinstance(value, gym.spaces.Dict) else value
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result

    def _unflatten(self, flat):
        result = {}
        for key, value in flat.items():
            parts = key.split('/')
            node = result
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return result

    def _convert(self, space):
        if hasattr(space, 'n'):
            return embodied.Space(np.int32, (), 0, space.n)
        return embodied.Space(space.dtype, space.shape, space.low, space.high)
