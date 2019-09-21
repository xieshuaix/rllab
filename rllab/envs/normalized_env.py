import numpy as np
from rllab.spaces import box
from rllab.misc.overrides import overrides
from rllab.envs import normalized_env_native


class NormalizedEnv(normalized_env_native.NormalizedEnvNative):
    def __init__(
            self,
            env,
            scale_reward=1.,
            normalize_obs=False,
            normalize_reward=False,
            obs_alpha=0.001,
            reward_alpha=0.001,
    ):
        super(NormalizedEnv, self).__init__(env=env, scale_reward=scale_reward, normalize_obs=normalize_obs,
                                            normalize_reward=normalize_reward, obs_alpha=obs_alpha,
                                            reward_alpha=reward_alpha)

    @property
    @overrides
    def action_space(self):
        if isinstance(self._wrapped_env.action_space, box.Box):
            ub = np.ones(self._wrapped_env.action_space.shape)
            return box.Box(-1 * ub, ub)
        return self._wrapped_env.action_space

normalize = NormalizedEnv
