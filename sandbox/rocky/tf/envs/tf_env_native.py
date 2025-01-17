from rllab.envs.proxy_env import ProxyEnv
from cached_property import cached_property
from rllab.envs.base import EnvSpec


class TfEnvNative(ProxyEnv):
    @cached_property
    def observation_space(self):
        return self.wrapped_env.observation_space

    @cached_property
    def action_space(self):
        return self.wrapped_env.action_space

    @cached_property
    def spec(self):
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

    @property
    def vectorized(self):
        return getattr(self.wrapped_env, "vectorized", False)

    def vec_env_executor(self, n_envs, max_path_length):
        return VecTfEnv(self.wrapped_env.vec_env_executor(n_envs=n_envs, max_path_length=max_path_length))

    @classmethod
    def wrap(cls, env_cls, **extra_kwargs):
        # Use a class wrapper rather than a lambda method for smoother serialization
        return WrappedCls(cls, env_cls, extra_kwargs)


class VecTfEnv(object):

    def __init__(self, vec_env):
        self.vec_env = vec_env

    def reset(self):
        return self.vec_env.reset()

    @property
    def num_envs(self):
        return self.vec_env.num_envs

    def step(self, action_n):
        return self.vec_env.step(action_n)

    def terminate(self):
        self.vec_env.terminate()


class WrappedCls(object):
    def __init__(self, cls, env_cls, extra_kwargs):
        self.cls = cls
        self.env_cls = env_cls
        self.extra_kwargs = extra_kwargs

    def __call__(self, *args, **kwargs):
        return self.cls(self.env_cls(*args, **dict(self.extra_kwargs, **kwargs)))
