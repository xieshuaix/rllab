from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.base import EnvSpec
from rllab.spaces.box import Box as TheanoBox
from rllab.spaces.discrete import Discrete as TheanoDiscrete
from rllab.spaces.product import Product as TheanoProduct
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.spaces.product import Product
from cached_property import cached_property
from sandbox.rocky.tf.envs import tf_env_native


def to_tf_space(space):
    if isinstance(space, TheanoBox):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, TheanoDiscrete):
        return Discrete(space.n)
    elif isinstance(space, TheanoProduct):
        return Product(list(map(to_tf_space, space.components)))
    else:
        raise NotImplementedError


class TfEnv(ProxyEnv):
    @cached_property
    def observation_space(self):
        return to_tf_space(self.wrapped_env.observation_space)

    @cached_property
    def action_space(self):
        return to_tf_space(self.wrapped_env.action_space)

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
        return tf_env_native.VecTfEnv(self.wrapped_env.vec_env_executor(n_envs=n_envs, max_path_length=max_path_length))

    @classmethod
    def wrap(cls, env_cls, **extra_kwargs):
        # Use a class wrapper rather than a lambda method for smoother serialization
        return tf_env_native.WrappedCls(cls, env_cls, extra_kwargs)
