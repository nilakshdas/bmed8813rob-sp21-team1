from typing import Tuple

import assistive_gym.envs
import gym
from assistive_gym.envs.agents import jaco, human
from assistive_gym.envs.agents.human import Human
from assistive_gym.envs.agents.jaco import Jaco
from gym.envs.registration import register as register_gym
from ray.tune.registry import register_env as register_ray

from riddbot.env.bedpan_env import BedPanEnv
from riddbot.env.setup import setup_camera

ROBOT_ARM = "right"
HUMAN_CONTROLLABLE_JOINT_INDICES = []


class BedPanJacoEnv(BedPanEnv):
    def __init__(self, reward_weights: dict):
        super(BedPanJacoEnv, self).__init__(
            robot=Jaco(ROBOT_ARM),
            human=Human(HUMAN_CONTROLLABLE_JOINT_INDICES, controllable=False),
            reward_weights=reward_weights,
        )


def make_env(reward_weights: dict) -> Tuple[str, gym.Env]:
    env_name = "BedPanJaco-v1"
    ray_name = "assistive_gym:" + env_name

    register_gym(
        id=env_name,
        entry_point="riddbot.gym:BedPanJacoEnv",
        kwargs={"reward_weights": reward_weights},
        max_episode_steps=200,
    )

    register_ray(ray_name, lambda config: BedPanJacoEnv(**config))

    env = gym.make(ray_name)
    setup_camera(env)

    return env_name, env
