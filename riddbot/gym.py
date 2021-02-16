from typing import Tuple

import assistive_gym.envs
import gym
from assistive_gym.envs.agents import jaco, human
from assistive_gym.envs.agents.human import Human
from assistive_gym.envs.agents.jaco import Jaco
from gym.envs.registration import register as register_gym
from ray.tune.registry import register_env as register_ray

from riddbot.env.bedpan_env import BedPanEnv

ROBOT_ARM = "right"
HUMAN_CONTROLLABLE_JOINT_INDICES = human.right_arm_joints


class BedPanJacoEnv(BedPanEnv):
    def __init__(self):
        super(BedPanJacoEnv, self).__init__(
            robot=Jaco(ROBOT_ARM),
            human=Human(HUMAN_CONTROLLABLE_JOINT_INDICES, controllable=False),
        )


def setup_camera(env: gym.Env):
    env.setup_camera(
        fov=60,
        camera_eye=[0.5, -0.75, 1.5],
        camera_target=[-0.2, 0, 0.75],
        camera_width=1920 // 4,
        camera_height=1080 // 4,
    )


def make_env() -> Tuple[str, gym.Env]:
    env_name = "BedPanJaco-v1"

    assistive_gym.envs.BedPanJacoEnv = BedPanJacoEnv
    register_ray("assistive_gym:" + env_name, lambda config: BedPanJacoEnv())

    register_gym(
        id=env_name,
        entry_point="assistive_gym.envs:" + BedPanJacoEnv.__name__,
        max_episode_steps=200,
    )

    env = gym.make("assistive_gym:" + env_name)
    setup_camera(env)

    return env_name, env
