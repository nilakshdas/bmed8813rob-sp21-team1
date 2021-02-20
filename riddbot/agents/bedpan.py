import os

import numpy as np
import pybullet as p
from numpy.random import RandomState

from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.agents.agent import Agent


class Bedpan(Agent):
    def init(self, env: AssistiveEnv, base_pos: np.ndarray):
        bedpan = p.loadURDF(
            os.path.join(env.directory, "dinnerware", "bowl.urdf"),
            basePosition=base_pos,
            baseOrientation=[0, 0, 0, 1],
            globalScaling=2,
            physicsClientId=env.id,
        )

        super().init(bedpan, env.id, env.np_random, indices=-1)
