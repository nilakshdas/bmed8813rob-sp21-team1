import os
from typing import Tuple

import numpy as np
import pybullet as p
from numpy.random import RandomState

from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.agents.agent import Agent


class DisposalBowl(Agent):
    def init(self, env: AssistiveEnv, base_pos: np.ndarray):
        disposal_bowl = p.loadURDF(
            os.path.join(env.directory, "dinnerware", "bowl.urdf"),
            basePosition=base_pos,
            baseOrientation=[0, 0, 0, 1],
            globalScaling=2.5,
            physicsClientId=env.id,
        )

        super().init(disposal_bowl, env.id, env.np_random, indices=-1)

    def set_original_pos_orient(self):
        self.original_pos, self.original_orient = self.get_pos_orient(self.base)

    @property
    def pos_orient_perturbation(self) -> Tuple[float, float]:
        curr_pos, curr_orient = self.get_pos_orient(self.base)
        return (
            np.linalg.norm(curr_pos - self.original_pos),
            np.linalg.norm(curr_orient - self.original_orient),
        )
