from typing import List

import numpy as np
from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.agents.agent import Agent


class Water(Agent):
    env: AssistiveEnv

    @property
    def is_inside_bedpan(self) -> bool:
        return np.sum(self.env.bedpan.get_contact_points(self)[-1]) > 0

    @property
    def is_on_bed(self) -> bool:
        return np.sum(self.env.furniture.get_contact_points(self)[-1]) > 0

    @property
    def is_on_human(self) -> bool:
        return np.sum(self.env.human.get_contact_points(self)[-1]) > 0

    @property
    def is_inside_disposal_bowl(self) -> bool:
        return np.sum(self.env.disposal_bowl.get_contact_points(self)[-1]) > 0

    @classmethod
    def set_env(cls, env: AssistiveEnv):
        cls.env = env

    @classmethod
    def from_sphere(cls, sphere: Agent) -> "Water":
        water = cls()
        water.init(sphere.body, sphere.id, sphere.np_random, indices=-1)
        return water

    @classmethod
    def from_spheres(cls, spheres: List[Agent]) -> List["Water"]:
        return [cls.from_sphere(sphere) for sphere in spheres]
