import numpy as np
import pybullet as p
from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.agents.human import Human
from assistive_gym.envs.agents.robot import Robot

from riddbot.env.observe import *
from riddbot.env.rewards import *
from riddbot.env.setup import *


class BedPanEnv(AssistiveEnv):
    def __init__(self, robot: Robot, human: Human, reward_weights: dict):

        if human.controllable:
            raise TypeError(human)

        obs_robot_len = len(robot.controllable_joint_indices) - (
            len(robot.wheel_joint_indices) if robot.mobile else 0
        )  # robot_joint_angles
        obs_robot_len += 3  # bedpan_pos
        obs_robot_len += 4  # bedpan_orient
        obs_robot_len += 3  # disposal_bowl_pos
        obs_robot_len += 1  # robot_force_on_human
        obs_robot_len += 1  # bedpan_force_on_human

        obs_human_len = len(human.controllable_joint_indices)

        super().__init__(
            robot=robot,
            human=human,
            task="bed_bathing",
            obs_robot_len=obs_robot_len,
            obs_human_len=obs_human_len,
        )

        self.reward_weights = reward_weights

    def _get_obs(self, agent=None):
        if agent != "robot" and agent is not None:
            raise ValueError(agent)

        robot_joint_angles = get_robot_observations(self)
        bedpan_pos, bedpan_orient = get_bedpan_observations(self)
        disposal_bowl_pos = get_disposal_bowl_observations(self)
        robot_force_on_human, bedpan_force_on_human = get_force_observations(self)

        self.total_force_on_human = robot_force_on_human + bedpan_force_on_human

        return np.concatenate(
            [
                robot_joint_angles,
                bedpan_pos,
                bedpan_orient,
                disposal_bowl_pos,
                [robot_force_on_human, bedpan_force_on_human],
            ]
        ).ravel()

    def reset(self):
        super().reset()

        # order of the setup calls DOES MATTER
        setup_bed(self)
        setup_patient(self)
        setup_bedpan(self)
        setup_robot(self)
        setup_sanitation_stand(self)
        setup_waters(self)

        setup_gravity(self)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()

        return self._get_obs()

    def step(self, action):
        self.take_step(action)

        obs = self._get_obs()

        rewards_dict = dict()
        rewards_dict.update(get_bed_rewards(self))
        rewards_dict.update(get_human_rewards(self))
        rewards_dict.update(get_robot_rewards(self, action))
        rewards_dict.update(get_sanitation_rewards(self))

        total_reward = sum(
            self.reward_weights[k] * rewards_dict[k] for k in rewards_dict.keys()
        )

        info = {
            "total_force_on_human": self.total_force_on_human,
            "action_robot_len": self.action_robot_len,
            "action_human_len": self.action_human_len,
            "obs_robot_len": self.obs_robot_len,
            "obs_human_len": self.obs_human_len,
        }
        info.update(rewards_dict)

        done = self.iteration >= 300

        return obs, total_reward, done, info
