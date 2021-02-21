import numpy as np
import pybullet as p

from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.agents.human import Human
from assistive_gym.envs.agents.robot import Robot

from riddbot.env.observe import *
from riddbot.env.setup import *


class BedPanEnv(AssistiveEnv):
    def __init__(self, robot, human: Human):

        if human.controllable:
            raise TypeError(human)

        obs_robot_len = (
            17
            + len(robot.controllable_joint_indices)
            - (len(robot.wheel_joint_indices) if robot.mobile else 0)
        )
        obs_human_len = 18 + len(human.controllable_joint_indices)

        super().__init__(
            robot=robot,
            human=human,
            task="bed_bathing",
            obs_robot_len=obs_robot_len,
            obs_human_len=obs_human_len,
        )

    def _get_obs(self, agent=None):
        if agent != "robot" and agent is not None:
            raise ValueError(agent)

        tool_pos_real, tool_orient_real = get_tool_observations(self)
        robot_joint_angles = get_robot_observations(self)
        shoulder_pos_real, elbow_pos_real, wrist_pos_real = get_human_observations(self)

        (
            self.tool_force,
            self.tool_force_on_human,
            self.total_force_on_human,
            self.new_contact_points,
        ) = get_force_observations(self)

        return np.concatenate(
            [
                tool_pos_real,
                tool_orient_real,
                robot_joint_angles,
                shoulder_pos_real,
                elbow_pos_real,
                wrist_pos_real,
                [self.tool_force],
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

        setup_gravity(self)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()

        return self._get_obs()

    def step(self, action):
        self.take_step(action)

        obs = self._get_obs()

        # Get human preferences
        end_effector_velocity = np.linalg.norm(
            self.robot.get_velocity(self.robot.left_end_effector)
        )
        preferences_score = self.human_preferences(
            end_effector_velocity=end_effector_velocity,
            total_force_on_human=self.total_force_on_human,
            tool_force_at_target=self.tool_force_on_human,
        )

        reward_distance = -min(
            self.tool.get_closest_points(self.human, distance=5.0)[-1]
        )
        reward_action = -np.linalg.norm(action)  # Penalize actions
        reward_new_contact_points = (
            self.new_contact_points
        )  # Reward new contact points on a person

        reward = (
            self.config("distance_weight") * reward_distance
            + self.config("action_weight") * reward_action
            + self.config("wiping_reward_weight") * reward_new_contact_points
            + preferences_score
        )

        if self.gui and self.tool_force_on_human > 0:
            print(
                "Task success:",
                self.task_success,
                "Force at tool on human:",
                self.tool_force_on_human,
                reward_new_contact_points,
            )

        info = {
            "total_force_on_human": self.total_force_on_human,
            "action_robot_len": self.action_robot_len,
            "action_human_len": self.action_human_len,
            "obs_robot_len": self.obs_robot_len,
            "obs_human_len": self.obs_human_len,
        }

        done = self.iteration >= 200

        return obs, reward, done, info
