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

        setup_bed(self)
        setup_patient(self)
        setup_bedpan(self)
        setup_robot(self)
        setup_sanitation_stand(self)

        self.generate_targets()

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
            "task_success": int(
                self.task_success
                >= (self.total_target_count * self.config("task_success_threshold"))
            ),
            "action_robot_len": self.action_robot_len,
            "action_human_len": self.action_human_len,
            "obs_robot_len": self.obs_robot_len,
            "obs_human_len": self.obs_human_len,
        }

        done = self.iteration >= 200

        return obs, reward, done, info

    def generate_targets(self):
        self.target_indices_to_ignore = []
        if self.human.gender == "male":
            self.upperarm, self.upperarm_length, self.upperarm_radius = (
                self.human.right_shoulder,
                0.279,
                0.043,
            )
            self.forearm, self.forearm_length, self.forearm_radius = (
                self.human.right_elbow,
                0.257,
                0.033,
            )
        else:
            self.upperarm, self.upperarm_length, self.upperarm_radius = (
                self.human.right_shoulder,
                0.264,
                0.0355,
            )
            self.forearm, self.forearm_length, self.forearm_radius = (
                self.human.right_elbow,
                0.234,
                0.027,
            )

        self.targets_pos_on_upperarm = self.util.capsule_points(
            p1=np.array([0, 0, 0]),
            p2=np.array([0, 0, -self.upperarm_length]),
            radius=self.upperarm_radius,
            distance_between_points=0.03,
        )
        self.targets_pos_on_forearm = self.util.capsule_points(
            p1=np.array([0, 0, 0]),
            p2=np.array([0, 0, -self.forearm_length]),
            radius=self.forearm_radius,
            distance_between_points=0.03,
        )

        self.targets_upperarm = self.create_spheres(
            radius=0.01,
            mass=0.0,
            batch_positions=[[0, 0, 0]] * len(self.targets_pos_on_upperarm),
            visual=True,
            collision=False,
            rgba=[0, 1, 1, 1],
        )
        self.targets_forearm = self.create_spheres(
            radius=0.01,
            mass=0.0,
            batch_positions=[[0, 0, 0]] * len(self.targets_pos_on_forearm),
            visual=True,
            collision=False,
            rgba=[0, 1, 1, 1],
        )
        self.total_target_count = len(self.targets_pos_on_upperarm) + len(
            self.targets_pos_on_forearm
        )
        self.update_targets()

    def update_targets(self):
        upperarm_pos, upperarm_orient = self.human.get_pos_orient(self.upperarm)
        self.targets_pos_upperarm_world = []
        for target_pos_on_arm, target in zip(
            self.targets_pos_on_upperarm, self.targets_upperarm
        ):
            target_pos = np.array(
                p.multiplyTransforms(
                    upperarm_pos,
                    upperarm_orient,
                    target_pos_on_arm,
                    [0, 0, 0, 1],
                    physicsClientId=self.id,
                )[0]
            )
            self.targets_pos_upperarm_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])

        forearm_pos, forearm_orient = self.human.get_pos_orient(self.forearm)
        self.targets_pos_forearm_world = []
        for target_pos_on_arm, target in zip(
            self.targets_pos_on_forearm, self.targets_forearm
        ):
            target_pos = np.array(
                p.multiplyTransforms(
                    forearm_pos,
                    forearm_orient,
                    target_pos_on_arm,
                    [0, 0, 0, 1],
                    physicsClientId=self.id,
                )[0]
            )
            self.targets_pos_forearm_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])
