from typing import Tuple

import numpy as np
from assistive_gym.envs.env import AssistiveEnv


__all__ = [
    "get_robot_observations",
    "get_bedpan_observations",
    "get_disposal_bowl_observations",
    "get_force_observations",
]


def get_robot_observations(env: AssistiveEnv) -> np.ndarray:
    robot_joint_angles = env.robot.get_joint_angles(
        env.robot.controllable_joint_indices
    )

    # Fix joint angles to be in [-pi, pi]
    robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2 * np.pi) - np.pi

    # Don't include joint angles for the wheels
    if env.robot.mobile:
        robot_joint_angles = robot_joint_angles[len(env.robot.wheel_joint_indices) :]

    return robot_joint_angles


def get_bedpan_observations(env: AssistiveEnv) -> Tuple[np.ndarray, np.ndarray]:
    bedpan_pos, bedpan_orient = env.bedpan.get_pos_orient(env.bedpan.base)

    bedpan_pos_real, bedpan_orient_real = env.robot.convert_to_realworld(
        bedpan_pos, bedpan_orient
    )

    return bedpan_pos_real, bedpan_orient_real


def get_disposal_bowl_observations(env: AssistiveEnv) -> np.ndarray:
    disposal_bowl_pos, _ = env.disposal_bowl.get_pos_orient(env.disposal_bowl.base)
    disposal_bowl_pos_real, _ = env.robot.convert_to_realworld(disposal_bowl_pos)
    return disposal_bowl_pos_real


def get_force_observations(env: AssistiveEnv) -> Tuple[np.ndarray, np.ndarray]:
    robot_force_on_human = np.sum(env.robot.get_contact_points(env.human)[-1])
    bedpan_force_on_human = np.sum(env.bedpan.get_contact_points(env.human)[-1])
    return robot_force_on_human, bedpan_force_on_human
