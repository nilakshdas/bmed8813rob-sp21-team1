from typing import Tuple

import numpy as np
from assistive_gym.envs.env import AssistiveEnv


__all__ = [
    "get_tool_observations",
    "get_robot_observations",
    "get_human_observations",
    "get_force_observations",
]


def get_tool_observations(env: AssistiveEnv) -> Tuple[np.ndarray, np.ndarray]:
    tool_pos, tool_orient = env.tool.get_pos_orient(1)
    tool_pos_real, tool_orient_real = env.robot.convert_to_realworld(
        tool_pos, tool_orient
    )

    return tool_pos_real, tool_orient_real


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


def get_human_observations(
    env: AssistiveEnv,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    shoulder_pos = env.human.get_pos_orient(env.human.right_shoulder)[0]
    elbow_pos = env.human.get_pos_orient(env.human.right_elbow)[0]
    wrist_pos = env.human.get_pos_orient(env.human.right_wrist)[0]

    shoulder_pos_real, _ = env.robot.convert_to_realworld(shoulder_pos)
    elbow_pos_real, _ = env.robot.convert_to_realworld(elbow_pos)
    wrist_pos_real, _ = env.robot.convert_to_realworld(wrist_pos)

    return shoulder_pos_real, elbow_pos_real, wrist_pos_real


def get_force_observations(
    env: AssistiveEnv,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_force_on_human = np.sum(env.robot.get_contact_points(env.human)[-1])
    tool_force = np.sum(env.tool.get_contact_points()[-1])

    tool_force_on_human = 0
    new_contact_points = 0
    for linkA, linkB, posA, posB, force in zip(*env.tool.get_contact_points(env.human)):
        total_force_on_human += force
        if linkA in [1]:
            tool_force_on_human += force

    return tool_force, tool_force_on_human, total_force_on_human, new_contact_points
