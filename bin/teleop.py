import argparse
import sys
import json
from pathlib import Path

import gym
import numpy as np
import pybullet as p

# XXX: hack for Windows users
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from riddbot.gym import make_env


POS_MAGNITUDE = 0.01
RPY_MAGNITUDE = 0.05

# Map keys to position and orientation end effector movements
POS_KEYS_ACTIONS = {
    ord("j"): np.array([-POS_MAGNITUDE, 0, 0]),
    ord("l"): np.array([POS_MAGNITUDE, 0, 0]),
    ord("u"): np.array([0, -POS_MAGNITUDE, 0]),
    ord("o"): np.array([0, POS_MAGNITUDE, 0]),
    ord("k"): np.array([0, 0, -POS_MAGNITUDE]),
    ord("i"): np.array([0, 0, POS_MAGNITUDE]),
}
RPY_KEYS_ACTIONS = {
    ord("k"): np.array([-RPY_MAGNITUDE, 0, 0]),
    ord("i"): np.array([RPY_MAGNITUDE, 0, 0]),
    ord("u"): np.array([0, -RPY_MAGNITUDE, 0]),
    ord("o"): np.array([0, RPY_MAGNITUDE, 0]),
    ord("j"): np.array([0, 0, -RPY_MAGNITUDE]),
    ord("l"): np.array([0, 0, RPY_MAGNITUDE]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=Path, required=True)
    return parser.parse_args()


def read_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def setup_env(config_path: Path):
    config = read_config(config_path)
    return make_env(reward_weights=config["reward_weights"])


def main():
    args = parse_args()

    env_name, env = setup_env(args.config_path)
    env = gym.make(env_name)
    env.render()
    observation = env.reset()

    start_pos, orient = env.robot.get_pos_orient(env.robot.right_end_effector)
    start_rpy = env.get_euler(orient)
    target_pos_offset = np.zeros(3)
    target_rpy_offset = np.zeros(3)

    while True:
        keys = p.getKeyboardEvents(env.id)
        # Process position movement keys ('u', 'i', 'o', 'j', 'k', 'l')
        for key, action in POS_KEYS_ACTIONS.items():
            if p.B3G_SHIFT not in keys and key in keys and keys[key] & p.KEY_IS_DOWN:
                target_pos_offset += action
        # Process rpy movement keys (shift + movement keys)
        for key, action in RPY_KEYS_ACTIONS.items():
            if (
                p.B3G_SHIFT in keys
                and keys[p.B3G_SHIFT] & p.KEY_IS_DOWN
                and (key in keys and keys[key] & p.KEY_IS_DOWN)
            ):
                target_rpy_offset += action

        # print('Target position offset:', target_pos_offset, 'Target rpy offset:', target_rpy_offset)
        target_pos = start_pos + target_pos_offset
        target_rpy = start_rpy + target_rpy_offset

        # Use inverse kinematics to compute the joint angles for the robot's arm
        # so that its end effector moves to the target position.
        target_joint_angles = env.robot.ik(
            env.robot.right_end_effector,
            target_pos,
            env.get_quaternion(target_rpy),
            env.robot.right_arm_ik_indices,
            max_iterations=200,
            use_current_as_rest=True,
        )
        # Get current joint angles of the robot's arm
        current_joint_angles = env.robot.get_joint_angles(
            env.robot.right_arm_joint_indices
        )
        # Compute the action as the difference between target and current joint angles.
        action = (target_joint_angles - current_joint_angles) * 10
        # Step the simulation forward
        observation, reward, done, info = env.step(action)


if __name__ == "__main__":
    main()
