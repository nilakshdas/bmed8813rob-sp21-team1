import numpy as np
from assistive_gym.envs.env import AssistiveEnv


__all__ = [
    "get_bed_rewards",
    "get_human_rewards",
    "get_robot_rewards",
    "get_sanitation_rewards",
]


def get_bed_rewards(env: AssistiveEnv) -> dict:
    return dict(water_on_bed=sum(w.is_on_bed for w in env.waters))


def get_human_rewards(env: AssistiveEnv) -> dict:
    rewards = dict()

    end_effector_velocity = np.linalg.norm(
        env.robot.get_velocity(env.robot.left_end_effector)
    )

    rewards.update(
        human_preferences=env.human_preferences(
            end_effector_velocity=end_effector_velocity,
            total_force_on_human=env.total_force_on_human,
        )
    )

    rewards.update(
        distance_bedpan_human=min(env.bedpan.get_closest_points(env.human)[-1])
    )

    rewards.update(water_on_human=sum(w.is_on_human for w in env.waters))

    return rewards


def get_robot_rewards(env: AssistiveEnv, action: np.ndarray) -> dict:
    rewards = dict()

    rewards.update(robot_action=np.linalg.norm(action))

    _, bedpan_orient = env.bedpan.get_pos_orient(env.bedpan.base)
    rewards.update(
        bedpan_disorient=np.sum(np.abs(bedpan_orient - np.array([0, 0, 0, 1])))
    )

    return rewards


def get_sanitation_rewards(env: AssistiveEnv) -> dict:
    return dict(
        water_in_sanitation_bowl=sum(w.is_inside_disposal_bowl for w in env.waters)
    )
