import gym
import numpy as np
import pybullet as p
from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.agents.furniture import Furniture

from riddbot.agents.bedpan import Bedpan
from riddbot.agents.disposal import DisposalBowl


__all__ = [
    "setup_bed",
    "setup_patient",
    "setup_robot",
    "setup_bedpan",
    "setup_sanitation_stand",
    "setup_gravity",
]


def setup_bed(env: AssistiveEnv):
    env.build_assistive_env("bed", fixed_human_base=False)
    env.furniture.set_friction(env.furniture.base, friction=5)


def setup_patient(env: AssistiveEnv):
    # Setup human in the air and let them settle into a resting pose on the bed
    joints_positions = [(env.human.j_right_shoulder_x, 30)]
    env.human.setup_joints(
        joints_positions, use_static_joints=False, reactive_force=None
    )
    env.human.set_base_pos_orient([0.05, 0.2, 0.95], [-np.pi / 2.0, np.pi / 2.0, 0])

    # Let the person settle on the bed
    p.setGravity(0, 0, -1, physicsClientId=env.id)
    for _ in range(100):
        p.stepSimulation(physicsClientId=env.id)

    # Lock human joints and set velocities to 0
    joints_positions = []
    env.human.setup_joints(
        joints_positions,
        use_static_joints=True,
        reactive_force=None,
        reactive_gain=0.01,
    )
    env.human.set_mass(env.human.base, mass=0)
    env.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])


def setup_bedpan(env: AssistiveEnv):
    human_waist_pos = env.human.get_pos_orient(env.human.waist)[0]
    bedpan_pos = human_waist_pos + np.array([-0.4, -0.2, 2.0])

    env.bedpan = Bedpan()
    env.bedpan.init(env=env, base_pos=bedpan_pos)

    # Let the bedpan settle on the bed
    p.setGravity(0, 0, -1, physicsClientId=env.id)
    for _ in range(100):
        p.stepSimulation(physicsClientId=env.id)


def setup_robot(env: AssistiveEnv):
    # Initialize the tool in the robot's gripper
    env.tool.init(
        env.robot,
        env.task,
        env.directory,
        env.id,
        env.np_random,
        right=False,
        mesh_scale=[1] * 3,
    )

    # Initialize robot pose, get base position
    human_shoulder_pos = env.human.get_pos_orient(env.human.right_shoulder)[0]
    human_elbow_pos = env.human.get_pos_orient(env.human.right_elbow)[0]
    human_wrist_pos = env.human.get_pos_orient(env.human.right_wrist)[0]

    target_ee_pos = np.array([-0.6, 0.2, 1])
    target_ee_pos += env.np_random.uniform(-0.05, 0.05, size=3)
    target_ee_orient = env.get_quaternion(env.robot.toc_ee_orient_rpy[env.task])

    env.robot_base_position = env.init_robot_pose(
        target_ee_pos,
        target_ee_orient,
        [(target_ee_pos, target_ee_orient)],
        [(human_shoulder_pos, None), (human_elbow_pos, None), (human_wrist_pos, None)],
        arm="left",
        tools=[env.tool],
        collision_objects=[env.human, env.furniture],
        wheelchair_enabled=False,
    )

    # Load a nightstand in the environment for mounted arms
    if env.robot.wheelchair_mounted:
        env.nightstand = Furniture()
        env.nightstand.init("nightstand", env.directory, env.id, env.np_random)
        env.nightstand.set_base_pos_orient(
            env.robot_base_position + np.array([-0.9, 0.7, 0]), [0, 0, 0, 1]
        )

    # Open gripper to hold the tool
    env.robot.set_gripper_open_position(
        env.robot.left_gripper_indices,
        env.robot.gripper_pos[env.task],
        set_instantly=True,
    )


def setup_sanitation_stand(env: AssistiveEnv):
    robot_base_position = env.robot_base_position
    sanitation_stand_pos = robot_base_position + np.array([-0.9, 0.1, 0])
    disposal_bowl_pos = sanitation_stand_pos + np.array([0, 0, 0.8])

    env.sanitation_stand = Furniture()
    env.sanitation_stand.init("nightstand", env.directory, env.id, env.np_random)
    env.sanitation_stand.set_base_pos_orient(sanitation_stand_pos, [0, 0, 0, 1])

    env.disposal_bowl = DisposalBowl()
    env.disposal_bowl.init(env=env, base_pos=disposal_bowl_pos)

    # Let the disposal bowl settle on the sanitation stand
    p.setGravity(0, 0, -1, physicsClientId=env.id)
    for _ in range(100):
        p.stepSimulation(physicsClientId=env.id)


def setup_gravity(env: AssistiveEnv):
    p.setGravity(0, 0, -9.81, physicsClientId=env.id)
    if not env.robot.mobile:
        env.robot.set_gravity(0, 0, 0)
    env.human.set_gravity(0, 0, -1)
    env.tool.set_gravity(0, 0, 0)


def setup_camera(env: gym.Env):
    env.setup_camera(
        fov=60,
        camera_eye=[-2.0, -0.5, 1.5],
        camera_target=[-0.5, 0, 0.75],
        camera_width=1920 // 4,
        camera_height=1080 // 4,
    )
