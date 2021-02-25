import gym
import numpy as np
import pybullet as p
from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.agents.furniture import Furniture

from riddbot.agents.bedpan import Bedpan
from riddbot.agents.disposal import DisposalBowl
from riddbot.agents.water import Water


__all__ = [
    "setup_bed",
    "setup_patient",
    "setup_robot",
    "setup_bedpan",
    "setup_waters",
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


def setup_waters(env: AssistiveEnv):
    Water.set_env(env)

    num_waters = 3
    water_radius = 0.005
    water_mass = 0.001

    bedpan_pos = env.bedpan.get_pos_orient(env.bedpan.base)[0]

    water_positions = []
    for i in range(num_waters):
        for j in range(num_waters):
            for k in range(num_waters):
                water_pos = bedpan_pos + np.array(
                    [
                        i * 2 * water_radius - 0.06,
                        j * 2 * water_radius - 0.06,
                        k * 2 * water_radius + 0.2,
                    ]
                )
                water_positions.append(water_pos)

    env.waters = Water.from_spheres(
        env.create_spheres(
            radius=water_radius,
            mass=water_mass,
            batch_positions=water_positions,
            visual=False,
            collision=True,
        )
    )

    for w in env.waters:
        p.changeVisualShape(
            w.body, -1, rgbaColor=[0.25, 0.5, 1, 1], physicsClientId=env.id
        )

    #  Let the waters settle in the bedpan
    p.setGravity(0, 0, -1, physicsClientId=env.id)
    for _ in range(200):
        p.stepSimulation(physicsClientId=env.id)

    # Remove waters that may have got out of bedpan
    waters_to_remove = [w for w in env.waters if not w.is_inside_bedpan]
    env.waters = [w for w in env.waters if w not in waters_to_remove]
    for w in waters_to_remove:
        p.removeBody(w.body)
    del waters_to_remove


def setup_robot(env: AssistiveEnv):
    right_arm = True

    # Initialize robot pose, get base position
    target_ee_pos = np.array([-0.5, -0.2, 1.1])
    target_ee_orient = env.get_quaternion(env.robot.toc_ee_orient_rpy[env.task])
    bedpan_pos = env.bedpan.get_pos_orient(env.bedpan.base)[0]

    env.robot_base_position = env.init_robot_pose(
        target_ee_pos,
        target_ee_orient,
        [(target_ee_pos, target_ee_orient)],
        [(bedpan_pos, None)],
        arm="right" if right_arm else "left",
        collision_objects=[env.human, env.furniture],
        wheelchair_enabled=False,
    )

    # Open gripper to hold the bedpan
    env.robot.set_gripper_open_position(
        (
            env.robot.right_gripper_indices
            if right_arm
            else env.robot.left_gripper_indices
        ),
        positions=[0.4] * 3,
        set_instantly=True,
    )

    # Load a nightstand in the environment for mounted arms
    if env.robot.wheelchair_mounted:
        env.nightstand = Furniture()
        env.nightstand.init("nightstand", env.directory, env.id, env.np_random)
        env.nightstand.set_base_pos_orient(
            env.robot_base_position + np.array([-0.9, 0.7, 0]), [0, 0, 0, 1]
        )

    # Disable collisions between the bedpan and robot
    for j in (
        env.robot.right_gripper_collision_indices
        if right_arm
        else env.robot.left_gripper_collision_indices
    ):
        for tj in env.bedpan.all_joint_indices + [env.bedpan.base]:
            p.setCollisionFilterPair(
                env.robot.body, env.bedpan.body, j, tj, False, physicsClientId=env.id
            )

    # Create constraint that keeps the bedpan in the gripper
    robot_gripper_pos_offset = [0, 0, 0]
    robot_gripper_orient_offset = [0, -np.pi / 2.0, 0]
    constraint = p.createConstraint(
        env.robot.body,
        env.robot.right_tool_joint if right_arm else env.robot.left_tool_joint,
        env.bedpan.body,
        -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        parentFramePosition=robot_gripper_pos_offset,
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=robot_gripper_orient_offset,
        childFrameOrientation=[0, 0, 0, 1],
        physicsClientId=env.id,
    )
    p.changeConstraint(constraint, maxForce=500, physicsClientId=env.id)


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

    env.disposal_bowl.set_original_pos_orient()


def setup_gravity(env: AssistiveEnv):
    p.setGravity(0, 0, -9.81, physicsClientId=env.id)
    if not env.robot.mobile:
        env.robot.set_gravity(0, 0, 0)
    env.human.set_gravity(0, 0, -1)


def setup_camera(env: gym.Env):
    env.setup_camera(
        fov=60,
        camera_eye=[-2.0, -0.5, 1.5],
        camera_target=[-0.5, 0, 0.75],
        camera_width=1920 // 4,
        camera_height=1080 // 4,
    )
