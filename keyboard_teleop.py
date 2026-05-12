"""
键盘控制 Isaac Gym Allegro Hand 仿真
无需 VR 头盔，直接用键盘操作手指关节。

运行方式:
    cd Open-Teach
    export LD_LIBRARY_PATH=/home/hu/miniconda3/envs/openteach_v2/lib
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    export PATH=/home/hu/miniconda3/envs/openteach_v2/bin:$PATH
    /home/hu/miniconda3/envs/openteach_v2/bin/python keyboard_teleop.py

键盘控制说明:
    1-4: 选择手指 (1=食指, 2=中指, 3=无名指, 4=拇指)
    W/S: 当前手指弯曲/伸展 (MCP关节)
    A/D: 当前手指侧向摆动 (ABD关节)
    Q/E: 当前手指中间关节 弯曲/伸展
    Z/X: 当前手指末端关节 弯曲/伸展
    R:   重置所有关节到初始位置
    O:   打开所有手指（张开手）
    C:   关闭所有手指（握拳）
    G:   执行抓取动作序列
    ESC: 退出（或关闭viewer窗口）
"""

import os
os.environ['MESA_VK_DEVICE_SELECT'] = '10de:24b0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *
import numpy as np
import torch
import time

# ============================================================
# 配置
# ============================================================
STEP_SIZE = 0.05  # 每次按键关节变化量(弧度)

# Allegro Hand 16个关节的映射
# 食指: joints 0-3, 中指: joints 4-7, 无名指: joints 8-11, 拇指: joints 12-15
FINGER_NAMES = ["食指(Index)", "中指(Middle)", "无名指(Ring)", "拇指(Thumb)"]
FINGER_JOINTS = {
    0: [0, 1, 2, 3],    # 食指
    1: [4, 5, 6, 7],    # 中指
    2: [8, 9, 10, 11],  # 无名指
    3: [12, 13, 14, 15], # 拇指
}

# 关节限位 (近似值)
JOINT_LOWER = np.array([
    -0.47, -0.196, -0.174, -0.227,
    -0.47, -0.196, -0.174, -0.227,
    -0.47, -0.196, -0.174, -0.227,
     0.263, -0.105, -0.189, -0.162
])
JOINT_UPPER = np.array([
    0.47, 1.61, 1.709, 1.618,
    0.47, 1.61, 1.709, 1.618,
    0.47, 1.61, 1.709, 1.618,
    1.396, 1.163, 1.644, 1.719
])

# 初始姿态
HOME_POSITION = np.array([
    -0.00137183, -0.22922094, 0.7265581, 0.79128325,
     0.9890924,   0.37431374, 0.36866143, 0.77558154,
     0.00662423, -0.23064502, 0.73253167, 0.7449019,
     0.08261403, -0.15844858, 0.82595366, 0.7666822
])

# 张开手姿态
OPEN_POSITION = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.3, 0.0, 0.0, 0.0
])

# 握拳姿态
CLOSE_POSITION = np.array([
    0.0, 1.4, 1.5, 1.4,
    0.0, 1.4, 1.5, 1.4,
    0.0, 1.4, 1.5, 1.4,
    1.2, 1.0, 1.4, 1.5
])

# ============================================================
# Isaac Gym 键盘事件映射
# ============================================================
KEY_MAP = {
    gymapi.KEY_1: '1', gymapi.KEY_2: '2', gymapi.KEY_3: '3', gymapi.KEY_4: '4',
    gymapi.KEY_W: 'w', gymapi.KEY_S: 's',
    gymapi.KEY_A: 'a', gymapi.KEY_D: 'd',
    gymapi.KEY_Q: 'q', gymapi.KEY_E: 'e',
    gymapi.KEY_Z: 'z', gymapi.KEY_X: 'x',
    gymapi.KEY_R: 'r', gymapi.KEY_O: 'o',
    gymapi.KEY_C: 'c', gymapi.KEY_G: 'g',
}


def create_sim():
    """创建仿真环境"""
    gym = gymapi.acquire_gym()

    # 仿真参数
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0)
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create sim")

    # 地面
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)

    # Viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create viewer")

    cam_pos = gymapi.Vec3(1.06, 1.6, -0.02)
    cam_target = gymapi.Vec3(1.03, 1.3, -0.02)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # 订阅键盘事件
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_1, "finger_1")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_2, "finger_2")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_3, "finger_3")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_4, "finger_4")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "w")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "s")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "a")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "d")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "q")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_E, "e")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Z, "z")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_X, "x")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_O, "open")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_C, "close")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_G, "grasp")

    # 加载资产
    asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "openteach/components/environment/assets/urdf/")
    asset_file = "allegro_hand_description/urdf/model_only_hand.urdf"
    object_file = "objects/cube_multicolor.urdf"
    table_file = "allegro_hand_description/urdf/table.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = False
    asset_options.use_mesh_materials = True
    asset_options.disable_gravity = True

    table_options = gymapi.AssetOptions()
    table_options.fix_base_link = True
    table_options.flip_visual_attachments = False
    table_options.collapse_fixed_joints = True
    table_options.disable_gravity = False

    object_options = gymapi.AssetOptions()

    hand_asset = gym.load_urdf(sim, asset_root, asset_file, asset_options)
    table_asset = gym.load_urdf(sim, asset_root, table_file, table_options)
    object_asset = gym.load_urdf(sim, asset_root, object_file, object_options)

    # 创建环境
    spacing = 2.5
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # Allegro Hand
    actor_pose = gymapi.Transform()
    actor_pose.p = gymapi.Vec3(1, 1.2, 0)
    actor_pose.r = gymapi.Quat(-0.707, -0.707, 0, 0)
    actor_handle = gym.create_actor(env, hand_asset, actor_pose, "allegro_hand", 0, 1)

    # 桌子
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.7, 0.0, 0.3)
    table_pose.r = gymapi.Quat(-0.707107, 0, 0.0, 0.707)
    gym.create_actor(env, table_asset, table_pose, "table", 0, 1)

    # 方块
    object_pose = gymapi.Transform()
    object_pose.p = gymapi.Vec3(1, 1.3, 0.06)
    object_pose.r = gymapi.Quat(-1.3, -0.707, 0, 0)
    gym.create_actor(env, object_asset, object_pose, "cube", 0, 0, 0)

    # 手的颜色
    num_dofs = gym.get_asset_dof_count(hand_asset)
    for j in range(num_dofs + 13):
        if j != 20 and j != 15 and j != 10 and j != 5:
            gym.set_rigid_body_color(env, actor_handle, j, gymapi.MESH_VISUAL,
                                     gymapi.Vec3(0.15, 0.15, 0.15))

    # 关节属性
    props = gym.get_actor_dof_properties(env, actor_handle)
    props["stiffness"] = [3] * 16
    props["damping"] = [0.18] * 16
    props["friction"] = [0.01] * 16
    props["armature"] = [0.001] * 16
    props["velocity"] = [2.0] * 16
    for k in range(num_dofs):
        props["driveMode"][k] = gymapi.DOF_MODE_POS
    gym.set_actor_dof_properties(env, actor_handle, props)

    return gym, sim, viewer, env, actor_handle


def print_status(current_finger, joint_positions):
    """打印当前状态"""
    os.system('clear' if os.name == 'posix' else 'cls')
    print("=" * 60)
    print("  Allegro Hand 键盘遥操作")
    print("=" * 60)
    print(f"\n  当前选中手指: [{current_finger + 1}] {FINGER_NAMES[current_finger]}")
    print(f"  关节角度 (弧度):")
    for i, name in enumerate(FINGER_NAMES):
        joints = FINGER_JOINTS[i]
        marker = " >>>" if i == current_finger else "    "
        vals = [f"{joint_positions[j]:+.3f}" for j in joints]
        print(f"  {marker} {name}: {vals}")
    print(f"\n  控制说明:")
    print(f"    1-4: 选择手指    W/S: 弯曲/伸展(关节1)")
    print(f"    A/D: 侧摆       Q/E: 弯曲/伸展(关节2)")
    print(f"    Z/X: 末端弯曲/伸展(关节3)")
    print(f"    R: 重置  O: 张开  C: 握拳  G: 抓取动作")
    print(f"    ESC/关闭窗口: 退出")
    print("=" * 60)


def smooth_move(gym, sim, viewer, current_pos, target_pos, steps=30):
    """平滑移动到目标位置"""
    for i in range(steps):
        t = (i + 1) / steps
        interp = current_pos + t * (target_pos - current_pos)
        pos_tensor = torch.tensor(interp, dtype=torch.float32)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_tensor))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        if gym.query_viewer_has_closed(viewer):
            return interp
    return target_pos


def grasp_sequence(gym, sim, viewer, current_pos):
    """执行抓取动作序列"""
    # 1. 先张开手
    pos = smooth_move(gym, sim, viewer, current_pos, OPEN_POSITION, steps=40)
    time.sleep(0.3)
    # 2. 慢慢握住
    pos = smooth_move(gym, sim, viewer, pos, CLOSE_POSITION * 0.7, steps=60)
    time.sleep(0.5)
    # 3. 稍微松开
    target = CLOSE_POSITION * 0.5
    pos = smooth_move(gym, sim, viewer, pos, target, steps=30)
    return pos


def main():
    gym, sim, viewer, env, actor_handle = create_sim()

    # 初始化关节位置
    joint_positions = HOME_POSITION.copy()
    pos_tensor = torch.tensor(joint_positions, dtype=torch.float32)
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_tensor))

    current_finger = 0  # 当前选中的手指 (0-3)
    print_status(current_finger, joint_positions)

    # 主循环
    while not gym.query_viewer_has_closed(viewer):
        # 处理键盘事件
        events = gym.query_viewer_action_events(viewer)
        for event in events:
            if event.value == 0:  # 只处理按下事件
                continue

            action = event.action
            joints = FINGER_JOINTS[current_finger]
            updated = False

            if action == "finger_1":
                current_finger = 0
                updated = True
            elif action == "finger_2":
                current_finger = 1
                updated = True
            elif action == "finger_3":
                current_finger = 2
                updated = True
            elif action == "finger_4":
                current_finger = 3
                updated = True
            elif action == "a":  # 侧摆 -
                joint_positions[joints[0]] -= STEP_SIZE
                updated = True
            elif action == "d":  # 侧摆 +
                joint_positions[joints[0]] += STEP_SIZE
                updated = True
            elif action == "w":  # 关节1 弯曲
                joint_positions[joints[1]] += STEP_SIZE
                updated = True
            elif action == "s":  # 关节1 伸展
                joint_positions[joints[1]] -= STEP_SIZE
                updated = True
            elif action == "q":  # 关节2 弯曲
                joint_positions[joints[2]] += STEP_SIZE
                updated = True
            elif action == "e":  # 关节2 伸展
                joint_positions[joints[2]] -= STEP_SIZE
                updated = True
            elif action == "z":  # 关节3 弯曲
                joint_positions[joints[3]] += STEP_SIZE
                updated = True
            elif action == "x":  # 关节3 伸展
                joint_positions[joints[3]] -= STEP_SIZE
                updated = True
            elif action == "reset":
                joint_positions = HOME_POSITION.copy()
                updated = True
            elif action == "open":
                joint_positions = smooth_move(gym, sim, viewer, joint_positions, OPEN_POSITION, steps=40).copy()
                updated = True
            elif action == "close":
                joint_positions = smooth_move(gym, sim, viewer, joint_positions, CLOSE_POSITION, steps=40).copy()
                updated = True
            elif action == "grasp":
                joint_positions = grasp_sequence(gym, sim, viewer, joint_positions).copy()
                updated = True

            if updated:
                # 限位
                joint_positions = np.clip(joint_positions, JOINT_LOWER, JOINT_UPPER)
                print_status(current_finger, joint_positions)

        # 设置目标位置并仿真
        pos_tensor = torch.tensor(joint_positions, dtype=torch.float32)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_tensor))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    # 清理
    print("\nClosing simulation...")
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("Done!")


if __name__ == "__main__":
    main()
