"""
键盘控制 + 数据采集一体化脚本
用键盘操作 Allegro Hand 完成任务，同时录制数据。
保存格式与 OpenTeach 原版 data_collect.py 完全兼容。

运行方式:
    cd Open-Teach
    export LD_LIBRARY_PATH=/home/hu/miniconda3/envs/openteach_v2/lib
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    export PATH=/home/hu/miniconda3/envs/openteach_v2/bin:$PATH
    /home/hu/miniconda3/envs/openteach_v2/bin/python keyboard_data_collect.py --demo_num 1

或直接:
    ./run_data_collect.sh 1

保存的数据:
    extracted_data/demonstration_X/
    ├── cam_0_rgb_video.avi          # RGB 视频 (480x480, 60fps)
    ├── cam_0_rgb_video.metadata     # 视频元数据
    ├── cam_0_depth.h5               # 深度图 (HDF5)
    ├── allegro_joint_states.h5      # 实际关节角度 (HDF5)
    └── allegro_commanded_joint_states.h5  # 指令关节角度 (HDF5)

键盘控制:
    1-4: 选择手指 (1=食指, 2=中指, 3=无名指, 4=拇指)
    W/S: 弯曲/伸展 (MCP关节)
    A/D: 侧向摆动 (ABD关节)
    Q/E: 中间关节 弯曲/伸展
    Z/X: 末端关节 弯曲/伸展
    R: 重置    O: 张开    C: 握拳    G: 抓取动作
    SPACE: 开始/暂停录制
    ESC: 停止并保存数据
"""

import os
os.environ['MESA_VK_DEVICE_SELECT'] = '10de:24b0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *
import numpy as np
import torch
import time
import cv2
import h5py
import pickle
import argparse

# ============================================================
# 参数
# ============================================================
STEP_SIZE = 0.05
SIM_FPS = 60
IMAGE_SIZE = (480, 480)

FINGER_NAMES = ["食指(Index)", "中指(Middle)", "无名指(Ring)", "拇指(Thumb)"]
FINGER_JOINTS = {
    0: [0, 1, 2, 3],
    1: [4, 5, 6, 7],
    2: [8, 9, 10, 11],
    3: [12, 13, 14, 15],
}

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

HOME_POSITION = np.array([
    -0.00137183, -0.22922094, 0.7265581, 0.79128325,
     0.9890924,   0.37431374, 0.36866143, 0.77558154,
     0.00662423, -0.23064502, 0.73253167, 0.7449019,
     0.08261403, -0.15844858, 0.82595366, 0.7666822
])

OPEN_POSITION = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.3, 0.0, 0.0, 0.0
])

CLOSE_POSITION = np.array([
    0.0, 1.4, 1.5, 1.4,
    0.0, 1.4, 1.5, 1.4,
    0.0, 1.4, 1.5, 1.4,
    1.2, 1.0, 1.4, 1.5
])


# ============================================================
# 数据录制器
# ============================================================
class DataRecorder:
    """录制仿真数据，格式与 OpenTeach 兼容"""

    def __init__(self, storage_path):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

        # 视频录制器
        video_path = os.path.join(storage_path, 'cam_0_rgb_video.avi')
        self.video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            SIM_FPS,
            IMAGE_SIZE
        )

        # 数据容器
        self.rgb_timestamps = []
        self.depth_frames = []
        self.depth_timestamps = []
        self.joint_states = []          # 实际关节角度
        self.joint_timestamps = []
        self.commanded_states = []      # 指令关节角度
        self.commanded_timestamps = []

        self.is_recording = False
        self.num_frames = 0
        self.record_start_time = None

    def start_recording(self):
        self.is_recording = True
        self.record_start_time = time.time()
        print("\n🔴 开始录制!")

    def pause_recording(self):
        self.is_recording = False
        print("\n⏸️  暂停录制")

    def reset_recording(self):
        """重置录制，清空所有已录数据"""
        self.is_recording = False
        self.rgb_timestamps = []
        self.depth_frames = []
        self.depth_timestamps = []
        self.joint_states = []
        self.joint_timestamps = []
        self.commanded_states = []
        self.commanded_timestamps = []
        self.num_frames = 0
        self.record_start_time = None

        # 重新创建视频录制器
        self.video_writer.release()
        video_path = os.path.join(self.storage_path, 'cam_0_rgb_video.avi')
        self.video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            SIM_FPS,
            IMAGE_SIZE
        )
        print("\n🔄 录制已重置! 按 SPACE 重新开始录制")

    def record_frame(self, rgb_image, depth_image, actual_joints, commanded_joints, sim_time):
        """录制一帧数据"""
        if not self.is_recording:
            return

        # RGB
        self.video_writer.write(rgb_image)
        self.rgb_timestamps.append(sim_time)

        # Depth
        depth_uint16 = (depth_image * 1000).astype(np.uint16)  # 转换为毫米
        self.depth_frames.append(depth_uint16)
        self.depth_timestamps.append(sim_time)

        # Joint states
        self.joint_states.append(actual_joints.copy())
        self.joint_timestamps.append(sim_time)

        # Commanded states
        self.commanded_states.append(commanded_joints.copy())
        self.commanded_timestamps.append(sim_time)

        self.num_frames += 1

    def save(self):
        """保存所有数据"""
        if self.num_frames == 0:
            print("\n⚠️  没有录制到数据，跳过保存")
            return

        record_duration = time.time() - self.record_start_time if self.record_start_time else 0
        print(f"\n{'='*60}")
        print(f"  保存数据中...")
        print(f"  总帧数: {self.num_frames}")
        print(f"  录制时长: {record_duration:.1f}s")
        print(f"  平均帧率: {self.num_frames/max(record_duration,1):.1f} fps")
        print(f"{'='*60}")

        # 1. 保存 RGB 视频
        self.video_writer.release()
        metadata = {
            'num_frames': self.num_frames,
            'timestamps': self.rgb_timestamps,
            'duration': record_duration,
            'fps': SIM_FPS,
            'resolution': IMAGE_SIZE,
        }
        metadata_path = os.path.join(self.storage_path, 'cam_0_rgb_video.metadata')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"  ✅ RGB 视频: cam_0_rgb_video.avi ({self.num_frames} frames)")

        # 2. 保存深度数据
        depth_path = os.path.join(self.storage_path, 'cam_0_depth.h5')
        with h5py.File(depth_path, 'w') as f:
            stacked = np.array(self.depth_frames, dtype=np.uint16)
            f.create_dataset('depth_images', data=stacked, compression='gzip', compression_opts=6)
            timestamps = np.array(self.depth_timestamps, dtype=np.float64)
            f.create_dataset('timestamps', data=timestamps, compression='gzip', compression_opts=6)
        print(f"  ✅ 深度数据: cam_0_depth.h5")

        # 3. 保存实际关节角度
        joint_path = os.path.join(self.storage_path, 'allegro_joint_states.h5')
        with h5py.File(joint_path, 'w') as f:
            positions = np.array(self.joint_states, dtype=np.float32)
            f.create_dataset('positions', data=positions, compression='gzip', compression_opts=6)
            timestamps = np.array(self.joint_timestamps, dtype=np.float64)
            f.create_dataset('timestamps', data=timestamps, compression='gzip', compression_opts=6)
        print(f"  ✅ 关节状态: allegro_joint_states.h5")

        # 4. 保存指令关节角度
        cmd_path = os.path.join(self.storage_path, 'allegro_commanded_joint_states.h5')
        with h5py.File(cmd_path, 'w') as f:
            positions = np.array(self.commanded_states, dtype=np.float32)
            f.create_dataset('positions', data=positions, compression='gzip', compression_opts=6)
            timestamps = np.array(self.commanded_timestamps, dtype=np.float64)
            f.create_dataset('timestamps', data=timestamps, compression='gzip', compression_opts=6)
        print(f"  ✅ 指令状态: allegro_commanded_joint_states.h5")

        print(f"\n  📁 数据保存在: {self.storage_path}")
        print(f"{'='*60}")


# ============================================================
# 仿真环境
# ============================================================
def create_sim():
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / SIM_FPS
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

    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create viewer")

    cam_pos = gymapi.Vec3(1.06, 1.6, -0.02)
    cam_target = gymapi.Vec3(1.03, 1.3, -0.02)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # 键盘事件
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
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "toggle_record")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_T, "reset_record")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_F, "push")  # 长按持续推
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_B, "reset_scene")  # 重置整个场景

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

    # 创建相机传感器（用于录制）
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 100
    camera_props.width = IMAGE_SIZE[0]
    camera_props.height = IMAGE_SIZE[1]
    camera_props.enable_tensors = True
    camera_handle = gym.create_camera_sensor(env, camera_props)
    gym.set_camera_location(camera_handle, env,
                            gymapi.Vec3(1.06, 1.6, -0.02),
                            gymapi.Vec3(1.03, 1.3, -0.02))

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
    object_handle = gym.create_actor(env, object_asset, object_pose, "cube", 0, 0, 0)

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

    return gym, sim, viewer, env, actor_handle, camera_handle, object_handle


def get_camera_images(gym, sim, env, camera_handle):
    """获取 RGB 和深度图像"""
    # RGB
    color_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handle, gymapi.IMAGE_COLOR)
    color_image = gymtorch.wrap_tensor(color_tensor).cpu().numpy()
    color_image = color_image[:, :, [2, 1, 0]]  # RGBA -> BGR

    # Depth
    depth_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handle, gymapi.IMAGE_DEPTH)
    depth_image = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()

    return color_image, depth_image


def get_actual_joint_positions(gym, env, num_dofs=16):
    """获取实际关节角度"""
    positions = np.zeros(num_dofs)
    for i in range(num_dofs):
        positions[i] = gym.get_dof_position(env, i)
    return positions


def smooth_move(gym, sim, viewer, env, camera_handle, recorder, current_pos, target_pos, commanded_pos, steps=30):
    """平滑移动并录制"""
    for i in range(steps):
        if gym.query_viewer_has_closed(viewer):
            return current_pos
        t = (i + 1) / steps
        interp = current_pos + t * (target_pos - current_pos)
        pos_tensor = torch.tensor(interp, dtype=torch.float32)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_tensor))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

        # 录制
        if recorder.is_recording:
            rgb, depth = get_camera_images(gym, sim, env, camera_handle)
            actual = get_actual_joint_positions(gym, env)
            sim_time = gym.get_elapsed_time(sim)
            recorder.record_frame(rgb, depth, actual, interp, sim_time)

    return target_pos


def print_status(current_finger, joint_positions, recorder):
    """打印状态"""
    os.system('clear' if os.name == 'posix' else 'cls')
    print("=" * 60)
    print("  Allegro Hand 键盘遥操作 + 数据采集")
    print("=" * 60)

    # 录制状态
    if recorder.is_recording:
        print(f"  🔴 录制中... (已录 {recorder.num_frames} 帧)")
    else:
        print(f"  ⏹️  未录制 (按 SPACE 开始录制)")

    print(f"\n  当前手指: [{current_finger + 1}] {FINGER_NAMES[current_finger]}")
    print(f"  关节角度:")
    for i, name in enumerate(FINGER_NAMES):
        joints = FINGER_JOINTS[i]
        marker = " >>>" if i == current_finger else "    "
        vals = [f"{joint_positions[j]:+.3f}" for j in joints]
        print(f"  {marker} {name}: {vals}")
    print(f"\n  操作: 1-4选指 W/S弯曲 A/D侧摆 Q/E中节 Z/X末节")
    print(f"  动作: R重置手 O张开 C握拳 G抓取 F长按推 B重置场景")
    print(f"  录制: SPACE开始/暂停  T重置录制  ESC退出并保存")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_num', type=int, default=1, help='演示编号')
    parser.add_argument('--storage_path', type=str, default='extracted_data', help='数据存储路径')
    args = parser.parse_args()

    # 存储路径
    storage_path = os.path.join(args.storage_path, f'demonstration_{args.demo_num}')
    print(f"数据将保存到: {storage_path}")

    # 创建仿真
    gym, sim, viewer, env, actor_handle, camera_handle, object_handle = create_sim()

    # 获取 root state tensor 用于重置方块
    actor_root_state = gym.acquire_actor_root_state_tensor(sim)
    root_state_tensor = gymtorch.wrap_tensor(actor_root_state).view(-1, 13)
    object_idx = gym.get_actor_index(env, object_handle, gymapi.DOMAIN_SIM)

    # 创建录制器
    recorder = DataRecorder(storage_path)

    # 初始化
    joint_positions = HOME_POSITION.copy()
    commanded_positions = HOME_POSITION.copy()
    pos_tensor = torch.tensor(joint_positions, dtype=torch.float32)
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_tensor))

    # 启动图像张量访问
    gym.start_access_image_tensors(sim)

    current_finger = 0
    print_status(current_finger, joint_positions, recorder)

    # 长按状态
    push_held = False

    # 主循环
    while not gym.query_viewer_has_closed(viewer):
        # 处理键盘事件
        events = gym.query_viewer_action_events(viewer)
        for event in events:
            action = event.action

            # 长按 F 键：按下开始推，松开停止
            if action == "push":
                push_held = (event.value != 0)
                continue

            if event.value == 0:
                continue

            joints = FINGER_JOINTS[current_finger]
            updated = False

            if action == "finger_1":
                current_finger = 0; updated = True
            elif action == "finger_2":
                current_finger = 1; updated = True
            elif action == "finger_3":
                current_finger = 2; updated = True
            elif action == "finger_4":
                current_finger = 3; updated = True
            elif action == "a":
                joint_positions[joints[0]] -= STEP_SIZE; updated = True
            elif action == "d":
                joint_positions[joints[0]] += STEP_SIZE; updated = True
            elif action == "w":
                joint_positions[joints[1]] += STEP_SIZE; updated = True
            elif action == "s":
                joint_positions[joints[1]] -= STEP_SIZE; updated = True
            elif action == "q":
                joint_positions[joints[2]] += STEP_SIZE; updated = True
            elif action == "e":
                joint_positions[joints[2]] -= STEP_SIZE; updated = True
            elif action == "z":
                joint_positions[joints[3]] += STEP_SIZE; updated = True
            elif action == "x":
                joint_positions[joints[3]] -= STEP_SIZE; updated = True
            elif action == "reset":
                joint_positions = HOME_POSITION.copy(); updated = True
            elif action == "open":
                joint_positions = smooth_move(gym, sim, viewer, env, camera_handle, recorder,
                                             joint_positions, OPEN_POSITION, commanded_positions, steps=40).copy()
                updated = True
            elif action == "close":
                joint_positions = smooth_move(gym, sim, viewer, env, camera_handle, recorder,
                                             joint_positions, CLOSE_POSITION, commanded_positions, steps=40).copy()
                updated = True
            elif action == "grasp":
                # 抓取序列
                joint_positions = smooth_move(gym, sim, viewer, env, camera_handle, recorder,
                                             joint_positions, OPEN_POSITION, commanded_positions, steps=40).copy()
                joint_positions = smooth_move(gym, sim, viewer, env, camera_handle, recorder,
                                             joint_positions, CLOSE_POSITION * 0.7, commanded_positions, steps=60).copy()
                updated = True
            elif action == "toggle_record":
                if recorder.is_recording:
                    recorder.pause_recording()
                else:
                    recorder.start_recording()
                updated = True
            elif action == "reset_record":
                recorder.reset_recording()
                updated = True
            elif action == "reset_scene":
                # 重置手到初始姿态
                joint_positions = HOME_POSITION.copy()
                commanded_positions = HOME_POSITION.copy()
                # 重置方块到初始位置
                root_state_tensor[object_idx, 0:3] = torch.tensor([1.0, 1.3, 0.06])
                root_state_tensor[object_idx, 3:7] = torch.tensor([-1.3, -0.707, 0.0, 0.0])
                root_state_tensor[object_idx, 7:13] = 0  # 清零速度
                object_indices = torch.tensor([object_idx], dtype=torch.int32)
                gym.set_actor_root_state_tensor_indexed(sim,
                    gymtorch.unwrap_tensor(root_state_tensor),
                    gymtorch.unwrap_tensor(object_indices), 1)
                print("\n🔄 场景已重置!")
                updated = True

            if updated:
                joint_positions = np.clip(joint_positions, JOINT_LOWER, JOINT_UPPER)
                commanded_positions = joint_positions.copy()
                print_status(current_finger, joint_positions, recorder)

        # 长按 F：持续弯曲当前手指所有关节（推动方块）
        if push_held:
            joints = FINGER_JOINTS[current_finger]
            joint_positions[joints[1]] += STEP_SIZE * 0.3
            joint_positions[joints[2]] += STEP_SIZE * 0.3
            joint_positions[joints[3]] += STEP_SIZE * 0.2
            joint_positions = np.clip(joint_positions, JOINT_LOWER, JOINT_UPPER)
            commanded_positions = joint_positions.copy()

        # 仿真步进
        pos_tensor = torch.tensor(joint_positions, dtype=torch.float32)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_tensor))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

        # 录制当前帧
        if recorder.is_recording:
            rgb, depth = get_camera_images(gym, sim, env, camera_handle)
            actual = get_actual_joint_positions(gym, env)
            sim_time = gym.get_elapsed_time(sim)
            recorder.record_frame(rgb, depth, actual, joint_positions.copy(), sim_time)

    # 保存数据
    recorder.save()

    # 清理
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("\nDone!")


if __name__ == "__main__":
    main()
