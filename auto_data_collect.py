"""
自动数据采集脚本
自动执行多种方块翻转动作并录制数据，无需手动操作。

运行方式:
    cd Open-Teach
    export LD_LIBRARY_PATH=/home/hu/miniconda3/envs/openteach_v2/lib
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    export PATH=/home/hu/miniconda3/envs/openteach_v2/bin:$PATH
    /home/hu/miniconda3/envs/openteach_v2/bin/python auto_data_collect.py --num_demos 10

生成数据:
    extracted_data/demonstration_1/
    extracted_data/demonstration_2/
    ...
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
import random

# ============================================================
# 配置
# ============================================================
SIM_FPS = 60
IMAGE_SIZE = (480, 480)

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


# ============================================================
# 动作序列生成器
# ============================================================
def generate_flip_trajectory(variation=0):
    """生成一个抓取方块的关节轨迹
    任务：手从张开状态逐渐握住方块
    """
    speed_factor = random.uniform(0.8, 1.4)
    noise_scale = random.uniform(0.01, 0.02)
    
    if variation % 5 == 0:
        traj = _grasp_full_hand(speed_factor, noise_scale)
    elif variation % 5 == 1:
        traj = _grasp_pinch(speed_factor, noise_scale)
    elif variation % 5 == 2:
        traj = _grasp_three_finger(speed_factor, noise_scale)
    elif variation % 5 == 3:
        traj = _grasp_slow_close(speed_factor, noise_scale)
    elif variation % 5 == 4:
        traj = _grasp_open_close_open(speed_factor, noise_scale)
    
    return traj


def _interpolate(start, end, steps):
    frames = []
    for i in range(steps):
        t = (i + 1) / steps
        # 使用平滑插值（ease in-out）
        t_smooth = t * t * (3 - 2 * t)
        frame = start + t_smooth * (end - start)
        frames.append(frame.copy())
    return frames


def _add_noise(trajectory, scale):
    noisy = []
    for frame in trajectory:
        noise = np.random.randn(16) * scale
        noisy_frame = np.clip(frame + noise, JOINT_LOWER, JOINT_UPPER)
        noisy.append(noisy_frame)
    return noisy


def _grasp_full_hand(speed, noise):
    """全手抓取：张开 -> 握住"""
    traj = []
    
    # 张开姿态
    OPEN = np.zeros(16)
    OPEN[12] = 0.3
    
    # 抓取姿态（所有手指弯曲握住方块）
    GRASP = np.array([
        0.0, 1.2, 1.3, 1.0,      # 食指
        0.0, 1.2, 1.3, 1.0,      # 中指
        0.0, 1.2, 1.3, 1.0,      # 无名指
        1.2, 0.8, 1.2, 1.0       # 拇指
    ]) + np.random.randn(16) * 0.05
    GRASP = np.clip(GRASP, JOINT_LOWER, JOINT_UPPER)
    
    # 保持张开 (30帧)
    for _ in range(int(30 * speed)):
        traj.append(OPEN.copy())
    
    # 慢慢握住 (80帧)
    steps = int(80 * speed)
    traj.extend(_interpolate(OPEN, GRASP, steps))
    
    # 保持握住 (50帧)
    for _ in range(int(50 * speed)):
        traj.append(GRASP.copy())
    
    # 松开 (40帧)
    steps = int(40 * speed)
    traj.extend(_interpolate(GRASP, OPEN, steps))
    
    return _add_noise(traj, noise)


def _grasp_pinch(speed, noise):
    """捏取：拇指+食指"""
    traj = []
    
    OPEN = np.zeros(16)
    OPEN[12] = 0.3
    
    PINCH = np.zeros(16)
    PINCH[0] = random.uniform(-0.2, 0.2)
    PINCH[1] = 1.3 + random.uniform(-0.1, 0.1)
    PINCH[2] = 1.4 + random.uniform(-0.1, 0.1)
    PINCH[3] = 1.1
    PINCH[12] = 1.3
    PINCH[13] = 0.9 + random.uniform(-0.1, 0.1)
    PINCH[14] = 1.3 + random.uniform(-0.1, 0.1)
    PINCH[15] = 1.1
    PINCH = np.clip(PINCH, JOINT_LOWER, JOINT_UPPER)
    
    for _ in range(int(25 * speed)):
        traj.append(OPEN.copy())
    
    steps = int(70 * speed)
    traj.extend(_interpolate(OPEN, PINCH, steps))
    
    for _ in range(int(60 * speed)):
        traj.append(PINCH.copy())
    
    steps = int(40 * speed)
    traj.extend(_interpolate(PINCH, OPEN, steps))
    
    return _add_noise(traj, noise)


def _grasp_three_finger(speed, noise):
    """三指抓取：食指+中指+拇指"""
    traj = []
    
    OPEN = np.zeros(16)
    OPEN[12] = 0.3
    
    GRASP = np.zeros(16)
    GRASP[1] = 1.2 + random.uniform(-0.1, 0.1)
    GRASP[2] = 1.3 + random.uniform(-0.1, 0.1)
    GRASP[3] = 1.0
    GRASP[5] = 1.2 + random.uniform(-0.1, 0.1)
    GRASP[6] = 1.3 + random.uniform(-0.1, 0.1)
    GRASP[7] = 1.0
    GRASP[12] = 1.2
    GRASP[13] = 0.8
    GRASP[14] = 1.2
    GRASP[15] = 0.9
    GRASP = np.clip(GRASP, JOINT_LOWER, JOINT_UPPER)
    
    for _ in range(int(20 * speed)):
        traj.append(OPEN.copy())
    
    steps = int(75 * speed)
    traj.extend(_interpolate(OPEN, GRASP, steps))
    
    for _ in range(int(55 * speed)):
        traj.append(GRASP.copy())
    
    steps = int(35 * speed)
    traj.extend(_interpolate(GRASP, OPEN, steps))
    
    return _add_noise(traj, noise)


def _grasp_slow_close(speed, noise):
    """非常慢地合拢所有手指"""
    traj = []
    
    OPEN = np.zeros(16)
    OPEN[12] = 0.3
    
    GRASP = np.array([
        0.0, 1.0, 1.1, 0.8,
        0.0, 1.0, 1.1, 0.8,
        0.0, 1.0, 1.1, 0.8,
        1.0, 0.7, 1.0, 0.8
    ]) + np.random.randn(16) * 0.05
    GRASP = np.clip(GRASP, JOINT_LOWER, JOINT_UPPER)
    
    for _ in range(int(30 * speed)):
        traj.append(OPEN.copy())
    
    # 非常慢地握
    steps = int(120 * speed)
    traj.extend(_interpolate(OPEN, GRASP, steps))
    
    for _ in range(int(60 * speed)):
        traj.append(GRASP.copy())
    
    steps = int(50 * speed)
    traj.extend(_interpolate(GRASP, OPEN, steps))
    
    return _add_noise(traj, noise)


def _grasp_open_close_open(speed, noise):
    """张开-握住-松开-再握住"""
    traj = []
    
    OPEN = np.zeros(16)
    OPEN[12] = 0.3
    
    GRASP = np.array([
        0.0, 1.3, 1.4, 1.1,
        0.0, 1.3, 1.4, 1.1,
        0.0, 1.3, 1.4, 1.1,
        1.2, 0.9, 1.3, 1.0
    ]) + np.random.randn(16) * 0.04
    GRASP = np.clip(GRASP, JOINT_LOWER, JOINT_UPPER)
    
    HALF = OPEN + (GRASP - OPEN) * 0.4
    
    # 张开
    for _ in range(int(15 * speed)):
        traj.append(OPEN.copy())
    
    # 半握
    steps = int(40 * speed)
    traj.extend(_interpolate(OPEN, HALF, steps))
    
    # 松开一点
    steps = int(25 * speed)
    traj.extend(_interpolate(HALF, OPEN, steps))
    
    # 完全握住
    steps = int(60 * speed)
    traj.extend(_interpolate(OPEN, GRASP, steps))
    
    # 保持
    for _ in range(int(40 * speed)):
        traj.append(GRASP.copy())
    
    # 松开
    steps = int(35 * speed)
    traj.extend(_interpolate(GRASP, OPEN, steps))
    
    return _add_noise(traj, noise)


# ============================================================
# 数据录制器 (同 keyboard_data_collect.py)
# ============================================================
class DataRecorder:
    def __init__(self, storage_path):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

        video_path = os.path.join(storage_path, 'cam_0_rgb_video.avi')
        self.video_writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*'XVID'), SIM_FPS, IMAGE_SIZE
        )
        self.rgb_timestamps = []
        self.depth_frames = []
        self.depth_timestamps = []
        self.joint_states = []
        self.joint_timestamps = []
        self.commanded_states = []
        self.commanded_timestamps = []
        self.num_frames = 0

    def record_frame(self, rgb_image, depth_image, actual_joints, commanded_joints, sim_time):
        self.video_writer.write(rgb_image)
        self.rgb_timestamps.append(sim_time)
        depth_uint16 = (depth_image * 1000).astype(np.uint16)
        self.depth_frames.append(depth_uint16)
        self.depth_timestamps.append(sim_time)
        self.joint_states.append(actual_joints.copy())
        self.joint_timestamps.append(sim_time)
        self.commanded_states.append(commanded_joints.copy())
        self.commanded_timestamps.append(sim_time)
        self.num_frames += 1

    def save(self):
        if self.num_frames == 0:
            print("  ⚠️  没有数据")
            return
        
        self.video_writer.release()
        
        # metadata
        metadata = {'num_frames': self.num_frames, 'timestamps': self.rgb_timestamps, 'fps': SIM_FPS}
        with open(os.path.join(self.storage_path, 'cam_0_rgb_video.metadata'), 'wb') as f:
            pickle.dump(metadata, f)
        
        # depth
        with h5py.File(os.path.join(self.storage_path, 'cam_0_depth.h5'), 'w') as f:
            f.create_dataset('depth_images', data=np.array(self.depth_frames, dtype=np.uint16), compression='gzip', compression_opts=6)
            f.create_dataset('timestamps', data=np.array(self.depth_timestamps, dtype=np.float64), compression='gzip', compression_opts=6)
        
        # joint states
        with h5py.File(os.path.join(self.storage_path, 'allegro_joint_states.h5'), 'w') as f:
            f.create_dataset('positions', data=np.array(self.joint_states, dtype=np.float32), compression='gzip', compression_opts=6)
            f.create_dataset('timestamps', data=np.array(self.joint_timestamps, dtype=np.float64), compression='gzip', compression_opts=6)
        
        # commanded states
        with h5py.File(os.path.join(self.storage_path, 'allegro_commanded_joint_states.h5'), 'w') as f:
            f.create_dataset('positions', data=np.array(self.commanded_states, dtype=np.float32), compression='gzip', compression_opts=6)
            f.create_dataset('timestamps', data=np.array(self.commanded_timestamps, dtype=np.float64), compression='gzip', compression_opts=6)

        print(f"  ✅ 保存完成: {self.num_frames} 帧")


# ============================================================
# 仿真
# ============================================================
def create_sim(headless=False):
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
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)

    viewer = None
    if not headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(1.06, 1.6, -0.02)
        cam_target = gymapi.Vec3(1.03, 1.3, -0.02)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # 资产
    asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "openteach/components/environment/assets/urdf/")

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = False
    asset_options.use_mesh_materials = True
    asset_options.disable_gravity = True

    table_options = gymapi.AssetOptions()
    table_options.fix_base_link = True
    table_options.collapse_fixed_joints = True
    table_options.disable_gravity = False

    hand_asset = gym.load_urdf(sim, asset_root, "allegro_hand_description/urdf/model_only_hand.urdf", asset_options)
    table_asset = gym.load_urdf(sim, asset_root, "allegro_hand_description/urdf/table.urdf", table_options)
    object_asset = gym.load_urdf(sim, asset_root, "objects/cube_multicolor.urdf", gymapi.AssetOptions())

    spacing = 2.5
    env = gym.create_env(sim, gymapi.Vec3(-spacing, 0, -spacing), gymapi.Vec3(spacing, spacing, spacing), 1)

    # 相机
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 100
    camera_props.width = IMAGE_SIZE[0]
    camera_props.height = IMAGE_SIZE[1]
    camera_props.enable_tensors = True
    camera_handle = gym.create_camera_sensor(env, camera_props)
    gym.set_camera_location(camera_handle, env, gymapi.Vec3(1.06, 1.6, -0.02), gymapi.Vec3(1.03, 1.3, -0.02))

    # Actors
    actor_pose = gymapi.Transform()
    actor_pose.p = gymapi.Vec3(1, 1.2, 0)
    actor_pose.r = gymapi.Quat(-0.707, -0.707, 0, 0)
    actor_handle = gym.create_actor(env, hand_asset, actor_pose, "hand", 0, 1)

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.7, 0.0, 0.3)
    table_pose.r = gymapi.Quat(-0.707107, 0, 0.0, 0.707)
    gym.create_actor(env, table_asset, table_pose, "table", 0, 1)

    object_pose = gymapi.Transform()
    object_pose.p = gymapi.Vec3(1, 1.3, 0.06)
    object_pose.r = gymapi.Quat(-1.3, -0.707, 0, 0)
    object_handle = gym.create_actor(env, object_asset, object_pose, "cube", 0, 0, 0)

    # 保存初始状态用于场景重置
    initial_object_pos = [1.0, 1.3, 0.06]
    initial_object_rot = [-1.3, -0.707, 0.0, 0.0]

    # 手颜色
    num_dofs = gym.get_asset_dof_count(hand_asset)
    for j in range(num_dofs + 13):
        if j != 20 and j != 15 and j != 10 and j != 5:
            gym.set_rigid_body_color(env, actor_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.15, 0.15, 0.15))

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

    # root state for reset
    actor_root_state = gym.acquire_actor_root_state_tensor(sim)
    root_state_tensor = gymtorch.wrap_tensor(actor_root_state).view(-1, 13)
    object_idx = gym.get_actor_index(env, object_handle, gymapi.DOMAIN_SIM)

    gym.start_access_image_tensors(sim)

    return gym, sim, viewer, env, actor_handle, camera_handle, root_state_tensor, object_idx


def reset_scene(gym, sim, env, root_state_tensor, object_idx):
    """重置方块到初始位置"""
    root_state_tensor[object_idx, 0:3] = torch.tensor([1.0, 1.3, 0.06])
    root_state_tensor[object_idx, 3:7] = torch.tensor([-1.3, -0.707, 0.0, 0.0])
    root_state_tensor[object_idx, 7:13] = 0
    object_indices = torch.tensor([object_idx], dtype=torch.int32)
    gym.set_actor_root_state_tensor_indexed(sim,
        gymtorch.unwrap_tensor(root_state_tensor),
        gymtorch.unwrap_tensor(object_indices), 1)


def run_demo(gym, sim, viewer, env, actor_handle, camera_handle, root_state_tensor, object_idx, trajectory, recorder):
    """执行一个演示轨迹并录制"""
    # 重置场景
    reset_scene(gym, sim, env, root_state_tensor, object_idx)
    
    # 先设置home并稳定几帧
    pos = torch.tensor(HOME_POSITION, dtype=torch.float32)
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos))
    for _ in range(30):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        if viewer:
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

    # 执行轨迹并录制
    for frame_joints in trajectory:
        frame_joints = np.clip(frame_joints, JOINT_LOWER, JOINT_UPPER)
        pos = torch.tensor(frame_joints, dtype=torch.float32)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos))
        
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        
        if viewer:
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
            if gym.query_viewer_has_closed(viewer):
                return False

        # 录制
        color_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handle, gymapi.IMAGE_COLOR)
        rgb = gymtorch.wrap_tensor(color_tensor).cpu().numpy()[:, :, [2, 1, 0]]
        
        depth_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handle, gymapi.IMAGE_DEPTH)
        depth = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()
        
        actual = np.zeros(16)
        for i in range(16):
            actual[i] = gym.get_dof_position(env, i)
        
        sim_time = gym.get_elapsed_time(sim)
        recorder.record_frame(rgb, depth, actual, frame_joints, sim_time)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_demos', type=int, default=10, help='要生成的演示数量')
    parser.add_argument('--storage_path', type=str, default='extracted_data', help='存储路径')
    parser.add_argument('--headless', action='store_true', help='无窗口模式（更快）')
    parser.add_argument('--start_from', type=int, default=1, help='起始编号')
    args = parser.parse_args()

    print("=" * 60)
    print(f"  自动数据采集: 生成 {args.num_demos} 个演示")
    print(f"  存储路径: {args.storage_path}/")
    print(f"  模式: {'无窗口(快速)' if args.headless else '有窗口(可视化)'}")
    print("=" * 60)

    # 创建仿真
    gym, sim, viewer, env, actor_handle, camera_handle, root_state_tensor, object_idx = create_sim(headless=args.headless)

    # 生成演示
    for demo_id in range(args.start_from, args.start_from + args.num_demos):
        storage = os.path.join(args.storage_path, f'demonstration_{demo_id}')
        recorder = DataRecorder(storage)
        
        # 生成轨迹
        trajectory = generate_flip_trajectory(variation=demo_id - 1)
        
        print(f"\n  [{demo_id}/{args.start_from + args.num_demos - 1}] 录制演示 {demo_id} ({len(trajectory)} 帧)...")
        
        success = run_demo(gym, sim, viewer, env, actor_handle, camera_handle, 
                          root_state_tensor, object_idx, trajectory, recorder)
        
        if not success:
            print("  窗口关闭，停止采集")
            break
        
        recorder.save()

    # 清理
    if viewer:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    
    print(f"\n{'=' * 60}")
    print(f"  采集完成! 共 {args.num_demos} 个演示")
    print(f"  数据在: {args.storage_path}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
