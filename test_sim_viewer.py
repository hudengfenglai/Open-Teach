"""
独立测试 Isaac Gym Allegro Hand 仿真环境（带 Viewer 窗口）
不需要 VR 连接，直接显示仿真画面。

运行方式:
    export LD_LIBRARY_PATH=/home/hu/miniconda3/envs/openteach_v2/lib
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    /home/hu/miniconda3/envs/openteach_v2/bin/python test_sim_viewer.py
"""

import os
os.environ['MESA_VK_DEVICE_SELECT'] = '10de:24b0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *
import numpy as np
import torch

print("=" * 60)
print("  Isaac Gym Allegro Hand Simulation Viewer Test")
print("=" * 60)

# 初始化 Gym
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

# 创建仿真
print("Creating simulation...")
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# 添加地面
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# 创建 Viewer
print("Creating viewer...")
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# 设置相机位置
cam_pos = gymapi.Vec3(1.06, 1.6, -0.02)
cam_target = gymapi.Vec3(1.03, 1.3, -0.02)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

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

print(f"Loading assets from: {asset_root}")
hand_asset = gym.load_urdf(sim, asset_root, asset_file, asset_options)
table_asset = gym.load_urdf(sim, asset_root, table_file, table_options)
object_asset = gym.load_urdf(sim, asset_root, object_file, object_options)

# 创建环境
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, env_lower, env_upper, 1)

# 放置 Allegro Hand
actor_pose = gymapi.Transform()
actor_pose.p = gymapi.Vec3(1, 1.2, 0)
actor_pose.r = gymapi.Quat(-0.707, -0.707, 0, 0)
actor_handle = gym.create_actor(env, hand_asset, actor_pose, "allegro_hand", 0, 1)

# 放置桌子
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.7, 0.0, 0.3)
table_pose.r = gymapi.Quat(-0.707107, 0, 0.0, 0.707)
table_handle = gym.create_actor(env, table_asset, table_pose, "table", 0, 1)

# 放置方块
object_pose = gymapi.Transform()
object_pose.p = gymapi.Vec3(1, 1.3, 0.06)
object_pose.r = gymapi.Quat(-1.3, -0.707, 0, 0)
object_handle = gym.create_actor(env, object_asset, object_pose, "cube", 0, 0, 0)

# 给手上色
num_dofs = gym.get_asset_dof_count(hand_asset)
for j in range(num_dofs + 13):
    if j != 20 and j != 15 and j != 10 and j != 5:
        gym.set_rigid_body_color(env, actor_handle, j, gymapi.MESH_VISUAL, 
                                 gymapi.Vec3(0.15, 0.15, 0.15))

# 设置关节属性
props = gym.get_actor_dof_properties(env, actor_handle)
props["stiffness"] = [3] * 16
props["damping"] = [0.18] * 16
props["friction"] = [0.01] * 16
props["armature"] = [0.001] * 16
props["velocity"] = [2.0] * 16
for k in range(num_dofs):
    props["driveMode"][k] = gymapi.DOF_MODE_POS
gym.set_actor_dof_properties(env, actor_handle, props)

# 设置初始姿态
home_position = torch.tensor([-0.00137183, -0.22922094, 0.7265581, 0.79128325,
                               0.9890924, 0.37431374, 0.36866143, 0.77558154,
                               0.00662423, -0.23064502, 0.73253167, 0.7449019,
                               0.08261403, -0.15844858, 0.82595366, 0.7666822])
gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(home_position))

print("\n" + "=" * 60)
print("  仿真窗口已启动!")
print("  - 鼠标左键拖动: 旋转视角")
print("  - 鼠标中键拖动: 平移视角")  
print("  - 滚轮: 缩放")
print("  - ESC 或关闭窗口: 退出")
print("=" * 60 + "\n")

# 主循环
while not gym.query_viewer_has_closed(viewer):
    # 仿真步进
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    # 渲染
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

# 清理
print("Closing simulation...")
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("Done!")
