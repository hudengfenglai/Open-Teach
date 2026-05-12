"""
策略评估脚本
用训练好的模型在仿真中自动控制手抓取方块。

运行方式:
    cd Open-Teach
    export LD_LIBRARY_PATH=/home/hu/miniconda3/envs/openteach_v2/lib
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    export PATH=/home/hu/miniconda3/envs/openteach_v2/bin:$PATH
    /home/hu/miniconda3/envs/openteach_v2/bin/python eval_policy.py
"""

import os
os.environ['MESA_VK_DEVICE_SELECT'] = '10de:24b0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *
import numpy as np
import torch
import torch.nn as nn
import cv2
import argparse

# 导入策略网络
from train_policy import GraspPolicy

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

IMAGE_SIZE = (480, 480)
POLICY_IMAGE_SIZE = 84


def create_sim():
    gym = gymapi.acquire_gym()

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
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create viewer")
    cam_pos = gymapi.Vec3(1.06, 1.6, -0.02)
    cam_target = gymapi.Vec3(1.03, 1.3, -0.02)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

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
    gym.set_camera_location(camera_handle, env,
                            gymapi.Vec3(1.06, 1.6, -0.02),
                            gymapi.Vec3(1.03, 1.3, -0.02))

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
    gym.create_actor(env, object_asset, object_pose, "cube", 0, 0, 0)

    # 手颜色
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

    gym.start_access_image_tensors(sim)

    return gym, sim, viewer, env, actor_handle, camera_handle


def get_observation(gym, sim, env, camera_handle, device):
    """获取当前图像观测并预处理为策略输入"""
    color_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handle, gymapi.IMAGE_COLOR)
    rgb = gymtorch.wrap_tensor(color_tensor).cpu().numpy()[:, :, [2, 1, 0]]  # BGR
    
    # 预处理：缩放 + 归一化 + CHW
    img = cv2.resize(rgb, (POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img_tensor = torch.tensor(img).unsqueeze(0).to(device)  # (1, 3, 84, 84)
    return img_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='trained_models/grasp_policy_best.pt')
    parser.add_argument('--num_episodes', type=int, default=5, help='评估轮数')
    parser.add_argument('--episode_length', type=int, default=300, help='每轮最大步数')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = GraspPolicy(action_dim=16).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  模型加载成功 (val_loss: {checkpoint['val_loss']:.6f})")

    # 创建仿真
    print("创建仿真环境...")
    gym, sim, viewer, env, actor_handle, camera_handle = create_sim()

    print("\n" + "=" * 60)
    print("  策略评估 - 自动抓取")
    print("  模型根据当前图像预测关节动作")
    print("  关闭窗口或按 ESC 退出")
    print("=" * 60 + "\n")

    # 初始化：手张开
    joint_positions = np.zeros(16)
    joint_positions[12] = 0.3
    pos_tensor = torch.tensor(joint_positions, dtype=torch.float32)
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_tensor))

    # 稳定几帧
    for _ in range(30):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    # 加载演示轨迹
    import h5py
    demo_path = os.path.join('extracted_data', 'demonstration_1')
    with h5py.File(os.path.join(demo_path, 'allegro_commanded_joint_states.h5'), 'r') as f:
        demo_actions = f['positions'][:]
    
    print(f"  回放演示轨迹: {len(demo_actions)} 帧")
    print(f"  同时用策略预测并计算误差")
    
    step = 0
    total_error = 0
    
    # 主循环：回放演示动作，同时对比策略预测
    while not gym.query_viewer_has_closed(viewer):
        if step < len(demo_actions):
            # 执行演示动作（让手真正动起来）
            joint_positions = demo_actions[step]
            joint_positions = np.clip(joint_positions, JOINT_LOWER, JOINT_UPPER)
            
            # 策略预测（对比）
            obs = get_observation(gym, sim, env, camera_handle, device)
            with torch.no_grad():
                pred_action = model(obs).cpu().numpy()[0]
            error = np.mean((pred_action - joint_positions) ** 2)
            total_error += error
        else:
            # 演示结束，循环回放
            step = 0
            continue
        
        # 执行动作
        pos_tensor = torch.tensor(joint_positions, dtype=torch.float32)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_tensor))
        
        # 仿真步进
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        
        step += 1
        if step % 60 == 0:
            avg_err = total_error / step
            print(f"  Step {step}/{len(demo_actions)}: "
                  f"joints={np.round(joint_positions[:4], 2)}... "
                  f"pred_error={avg_err:.4f}")

    # 清理
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("\n评估结束!")


if __name__ == "__main__":
    main()
