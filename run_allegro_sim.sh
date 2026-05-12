#!/bin/bash
# OpenTeach Allegro Sim 启动脚本
# 使用方法: ./run_allegro_sim.sh

# 设置环境变量
export LD_LIBRARY_PATH=/home/hu/miniconda3/envs/openteach_v2/lib
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PATH=/home/hu/miniconda3/envs/openteach_v2/bin:$PATH

# 激活 conda 环境并运行
echo "Starting Allegro Sim with Isaac Gym..."
echo "确保已修改 configs/network.yaml 中的 host_address 为本机 IP"
echo ""

/home/hu/miniconda3/envs/openteach_v2/bin/python teleop.py robot=allegro_sim sim_env=True
