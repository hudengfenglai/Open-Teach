#!/bin/bash
# Allegro Hand 键盘遥操作启动脚本
cd "$(dirname "$0")"

export LD_LIBRARY_PATH=/home/hu/miniconda3/envs/openteach_v2/lib
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PATH=/home/hu/miniconda3/envs/openteach_v2/bin:$PATH

echo "启动 Allegro Hand 键盘遥操作..."
echo ""
/home/hu/miniconda3/envs/openteach_v2/bin/python keyboard_teleop.py
