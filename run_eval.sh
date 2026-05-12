#!/bin/bash
# 策略评估启动脚本
cd "$(dirname "$0")"

export LD_LIBRARY_PATH=/home/hu/miniconda3/envs/openteach_v2/lib
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PATH=/home/hu/miniconda3/envs/openteach_v2/bin:$PATH

echo "=========================================="
echo "  策略评估 - 自动抓取"
echo "  模型: trained_models/grasp_policy_best.pt"
echo "=========================================="
echo ""

/home/hu/miniconda3/envs/openteach_v2/bin/python eval_policy.py "$@"
