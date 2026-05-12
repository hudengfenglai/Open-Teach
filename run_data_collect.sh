#!/bin/bash
# Allegro Hand 键盘数据采集启动脚本
# 用法: ./run_data_collect.sh [demo_num]
# 例如: ./run_data_collect.sh 1

cd "$(dirname "$0")"

DEMO_NUM=${1:-1}

export LD_LIBRARY_PATH=/home/hu/miniconda3/envs/openteach_v2/lib
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PATH=/home/hu/miniconda3/envs/openteach_v2/bin:$PATH

echo "=========================================="
echo "  Allegro Hand 键盘数据采集"
echo "  演示编号: $DEMO_NUM"
echo "  数据保存: extracted_data/demonstration_$DEMO_NUM/"
echo "=========================================="
echo ""
echo "操作提示:"
echo "  1. 窗口弹出后点击获取焦点"
echo "  2. 按 SPACE 开始录制"
echo "  3. 用键盘操作手完成任务"
echo "  4. 关闭窗口或按 ESC 保存数据"
echo ""

/home/hu/miniconda3/envs/openteach_v2/bin/python keyboard_data_collect.py --demo_num $DEMO_NUM
