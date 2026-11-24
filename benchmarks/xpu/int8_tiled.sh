#!/usr/bin/env bash
cd "$(dirname "$0")"; export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0} HF_HUB_OFFLINE=1
P=../../.venv-xpu/bin/python
$P flux_int8.py --tag bf16_tiled 2>&1 | grep -E "^bf16|vs|Error|error"
$P flux_int8.py --tag int8_tiled --int8 2>&1 | grep -E "^int8|vs|Error|error"
SUB=variants/sd DEV=xpu $P flux_phases.py --compile 2>&1 | grep -E "^compile|^  [a-zDt ]|Error"
echo TILED DONE
