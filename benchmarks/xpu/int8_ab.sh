#!/usr/bin/env bash
cd "$(dirname "$0")"; export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0} HF_HUB_OFFLINE=1
P=../../.venv-xpu/bin/python
$P flux_int8.py --tag bf16 2>&1 | grep -E "^(bf16|int8|eager)|Error"
$P flux_int8.py --tag bf16_rerun 2>&1 | grep -E "^(bf16|int8)|vs|Error"
$P flux_int8.py --tag int8 --int8 2>&1 | grep -E "^int8|vs|Error"
echo INT8 AB DONE
