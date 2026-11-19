#!/usr/bin/env bash
# Reproduces every number in the README (CPU, ~1 h). Run from the project folder.
set -e
python -u finetune.py --mode lora --rank 8 > results/log-lora-r8.txt 2>&1
python -u finetune.py --mode head > results/log-head.txt 2>&1
python -u finetune.py --mode qlora --rank 8 > results/log-qlora-r8.txt 2>&1
python -u finetune.py --mode lora --rank 1 > results/log-lora-r1.txt 2>&1
python -u finetune.py --mode lora --rank 8 --n-train 1000 --name lora-r8-c_attn-n1000 > results/log-lora-r8-n1000.txt 2>&1
python -u compare_full_ft.py > results/log-full-vs-lora-minigpt.txt 2>&1
python -u merge_and_swap.py > results/log-merge-and-swap.txt 2>&1
