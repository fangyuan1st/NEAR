#!/bin/bash
model_dir="/home/xiaotong/src/logs/filter_1.0_-0.001/model"
cp -r ${model_dir} .
python runevaluateattr.py --data_dir /home/xiaotong/data/ego-facebook