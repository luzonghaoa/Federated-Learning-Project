#!/bin/bash -i

#eval "$(conda shell.bash hook)"
conda activate pytorch
nohup top -b -n600 -d1 < /dev/null > top_out.txt 2>&1 &
python client.py 
