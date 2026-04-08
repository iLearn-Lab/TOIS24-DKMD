#!/usr/bin/env bash
# ./train.sh <gpu_id> <task_name> <model_file> <output_file>
# nohup 不挂断的运行命令
# >表示覆盖式重定向，正常输出是把内容输出到显示器上，重定向是把内容输出到文件中
# >>表示追加式重定向。command>>xxx.log，将输出重定向追加到XXX.log文件中
# 2>&1 2是标准错误输出，1是标准输出，这里的&表示引用的意思，对标准输出的引用。这个命令表示将标准错误输出也重定向到标准输出指向的文件中。
# 所以这个命令是首先指定GPU卡号，运行python程序，2和3是输入，将输出打印到4中。
# nohup python train.py  result.txt 2>&1  & 
CUDA_VISIBLE_DEVICES=$1 nohup python -u train.py $2 $3 >> $4 2>&1 &  # -u要保留！ python -u参数的作用，是取消stdout和stderr的缓存，如果执行脚本中有向stdout和stderr的输出，无论在什么情况下，都不再有缓冲。
# CUDA_VISIBLE_DEVICES=1,0 nohup python -u train.py $1 $2 >> $3 2>&1 &
# CUDA_VISIBLE_DEVICES=$1 nohup python -u -W ignore train.py $2 $3 >> $4 2>&1 &
