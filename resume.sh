#!/bin/bash
lambda1_array=(1.0 1.0 1.0)
task='link'
prefix="filter"
epoch_base=30
epochs=20
for lambda1 in ${lambda1_array[@]}
do
    name="${prefix}_${lambda1}"
    rm -rf model
    rm -rf summary
    rm -f nohup.out
    rm -f log
    mv ${name}/log .
    mv ${name}/model .
    rm -rf ${name}/${name}
    mv ${name} summary
    if [ $task = 'link' ]; do
        python run.py --resume --task link_predict --epochs $epochs --epoch_base $epoch_base --lambda1 $lambda1
    elif [ $task = 'node' ]; do
        python run.py --resume --task node_classify --epochs $epochs --epoch_base $epoch_base --lambda1 $lambda1
    elif [ $task = 'all' ]; do
        python run.py --resume --task all --epochs $epochs --epoch_base $epoch_base --lambda1 $lambda1
    fi
    mv summary $name
    mv log $name
    mv model $name/.
    mkdir ${name}/${name}
    mv *.csv ${name}/${name}
    echo "$name finish"
done
