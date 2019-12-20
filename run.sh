#!/bin/bash
lambda1_array=(1.0 1.0 1.0)
task='link'
prefix="filter"
epochs=500
for lambda1 in ${lambda1_array[@]}
do
    name="${prefix}_${lambda1}"
    rm -rf summary
    rm -rf model
    rm -f nohup.out
    rm -f log
    mkdir model
    if [ $task = 'link' ]; then
        python run.py --train --epochs $epochs --task link_predict --lambda1 $lambda1
    elif [ $task = 'node' ]; then
        python run.py --train --epochs $epochs --task node_classify --lambda1 $lambda1
    elif [ $task = 'all' ]; then
        python run.py --train --epochs $epochs --task all --lambda1 $lambda1
    fi
    mv summary $name
    mv log $name
    mv model $name/.
    mkdir ${name}/${name}
    mv *.csv ${name}/${name}
    echo "$name finish"
done
