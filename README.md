## Requirement
* Python 3.6.5
* tensorflow 1.9.0
* sklearn 0.19.2
* scipy 1.1.0

## Usage

Before running any scripts, "prepare" and "model" directory must exist in the current directory, if not, please use `mkdir prepare` and `mkdir model` to create.

1. Prepare data for training and testing 

    ### Cora Dataset

    ` python run.py --prepare --task all --dataset cora --data_dir <cora directory>`

    ### Citeseer Dataset
    ` python run.py --prepare --task all --dataset citeseer --data_dir <citeseer directory>`

    ### Pubmed Dataset
    ` python run.py --prepare --task all --dataset pubmed --data_dir <pubmed directory>`

    ### Facebook Dataset
    ` python run.py --prepare --task link_predict --dastaset facebook --data_dir <facebook directory>`

    ### Amazon Dataset
    ` python run.py --prepare --task link_predict --dastaset amazon --data_dir <amazon directory>`

    to add noise, please use `--noise 0.05` argument.

2. Train model (automatically predict on the last model and generate csv files)

   `python run.py --train --task <task> --lambda1 1.0 --epochs 500`

    \<task\> should be the same as in prepare stage.

   We recommend to train 500 epochs on Cora and Citeseer Dataset, 100epochs on Pubmed Dataset, 50 epochs on Facebook and Amazon Dataset.

   Log will be generated at current directory, where you can find train_loss, valid_loss, grid_loss for each epoch and the prediction result at the end of training.

   Summary and graph used by tensorboard will be generated in summary folder. 

   Two models(best_model, with the smallest valid loss, and model, the last model in training) will be saved in model folder. 
   
   Several .csv files will be generate in the current folder. In valid*.csv files, the first column on each line is the label of embedding, followed by the embedding. In test*.csv files, the first column on each line is the truth label and the second column is the predicted label by Logistic Regression followed by embedding.

3. Predict model (if necessary)
   
   `python run.py --predict --task <task> --model_dir <model_dir>`

   \<task\> should be the same as in prepare stage.

   \<model_dir\> is the directory where 'model' directory locates. By default, \<model_dir\> is the current directory.

4. Resume training (if necessary)
   
   `python run.py --resume --task <task> --epochs <epochs> --epoch_base <epoch_base>`

   \<task\> should be the same as in prepare stage. We recommend you to reprepare the dataset before every resume of traning to avoid trainging on different dataset. 

   \<epochs\> is the number of epochs to be further trained.

   \<epoch_base\> is the last number of epoch for the last trainning. Used in log.

## Argument

* --dataset 
    
    can be chosen from 'cora' 'citeseer' 'pubmed' 'facebook' 'amazon'

* --task

    can be chosen from 'link_predict' 'node_classify' 'all' 'none', should be same in prepare, train, predict, resume.

* --epochs

    how many epochs to train the model

* --noise
  
    add how many noise to the dataset. Noise is added by randomly change 0 attributes to 1 attributes.

* --embedsize

    the result dimensions of embedding, used in train, predict, resume.

* --lambda1

    parameter controlling struct_attr loss

## Different version and models

Ours: `git checkout filter/AA/dev`

onlyC: `git checkout onlyC/knot`

onlyC_attr: `git checkout onlyC/attr/dev`

nofilter: `git checkout nofilter`

## Notice

find.py view.py maybe adjusted to each version
resume.sh is not tested
noise filter's histogram is not tested
noise_filter attr_filter function in evaluate.py are not tested