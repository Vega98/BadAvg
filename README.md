# [BadAvg](link-to-paper)


This repository contains the code of [BadAvg](link-to-paper), an aggregation-aware backdoor attack on Federated Contrastive Learning (FCL) built on [BadEncoder](https://github.com/jinyuan-jia/BadEncoder) that integrates the federated averaging process into the attack optimization to achieve high effectiveness, stealth, and efficiency while evading state-of-the-art defenses. Here is an overview of BadAvg: 



<div align="center">
<img width="100%" alt="BadEncoder Illustration" src="figure.png">
</div>

## Citation

If you use this code, please cite the following [paper](link-to-paper):
```
paper
```


## Required python packages

Our code is tested under the following environment: Ubuntu 24.04.2 LTS, Python 3.13.7.

To set up the Python environment, first create and activate a virtual environment (e.g., with python ``` -m venv .venv and source .venv/bin/activate ``` on Linux/Mac or ```.venv\Scripts\activate``` on Windows). Then install all required dependencies by running:

``` 
pip install -r requirements.txt
``` 

## Running a federated experiment

The file run_federated.py is used to run a federated experiment, pre-training an image encoder on the selected dataset partitions and applying the BadAvg attack if specified.

To control the parameters of the federated experiment, you have to edit these specific code lines. These are the main knobs to turn:

235 ```num_rounds = ...``` is the number of overall federated rounds.
236 ```bad_round = ...``` run poison attack every ```bad_round``` rounds. Set 0 if you don't want to perform any attack.
245 ```base_output_dir = ...``` directory of the results of the federated experiment (logs, weights, plots...)
328&344 ```dataset_paths = ...``` to choose which dataset partitions to use respectively for clean and attack rounds. E.g. ```./data/cifar10/partitions/iid/``` will use CIFAR-10 iid partitions, ```./data/stl10/partitions/dirichlet/``` will use STL-10 non-iid partitions and so on...

You can edit further the code to further customize the federated experiment (number of client rounds, type of attack, if to apply defense or not etc...)

Before running an experiment on CIFAR10 or STL10, you could first download the pre-partitioned data from the following link [data](drive-link) (put the data folder in the main directory). Then, you could run the run_federated.py script: 

```
python3 scripts/run_federated.py
```


## Training downstream classifiers

The file training\_downstream\_classifier.py can be used to train a downstream classifier on a downstream task using an image encoder. Here is an example scripts:

```
python3 scripts/run_cifar10_training_downstream_classifier.py
```


## Experimental results
...



We refer to the following code in our implementation:
https://github.com/jinyuan-jia/BadEncoder




