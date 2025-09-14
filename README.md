# [BadAvg](link-to-paper)


This repository contains the code of [BadAvg](link-to-paper), an aggregation-aware backdoor attack on Federated Contrastive Learning (FCL) built on [BadEncoder](https://github.com/jinyuan-jia/BadEncoder) that integrates the federated averaging process into the attack optimization to achieve high effectiveness, stealth, and efficiency while evading state-of-the-art defenses. Here is an overview of BadAvg: 



<div align="center">
<img width="100%" alt="BadAvg Illustration" src="BadAvg.png">
</div>

## Citation

If you use this code, please cite the following [paper](link-to-paper):
```
paper
```


## Required python packages

Our code is tested under the following environment: Ubuntu 24.04.2 LTS, Python 3.13.7.

To set up the Python environment, first create and activate a virtual environment (e.g., with ```python -m venv .venv``` and ```source .venv/bin/activate ``` on Linux/Mac or ```.venv\Scripts\activate``` on Windows). Then install all required dependencies by running:

``` 
pip install -r requirements.txt
``` 

## Running a Federated Experiment

The script `run_federated.py` launches a federated experiment: it pre-trains an image encoder on the chosen dataset partitions and (optionally) applies the BadAvg attack.

Control the experiment by editing the indicated lines in the script. The main knobs are:

- **Line 235** — `num_rounds = ...`  
  Number of federated rounds to run.

- **Line 236** — `bad_round = ...`  
  Run the poisoning attack every `bad_round` rounds. Set to `0` to disable the attack.

- **Line 245** — `base_output_dir = ...`  
  Directory where experiment outputs (logs, weights, plots) are saved.

- **Lines 328 & 344** — `dataset_paths = ...`  
  Choose which dataset partitions to use for clean vs. attack rounds. Examples:  
  ```text
  ./data/cifar10/partitions/iid/        # CIFAR-10 iid partitions
  ./data/stl10/partitions/dirichlet/   # STL-10 non-iid (Dirichlet) partitions
  
You can further customize the experiment by editing other parts of the script (number of clients per round, local training epochs, attack type, whether to enable defenses, etc.).

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




