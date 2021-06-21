# H-Mem: Harnessing synaptic plasticity with Hebbian Memory Networks

This repository contains a PyTorch implementation of H-Mem from the paper ["H-Mem: Harnessing synaptic plasticity with Hebbian Memory Networks"](https://www.biorxiv.org/content/10.1101/2020.07.01.180372v2) for training in the bAbI question-answering tasks.

## Implementation Details

I followed the [original implementation](https://github.com/IGITUGraz/H-Mem) and the details mentioned in the paper. You can find all the hyperparameters in the `config.yaml` file.

## Usage

Run the following command

``python run_babi.py --data_dir=tasks_1-20_v1-2/en-10k --task_id=1 ``

You have to define the `data_dir` and `task_id` parameters to specify the dataset's directory (1k or 10k) and the task's id, respectively.

You can also run the "Memory-dependent memorization" version by setting the `read_before_write` parameter in the` config.yaml` file.

## References
* [H-Mem: Harnessing synaptic plasticity with Hebbian Memory Networks](https://www.biorxiv.org/content/10.1101/2020.07.01.180372v2)
* Part of the code is borrowed from https://github.com/thaihungle/SAM
* Part of the code is borrowed from https://github.com/anantzoid/Recurrent-Entity-Networks-pytorch
