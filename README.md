# On Using Certified Training towards Empirical Robustness

Codebase associated to "[On Using Certified Training towards Empirical Robustness](https://openreview.net/forum?id=UaaT2fI9DC)", showing that certified training using expressive losses ([DePalma2024](https://openreview.net/pdf?id=mzyZ4wzKlM)) as well as a relaxed version using the absolute value of the weights (*ForwAbs*) can help improve empirical robustness and prevent catastrophic overfitting.


If you use our code please cite the corresponding paper:

```bib
@article{
DePalma2025,
title={On Using Certified Training towards Empirical Robustness},
author={De Palma, Alessandro and Durand, Serge and Chihani, Zakaria and Terrier, François and Urban, Caterina},
journal={Transactions on Machine Learning Research},
year={2025},
url={https://openreview.net/forum?id=UaaT2fI9DC},
note={}
}
```

## Installation

Using mamba or conda you can first create the environment and install pytorch, with versions compatible with [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA/blob/bfb7997115c5e66327de81b43c42b23c8a0ffb1e/setup.py#L5)
```bash
conda create -n certified-training-emp-robustness python=3.8 'setuptools<70.0'
conda activate certified-training-emp-robustness
# pytorch
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
```

Install auto_LiRPA, used for the IBP training and autoattack for the AA evaluations.
```bash
git clone https://github.com/Verified-Intelligence/auto_LiRPA
cd auto_LiRPA
python setup.py install
cd ../certified-training-emp-robustness
pip install git+https://github.com/fra31/auto-attack
```

Then, install this code by running:
```bash
pip install .
```

Alternatively to the above process, one can use `conda env create -f env.yaml`, and then only proceed to install `auto_LiRPA` and `AutoAttack`.

Note that the two main scripts (`train.py` and `evaluate_autoattack.py`) optionally rely on `wandb` for logging purposes.
In order to activate its logging, use a non-null `--wandb_label` argument. 
By default, `wandb` operates in `offline` mode (this can be changed by setting `--wandb_options online`).
In order to use `wandb`, the `your-project` and `your-entity` strings to be fed to `wandb.init()` need to be appropriately replaced.
Finally, in order to operate in `offline` mode, `your_key` needs to be replaced by the user's key.

## Usage

### Training:

Example commands to run the experiments on a single seed across all methods (Exp-IBP, MTL-IBP, ForwAbs, N-FGSM, ELLE-A) on cifar10
with the short cyclic learning rate schedule, using preactresnet18, can be found in scripts/cifar10_cyclic.

Example commands for the long schedule are provided scripts/cifar10_long.

### AA Evaluation:

The training code performs a PGD-50 evaluation before terminating.
In order to instead compute the AutoAttack accuracy, a separate scripts needs to be run, using for instance the following command:
```bash
python evaluate_autoattack.py --config=config/cifar.json --eps 0.047058823529411764 --model=preactresnet18 --load path_to_model_ckpt --test_att_n_steps 50 --test_att_step_size 0.1 --test-batch-size 128
```
The `evaluate_autoattack.py` script proceeds by trying pgd-50 first on correctly classified inputs, then attempting a cheap verification with IBP, and finally using AutoAttack on the remaining inputs that were neither verified nor successfully attacked.
