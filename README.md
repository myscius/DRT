# DRT : Dynamic Replay Training for class-incremental learning
---

<p align="center">
  <a href="#Introduction">Introduction</a> •
  <a href="#Usage">Usage</a> •
  <a href="#Acknowledgments">Acknowledgments</a> •  
  <a href="#Contact">Contact</a> •
</p>

---


Created by Yan Yang from Nanjing University.\
This project is built upon the PyCIL library developed by Da-Wei Zhou. \
Link to the PyCIL library: [PyCIL: A Python Toolbox for Class-Incremental Learning](https://github.com/G-U-N/PyCIL)
## Introduction
DRT aims to reduce catastrophic forgetting caused by class imbalance in class-incremental learning by dynamically replaying the memory data.\
A more detailed abstract can be found in the paper. This code implements our method. We use Python and Pytorch for the experiments.
## Usage
The code has been tested on Linux (Linux version 3.10.0-1160.25.1.el7.x86_64)
### Dependencies
1. [torch 1.81](https://github.com/pytorch/pytorch)
2. [torchvision 0.6.0](https://github.com/pytorch/vision)
3. [tqdm](https://github.com/tqdm/tqdm)
4. [numpy](https://github.com/numpy/numpy)
5. [scipy](https://github.com/scipy/scipy)
6. [quadprog](https://github.com/quadprog/quadprog)

### Run experiment
1. Edit the `exps/[MODEL NAME].json` file for global settings.
2. As for our method, you can run:

```bash
python main.py --config=./exps/drt.json
```
### Datasets

The code has implemented the pre-processing of `CIFAR100` and `imagenet100`. When training on `CIFAR100`, this framework will automatically download it.  When training on `imagenet100`, you should specify the folder of your dataset in `utils/data.py`.

```python
    def download_data(self):
        assert 0,"You should specify the folder of your dataset"
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'
```

## Acknowledgments

We appreciate the open source code framework provided by the [PyCIL](https://github.com/G-U-N/PyCIL) project.

## Contact

If there are any questions, please feel free to propose new features by opening an issue  or contact with the author: **Yan Yang** (yangyan@smail.nju.edu.cn). Enjoy the code.
