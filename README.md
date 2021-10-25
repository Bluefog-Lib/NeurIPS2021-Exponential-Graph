# Exponential Graph is Provably Efficient for Decentralized Deep Training

This code repository is for the paper
**Exponential Graph is Provably Efficient for Decentralized Deep Training** to
be appeared in NeurIPS 2021. If you feel this work helps in your research,
please consider cite

Bicheng Ying, Kun Yuan, Yiming Chen, Hanbin Hu, Pan Pan, and Wotao Yin,
**Exponential Graph is Provably Efficient for Decentralized Deep Training**.
Advances in Neural Information Processing Systems (NeurIPS), 2021.

```txt
@inproceedings{Ying_2021_ExpoGraph,
    title = {Exponential Graph is Provably Efficient for Decentralized Deep Training},
    author = {Bicheng Ying and Kun Yuan and Yiming Chen and Hanbin Hu and Pan Pan and Wotao Yin},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year = 2021
}
```

## Pre-requisites

This code was trained and tested with

1. Python 3.7.5
2. PyTorch 1.4.0
3. Torchvision 0.5.0
4. tqdm 4.62.3
5. tensorboardX 2.4
6. [bluefog](https://github.com/Bluefog-Lib/bluefog) 0.3.0

You can also use the Bluefog
[docker image](https://bluefog-lib.github.io/bluefog/docker.html) for testing.

## Static and One-peer exponential graphs comparison on CIFAR-10

Here gives a simple code snippet for running the CIFAR-10 experiments on
a decentralized network of size 4 using the one-peer exponential graph.
You can find the script for testing in `run.sh` as well.

```bash
$ bfrun -np 4 python train_cifar10.py --epochs 200 --model resnet56 --batch-size 128 --base-lr 0.01
```

### Expected performance on CIFAR-10 experiments

| Method  | Accuracy |
|--------|------|
| Static exponential graph | 91.21% |
| One-peer exponential graph | 91.84% |