# Modifications copyright (C) 2020 Bluefog Team. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function

from bluefog.common import topology_util
import bluefog.torch as bf
import argparse
import os
import sys
import warnings
import resnet
warnings.simplefilter('ignore')

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms
import tensorboardX
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

cwd_folder_loc = os.path.dirname(os.path.abspath(__file__))
# Training settings
parser = argparse.ArgumentParser(
    description="PyTorch CIFAR10 Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--model', type=str, default='resnet56',
                    help='model to benchmark')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=128,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01,
                    help='learning rate for a single GPU')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay')

parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument('--disable-dynamic-topology', action='store_true',
                    default=False, help=('Disable each iteration to transmit one neighbor ' +
                                         'per iteration dynamically.'))

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
allreduce_batch_size = args.batch_size

bf.init()
bf.set_topology(bf.ExponentialTwoGraph(bf.size()))
torch.manual_seed(args.seed)
if args.cuda:
    # Bluefog: pin GPU to local rank.
    device_id = bf.local_rank() if bf.nccl_built() else bf.local_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    torch.cuda.manual_seed(args.seed)
cudnn.benchmark = True

# Bluefog: print logs on the first worker.
verbose = 1 if bf.rank() == 0 else 0

# Bluefog: write TensorBoard logs on first worker.
log_writer = tensorboardX.SummaryWriter(
    args.log_dir) if bf.rank() == 0 else None


kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}
train_dataset = datasets.CIFAR10(
    os.path.join(cwd_folder_loc, "..", "data", "data-%d" % bf.rank()),
    train=True,
    download=True,
    transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010]),
    ]),
)
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=bf.size(), rank=bf.rank()
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=allreduce_batch_size, sampler=train_sampler, **kwargs
)

val_dataset = datasets.CIFAR10(
    os.path.join(cwd_folder_loc, "..", "data", "data-%d" % bf.rank()),
    train=False,
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010]),
    ])
)
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=bf.size(), rank=bf.rank()
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.val_batch_size, sampler=val_sampler, **kwargs
)

model = resnet.__dict__[args.model]()

if args.cuda:
    # Move model to GPU.
    model.cuda()

def add_weight_decay(model, weight_decay, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]
wd = args.wd
if wd:
    parameters = add_weight_decay(model, wd)
    wd = 0.
else:
    parameters = model.parameters()

# Bluefog: scale learning rate by the number of GPUs.
optimizer = optim.SGD(
    parameters,
    lr=(args.base_lr * bf.size()),
    momentum=args.momentum,
    weight_decay=wd,
)

# Bluefog: wrap optimizer with DistributedOptimizer.
optimizer = bf.DistributedAdaptThenCombineOptimizer(optimizer, model=model,
    communication_type=bf.CommunicationType.neighbor_allreduce)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
# Bluefog: broadcast parameters & optimizer state.
bf.broadcast_parameters(model.state_dict(), root_rank=0)
bf.broadcast_optimizer_state(optimizer, root_rank=0)


def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric("train_loss")
    train_accuracy = Metric("train_accuracy")

    with tqdm(total=len(train_loader), desc="Train Epoch     #{}".format(epoch + 1),
              disable=not verbose,) as t:
        for data, target in train_loader:
            if not args.disable_dynamic_topology:
                dynamic_topology_update()

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            train_accuracy.update(accuracy(output, target))
            loss = F.cross_entropy(output, target)
            train_loss.update(loss)
            loss.backward()
            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix(
                {
                    "loss": train_loss.avg.item(),
                    "accuracy": 100.0 * train_accuracy.avg.item(),
                }
            )
            t.update(1)

    if log_writer:
        log_writer.add_scalar("train/loss", train_loss.avg, epoch)
        log_writer.add_scalar("train/accuracy", train_accuracy.avg, epoch)
    return train_loss.avg, train_accuracy.avg


def validate(epoch):
    model.eval()
    val_loss = Metric("val_loss")
    val_accuracy = Metric("val_accuracy")

    with tqdm(total=len(val_loader), desc="Validate Epoch  #{}".format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix(
                    {
                        "loss": val_loss.avg.item(),
                        "accuracy": 100.0 * val_accuracy.avg.item(),
                    }
                )
                t.update(1)

    if log_writer:
        log_writer.add_scalar("val/loss", val_loss.avg, epoch)
        log_writer.add_scalar("val/accuracy", val_accuracy.avg, epoch)
    return val_loss.avg, val_accuracy.avg

if not args.disable_dynamic_topology:
    if bf.is_homogeneous() and bf.size() > bf.local_size():
        dynamic_neighbor_allreduce_gen = topology_util.GetInnerOuterExpo2DynamicSendRecvRanks(
            bf.size(),
            local_size=bf.local_size(),
            self_rank=bf.rank())
    else:
        dynamic_neighbor_allreduce_gen = topology_util.GetDynamicOnePeerSendRecvRanks(
            bf.load_topology(), bf.rank())

def dynamic_topology_update():
    send_neighbors, recv_neighbors = next(dynamic_neighbor_allreduce_gen)
    optimizer.dst_weights = send_neighbors
    optimizer.src_weights = {r: 1/(len(recv_neighbors) + 1) for r in recv_neighbors}
    optimizer.self_weight = 1 / (len(recv_neighbors) + 1)
    optimizer.enable_topo_check = False

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()

# Bluefog: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.0)  # pylint: disable=not-callable
        self.n = torch.tensor(0.0)  # pylint: disable=not-callable

    def update(self, val):
        self.sum += bf.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

if __name__ == "__main__":
    test_record = []
    for epoch in range(args.epochs):
        train_loss, train_acc = train(epoch)
        val_loss, val_acc = validate(epoch)
        test_record.append((val_loss, val_acc))
        scheduler.step()
    
    bf.barrier()
    if bf.rank() == 0:
        print()
        for epoch, (loss, acc) in  enumerate(test_record):
            print(f'[Epoch {epoch+1:2d}] Loss: {loss}, acc: {acc*100}%')
