#!/usr/bin/env python

import argparse
import os
import time

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from filelock import FileLock
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.utils import override
from torch.utils.data import DataLoader
from tqdm import trange

from models.convnet.safe_vgg16 import SafeVGG16
from utils.utils import zero_grad, AverageMeter, is_debug_session, load_config_yml

# Debugging options
# from pytorch_memlab import profile

# wandb
# enable dryrun to turn off wandb syncing completely
# os.environ['WANDB_MODE'] = 'dryrun'
# prevent wandb uploading pth to cloud
os.environ['WANDB_IGNORE_GLOBS'] = '*.pth'


def build_model(model_name, num_classes):
    if model_name == 'vgg16':
        return SafeVGG16(num_classes=num_classes)

    exit('{} model is not supported'.format(model_name))


def build_dataset(config):
    if config['dataset_name'] == "imagenet":
        # setup transforms
        transform = transforms.Compose([
            transforms.Resize(config.get("resolution")),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        train_dataset = torchvision.datasets.ImageNet(config['train_dataset_path'], transform=transform)
        val_dataset = torchvision.datasets.ImageNet(config['val_dataset_path'], transform=transform)
        return train_dataset, val_dataset

    if config['dataset_name'] == "cifar10":
        # setup transforms
        transform = transforms.Compose([
            transforms.Resize(config.get("resolution")),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = torchvision.datasets.CIFAR10(config['train_dataset_path'], transform=transform, download=True)
        val_dataset = torchvision.datasets.CIFAR10(config['train_dataset_path'], transform=transform, download=True)
        return train_dataset, val_dataset

    exit('{} dataset is not supported'.format(config['dataset_name']))


def initialization_hook():
    # torch
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    # cudnn.deterministic = False
    # torch.autograd.detect_anomaly()

    # nccl
    # os.environ["NCCL_DEBUG"] = "INFO"
    # print("NCCL DEBUG SET")
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo,eth1"
    os.environ["NCCL_LL_THRESHOLD"] = "0"


class ConvNetTrainingOperator(TrainingOperator):
    def setup(self, config):
        # setup wandb
        if self.world_rank == 0:
            wandb.init(project=os.getenv("PROJECT"), dir=os.getenv("LOG"), config=config, group=config['wandb_group'])

        # init dataset
        with FileLock(".ray.lock"):
            # init dataset under the lock because it may have to download dataset (in our case it doesnt)
            train_dataset, val_dataset = build_dataset(config)
            self.classes = train_dataset.classes

            # construct the models
            model = build_model(config['model_name'], len(self.classes))

        kwargs = {}
        if torch.cuda.is_available() and not is_debug_session():
            kwargs = {'num_workers': self.config['num_cpus_per_worker'], 'pin_memory': True}

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=config.get("train_batch_size"),
                                      shuffle=True,
                                      **kwargs)

        val_dataloader = DataLoader(val_dataset,
                                    batch_size=config.get("val_batch_size"),
                                    **kwargs)

        self.num_batches = len(train_dataloader)

        # setup optimizer
        # optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        optimizer = optim.SGD(model.parameters(), lr=self.config['learning_rate'], momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        self.model, self.optimizer, self.criterion = self.register(models=model,
                                                                   optimizers=optimizer,
                                                                   criterion=criterion)
        self.register_data(train_loader=train_dataloader, validation_loader=val_dataloader)

        # self.amp_enabled = self.config['use_fp16']
        # self.scaler = GradScaler(enabled=self.amp_enabled)

    @override(TrainingOperator)
    def train_batch(self, batch, batch_info):
        """Trains on one batch of data from the data creator."""

        step = batch_info["batch_idx"]
        start_time = time.time()
        images = batch[0]
        labels = batch[1]
        batch_size = images.shape[0]

        # Create non_blocking tensors for distributed training
        if self.use_gpu:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # forward
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        if self.world_rank == 0 and step % self.config['train_log_freq'] == 0:
            # logs
            _, predicted = torch.max(outputs, dim=1)

            # detach vars
            images = images.detach().cpu()
            labels = labels.detach().cpu().numpy()
            predicted = predicted.detach().cpu().numpy()

            # log random samples
            sample_idx = np.random.choice(self.config['train_batch_size'])
            sample_img = images[sample_idx]
            gt_caption = self.classes[labels[sample_idx]]
            predicted_caption = self.classes[predicted[sample_idx]]
            caption = 'Prediction: {}\nGround Truth: {}'.format(predicted_caption, gt_caption)

            # time
            end_time = time.time()
            log_dict = {
                "train/loss": loss.item(),
                "train/sample": wandb.Image(sample_img, caption=caption),
                "train/sec_per_kimg": ((end_time - start_time) * 1000) / (batch_size * self.config['num_workers'])
            }
            wandb.log(log_dict, step=self.global_step)

        stats = {
            "loss": loss.item()
        }
        return stats

    @override(TrainingOperator)
    def validate(self, val_dataloader, info=None):
        self.model.eval()
        log_sample = True
        log_dict = dict()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        with torch.no_grad():
            for samples in val_dataloader:
                images = samples[0]
                labels = samples[1]

                # Create non_blocking tensors for distributed training
                if self.use_gpu:
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

                # forward
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss_meter.update(loss.item())

                _, predicted = torch.max(outputs, dim=1)

                # detach vars
                images = images.detach().cpu()
                labels = labels.detach().cpu().numpy()
                predicted = predicted.detach().cpu().numpy()

                # log image in first iter only
                if log_sample and self.world_rank == 0:
                    # log random samples
                    sample_idx = np.random.choice(self.config['val_batch_size'])
                    sample_img = images[sample_idx]
                    gt_caption = self.classes[labels[sample_idx]]
                    predicted_caption = self.classes[predicted[sample_idx]]
                    caption = 'Prediction: {}\nGround Truth: {}'.format(predicted_caption, gt_caption)
                    log_dict['val/sample'] = wandb.Image(sample_img, caption=caption)
                    log_sample = False

                acc_meter.update(np.sum(predicted == labels), n=labels.shape[0])

        val_loss = loss_meter.avg
        val_acc = acc_meter.avg

        if self.world_rank == 0:
            log_dict['val/loss'] = val_loss
            log_dict['val/acc'] = val_acc
            wandb.log(log_dict, step=self.global_step)

        stats = {
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        return stats

    @override(TrainingOperator)
    def load_state_dict(self, state_dict):
        """Override this to load the representation of the operator state.
        Anything passed into self.register and self.register_data will
        automatically be loaded. Use this method to load any additional state.
        Args:
            state_dict (dict): State dict as returned by the operator. """

        # Note: If trainer.load is called optimizers state will be always loaded during self.register call
        # therefore, if we don't want optim state to be loaded we need to recreate them again
        if not self.config['load_optim_state']:
            pass


def train(args):
    # load config
    config = load_config_yml(args.config)

    # append config with ray args
    config['num_workers'] = args.num_workers
    config['num_cpus_per_worker'] = args.num_cpus_per_worker
    config['address'] = args.address
    config['use_gpu'] = args.use_gpu
    config['use_fp16'] = args.use_fp16
    config['use_tqdm'] = args.use_tqdm

    # setup wandb
    config['wandb_group'] = wandb.util.generate_id()
    wandb.init(project=os.getenv("PROJECT"), dir=os.getenv("LOG"), config=config, group=config['wandb_group'])

    # setup torch trainer
    trainer = TorchTrainer(
        training_operator_cls=ConvNetTrainingOperator,
        initialization_hook=initialization_hook,
        num_workers=args.num_workers,
        num_cpus_per_worker=args.num_cpus_per_worker,
        config=config,
        use_gpu=args.use_gpu,
        use_tqdm=args.use_tqdm,
        backend="nccl")

    # load checkpoints
    if config['train_restore_file']:
        trainer.load(config['train_restore_file'])
        val_stats = trainer.validate()
        print('Models successfully restored from {}'.format(config['train_restore_file']))
        print(val_stats)

    num_epochs = config.get("train_num_epochs")
    pbar = trange(num_epochs, unit="epoch")
    for itr in pbar:
        # read stats
        stats = trainer.train(info=dict(epoch_idx=itr, num_epochs=num_epochs))
        val_stats = trainer.validate(info=dict(epoch_idx=itr, num_epochs=num_epochs))

        # log
        pbar.set_postfix(dict(train_loss=stats["loss"], vaL_loss=val_stats["val_loss"], vaL_acc=val_stats["val_acc"]))

        # save models
        with torch.no_grad():
            checkpoint_filename = "{}_{}_{}.pth".format(config['model_name'], wandb.run.id, itr)
            checkpoint_path = os.path.join(os.getenv("LOG"), wandb.run.dir, checkpoint_filename)
            trainer.save(checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config YML")
    parser.add_argument("--address", type=str, help="Ray address to use to connect to a cluster.")
    parser.add_argument("--num-workers", type=int, default=1, help="Sets number of workers for training.")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="Enables GPU training")
    parser.add_argument("--use-fp16", action="store_true", default=False, help="Enables mixed precision training")
    parser.add_argument("--num-cpus-per-worker", type=int, default=1, help="Sets number of cpus per worker")
    parser.add_argument("--use-tqdm", action="store_true", default=False, help="Enables tqdm")
    args = parser.parse_args()
    ray.init(address=args.address, local_mode=is_debug_session())
    train(args)
