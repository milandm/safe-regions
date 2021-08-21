#!/usr/bin/env python

import argparse
import os

import torch
import torch.nn as nn
import tqdm
import wandb
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from models.convnet.convnet_train import build_model
from utils.utils import AverageMeter, is_debug_session, load_config_yml, accuracy
from modules.saferegion_utils import collect_safe_regions_test_stats
from datasets.cifar10c import CIFAR10C


def build_dataset(config):
    if config['dataset_name'] == "cifar10":
        # setup transforms
        transform = transforms.Compose([
            transforms.Resize(config.get("resolution")),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_dataset = torchvision.datasets.CIFAR10(config['test_dataset_path'], train=False, transform=transform, download=True)
        return test_dataset

    if config['dataset_name'] == "cifar10c":
        # setup transforms
        transform = transforms.Compose([
            transforms.Resize(config.get("resolution")),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_dataset = CIFAR10C(config['test_dataset_path'],
                                corruptions=config['corruptions'],
                                severities=config['severities'],
                                train=False,
                                transform=transform)
        return test_dataset

    exit('{} dataset is not supported'.format(config['dataset_name']))


def test(config, use_gpu=True):
    # init wandb
    wandb.init(project=os.getenv("PROJECT"), dir=os.getenv("LOG"), config=config)

    # setup dataset
    test_dataset = build_dataset(config)

    kwargs = {}
    if torch.cuda.is_available() and not is_debug_session():
        kwargs = {'num_workers': 2, 'pin_memory': True}

    # construct the model
    num_classes = len(test_dataset.classes)
    model = build_model(config['model_name'], num_classes)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(config['train_restore_file'])['models'][0])
    if use_gpu:
        model = model.cuda()
    model.eval()

    with torch.no_grad():
        # print('accuracy ...')
        # loss_meter = AverageMeter()
        # top1_meter = AverageMeter()
        # top5_meter = AverageMeter()
        # test_dataloader = DataLoader(test_dataset,
        #                              batch_size=128,
        #                              **kwargs)
        # for i, samples in enumerate(test_dataloader):
        #     images = samples[0]
        #     labels = samples[1]
        #
        #     # Create non_blocking tensors for distributed training
        #     if use_gpu:
        #         images = images.cuda(non_blocking=True)
        #         labels = labels.cuda(non_blocking=True)
        #
        #     # forward
        #     outputs = model(images)
        #     loss = criterion(outputs, labels)
        #
        #     acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        #     loss_meter.update(loss.item())
        #     top1_meter.update(acc1.item(), n=images.size(0))
        #     top5_meter.update(acc5.item(), n=images.size(0))
        #
        # log_dict = dict()
        # log_dict['test/loss'] = loss_meter.avg
        # log_dict['test/top1'] = top1_meter.avg
        # log_dict['test/top5'] = top5_meter.avg
        # wandb.log(log_dict)

        print('plotting ...')
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=config.get("test_batch_size"),
                                     **kwargs)
        for i, samples in enumerate(test_dataloader):
            images = samples[0]
            labels = samples[1]

            # Create non_blocking tensors for distributed training
            if use_gpu:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, dim=1)

            # detach vars
            images = images.detach().cpu()
            labels = labels.detach().cpu().numpy()
            predicted = predicted.detach().cpu().numpy()

            # plot sample stats
            sample_idx = 0
            sample_img = images[sample_idx]
            gt_caption = test_dataset.classes[labels[sample_idx]]
            predicted_caption = test_dataset.classes[predicted[sample_idx]]
            caption = 'Prediction: {}\nGround Truth: {}'.format(predicted_caption, gt_caption)

            log_dict = {
                "test/sample": wandb.Image(sample_img, caption=caption),
            }
            safe_regions_stats = collect_safe_regions_test_stats(model, wandb.run.dir, i)
            log_dict.update(safe_regions_stats)
            wandb.log(log_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config YML")
    # parser.add_argument("--address", type=str, help="Ray address to use to connect to a cluster.")
    # parser.add_argument("--num-workers", type=int, default=1, help="Sets number of workers for training.")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="Enables GPU training")
    parser.add_argument("--use-fp16", action="store_true", default=False, help="Enables mixed precision training")
    # parser.add_argument("--num-cpus-per-worker", type=int, default=1, help="Sets number of cpus per worker")
    # parser.add_argument("--use-tqdm", action="store_true", default=False, help="Enables tqdm")
    args = parser.parse_args()
    # ray.init(address=args.address, local_mode=is_debug_session())
    config = load_config_yml(args.config)
    test(config, args.use_gpu)
