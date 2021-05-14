import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision import datasets, transforms

from modules.saferegion import SafeRegion
from utils.utils import is_debug_session, load_config_yml

os.environ['WANDB_IGNORE_GLOBS'] = '*.pth'


class MLP(nn.Module):
    def __init__(self, arch):
        super(MLP, self).__init__()
        assert arch is not None
        assert len(arch) > 2
        self.arch = arch
        self.net = nn.Sequential()
        n_in = arch.pop(0)
        n_out = arch.pop()
        prev = n_in
        for i, m in enumerate(arch):
            self.net.add_module(f'dense{i}', nn.Linear(prev, int(m)))
            self.net.add_module(f'dense{i}_saferegion', SafeRegion(int(m)))
            self.net.add_module(f'dense{i}_act', nn.ReLU())
            prev = int(m)
        self.net.add_module(f'dense_{len(arch)}', nn.Linear(prev, n_out))
        self.net.add_module(f'dense{n_out}_saferegion', SafeRegion(n_out))

    def forward(self, x):
        return self.net(x)


def train(config):
    # init wandb
    wandb.init(project='saferegions', config=config, dir=os.getenv('LOG'))
    model = MLP(config['arch'])
    model.train()

    kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() and not is_debug_session() else {}
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.MNIST(os.getenv('DATASETS'),
                             train=True,
                             download=True,
                             transform=transform)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[len(dataset) - 100, 100])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config['train_batch_size'],
                                               shuffle=True,
                                               **kwargs)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config['val_batch_size'],
                                             shuffle=False,
                                             **kwargs)

    device = config['device']
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    step = 0
    for epoch in range(config['num_epochs']):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            in_dim = data.shape[2] * data.shape[3]
            inputs_vec = data.reshape((data.shape[0], in_dim))
            optimizer.zero_grad()
            output = model(inputs_vec)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            p = torch.softmax(output, dim=1)
            c = torch.argmax(p, dim=1)
            if step % config['train_log_freq'] == 0:
                # log scalars
                wandb.log({"train/loss": loss.item()}, step=step)
                # log random samples
                sample_idx = np.random.choice(config['train_batch_size'])
                predicted_caption = str(c[sample_idx].item())
                gt_caption = str(target[sample_idx].item())
                caption = 'Prediction: {}\nGround Truth: {}'.format(predicted_caption, gt_caption)
                sample_img = data[sample_idx].detach().cpu()
                wandb.log({"train/samples": wandb.Image(sample_img, caption=caption)}, step=step)
            step += 1

    with torch.no_grad():
        model_name = '{}-{}.pth'.format(config['model'], step)
        checkpoint_file = os.path.join(wandb.run.dir, model_name)
        torch.save(model.state_dict(), checkpoint_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool for experimenting with SafeRegion module')
    parser.add_argument('--config', type=str, default='config/dev_config.yml', help='Path to yml config')
    args = parser.parse_args()
    config = load_config_yml(args.config)
    train(config)
