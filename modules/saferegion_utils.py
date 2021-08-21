import os

import matplotlib.pyplot as plt
import numpy as np
import wandb
from torch.nn.parallel import DistributedDataParallel

from modules.saferegion import _SafeRegion


def collect_safe_regions_stats(model, run_dir=None, step=None, plot_graphs_to_wandb_log_freq=0):
    temp_model = model
    if type(model) is DistributedDataParallel:
        temp_model = model.module

    stats = dict()
    idx = 0

    fig = plt.figure()

    for module in temp_model.modules():
        if isinstance(module, _SafeRegion) or isinstance(module, _SafeRegion) or isinstance(module, _SafeRegion):
            # stats[f'safe_regions_stats/{idx}/avg_running_mean'] = module.running_mean.mean().item()
            # stats[f'safe_regions_stats/{idx}/avg_running_var'] = module.running_var.mean().item()
            # stats[f'safe_regions_stats/{idx}/avg_running_min'] = module.running_min.mean().item()
            # stats[f'safe_regions_stats/{idx}/avg_running_max'] = module.running_max.mean().item()
            # stats[f'safe_regions_stats/{idx}/num_samples_tracked'] = module.num_samples_tracked.item()
            # stats[f'safe_regions_stats/{idx}/num_batches_tracked'] = module.num_batches_tracked.item()

            # bar plot
            # labels = [f"unit{i}" for i in range(module.num_features)]
            # values = module.running_mean.detach().cpu().numpy()
            # data = [[label, val] for (label, val) in zip(labels, values)]
            # table = wandb.Table(data=data, columns=["unit", "mean"])
            # stats[f"safe_regions_stats/{idx}"] = wandb.plot.bar(table, "unit", "mean", title="Safe regions mean per unit")

            layer_name = 'layer' + str(idx).zfill(3)

            # wandb plot
            xs = np.arange(module.num_features)
            ys_mean = module.running_mean.detach().cpu().numpy()
            ys_min = module.running_min.detach().cpu().numpy()
            ys_max = module.running_max.detach().cpu().numpy()

            if plot_graphs_to_wandb_log_freq != 0 and step % plot_graphs_to_wandb_log_freq == 0:
                plot = wandb.plot.line_series(xs=xs,
                                              ys=[ys_min, ys_mean, ys_max],
                                              keys=["running_min", "running_mean", "running_max"],
                                              title=layer_name,
                                              xname="step")
                stats[f"safe_regions_stats/{layer_name}/multi_line_plot"] = plot

            layer_dir = os.path.join(run_dir, 'plots', layer_name)
            if not os.path.exists(layer_dir):
                os.makedirs(layer_dir)

            # matplotlib plot
            ax = fig.add_subplot()
            ax.plot(xs, ys_max, label="running_max", linestyle="-")
            ax.plot(xs, ys_mean, label="running_mean", linestyle="-")
            ax.plot(xs, ys_min, label="running_min", linestyle="-")
            ax.set_xlabel('neural unit')
            ax.set_ylabel('recorded value')
            ax.set_title(f'Safe region per neural unit at step {step}')
            max_bound = 20
            min_bound = 20
            ax.set_ylim((-min_bound, max_bound))
            ax.legend()
            fig_name = os.path.join(layer_dir, f"{str(step).zfill(7)}.png")
            fig.savefig(fig_name)
            plt.clf()
            idx += 1
    return stats


def collect_safe_regions_test_stats(model, run_dir=None, step=None):
    temp_model = model
    if type(model) is DistributedDataParallel:
        temp_model = model.module

    stats = dict()
    idx = 0

    fig = plt.figure()
    # bar = plt.figure()
    for module in temp_model.modules():
        if isinstance(module, _SafeRegion) or isinstance(module, _SafeRegion) or isinstance(module, _SafeRegion):
            layer_name = 'layer' + str(idx).zfill(3)

            # wandb plot
            xs = np.arange(module.num_features)
            ys_mean = module.running_mean.detach().cpu().numpy()
            ys_min = module.running_min.detach().cpu().numpy()
            ys_max = module.running_max.detach().cpu().numpy()
            sample_x_max = module.last_x_max.detach().cpu().numpy()
            sample_x_min = module.last_x_min.detach().cpu().numpy()
            sample_out = module.last_x_sum.detach().cpu().numpy()

            layer_dir = os.path.join(run_dir, 'plots', layer_name)
            if not os.path.exists(layer_dir):
                os.makedirs(layer_dir)

            # matplotlib plot
            ax = fig.add_subplot()
            ax.plot(xs, ys_max, label="running_max", linestyle="-")
            ax.plot(xs, ys_mean, label="running_mean", linestyle="-")
            ax.plot(xs, ys_min, label="running_min", linestyle="-")
            ax.plot(xs, sample_x_max, label="x_max", linestyle="-")
            ax.plot(xs, sample_x_min, label="x_min", linestyle="-")
            ax.set_xlabel('neural unit')
            ax.set_ylabel('recorded value')
            ax.set_title(f'Safe region at {layer_name} for sample {step}')
            max_bound = np.max(np.stack([ys_max, sample_x_max])) + 1
            min_bound = np.min(np.stack([ys_min, sample_x_min])) - 1
            ax.set_ylim((min_bound, max_bound))
            ax.legend()
            fig_name = os.path.join(layer_dir, f"{str(step).zfill(7)}.png")
            fig.savefig(fig_name)
            stats[f'saferegions/{layer_name}/chart'] = wandb.Image(fig)
            plt.clf()

            # plot bar
            bar_ax = fig.add_subplot()
            bar_ax.bar(xs, sample_out)
            bar_name = os.path.join(layer_dir, f"bar_{str(step).zfill(7)}.png")
            fig.savefig(bar_name)
            stats[f'saferegions/{layer_name}/bar'] = wandb.Image(fig)

            plt.clf()

            idx += 1

    return stats
