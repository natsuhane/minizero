#!/usr/bin/env python

import sys
import os
import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from minizero.network.py.create_network import create_network
from tools.analysis import analysis
import build.go.minizero_py as py


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs, flush=True)


class MinizeroDataLoader:
    def __init__(self, conf_file_name):
        self.data_loader = py.DataLoader(conf_file_name)
        self.data_loader.initialize()
        self.data_list = []

        self.anchor_channels = 72
        self.board_channels = 2

        self.anchor = np.zeros(py.get_batch_size() * self.anchor_channels * py.get_nn_input_channel_height() * py.get_nn_input_channel_width(), dtype=np.float32)
        self.positive = np.zeros(py.get_batch_size() * self.board_channels * py.get_nn_input_channel_height() * py.get_nn_input_channel_width(), dtype=np.float32)
        self.negative = np.zeros(py.get_batch_size() * py.get_siamese_eval_num_negatives() * self.board_channels * py.get_nn_input_channel_height() * py.get_nn_input_channel_width(), dtype=np.float32)
        self.sampled_index = np.zeros(py.get_batch_size() * 2, dtype=np.int32)

    def load_data(self, testing_dataset):
        file_name = f"{testing_dataset}/1.sgf"
        print(f"Loading testing data from: {file_name}")
        self.data_loader.load_data_from_file(file_name)
        self.data_list.append(file_name)

    def sample_data(self, device='cpu'):
        self.data_loader.sample_iig_data(self.anchor, self.positive, self.negative, self.sampled_index)
        anchor = torch.FloatTensor(self.anchor).view(py.get_batch_size(), self.anchor_channels, py.get_nn_input_channel_height(), py.get_nn_input_channel_width()).to(device)
        positive = torch.FloatTensor(self.positive).view(py.get_batch_size(), self.board_channels, py.get_nn_input_channel_height(), py.get_nn_input_channel_width()).to(device)
        # multiple negatives
        negative = torch.FloatTensor(
            self.negative).view(
                py.get_batch_size(),
                py.get_siamese_eval_num_negatives(),
                self.board_channels,
                py.get_nn_input_channel_height(),
                py.get_nn_input_channel_width()).to(device)
        # move numbers (pos values at odd indices)
        move_numbers = self.sampled_index[1::2].copy()
        return anchor, positive, negative, move_numbers


class Model:
    def __init__(self):
        self.network = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_model(self, training_dir, model_file):
        anchor_channels = 72
        self.network = create_network(py.get_game_name(),
                                      anchor_channels,  # Use 72 for anchor input channels
                                      py.get_nn_input_channel_height(),
                                      py.get_nn_input_channel_width(),
                                      py.get_nn_num_hidden_channels(),
                                      py.get_nn_hidden_channel_height(),
                                      py.get_nn_hidden_channel_width(),
                                      py.get_nn_num_action_feature_channels(),
                                      py.get_nn_num_blocks(),
                                      py.get_nn_action_size(),
                                      py.get_nn_num_value_hidden_channels(),
                                      py.get_nn_discrete_value_size(),
                                      py.get_nn_type_name())
        self.network.to(self.device)

        if model_file:
            snapshot = torch.load(f"{training_dir}/model/{model_file}", map_location=torch.device('cpu'))
            self.network.load_state_dict(snapshot['network'])

        # for multi-gpu
        self.network = nn.DataParallel(self.network)


def bin_move_number(move_num, bin_size=10):
    """Bin move numbers: 0-9 -> 0, 10-19 -> 10, etc."""
    return (move_num // bin_size) * bin_size


def plot_metrics_by_move(move_stats, output_dir):
    """Generate success rate, margin, and loss charts binned by 10 steps."""
    os.makedirs(output_dir, exist_ok=True)

    # Sort bins
    bins = sorted(move_stats.keys())
    if not bins:
        print("No data to plot.")
        return

    success_rates = [np.mean(move_stats[b]['successes']) for b in bins]
    margins = [np.mean(move_stats[b]['margins']) for b in bins]
    # Compute loss: triplet loss = max(0, 1.0 - margin)
    losses = [np.mean([max(0, 1.0 - m) for m in move_stats[b]['margins']]) for b in bins]

    # Compute overall metrics (across all bins)
    all_successes = [s for b in bins for s in move_stats[b]['successes']]
    all_margins = [m for b in bins for m in move_stats[b]['margins']]
    overall_success_rate = np.mean(all_successes)
    overall_margin = np.mean(all_margins)
    overall_loss = np.mean([max(0, 1.0 - m) for m in all_margins])

    # Chart 1: Success Rate
    plt.figure(figsize=(12, 8))
    plt.plot(bins, success_rates, marker='o', linewidth=2, color='steelblue')
    plt.axhline(y=overall_success_rate, color='gray', linestyle='--',
                label=f'Overall: {overall_success_rate:.4f}')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('Success Rate by Step (Binned by 10)', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_by_step.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/success_rate_by_step.png")

    # Chart 2: Margin
    plt.figure(figsize=(12, 8))
    plt.plot(bins, margins, marker='o', linewidth=2, color='coral')
    plt.axhline(y=1.0, color='red', linestyle='--', label='Threshold (1.0)')
    plt.axhline(y=overall_margin, color='gray', linestyle='--',
                label=f'Overall: {overall_margin:.4f}')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Average Margin (d_an - d_ap)', fontsize=12)
    plt.title('Average Margin by Step (Binned by 10)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'margin_by_step.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/margin_by_step.png")

    # Chart 3: Loss
    plt.figure(figsize=(12, 8))
    plt.plot(bins, losses, marker='o', linewidth=2, color='green')
    plt.axhline(y=overall_loss, color='gray', linestyle='--',
                label=f'Overall: {overall_loss:.4f}')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title('Triplet Loss by Step (Binned by 10)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_by_step.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/loss_by_step.png")


def plot_from_log(log_path, output_dir):
    """Parse eval.log and generate charts."""
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    move_stats = defaultdict(lambda: {'margins': [], 'successes': []})

    with open(log_path, 'r') as f:
        for line in f:
            # Parse: move:42 margin:1.234567 success:1
            match = re.match(r'move:(\d+) margin:([-\d.]+) success:(\d)', line)
            if match:
                move = int(match.group(1))
                margin = float(match.group(2))
                success = int(match.group(3))

                binned = bin_move_number(move)
                move_stats[binned]['margins'].append(margin)
                move_stats[binned]['successes'].append(success)

    if not move_stats:
        print("No data found in log file.")
        return

    plot_metrics_by_move(move_stats, output_dir)


def evaluate(model, testing_dataset, data_loader, training_dir):
    data_loader.load_data(testing_dataset)

    # Use train mode instead of eval mode for BatchNorm layers.
    model.network.train()
    # - In eval mode, BatchNorm uses pre-computed running_mean/running_var statistics
    # - These running statistics were incorrectly accumulated during DataParallel training
    #   (each GPU only sees 1/N of data, causing statistics to be unrepresentative)
    # - In train mode, BatchNorm computes mean/var from the current batch, which is correct
    # - torch.no_grad() ensures no gradient computation, so this is still inference-only

    # Create output directory and log file
    output_dir = f"{training_dir}/eval_analysis"
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'eval.log')

    with open(log_path, 'w') as eval_log:
        with torch.no_grad():
            for i in range(1, 10000):
                anchor, positive, negative, move_numbers = data_loader.sample_data(model.device)

                for n in range(py.get_siamese_eval_num_negatives()):
                    negative_n = negative[:, n, :, :, :]
                    if negative_n.sum() == 0:  # filter
                        continue

                    anchor_emb, positive_emb, negative_emb = model.network(anchor, positive, negative_n)

                    # Compute distance metrics
                    d_ap = torch.norm(anchor_emb - positive_emb, dim=1)
                    d_an = torch.norm(anchor_emb - negative_emb, dim=1)

                    # distance difference
                    margin = d_an - d_ap

                    # Write to log file (per-sample with move number)
                    for b in range(len(move_numbers)):
                        success = 1 if margin[b] > 1.0 else 0
                        eval_log.write(f"move:{move_numbers[b]} margin:{margin[b].item():.6f} success:{success}\n")
                    eval_log.flush()  # Ensure data is written immediately

                    # Success rate (margin > 1.0)
                    success_rate = (margin > 1.0).float().mean().item()

                    # print progress
                    print(f"Step {i}, Negative {n}:")
                    print(f"  Margin (d_an - d_ap): {margin.mean().item():.4f}")
                    print(f"  Distance AP (d_ap): {d_ap.mean().item():.4f} ± {d_ap.std().item():.4f}")
                    print(f"  Distance AN (d_an): {d_an.mean().item():.4f} ± {d_an.std().item():.4f}")
                    print(f"  Success Rate: {success_rate:.4f}")

    print(f"\nLog saved to: {log_path}")


def get_last_model_file(training_dir):
    model_dir = f"{training_dir}/model"
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not model_files:
        return None

    model_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    return model_files[-1]


if __name__ == '__main__':
    # Mode 1: Generate charts from existing log
    # python eval_siamese.py --plot training_dir
    if len(sys.argv) == 3 and sys.argv[1] == '--plot':
        training_dir = sys.argv[2]
        output_dir = f"{training_dir}/eval_analysis"
        log_path = os.path.join(output_dir, 'eval.log')
        print(f"Generating charts from: {log_path}")
        plot_from_log(log_path, output_dir)
        print(f"Charts saved to: {output_dir}")
        exit(0)

    # Mode 2: Run evaluation (writes log + generates charts)
    # python eval_siamese.py game_type training_dir conf_file testing_dataset
    if len(sys.argv) == 5:
        game_type = sys.argv[1]
        training_dir = sys.argv[2]
        conf_file_name = sys.argv[3]
        testing_dataset = sys.argv[4]
    else:
        eprint("Usage:")
        eprint("  python eval_siamese.py game_type training_dir conf_file testing_dataset")
        eprint("  python eval_siamese.py --plot training_dir")
        exit(0)

    py.load_config_file(conf_file_name)
    data_loader = MinizeroDataLoader(conf_file_name)
    model = Model()

    output_dir = f"{training_dir}/eval_analysis"
    log_path = os.path.join(output_dir, 'eval.log')

    try:
        model_file = get_last_model_file(training_dir)
        print(f"Evaluating model file: {model_file}")
        if model.network is None:
            model.load_model(training_dir, model_file)

        evaluate(model, testing_dataset, data_loader, training_dir)

    except (KeyboardInterrupt, EOFError) as e:
        eprint("\nEvaluation interrupted.")

    # Generate charts from log (works even if interrupted)
    print("\nGenerating charts from log...")
    plot_from_log(log_path, output_dir)
    print(f"Charts saved to: {output_dir}")
