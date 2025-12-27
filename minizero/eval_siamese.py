#!/usr/bin/env python

import sys
import os
import torch
import torch.nn as nn
import numpy as np
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

    def load_data(self, testing_dataset):
        file_name = f"{testing_dataset}/1.sgf"
        print(f"Loading testing data from: {file_name}")
        self.data_loader.load_data_from_file(file_name)
        self.data_list.append(file_name)

    def sample_data(self, device='cpu'):
        self.data_loader.sample_iig_data(self.anchor, self.positive, self.negative)
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
        return anchor, positive, negative


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


def evaluate(model, testing_dataset, data_loader):
    data_loader.load_data(testing_dataset)

    # Use train mode instead of eval mode for BatchNorm layers.
    model.network.train()
    # - In eval mode, BatchNorm uses pre-computed running_mean/running_var statistics
    # - These running statistics were incorrectly accumulated during DataParallel training
    #   (each GPU only sees 1/N of data, causing statistics to be unrepresentative)
    # - In train mode, BatchNorm computes mean/var from the current batch, which is correct
    # - torch.no_grad() ensures no gradient computation, so this is still inference-only

    with torch.no_grad():
        for i in range(1, 10000):
            anchor, positive, negative = data_loader.sample_data(model.device)

            for n in range(py.get_siamese_eval_num_negatives()):
                negative_n = negative[:, n, :, :, :]
                if negative_n.sum() == 0:  # fillter
                    continue

                anchor_emb, positive_emb, negative_emb = model.network(anchor, positive, negative_n)

                # Compute distance metrics
                d_ap = torch.norm(anchor_emb - positive_emb, dim=1)
                d_an = torch.norm(anchor_emb - negative_emb, dim=1)

                # distance difference
                margin = d_an - d_ap

                # Success rate (margin > 1.0)
                success_rate = (margin > 1.0).float().mean().item()

                # print
                print(f"Step {i}, Negative {n}:")
                print(f"  Margin (d_an - d_ap): {margin.mean().item():.4f}")
                print(f"  Distance AP (d_ap): {d_ap.mean().item():.4f} ± {d_ap.std().item():.4f}")
                print(f"  Distance AN (d_an): {d_an.mean().item():.4f} ± {d_an.std().item():.4f}")
                print(f"  Success Rate: {success_rate:.4f}")


def get_last_model_file(training_dir):
    model_dir = f"{training_dir}/model"
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not model_files:
        return None

    model_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    return model_files[-1]


if __name__ == '__main__':
    if len(sys.argv) == 5:
        game_type = sys.argv[1]
        training_dir = sys.argv[2]
        conf_file_name = sys.argv[3]
        testing_dataset = sys.argv[4]
    else:
        eprint("python eval_siamese.py game_type training_dir conf_file testing_dataset")
        exit(0)

    py.load_config_file(conf_file_name)
    data_loader = MinizeroDataLoader(conf_file_name)
    model = Model()

    try:
        model_file = get_last_model_file(training_dir)
        print(f"Evaluating model file: {model_file}")
        if model.network is None:
            model.load_model(training_dir, model_file)

        evaluate(model, testing_dataset, data_loader)

    except (KeyboardInterrupt, EOFError) as e:
        eprint("Evaluation interrupted.")
