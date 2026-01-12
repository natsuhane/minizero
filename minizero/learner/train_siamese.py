#!/usr/bin/env python

import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from minizero.network.py.create_network import create_network
from tools.analysis import analysis


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs, flush=True)


class TripletLoss(nn.Module):
    """
    Triplet Loss with margin
    L = max(0, d(a,p) - d(a,n) + margin)
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor_emb, positive_emb, negative_emb):
        d_ap = torch.norm(anchor_emb - positive_emb, dim=1)
        d_an = torch.norm(anchor_emb - negative_emb, dim=1)
        loss = torch.relu(self.margin + d_ap - d_an)
        return loss.mean()


class MinizeroDataLoader:
    def __init__(self, conf_file_name):
        self.data_loader = py.DataLoader(conf_file_name)
        self.data_loader.initialize()
        self.data_list = []

        # allocate memory

        self.anchor_channels = 72
        self.board_channels = 4  # black piece, white piece, black's turn?, white's turn?

        self.anchor = np.zeros(py.get_batch_size() * self.anchor_channels * py.get_nn_input_channel_height() * py.get_nn_input_channel_width(), dtype=np.float32)
        self.positive = np.zeros(py.get_batch_size() * self.board_channels * py.get_nn_input_channel_height() * py.get_nn_input_channel_width(), dtype=np.float32)
        self.negative = np.zeros(py.get_batch_size() * self.board_channels * py.get_nn_input_channel_height() * py.get_nn_input_channel_width(), dtype=np.float32)
        self.sampled_index = np.zeros(py.get_batch_size() * 2, dtype=np.int32)

    def load_data(self, training_dir, start_iter, end_iter):
        for i in range(start_iter, end_iter + 1):
            file_name = f"{training_dir}/sgf/{i}.sgf"
            if file_name in self.data_list:
                continue
            self.data_loader.load_data_from_file(file_name)
            self.data_list.append(file_name)
            if len(self.data_list) > py.get_zero_replay_buffer():
                self.data_list.pop(0)

    def sample_data(self, device='cpu'):
        self.data_loader.sample_iig_data(self.anchor, self.positive, self.negative, self.sampled_index)
        anchor = torch.FloatTensor(self.anchor).view(py.get_batch_size(), self.anchor_channels, py.get_nn_input_channel_height(), py.get_nn_input_channel_width()).to(device)
        positive = torch.FloatTensor(self.positive).view(py.get_batch_size(), self.board_channels, py.get_nn_input_channel_height(), py.get_nn_input_channel_width()).to(device)
        negative = torch.FloatTensor(self.negative).view(py.get_batch_size(), self.board_channels, py.get_nn_input_channel_height(), py.get_nn_input_channel_width()).to(device)

        return anchor, positive, negative


class Model:
    def __init__(self):
        self.training_step = 0
        self.network = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.network_type = None

    def load_model(self, training_dir, model_file):
        self.training_step = 0
        self.network_type = py.get_siamese_nn_type_name()
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
                                      self.network_type)
        self.network.to(self.device)

        # Set loss function based on network type
        if self.network_type == "siamese":
            self.loss_fn = TripletLoss(margin=1.0)
        elif self.network_type == "binary_cnn":
            self.loss_fn = nn.BCEWithLogitsLoss()
        if py.get_optimizer().lower() == "adam":
            self.optimizer = optim.Adam(self.network.parameters(),
                                        lr=py.get_learning_rate(),
                                        weight_decay=py.get_weight_decay())
        elif py.get_optimizer().lower() == "adamw":
            self.optimizer = optim.AdamW(self.network.parameters(),
                                         lr=py.get_learning_rate(),
                                         weight_decay=py.get_weight_decay())
        else:
            self.optimizer = optim.SGD(self.network.parameters(),
                                       lr=py.get_learning_rate(),
                                       momentum=py.get_momentum(),
                                       weight_decay=py.get_weight_decay())
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000000, gamma=0.1)

        if model_file:
            snapshot = torch.load(f"{training_dir}/model/{model_file}", map_location=torch.device('cpu'))
            self.training_step = snapshot['training_step']
            self.network.load_state_dict(snapshot['network'])
            self.optimizer.load_state_dict(snapshot['optimizer'])
            self.optimizer.param_groups[0]["lr"] = py.get_learning_rate()
            self.scheduler.load_state_dict(snapshot['scheduler'])

        # for multi-gpu
        self.network = nn.DataParallel(self.network)

    def save_model(self, training_dir):
        snapshot = {'training_step': self.training_step,
                    'network': self.network.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()}
        torch.save(snapshot, f"{training_dir}/model/weight_iter_{self.training_step}.pkl")
        torch.jit.script(self.network.module).save(f"{training_dir}/model/weight_iter_{self.training_step}.pt")


def add_training_info(training_info, key, value):
    if key not in training_info:
        training_info[key] = 0
    training_info[key] += value


def train(model, training_dir, data_loader, start_iter, end_iter):
    if start_iter == -1:
        model.save_model(training_dir)
        return

    # load data
    data_loader.load_data(training_dir, start_iter, end_iter)

    training_info = {}
    for i in range(1, py.get_training_step() + 1):
        model.optimizer.zero_grad()
        anchor, positive, negative = data_loader.sample_data(model.device)

        if model.network_type == "siamese":
            # Siamese Network: Triplet Loss
            anchor_emb, positive_emb, negative_emb = model.network(anchor, positive, negative)
            loss = model.loss_fn(anchor_emb, positive_emb, negative_emb)
        elif model.network_type == "binary_cnn":
            # Binary CNN: BCE Loss
            B = anchor.shape[0]
            pos_logits = model.network(anchor, positive)  # (B, 1)
            neg_logits = model.network(anchor, negative)  # (B, 1)
            logits = torch.cat([pos_logits, neg_logits], dim=0)  # (2B, 1)
            labels = torch.cat([torch.ones(B), torch.zeros(B)]).to(model.device)
            loss = model.loss_fn(logits.squeeze(), labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.network.parameters(), max_norm=1.0)
        model.optimizer.step()
        model.scheduler.step()

        model.training_step += 1

        # Compute metrics based on network type
        if model.network_type == "siamese":
            add_training_info(training_info, 'triplet_loss', loss.item())
            with torch.no_grad():
                d_ap = torch.norm(anchor_emb - positive_emb, dim=1)
                d_an = torch.norm(anchor_emb - negative_emb, dim=1)

                # distance difference
                margin = d_an - d_ap
                add_training_info(training_info, 'margin', margin.mean().item())

                success_rate = (margin > model.loss_fn.margin).float().mean().item()

                # Success rate (margin > 1.0)
                add_training_info(training_info, 'success_rate', success_rate)

                add_training_info(training_info, 'dist_ap', d_ap.mean().item())
                add_training_info(training_info, 'dist_an', d_an.mean().item())
                add_training_info(training_info, 'dist_ap_std', d_ap.std().item())
                add_training_info(training_info, 'dist_an_std', d_an.std().item())
        elif model.network_type == "binary_cnn":
            # Use loss_xxx and accuracy_xxx format for analysis.py compatibility
            add_training_info(training_info, 'loss_bce', loss.item())
            with torch.no_grad():
                probs = torch.sigmoid(logits.squeeze())
                preds = (probs > 0.5).float()
                accuracy = (preds == labels).float().mean().item()
                add_training_info(training_info, 'accuracy_total', accuracy)

                # Separate accuracy for positive and negative samples
                pos_acc = (preds[:B] == labels[:B]).float().mean().item()
                neg_acc = (preds[B:] == labels[B:]).float().mean().item()
                add_training_info(training_info, 'accuracy_pos', pos_acc)
                add_training_info(training_info, 'accuracy_neg', neg_acc)

        if model.training_step != 0 and model.training_step % py.get_training_display_step() == 0:
            eprint("[{}] nn step {}, lr: {}.".format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                model.training_step,
                round(model.optimizer.param_groups[0]["lr"], 6)
            ))
            for loss_key in training_info:
                eprint("\t{}: {}".format(
                    loss_key,
                    round(training_info[loss_key] / py.get_training_display_step(), 5)
                ))
            training_info = {}

        if py.get_nn_snapshot_interval() > 0 and model.training_step % py.get_nn_snapshot_interval() == 0:
            model.save_model(training_dir)
            print("Snapshot model", model.training_step, flush=True)
            eprint("Snapshot model", model.training_step)
            analysis(training_dir, "analysis")

    model.save_model(training_dir)
    print("Optimization_Done", model.training_step, flush=True)
    eprint("Optimization_Done", model.training_step)
    analysis(training_dir, "analysis")


if __name__ == '__main__':
    if len(sys.argv) == 4:
        game_type = sys.argv[1]
        training_dir = sys.argv[2]
        conf_file_name = sys.argv[3]

        # import pybind library
        _temps = __import__(f'build.{game_type}', globals(), locals(), ['minizero_py'], 0)
        py = _temps.minizero_py
    else:
        eprint("python train.py game_type training_dir conf_file")
        exit(0)

    py.load_config_file(conf_file_name)
    data_loader = MinizeroDataLoader(conf_file_name)
    model = Model()

    while True:
        try:
            command = input()
            command_prefix = command.split()[0]
            if command == "keep_alive":
                continue

            eprint("[{}] [command] {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), command))
            if command_prefix == "update_config":
                conf_str = command.split(" ", 1)[-1]
                if not py.load_config_string(conf_str):
                    eprint("Failed to load configuration string.")
                    exit(0)
            elif command_prefix == "train":
                _, model_file, start_iter, end_iter = command.split()
                model_file = model_file.replace('"', '')

                # skip loading model if the model is loaded
                if model.network is None:
                    model.load_model(training_dir, model_file)

                train(model, training_dir, data_loader, int(start_iter), int(end_iter))
            elif command_prefix == "quit":
                exit(0)

        except (KeyboardInterrupt, EOFError) as e:
            break
