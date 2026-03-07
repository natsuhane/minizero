import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_unit import ResidualBlock, PolicyNetwork, ValueNetwork, DiscreteValueNetwork


class ResidualBlockInstanceNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(channels)
        self.activation = nn.ELU(inplace=True)

    def forward(self, x):
        return x + self.activation(self.norm(self.conv(x)))


class EmbeddingNetwork(nn.Module):
    def __init__(self, in_channels, out_channels=128, hidden_channels=64, num_layers=5):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.hidden = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
            for _ in range(num_layers)
        ])
        self.output_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = F.elu(self.input_conv(x), inplace=True)
        for layer in self.hidden:
            x = F.elu(layer(x), inplace=True)
        x = torch.tanh(self.output_conv(x))
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, game_name, num_input_channels, input_channel_height, input_channel_width,
                 num_hidden_channels, hidden_channel_height, hidden_channel_width,
                 num_blocks, action_size, num_value_hidden_channels, discrete_value_size):
        super().__init__()
        self.game_name = game_name
        self.num_input_channels = num_input_channels
        self.input_channel_height = input_channel_height
        self.input_channel_width = input_channel_width
        self.num_hidden_channels = num_hidden_channels
        self.hidden_channel_height = hidden_channel_height
        self.hidden_channel_width = hidden_channel_width
        self.num_blocks = num_blocks
        self.action_size = action_size
        self.num_value_hidden_channels = num_value_hidden_channels
        self.discrete_value_size = discrete_value_size

        pre_embed_dim = 128
        embed_dim = 512
        hidden_channels = 64
        self.board_in_channels = 4

        # Embedding networks (separate for anchor and board)
        self.anchor_embed = EmbeddingNetwork(num_input_channels, pre_embed_dim, hidden_channels, num_layers=5)
        self.board_embed = EmbeddingNetwork(self.board_in_channels, pre_embed_dim, hidden_channels, num_layers=5)

        # Shared trunk (the TRUE Siamese part)
        self.first_block = nn.Sequential(
            nn.Conv2d(pre_embed_dim, pre_embed_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(pre_embed_dim),
            nn.ELU(inplace=True),
        )
        self.residual_blocks = nn.ModuleList([
            ResidualBlockInstanceNorm(pre_embed_dim) for _ in range(num_blocks)
        ])
        self.last_block = nn.Sequential(
            nn.Conv2d(pre_embed_dim, 64, kernel_size=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, bias=False),
            nn.InstanceNorm2d(1),
            nn.ELU(inplace=True),
        )
        self.output = nn.Linear(input_channel_height * input_channel_width, embed_dim)

    def _shared_trunk(self, x):
        x = self.first_block(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.last_block(x)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.output(x))
        return x

    def encode_anchor(self, anchor):
        x = self.anchor_embed(anchor)
        return self._shared_trunk(x)

    def encode_board(self, board):
        x = self.board_embed(board)
        return self._shared_trunk(x)

    @torch.jit.export
    def get_type_name(self):
        return "siamese"

    @torch.jit.export
    def get_game_name(self):
        return self.game_name

    @torch.jit.export
    def get_num_input_channels(self):
        return self.num_input_channels

    @torch.jit.export
    def get_input_channel_height(self):
        return self.input_channel_height

    @torch.jit.export
    def get_input_channel_width(self):
        return self.input_channel_width

    @torch.jit.export
    def get_num_hidden_channels(self):
        return self.num_hidden_channels

    @torch.jit.export
    def get_hidden_channel_height(self):
        return self.hidden_channel_height

    @torch.jit.export
    def get_hidden_channel_width(self):
        return self.hidden_channel_width

    @torch.jit.export
    def get_num_blocks(self):
        return self.num_blocks

    @torch.jit.export
    def get_action_size(self):
        return self.action_size

    @torch.jit.export
    def get_num_value_hidden_channels(self):
        return self.num_value_hidden_channels

    @torch.jit.export
    def get_discrete_value_size(self):
        return self.discrete_value_size

    def forward(self, inputs):
        if inputs.size(1) > self.board_in_channels:
            embeddings = self.encode_anchor(inputs)
        else:
            embeddings = self.encode_board(inputs)
        return {"embeddings": embeddings}
