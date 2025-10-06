import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_unit import ResidualBlock, PolicyNetwork, ValueNetwork, DiscreteValueNetwork


class SiameseNetwork(nn.Module):
    def __init__(self,
                 game_name,
                 num_input_channels,
                 input_channel_height,
                 input_channel_width,
                 num_hidden_channels,
                 num_blocks):
        super(SiameseNetwork, self).__init__()
        self.game_name = game_name
        self.num_input_channels = num_input_channels
        self.input_channel_height = input_channel_height
        self.input_channel_width = input_channel_width
        self.num_hidden_channels = num_hidden_channels
        self.num_blocks = num_blocks

        # TODO: change network?
        self.conv = nn.Conv2d(num_input_channels, num_hidden_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_hidden_channels)
        self.residual_blocks = nn.ModuleList([ResidualBlock(num_hidden_channels) for _ in range(num_blocks)])

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
    def get_num_blocks(self):
        return self.num_blocks

    def forward(self, state):
        x = self.conv(state)
        x = self.bn(x)
        x = F.relu(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)

        return x
