import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class PhantomGoBinaryCNN(nn.Module):
    """
    Binary classifier for Phantom Go.
    Input: concatenated anchor (72 ch) + board (4 ch) = 76 channels
    Output: 1 logit (probability of being true board)
    """

    def __init__(self, anchor_channels: int = 72, board_channels: int = 4):
        super().__init__()
        self.anchor_channels = anchor_channels
        self.board_channels = board_channels
        in_channels = anchor_channels + board_channels  # 76

        # Same architecture depth as Siamese (7 conv blocks + 2 downsamples)
        self.features = nn.Sequential(
            conv_block(in_channels, 64),
            conv_block(64, 64),
            conv_block(64, 128, s=2),     # downsample
            conv_block(128, 128),
            conv_block(128, 256, s=2),    # downsample
            conv_block(256, 256),
            conv_block(256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),  # Single output logit
        )

    def forward(self, anchor: torch.Tensor, board: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            anchor: (B, 72, N, N) observation history
            board: (B, 4, N, N) board state (positive or negative)
        Returns:
            (B, 1) logits
        """
        x = torch.cat([anchor, board], dim=1)  # (B, 76, N, N)
        x = self.features(x)
        x = self.classifier(x)
        return x


class BinaryCNNNetwork(nn.Module):
    def __init__(self,
                 game_name,
                 num_input_channels,
                 input_channel_height,
                 input_channel_width,
                 num_hidden_channels,
                 num_blocks):
        super().__init__()
        self.game_name = game_name
        self.num_input_channels = num_input_channels  # 72 (anchor channels)
        self.input_channel_height = input_channel_height
        self.input_channel_width = input_channel_width
        self.num_hidden_channels = num_hidden_channels
        self.num_blocks = num_blocks

        self.network = PhantomGoBinaryCNN(anchor_channels=num_input_channels)

    @torch.jit.export
    def get_type_name(self):
        return "binary_cnn"

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

    def forward(self, anchor, board):
        return self.network(anchor, board)
