import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_unit import ResidualBlock, PolicyNetwork, ValueNetwork, DiscreteValueNetwork


def conv_block(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class PhantomGoSiamese(nn.Module):
    """
    Siamese network for Phantom Go with two encoders:
    - anchor_encoder: encodes observation history (H*6 x N x N)
    - board_encoder: encodes board state (2 x N x N)
    Both produce L2-normalized embeddings in a shared space.
    """

    def __init__(self, obs_in_channels: int, board_in_channels: int = 2, embed_dim: int = 512):
        super().__init__()
        self.obs_in_channels = obs_in_channels
        self.board_in_channels = board_in_channels
        self.embed_dim = embed_dim

        # Anchor encoder
        self.anchor_feat = nn.Sequential(
            conv_block(obs_in_channels, 64),
            conv_block(64, 64),
            conv_block(64, 128, s=2),     # downsample
            conv_block(128, 128),
            conv_block(128, 256, s=2),    # downsample
            conv_block(256, 256),
            conv_block(256, 256),
        )
        self.anchor_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(256, embed_dim, bias=False),
        )

        # Board encoder
        self.board_feat = nn.Sequential(
            conv_block(board_in_channels, 64),
            conv_block(64, 64),
            conv_block(64, 128, s=2),
            conv_block(128, 128),
            conv_block(128, 256, s=2),
            conv_block(256, 256),
        )
        self.board_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, embed_dim, bias=False),
        )

    def encode_anchor(self, anchor: torch.Tensor) -> torch.Tensor:
        """
        Encode observation history
        Args:
            anchor: (B, H*6, N, N)
        Returns:
            (B, D) L2-normalized embeddings
        """
        x = self.anchor_feat(anchor)
        x = self.anchor_head(x)
        x = F.normalize(x, p=2.0, dim=1)
        return x

    def encode_board(self, board: torch.Tensor) -> torch.Tensor:
        """
        Encode board state
        Args:
            board: (B, 2, N, N) - black and white stone positions
        Returns:
            (B, D) L2-normalized embeddings
        """
        x = self.board_feat(board)
        x = self.board_head(x)
        x = F.normalize(x, p=2.0, dim=1)
        return x

    def forward(self, anchor, positive, negative):
        """
        Forward pass for training
        Args:
            anchor: (B, H*6, N, N)
            positive: (B, 2, N, N)
            negative: (B, 2, N, N)
        Returns:
            Tuple of (anchor_emb, positive_emb, negative_emb)
        """
        anchor_emb = self.encode_anchor(anchor)
        positive_emb = self.encode_board(positive)
        negative_emb = self.encode_board(negative)
        return anchor_emb, positive_emb, negative_emb

    @torch.no_grad()
    def compute_distances(self, anchor: torch.Tensor, boards: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise L2 distances between anchors and multiple boards
        Args:
            anchor: (B, H*6, N, N)
            boards: (B, K, 2, N, N) - K candidate boards per anchor
        Returns:
            (B, K) distances
        """
        B, K = boards.shape[0], boards.shape[1]
        anc_emb = self.encode_anchor(anchor)  # (B, D)

        # Flatten and encode all boards
        boards_flat = boards.view(B * K, *boards.shape[2:])
        brd_emb = self.encode_board(boards_flat).view(B, K, -1)  # (B, K, D)

        # Compute L2 distances
        anc_exp = anc_emb.unsqueeze(1).expand_as(brd_emb)  # (B, K, D)
        distances = torch.norm(anc_exp - brd_emb, dim=2)   # (B, K)
        return distances

    @staticmethod
    def softmin_weights(distances: torch.Tensor, temperature: float = 10.0) -> torch.Tensor:
        """
        Convert distances to soft weights (closer = higher weight)
        Args:
            distances: (B, K)
            temperature: scaling factor
        Returns:
            (B, K) weights that sum to 1
        """
        return torch.softmax(-distances / temperature, dim=1)


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

        self.network = PhantomGoSiamese(
            obs_in_channels=num_input_channels,
            board_in_channels=2,
            embed_dim=512
        )

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

    def forward(self, anchor, positive, negative):
        return self.network(anchor, positive, negative)
