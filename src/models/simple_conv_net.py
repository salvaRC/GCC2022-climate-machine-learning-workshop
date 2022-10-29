import torch
from torch import nn

from src.models.base_model import BaseModel


class ConvBlock(nn.Module):
    """ A simple convolutional block with BatchNorm and GELU activation. """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)  # a normalization layer for improved/more stable training
        self.activation = nn.GELU()  # a non-linearity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

# The model:
class ConvNet(BaseModel):
    """ A simple convolutional network. """

    def __init__(self, channels_hidden: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = channels_hidden
        # Define the convolutional layers
        self.conv1 = ConvBlock(self.num_input_channels, dim, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(dim, dim, kernel_size=3, padding=1)
        self.conv3 = ConvBlock(dim, dim // 2, kernel_size=3, padding=1)
        self.head = nn.Conv2d(dim // 2, self.num_output_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: a batch of images of shape (batch_size, channels_in, height, width)

        Returns:
            y: a batch of images of shape (batch_size, channels_out, height, width)
        """
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h2 = h1 + h2  # Residual connection
        h3 = self.conv3(h2)
        h4 = self.head(h3)
        return h4
