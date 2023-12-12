from torch import nn
from einops.layers.torch import Rearrange
from spikingjelly.activation_based import neuron


class PatchEmbedding(nn.Module):
    def __init__(self, time_steps=16, in_channels=2, out_channels=512):
        super().__init__()
        self.patch_splitter = nn.Sequential(
            # block 1
            Rearrange('T B C H W -> (T B) C H W'),
            nn.Conv2d(in_channels, out_channels // 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels // 8),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps),
            neuron.LIFNode(detach_reset=True),
            Rearrange('T B C H W -> (T B) C H W'),
            nn.MaxPool2d(3, stride=2, padding=1),
            # block 2
            nn.Conv2d(out_channels // 8, out_channels // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps),
            neuron.LIFNode(detach_reset=True),
            Rearrange('T B C H W -> (T B) C H W'),
            nn.MaxPool2d(3, stride=2, padding=1),
            # block 3
            nn.Conv2d(out_channels // 4, out_channels // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps),
            neuron.LIFNode(detach_reset=True),
            Rearrange('T B C H W -> (T B) C H W'),
            nn.MaxPool2d(3, stride=2, padding=1),
            # block 4
            nn.Conv2d(out_channels // 2, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(3, stride=2, padding=1),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps)
        )
        self.position_encoder = nn.Sequential(
            neuron.LIFNode(detach_reset=True),
            Rearrange('T B C H W -> (T B) C H W'),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps)
        )

    def forward(self, input):
        output = self.patch_splitter(input)
        output = self.position_encoder(output) + output
        return output
