from torch import nn
from spikingjelly.activation_based import layer, neuron


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=2, out_channels=512):
        super().__init__()
        self.patch_splitter = nn.Sequential(
            # block 1
            layer.Conv2d(in_channels, out_channels // 8, 3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels // 8),
            neuron.LIFNode(detach_reset=True),
            layer.MaxPool2d(3, stride=2, padding=1),
            # block 2
            layer.Conv2d(out_channels // 8, out_channels // 4, 3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels // 4),
            neuron.LIFNode(detach_reset=True),
            layer.MaxPool2d(3, stride=2, padding=1),
            # block 3
            layer.Conv2d(out_channels // 4, out_channels // 2, 3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels // 2),
            neuron.LIFNode(detach_reset=True),
            layer.MaxPool2d(3, stride=2, padding=1),
            # block 4
            layer.Conv2d(out_channels // 2, out_channels, 3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels),
            layer.MaxPool2d(3, stride=2, padding=1)
        )
        self.position_encoder = nn.Sequential(
            neuron.LIFNode(detach_reset=True),
            layer.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels)
        )

    def forward(self, input):
        output = self.patch_splitter(input)
        output = self.position_encoder(output) + output
        return output
