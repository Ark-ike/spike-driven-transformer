from torch import nn
from einops.layers.torch import Rearrange, Reduce
from spikingjelly.activation_based import neuron


class FeedForward(nn.Module):
    def __init__(self, time_steps=16, in_channels=512, out_channels=512):
        super().__init__()
        self.feed_forward = nn.Sequential(
            # block 1
            neuron.LIFNode(detach_reset=True),
            Rearrange('T B C H W -> (T B) C H W'),
            nn.Conv2d(in_channels, in_channels * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels * 4),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps),
            # block 2
            neuron.LIFNode(detach_reset=True),
            Rearrange('T B C H W -> (T B) C H W'),
            nn.Conv2d(in_channels * 4, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps)
        )

    def forward(self, input):
        output = self.feed_forward(input) + input
        return output


class SelfAttention(nn.Module):
    def __init__(self, time_steps=16, num_heads=8, num_ceils=8, num_channels=512):
        super().__init__()
        self.make_query = nn.Sequential(
            Rearrange('T B C H W -> (T B) C H W'),
            nn.Conv2d(num_channels, num_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(num_channels),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps),
            neuron.LIFNode(detach_reset=True),
            Rearrange('T B (K D) H W -> T B K (H W) D', K=num_heads)
        )
        self.make_key = nn.Sequential(
            Rearrange('T B C H W -> (T B) C H W'),
            nn.Conv2d(num_channels, num_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(num_channels),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps),
            neuron.LIFNode(detach_reset=True),
            Rearrange('T B (K D) H W -> T B K (H W) D', K=num_heads)
        )
        self.make_value = nn.Sequential(
            Rearrange('T B C H W -> (T B) C H W'),
            nn.Conv2d(num_channels, num_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(num_channels),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps),
            neuron.LIFNode(detach_reset=True),
            Rearrange('T B (K D) H W -> T B K (H W) D', K=num_heads)
        )
        self.compute_weight = nn.Sequential(
            Reduce('T B K N D -> T B K 1 D', 'sum'),
            neuron.LIFNode(v_threshold=0.5, detach_reset=True),
        )
        self.make_output = nn.Sequential(
            Rearrange('T B K (H W) D -> (T B) (K D) H W', H=num_ceils),
            nn.Conv2d(num_channels, num_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(num_channels),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps)
        )

    def forward(self, input):
        query = self.make_query(input)
        key = self.make_key(input)
        value = self.make_value(input)
        weight = self.compute_weight(key * value)
        output = self.make_output(query * weight) + input
        return output
