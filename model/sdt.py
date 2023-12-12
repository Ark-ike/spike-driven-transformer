from torch import nn
from einops.layers.torch import Rearrange, Reduce
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
            neuron.LIFNode(detach_reset=True, step_mode='m'),
            Rearrange('T B C H W -> (T B) C H W'),
            nn.MaxPool2d(3, stride=2, padding=1),
            # block 2
            nn.Conv2d(out_channels // 8, out_channels // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps),
            neuron.LIFNode(detach_reset=True, step_mode='m'),
            Rearrange('T B C H W -> (T B) C H W'),
            nn.MaxPool2d(3, stride=2, padding=1),
            # block 3
            nn.Conv2d(out_channels // 4, out_channels // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps),
            neuron.LIFNode(detach_reset=True, step_mode='m'),
            Rearrange('T B C H W -> (T B) C H W'),
            nn.MaxPool2d(3, stride=2, padding=1),
            # block 4
            nn.Conv2d(out_channels // 2, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(3, stride=2, padding=1),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps)
        )
        self.position_encoder = nn.Sequential(
            neuron.LIFNode(detach_reset=True, step_mode='m'),
            Rearrange('T B C H W -> (T B) C H W'),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps)
        )

    def forward(self, input):
        output = self.patch_splitter(input)
        output = self.position_encoder(output) + output
        return output


class FeedForward(nn.Module):
    def __init__(self, time_steps=16, in_channels=512, out_channels=512):
        super().__init__()
        self.feed_forward = nn.Sequential(
            # block 1
            neuron.LIFNode(detach_reset=True, step_mode='m'),
            Rearrange('T B C H W -> (T B) C H W'),
            nn.Conv2d(in_channels, in_channels * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels * 4),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps),
            # block 2
            neuron.LIFNode(detach_reset=True, step_mode='m'),
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
            neuron.LIFNode(detach_reset=True, step_mode='m'),
            Rearrange('T B (K D) H W -> T B K (H W) D', K=num_heads)
        )
        self.make_key = nn.Sequential(
            Rearrange('T B C H W -> (T B) C H W'),
            nn.Conv2d(num_channels, num_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(num_channels),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps),
            neuron.LIFNode(detach_reset=True, step_mode='m'),
            Rearrange('T B (K D) H W -> T B K (H W) D', K=num_heads)
        )
        self.make_value = nn.Sequential(
            Rearrange('T B C H W -> (T B) C H W'),
            nn.Conv2d(num_channels, num_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(num_channels),
            Rearrange('(T B) C H W -> T B C H W', T=time_steps),
            neuron.LIFNode(detach_reset=True, step_mode='m'),
            Rearrange('T B (K D) H W -> T B K (H W) D', K=num_heads)
        )
        self.compute_weight = nn.Sequential(
            Reduce('T B K N D -> T B K 1 D', 'sum'),
            neuron.LIFNode(v_threshold=0.5, detach_reset=True, step_mode='m'),
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


class EncoderBlock(nn.Module):
    def __init__(self, time_steps=16, num_heads=8, num_ceils=8, num_channels=512):
        super().__init__()
        self.self_attention = SelfAttention(time_steps, num_heads, num_ceils, num_channels)
        self.feed_forward = FeedForward(time_steps, num_channels, num_channels)

    def forward(self, input):
        output = self.self_attention(input)
        output = self.feed_forward(output)
        return output


class SpikingTransformer(nn.Module):
    def __init__(self, time_steps=16, in_channels=2, num_classes=10, num_layers=8, num_heads=8, num_ceils=8, num_channels=512):
        super().__init__()
        self.patch_embedding = PatchEmbedding(time_steps, in_channels, num_channels)
        self.encoder_block = nn.Sequential(
            *[EncoderBlock(time_steps, num_heads, num_ceils, num_channels) for _ in range(num_layers)]
        )
        self.classifier_head = nn.Sequential(
            Reduce('T B C H W -> T B C', 'mean'),
            nn.Linear(num_channels, num_classes),
            Reduce('T B C -> B C', 'mean')
        )

    def forward(self, input):
        output = self.patch_embedding(input)
        output = self.encoder_block(output)
        output = self.classifier_head(output)
        return output
