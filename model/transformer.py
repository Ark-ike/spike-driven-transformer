from torch import nn
from einops.layers.torch import Reduce

from model.embedding import PatchEmbedding
from model.attention import FeedForward, SelfAttention


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
