from torch import nn
from External.EGSST_PAPER.detector.efvit.nn.ops import ConvLayer, IdentityLayer, ResidualBlock

class EnchancedCNNBaseBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 2):
        super().__init__()
        self.conv = ConvLayer(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 3,
            act_func = "silu"
        )

        self.dilated_conv = ConvLayer(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 3,
            dilation = dilation,
            act_func = None,
            norm = None
        )

class EnchancedCNN(nn.Module):
    def __init__(self, channels: int, target_size):
        super().__init__()
        self.layer_1 = EnchancedCNNBaseBlock(channels)

        self.layer_2 = ResidualBlock(
            main = EnchancedCNNBaseBlock(channels),
            shortcut = IdentityLayer()
        )

        self.upsample = nn.Upsample(size = target_size)
        self.layer_3 = ConvLayer(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 1,
            act_func = None,
            norm = None
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.upsample(x)
        x = self.layer_3(x)
        return x