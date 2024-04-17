import torch
import torch.nn as nn


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        # x = self.sigmoid(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
        features = [32, 64, 128, 256]

        self.inc = InConv(in_channels, features[0])  # 输出 (160 - 3 + 2x1)/1 +1 = 160
        self.down1 = Down(features[0], features[1])  # 输出 (160 - 3 + 2x2)/2 +1 = 80
        self.down2 = Down(features[1], features[2])  # 输出 (80 - 3 + 2x2)/2 +1 = 40
        self.down3 = Down(features[2], features[3])  # 输出 (40 - 3 + 2x4)/4 +1 = 20
        self.down4 = Down(features[3], features[3])

        self.up1 = Up(features[3], features[3], features[2])
        self.up2 = Up(features[2], features[2], features[1])
        self.up3 = Up(features[1], features[1], features[0])
        self.up4 = Up(features[0], features[0], features[0])
        self.outc = OutConv(features[0], num_classes)

    def forward(self, x):
        d1 = self.inc(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)

        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        out = self.outc(u4)
        return out


# if __name__ == '__main__':
#     device = 'cuda'
#     ## ===========Unet======================
#     x = torch.randn(1, 4, 224, 224, 144).to(device)
#     # net = UNet(in_channels=4, num_classes=4)
    
#     ## ==========SegResNet==================
#     # sample = {'image': torch.randn(4, 240, 240, 155),
#     #             'label': torch.randn(3, 240, 240, 155)}
    
#     net = SegResNet(
#     blocks_down=[1, 2, 2, 4],
#     blocks_up=[1, 1, 1],
#     init_filters=16,
#     in_channels=4,
#     out_channels=3,
#     dropout_prob=0.2,
#     ).to(device)

#     # print(net)
#     y = net(x)
#     print("params: ", sum(p.numel() for p in net.parameters()))
#     print(y.shape)
#     print(net)