import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings

from TeamCode.src.ecg_cbam import CBAM


class ConvNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 groups=1, bias=False):
        super(ConvNBlock, self).__init__()

        self.convn = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=int((kernel_size - 1) / 2),
            groups=groups, bias=bias
        )
        return

    def forward(self, x):
        out = self.convn(x)
        return out


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, cbam=False, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvNBlock(in_channels, out_channels, 3, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = ConvNBlock(out_channels, out_channels, 3, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.cbam = CBAM(out_channels, 8, False) if cbam else None

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                ConvNBlock(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        return

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out += identity
        out = self.act2(out)
        return out


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, cbam=False, stride=1,
                 groups=1, expansion=4, base_channels=64,):
        super(BottleneckBlock, self).__init__()

        group_channels = out_channels * base_channels / 64
        pro_channels = int(group_channels) * groups
        out_channels *= expansion

        self.conv1 = ConvNBlock(in_channels, pro_channels, 1, 1, 1)
        self.bn1 = nn.BatchNorm2d(pro_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = ConvNBlock(pro_channels, pro_channels, 3, stride, groups)
        self.bn2 = nn.BatchNorm2d(pro_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = ConvNBlock(pro_channels, out_channels, 1, 1, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        self.cbam = CBAM(out_channels, 8, False) if cbam else None
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                ConvNBlock(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        return

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activate(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activate(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out += identity
        out = self.activate(out)
        return out


class BasicEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, n_blocks, cbam=False, stride=1):
        super(BasicEncoder, self).__init__()

        layers = [BasicBlock(in_channels, out_channels, cbam, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_channels, out_channels, cbam, 1))

        self.encoder = nn.Sequential(*layers)
        return

    def forward(self, x):
        out = self.encoder(x)
        return out


class BottleneckEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, n_blocks, cbam=False, stride=1,
                 groups=1, expansion=4, base_channels=64):
        super(BottleneckEncoder, self).__init__()

        layers = [BottleneckBlock(in_channels, out_channels, cbam, stride,
                                  groups, expansion, base_channels)]
        in_channels = out_channels * expansion
        for _ in range(1, n_blocks):
            layers.append(BottleneckBlock(in_channels, out_channels, cbam, 1,
                                          groups, expansion, base_channels))

        self.encoder = nn.Sequential(*layers)
        return

    def forward(self, x):
        out = self.encoder(x)
        return out


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, n_blocks=1, cbam=False):
        super(DecoderBlock, self).__init__()

        layers = [BasicBlock(in_channels, out_channels, cbam, 1)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_channels, out_channels, cbam, 1))

        self.decoder = nn.Sequential(*layers)
        return

    def forward(self, x):
        out = self.decoder(x)
        return out


class UpConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, factor=2):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True),
            ConvNBlock(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return

    def forward(self, x):
        out = self.upconv(x)
        return out
    
class STNBlock(nn.Module):
    def __init__(self, output_size):
        super(STNBlock, self).__init__()
        self.output_size = output_size  # Output size (height, width)
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.AdaptiveMaxPool2d(1),
            nn.ReLU(True)
        )
        # Fully connected layer for transformation parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(10, 4) # （10， 3 * 2）if with tilting
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[0].weight.data.zero_()
        self.fc_loc[0].bias.data.copy_(torch.tensor([1, 1, 0, 0], dtype=torch.float))
    
    
    def clamp_to_nearest_power_of_2(self, n):
        if n <= 0:
            warnings.warn("clamp_to_nearest_power_of_2: n must be greater than 0, returning default value 1, this might lead to unpredictable results", UserWarning)
            return torch.tensor(1.0)

        log2n = torch.log2(n.float())
        upper_power_of_2 = torch.pow(2, torch.ceil(log2n))
        
        
        return upper_power_of_2

    def forward(self, x):
        # get the x size
        h,w = x.size()[2], x.size()[3]
        x1, y1, x2, y2 = 79, 154, 2045, 199
        # x1, y1, x2, y2 = 79, 404, 569, 431
        # x1, y1, x2, y2 = 1063, 381, 1553, 464
        # x1, y1, x2, y2 = 79, 982, 569, 1042
        y1 = h - y1
        y2 = h - y2
        
        assert x1 != x2 and y1 != y2, "Bounding box is invalid"
        
        y1_tensor = torch.tensor(y1)
        y2_tensor = torch.tensor(y2)
        x1_tensor = torch.tensor(x1)
        x2_tensor = torch.tensor(x2)
        
        xs = self.localization(x)
        xs = self.fc_loc(xs.view(-1, 10))
        
        scalex = xs[:, 0].view(-1)
        scaley = xs[:, 1].view(-1) # Squeeze or view to flatten the tensor
        tx = xs[:, 2].view(-1)     # Squeeze or view to flatten the tensor
        ty = xs[:, 3].view(-1)
        
        extentx = self.clamp_to_nearest_power_of_2(torch.abs(x1_tensor-x2_tensor))
        extenty = self.clamp_to_nearest_power_of_2(torch.abs(y1_tensor-y2_tensor))

        theta = torch.zeros((x.size(0), 2, 3)).to(x.device)
        theta[:, 0, 0] = extentx.item() / w#self.output_size[1]/w  # Scale x
        theta[:, 1, 1] = extenty.item() / h#self.output_size[0]/h  # Scale y
        theta[:, 0, 2] = (x2 + x1) / w - 1 #tx     # Translate x, x2+x1 for bbox or use 2 * x_mean instead
        theta[:, 1, 2] = (y2 + y1) / h - 1 #ty     # Translate y, y2+y1 for bbox or use 2 * y_median instead
        # xs = xs.view(-1, 10) # if with tilting
        # theta = self.fc_loc(xs)
        # theta = theta.view(-1, 2, 3)

        # Use output_size to generate the output grid
        grid = F.affine_grid(theta, torch.Size([1, 1, int(extenty.item()), int(extentx.item())]), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x, theta


class BasicResUNet(nn.Module):

    def __init__(self, in_channels, out_channels, nbs=[1, 1, 1, 1],
                 init_channels=16, cbam=False, crop=128):
        super(BasicResUNet, self).__init__()
        
        

        # number of kernels
        nks = [init_channels * (2 ** i) for i in range(0, len(nbs) + 1)]

        self.input = nn.Sequential(
            ConvNBlock(in_channels, nks[0], 7),
            nn.BatchNorm2d(nks[0]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.econv1 = BasicEncoder(nks[0], nks[1], nbs[0], cbam, 2)
        self.econv2 = BasicEncoder(nks[1], nks[2], nbs[1], cbam, 2)
        self.econv3 = BasicEncoder(nks[2], nks[3], nbs[2], cbam, 2)
        self.econv4 = BasicEncoder(nks[3], nks[4], nbs[3], cbam, 2)



        self.uconv4 = UpConvBlock(nks[4], nks[3], 2)
        self.uconv3 = UpConvBlock(nks[3], nks[2], 2)
        self.uconv2 = UpConvBlock(nks[2], nks[1], 2)
        self.uconv1 = UpConvBlock(nks[1], nks[0], 2)

        self.dconv4 = DecoderBlock(nks[3] * 2, nks[3], 1, cbam)
        self.dconv3 = DecoderBlock(nks[2] * 2, nks[2], 1, cbam)
        self.dconv2 = DecoderBlock(nks[1] * 2, nks[1], 1, cbam)
        self.dconv1 = DecoderBlock(nks[0] * 2, nks[0], 1, cbam)

        self.output = ConvNBlock(nks[0], out_channels, 1)


        return

    def forward(self, x):
        x = self.input(x)
        e1 = self.econv1(x)
        e2 = self.econv2(e1)
        e3 = self.econv3(e2)
        e4 = self.econv4(e3)

        u4 = self.uconv4(e4)
        c4 = torch.cat([e3, u4], dim=1)
        d4 = self.dconv4(c4)

        u3 = self.uconv3(d4)
        c3 = torch.cat([e2, u3], dim=1)
        d3 = self.dconv3(c3)

        u2 = self.uconv2(d3)
        c2 = torch.cat([e1, u2], dim=1)
        d2 = self.dconv2(c2)

        u1 = self.uconv1(d2)
        c1 = torch.cat([x, u1], dim=1)
        d1 = self.dconv1(c1)

        out = self.output(d1)     
        
        return out

class STNResUNet(nn.Module):
    
    def __init__(self, unet):
        super(STNResUNet, self).__init__()
        
        # self.stn = STNBlock((256, 1024))
        self.stn = STNBlock((340, 2200))
        
        # operations to go from 200*1000 to 128*128
        # self.pooling = nn.AdaptiveAvgPool2d((128, 128))

        self.unet = unet
        
       

    def forward(self, x):
        out, theta = self.stn(x)
        # print(x.size())
        # out = self.unet(x)
        # print(out.size())
        return out, theta
    
    
class BottleneckResUNet(nn.Module):

    def __init__(self, in_channels, out_channels, nbs=[1, 1, 1, 1],
                 init_channels=16, cbam=False):
        super(BottleneckResUNet, self).__init__()

        # number of kernels
        nks = [init_channels * (2 ** i) for i in range(0, len(nbs) + 1)]

        self.input = nn.Sequential(
            ConvNBlock(in_channels, nks[1], 7),
            nn.BatchNorm2d(nks[1]),
            nn.LeakyReLU(0.2, inplace=True)
        )
        

        self.econv1 = BottleneckEncoder(nks[1], nks[0], nbs[0], cbam, 2)
        self.econv2 = BottleneckEncoder(nks[2], nks[1], nbs[1], cbam, 2)
        self.econv3 = BottleneckEncoder(nks[3], nks[2], nbs[2], cbam, 2)
        self.econv4 = BottleneckEncoder(nks[4], nks[3], nbs[3], cbam, 2)

        self.uconv4 = UpConvBlock(nks[4] * 2, nks[4], 2)
        self.uconv3 = UpConvBlock(nks[3] * 2, nks[3], 2)
        self.uconv2 = UpConvBlock(nks[2] * 2, nks[2], 2)
        self.uconv1 = UpConvBlock(nks[1] * 2, nks[1], 2)

        self.dconv4 = DecoderBlock(nks[4] * 2, nks[4], 1, cbam)
        self.dconv3 = DecoderBlock(nks[3] * 2, nks[3], 1, cbam)
        self.dconv2 = DecoderBlock(nks[2] * 2, nks[2], 1, cbam)
        self.dconv1 = DecoderBlock(nks[1] * 2, nks[1], 1, cbam)

        self.output = ConvNBlock(nks[1], out_channels, 1)
        return

    def forward(self, x):
        x = self.input(x)
        e1 = self.econv1(x)
        e2 = self.econv2(e1)
        e3 = self.econv3(e2)
        e4 = self.econv4(e3)

        u4 = self.uconv4(e4)
        c4 = torch.cat([e3, u4], dim=1)
        d4 = self.dconv4(c4)

        u3 = self.uconv3(d4)
        c3 = torch.cat([e2, u3], dim=1)
        d3 = self.dconv3(c3)

        u2 = self.uconv2(d3)
        c2 = torch.cat([e1, u2], dim=1)
        d2 = self.dconv2(c2)

        u1 = self.uconv1(d2)
        c1 = torch.cat([x, u1], dim=1)
        d1 = self.dconv1(c1)

        out = self.output(d1)
        return out

class BasicResUNetWithBBox(nn.Module):

    def __init__(self, in_channels, out_channels, nbs=[1, 1, 1, 1],
                 init_channels=16, cbam=False, crop=128):
        super(BasicResUNetWithBBox, self).__init__()

        # number of kernels
        nks = [init_channels * (2 ** i) for i in range(0, len(nbs) + 1)]

        self.input = nn.Sequential(
            ConvNBlock(in_channels, nks[0], 7),
            nn.BatchNorm2d(nks[0]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.econv1 = BasicEncoder(nks[0], nks[1], nbs[0], cbam, 2)
        self.econv2 = BasicEncoder(nks[1], nks[2], nbs[1], cbam, 2)
        self.econv3 = BasicEncoder(nks[2], nks[3], nbs[2], cbam, 2)
        self.econv4 = BasicEncoder(nks[3], nks[4], nbs[3], cbam, 2)

        self.uconv4 = UpConvBlock(nks[4], nks[3], 2)
        self.uconv3 = UpConvBlock(nks[3], nks[2], 2)
        self.uconv2 = UpConvBlock(nks[2], nks[1], 2)
        self.uconv1 = UpConvBlock(nks[1], nks[0], 2)

        self.dconv4 = DecoderBlock(nks[3] * 2, nks[3], 1, cbam)
        self.dconv3 = DecoderBlock(nks[2] * 2, nks[2], 1, cbam)
        self.dconv2 = DecoderBlock(nks[1] * 2, nks[1], 1, cbam)
        self.dconv1 = DecoderBlock(nks[0] * 2, nks[0], 1, cbam)

        # Output layers for segmentation and bounding box prediction
        self.seg_output = ConvNBlock(nks[0], out_channels, 1)
        self.bbox_output = nn.Linear(nks[4] * 2 * (crop // 16) * (crop // 16), 4) # predicting 4 bbox coordinates

        return

    def forward(self, x):
        x = self.input(x)
        e1 = self.econv1(x)
        e2 = self.econv2(e1)
        e3 = self.econv3(e2)
        e4 = self.econv4(e3)

        u4 = self.uconv4(e4)
        c4 = torch.cat([e3, u4], dim=1)
        d4 = self.dconv4(c4)

        u3 = self.uconv3(d4)
        c3 = torch.cat([e2, u3], dim=1)
        d3 = self.dconv3(c3)

        u2 = self.uconv2(d3)
        c2 = torch.cat([e1, u2], dim=1)
        d2 = self.dconv2(c2)

        u1 = self.uconv1(d2)
        c1 = torch.cat([x, u1], dim=1)
        d1 = self.dconv1(c1)

        seg_out = self.seg_output(d1)

        # Flatten for bbox prediction
        flattened = d1.view(d1.size(0), -1)
        bbox_out = self.bbox_output(flattened)

        return seg_out, bbox_out

def build_model(model_name, cbam=True):

    if model_name == 'resunet10':
        model = BasicResUNet(1, 1, [1, 1, 1, 1], 16, cbam)
    elif model_name == 'resunet18':
        model = BasicResUNet(1, 1, [2, 2, 2, 2], 64, cbam)
    elif model_name == 'resunet34':
        model = BasicResUNet(1, 1, [3, 4, 6, 3], 64, cbam)
    elif model_name == 'resunet14':
        model = BottleneckResUNet(1, 1, [1, 1, 1, 1], 64, cbam)
    elif model_name == 'resunet26':
        model = BottleneckResUNet(1, 1, [2, 2, 2, 2], 64, cbam)
    elif model_name == 'resunet50':
        model = BottleneckResUNet(1, 1, [3, 4, 6, 3], 64, cbam)
    elif model_name == 'resunet101':
        model = BottleneckResUNet(1, 1, [3, 4, 23, 3], 64, cbam)
    elif model_name == 'stnresunet':
        model = STNResUNet(BasicResUNet(1, 1, [1, 1, 1, 1], 16, cbam))
    else:
        raise ValueError('Invalid model_name')

    return model


if __name__ == '__main__':
    from torchsummary import summary

    resunet10 = build_model('resunet10')
    summary(resunet10, (1, 128, 128), batch_size=1)

    # resunet18 = build_model('resunet18')
    # summary(resunet18, (1, 128, 128), batch_size=1)

    # resunet34 = build_model('resunet34')
    # summary(resunet34, (1, 128, 128), batch_size=1)

    # resunet14 = build_model('resunet14')
    # summary(resunet14, (1, 128, 128), batch_size=1)

    # resunet26 = build_model('resunet26')
    # summary(resunet26, (1, 128, 128), batch_size=1)

    # resunet50 = build_model('resunet50')
    # summary(resunet50, (1, 128, 128), batch_size=1)

    # resunet101 = build_model('resunet101')
    # summary(resunet101, (1, 128, 128), batch_size=1)
