import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from utils.arch_utils import LayerNorm2d

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class ResidualBlock(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.bn = nn.BatchNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn(self.conv1(x)), inplace=True)
        out = self.conv2(out)
        return identity + out

###########################################################################################################


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SGE(nn.Module):
    def __init__(self, dw_channel):
        super().__init__() 
        self.dwc = nn.Conv2d(in_channels=dw_channel //2, out_channels=dw_channel//2, kernel_size=3, padding=1, stride=1, groups=dw_channel//2, bias=True)
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwc(x1)
        return x1 * x2
    
class SpaBlock(nn.Module):
    def __init__(self, nc, DW_Expand = 2,  FFN_Expand=2, drop_out_rate=0.):
        super(SpaBlock, self).__init__()
        dw_channel = nc * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True) # the dconv
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=nc, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * nc
        self.conv4 = nn.Conv2d(in_channels=nc, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=nc, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(nc)
        self.norm2 = LayerNorm2d(nc)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, nc, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, nc, 1, 1)), requires_grad=True)

    def forward(self, x):

        x = self.norm1(x) # size [B, C, H, W]

        x = self.conv1(x) # size [B, 2*C, H, W]
        x = self.conv2(x) # size [B, 2*C, H, W]
        x = self.sg(x)    # size [B, C, H, W]
        x = x * self.sca(x) # size [B, C, H, W]
        x = self.conv3(x) # size [B, C, H, W]

        x = self.dropout1(x)

        y = x + x * self.beta # size [B, C, H, W]

        x = self.conv4(self.norm2(y)) # size [B, 2*C, H, W]
        x = self.sg(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]

        x = self.dropout2(x)

        return y + x * self.gamma

class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process2 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        return x_out+x

    
class SFBlock(nn.Module):
    def __init__(self, nc, DW_Expand = 2,  FFN_Expand=2):
        super(SFBlock, self).__init__()
        dw_channel = nc * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True) # the dconv
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=nc, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.fatt = FreBlock(dw_channel // 2)
        self.sge = SGE(dw_channel)

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * nc
        self.conv4 = nn.Conv2d(in_channels=nc, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=nc, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(nc)
        self.norm2 = LayerNorm2d(nc)

        self.beta = nn.Parameter(torch.zeros((1, nc, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, nc, 1, 1)), requires_grad=True)

    def forward(self, x):

        x = self.norm1(x) # size [B, C, H, W]

        x = self.conv1(x) # size [B, 2*C, H, W]
        x = self.conv2(x) # size [B, 2*C, H, W]
        x = self.sge(x)    # size [B, C, H, W]
      
        x = self.fatt(x)
        x = self.conv3(x) # size [B, C, H, W]

        y = x + x * self.beta # size [B, C, H, W]

        x = self.conv4(self.norm2(y)) # size [B, 2*C, H, W]
        x = self.sg(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]

        return y + x * self.gamma
    
class ProcessBlock(nn.Module):
    def __init__(self, in_nc, spatial = True):
        super(ProcessBlock,self).__init__()
        self.spatial = spatial
        self.spatial_process = SpaBlock(in_nc) if spatial else nn.Identity()
        self.frequency_process = FreBlock(in_nc)
        self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0) if spatial else nn.Conv2d(in_nc,in_nc,1,1,0)

    def forward(self, x):
        xori = x
        x_freq = self.frequency_process(x)
        x_spatial = self.spatial_process(x)
        xcat = torch.cat([x_spatial,x_freq],1)
        x_out = self.cat(xcat) if self.spatial else self.cat(x_freq)

        return x_out+xori

class SFNet(nn.Module):

    def __init__(self, nc,n=5):
        super(SFNet,self).__init__()

        self.list_block = list()
        for index in range(n):

            self.list_block.append(ProcessBlock(nc,spatial=False))
  
        self.block = nn.Sequential(*self.list_block)

    def forward(self, x):

        x_ori = x
        x_out = self.block(x_ori)
        xout = x_ori + x_out

        return xout

class AmplitudeNet_skip(nn.Module):
    def __init__(self, nc,n=1):
        super(AmplitudeNet_skip,self).__init__()
        
        self.conv_init = nn.Conv2d(3, nc, 1, 1, 0)
        self.conv1 = SFBlock (nc)
        self.conv2 = SFBlock (nc)
        self.conv3 = SFBlock (nc)
        self.conv_out = nn.Conv2d(nc, 3, 1, 1, 0)

    def forward(self, x):
        
        x_lr = F.interpolate(x, scale_factor=0.5, mode='bilinear') # Resize and Normalize SNR map
        
        x_lr = self.conv_init(x_lr)
        x_lr = self.conv1(x_lr)
        x_lr = self.conv2(x_lr)
        x_lr = self.conv3(x_lr)
        x_lr = self.conv_out(x_lr)
        
        xout = F.interpolate(x_lr, scale_factor=2, mode='bilinear') # Resize and Normalize SNR map
        
        return xout

    
###########################################################################################################

class SG(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    

class SGE(nn.Module):
    def __init__(self, dw_channel):
        super().__init__() 
        self.dwc = nn.Conv2d(in_channels=dw_channel //2, out_channels=dw_channel//2, kernel_size=3, padding=1, stride=1, groups=dw_channel//2, bias=True)
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwc(x1)
        return x1 * x2