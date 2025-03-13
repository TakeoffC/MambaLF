import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import nn as enn
from e2cnn import gspaces

from mambalf.utils import image_grid
from mambalf.network.VmambaBlock import VmambaBlock
from fvcore.nn import FlopCountAnalysis
from torchinfo import summary

class LCBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=3 // 2, groups=hidden_features)
        # self.norm = nn.BatchNorm2d(hidden_features)
        self.norm = nn.LayerNorm(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0)
        self.act = act_layer()

    def forward(self, x):
        input = x
        x = self.fc1(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] --> [B, C, H, W]
        x = self.act(x)
        x = self.fc2(x)
        x = input + x
        return x



def conv1x1(in_type, out_type, stride=1):
    """1x1 convolution without padding"""
    return enn.R2Conv(in_type, out_type, 1,
                      stride=stride,
                      padding=0,
                      bias=False)


def conv3x3(in_type, out_type, stride=1):
    """3x3 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 3,
                      stride=stride,
                      padding=1,
                      bias=False)

class BasicBlock(torch.nn.Module):
    def __init__(self, in_type, out_type, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_type, out_type, stride)
        self.conv2 = conv3x3(out_type, out_type)
        self.bn1 = enn.InnerBatchNorm(out_type)
        self.bn2 = enn.InnerBatchNorm(out_type)
        self.relu1 = enn.ReLU(out_type, inplace=True)
        self.relu2 = enn.ReLU(out_type, inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = torch.nn.Sequential(
                conv1x1(in_type, out_type, stride=stride),
                enn.InnerBatchNorm(out_type)
            )

    def forward(self, x):
        y = x
        y = self.relu1(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu2(x+y)

class LGIFBolck(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lc_block = LCBlock(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = VmambaBlock(
            depths=[1],
            dims_input=input_dim//4,
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))


    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C, H, W = x.shape
        assert C == self.input_dim
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        x = self.norm(x)
        input = x
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] --> [B, C, H, W]
        x = self.lc_block(x)

        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]

        x_norm = self.norm(x)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=3)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba_cat = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=3)

        x_mamba = input + x_mamba_cat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.permute(0, 3, 1, 2)   # [B, H, W, C] --> [B, C, H, W]
        return out



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class BasicLayer(nn.Module):
	"""
	  Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
	"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
		super().__init__()
		self.layer = nn.Sequential(
									  nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
									  nn.BatchNorm2d(out_channels, affine=False),
									  nn.ReLU(inplace = True),
									)

	def forward(self, x):
	  return self.layer(x)


class DilationConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilationConv3x3, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class InterestPointModule(nn.Module):
    def __init__(self, is_test=False):
        super(InterestPointModule, self).__init__()
        self.is_test = is_test

        #------escnn-------------#
        block = BasicBlock
        initial_dim = 32
        block_dims = [32, 64, 128, 256]
        # e.g. 4 for C4-symmetry
        nbr_rotations = 4
        e2_same_nbr_filters = True

        self.r2_act = gspaces.Rot2dOnR2(N=nbr_rotations)
        self.triv_in_type = enn.FieldType(self.r2_act,
                                          3*[self.r2_act.trivial_repr])
        if e2_same_nbr_filters:
            dim_reduction = nbr_rotations
        else:
            dim_reduction = 2
        self.in_type = enn.FieldType(self.r2_act,
                                     (initial_dim // dim_reduction) * [self.r2_act.regular_repr])
        # dummy variable used to track input types to each block
        self._in_type = enn.FieldType(self.r2_act,
                                      (initial_dim // dim_reduction) * [self.r2_act.regular_repr])
        reg_repr_blocks = [
            enn.FieldType(self.r2_act,
                          (bd // dim_reduction) * [self.r2_act.regular_repr])
            for bd in block_dims
        ]
        b3_triv_repr = enn.FieldType(self.r2_act,
                                     block_dims[2] * [self.r2_act.trivial_repr])
        b4_triv_repr = enn.FieldType(self.r2_act,
                                     block_dims[3] * [self.r2_act.trivial_repr])

        # Networks
        self.conv1 = enn.R2Conv(self.triv_in_type,
                                self.in_type, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = enn.InnerBatchNorm(self.in_type)
        self.relu1 = enn.ReLU(self.in_type, inplace=True)

        self.layer1 = self._make_layer(block, reg_repr_blocks[0], stride=1)  # 1
        self.layer2 = self._make_layer(block, reg_repr_blocks[1], stride=2)  # 1/2
        self.layer3 = self._make_layer(block, reg_repr_blocks[2], stride=2)  # 1/4
        self.layer4 = self._make_layer(block, reg_repr_blocks[3], stride=2)  # 1/8
        self.layer3_outconv = conv1x1(reg_repr_blocks[2], reg_repr_blocks[2])
        self.layer4_outconv = conv1x1(reg_repr_blocks[3], reg_repr_blocks[3])
        # 128到256
        self.layer3triv = enn.R2Conv(reg_repr_blocks[2], b3_triv_repr,
                                     kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4triv = enn.R2Conv(reg_repr_blocks[3], b4_triv_repr,
                                     kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, enn.R2Conv):
                pass  # TODO: deltaorth initiation?

        #-------------------------#

        # desc head
        self.desc1 = nn.Sequential(LGIFBolck(input_dim=256, output_dim=256))
        self.desc2 = nn.Sequential(LGIFBolck(input_dim=256, output_dim=512))
        self.ebn1 = nn.GroupNorm(4, 256)
        self.ebn2 = nn.GroupNorm(4, 512)

        self.maxpool2x2 = nn.MaxPool2d(2, 2)

        # score head
        self.score_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.score_norm = nn.BatchNorm2d(256)
        self.score_out = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=1)

        # location head
        self.loc_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.loc_norm = nn.BatchNorm2d(256)
        self.loc_out = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)

        # descriptor out
        self.convFaa = torch.nn.Sequential(torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                           torch.nn.BatchNorm2d(256))
        self.convFbb = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # cross_head:
        self.shift_out = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.upsample = torch.nn.PixelShuffle(upscale_factor=2)

    def _make_layer(self, block, out_type, stride=1):
        layer1 = block(self._in_type, out_type, stride=stride)
        layer2 = block(out_type, out_type, stride=1)
        layers = (layer1, layer2)

        self._in_type = out_type
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        B, _, H, W = x.shape    # torch.Size([16, 3, 240, 320])
        x = enn.GeometricTensor(x, self.triv_in_type)

        # ResNet Backbone
        x0 = self.relu1(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1
        x2 = self.layer2(x1)  # 1/2
        x3 = self.layer3(x2)  # 1/4
        x4 = self.layer4(x3)  # 1/8

        x3 = self.layer3_outconv(x3)
        x4 = self.layer4_outconv(x4)

        # rotation invariarize
        x3 = self.layer3triv(x3)
        x3 = x3.tensor
        skip = x3
        x4 = self.layer4triv(x4)
        x4 = x4.tensor

        B, _, Hc, Wc = x4.shape

        # score head
        score_x = self.score_out(self.relu(self.score_norm(self.score_conv(x4))))    # torch.Size([16, 3, 30, 40])
        score = score_x[:, 0, :, :].unsqueeze(1).sigmoid()   # torch.Size([16, 1, 30, 40])


        border_mask = torch.ones(B, Hc, Wc)
        border_mask[:, 0] = 0
        border_mask[:, Hc - 1] = 0
        border_mask[:, :, 0] = 0
        border_mask[:, :, Wc - 1] = 0
        border_mask = border_mask.unsqueeze(1)
        score = score * border_mask.to(score.device)

        # location head
        coord_x = self.relu(self.loc_norm(self.loc_conv(x4)))
        coord_cell = self.loc_out(coord_x).tanh()

        shift_ratio = self.shift_out(coord_x).sigmoid() * 2.0
        step = ((H / Hc) - 1) / 2.
        center_base = image_grid(B, Hc, Wc,
                                 dtype=coord_cell.dtype,
                                 device=coord_cell.device,
                                 ones=False, normalized=False).mul(H / Hc) + step

        coord_un = center_base.add(coord_cell.mul(shift_ratio * step))
        coord = coord_un.clone()
        coord[:, 0] = torch.clamp(coord_un[:, 0], min=0, max=W - 1)
        coord[:, 1] = torch.clamp(coord_un[:, 1], min=0, max=H - 1)

        # descriptor block
        desc1 = F.gelu(self.ebn1(self.desc1(x4)))
        desc2 = F.gelu(self.ebn2(self.desc2(desc1)))
        desc3 = self.upsample(desc2)
        desc4 = torch.cat([desc3, skip], dim=1)
        desc = self.relu(self.convFaa(desc4))
        desc = self.convFbb(desc)

        if self.is_test:
            coord_norm = coord[:, :2].clone()
            coord_norm[:, 0] = (coord_norm[:, 0] / (float(W - 1) / 2.)) - 1.
            coord_norm[:, 1] = (coord_norm[:, 1] / (float(H - 1) / 2.)) - 1.
            coord_norm = coord_norm.permute(0, 2, 3, 1)     # (B,H,W,C)  torch.Size([16, 30, 40, 2])

            # 坐标采样
            desc = torch.nn.functional.grid_sample(desc, coord_norm)      # torch.Size([16, 256, 30, 40])

            desc = desc.div(torch.unsqueeze(torch.norm(desc, p=2, dim=1), 1))  # Divide by norm to normalize.
            # 这里返回的是采样好的描述符
            return score, coord, desc

        """
        score : torch.Size([16, 1, 30, 40])
        coord : torch.Size([16, 2, 30, 40])
        desc_block : list[self.des_conv2(x2) : torch.Size([16, 256, 120, 160]) , 
                          self.des_conv3(x3) : torch.Size([16, 256, 60, 80]), 
                          aware : torch.Size([16, 2, 30, 40])]
        """
        return score, coord, desc


class CorrespondenceModule(nn.Module):
    def __init__(self, match_type='dual_softmax'):
        super(CorrespondenceModule, self).__init__()
        self.match_type = match_type

        if self.match_type == 'dual_softmax':
            self.temperature = 0.1
        else:
            raise NotImplementedError()

    def forward(self, source_desc, target_desc):
        b, c, h, w = source_desc.size()     # torch.Size([16, 256, 30, 40])

        source_desc = source_desc.div(torch.unsqueeze(torch.norm(source_desc, p=2, dim=1), 1)).view(b, -1, h * w)
        target_desc = target_desc.div(torch.unsqueeze(torch.norm(target_desc, p=2, dim=1), 1)).view(b, -1, h * w)

        if self.match_type == 'dual_softmax':
            sim_mat = torch.einsum("bcm, bcn -> bmn", source_desc, target_desc) / self.temperature
            confidence_matrix = F.softmax(sim_mat, 1) * F.softmax(sim_mat, 2)
        else:
            raise NotImplementedError()

        return confidence_matrix


if __name__ == '__main__':
    backbone = InterestPointModule(is_test=True)

    # summary(backbone, input_size=(1, 3, 320, 240))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block = backbone.to(device)


    B, C, H, W = 4, 3, 320, 240
    input_tensor = torch.rand(B, C, H, W).to(device)

    # output_tensor = block(input_tensor, (H, W))
    score, coord, desc = block(input_tensor)

    total_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params / 1000000.0:.2f}M")


    flops = FlopCountAnalysis(block, input_tensor)
    total_flops = flops.total()

    total_flops_gflops = total_flops / 1e9

    print(f"Total FLOPs: {total_flops_gflops:.2f} GFLOPs")


    print("Input tensor size:", input_tensor.size())
    print(score.size())
    print(coord.size())
    print(desc.size())


