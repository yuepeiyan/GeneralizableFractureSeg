import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchsummary import summary
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.utils import ensure_tuple_rep
from typing import Sequence, Tuple, Union


class CSA_UNETR_decoder(nn.Module):
    # def __init__(self, in_channels, out_channels, final_sigmoid_flag=False, init_channel_number=32):
    #     super(CSA_UNETR_decoder, self).__init__()
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        feature_size: int = 16,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims: int = 3,
        init_channel_number: int = 16,
        final_sigmoid_flag=False
    ) -> None:
        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.grid_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, patch_size))
        self.hidden_size = hidden_size

        # self.encoders = nn.ModuleList([
        #     Encoder(in_channels, init_channel_number, max_pool_flag=False),
        #     Encoder(init_channel_number, 2 * init_channel_number),
        #     Encoder(2 * init_channel_number, 4 * init_channel_number),
        #     Encoder(4 * init_channel_number, 8 * init_channel_number)
        # ])
        # spatial_dims = 3
        # feature_size = 16
        # norm_name = "instance"
        # res_block = True
        # hidden_size = 768
        # conv_block = True
        
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoders = nn.ModuleList([
            Decoder((4+8) * init_channel_number, 4 * init_channel_number),
            Decoder((2+4) * init_channel_number, 2 * init_channel_number),
            Decoder((1+2) * init_channel_number, init_channel_number)
        ])

        self.attentions = nn.ModuleList([
            AttentionBlock(4 * init_channel_number, 8 * init_channel_number, init_channel_number),
            AttentionBlock(2 * init_channel_number, 4 * init_channel_number, init_channel_number),
            None
        ])
        # 1×1×1 convolution reduces the number of output channels to the number of class
        self.final_conv = nn.Conv3d(init_channel_number, out_channels, 1)

        if final_sigmoid_flag:
            self.final_activation = nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)
    
    def proj_feat(self, x, hidden_size, grid_size):
        new_view = (x.size(0), *grid_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(grid_size)))
        x = x.permute(new_axes).contiguous()
        return x

    # def forward(self, x):
    def forward(self, x_in, x, hidden_states_out):
        enc1 = self.encoder1(x_in)
        
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.grid_size))
        
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.grid_size))
        
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.grid_size))
        
        dec4 = self.proj_feat(x, self.hidden_size, self.grid_size)
        dec3 = self.decoder5(dec4, enc4)
        
        encoders_features = [enc4, enc3, enc2, enc1]
        x = dec3

        # for encoder in self.encoders:
        #     x = encoder(x)
        #     encoders_features.insert(0, x)

        # extremely important!! remove the last encoder's output from the list
        first_layer_feature = encoders_features[-1]
        encoders_feature = encoders_features[1:]

        for decoder, attention, encoder_feature in zip(self.decoders, self.attentions, encoders_feature):
            if attention:
                features_after_att = attention(encoder_feature, x, first_layer_feature)
            else:    # no attention opr in first layer
                features_after_att = first_layer_feature
            x = decoder(features_after_att, x)

        x = self.final_conv(x)
        if hasattr(CSA_UNETR_decoder, 'final_activation'):
            x = self.final_activation(x)
        return x


# class Encoder(nn.Module):
#     def __init__(self, in_channels, out_channels, conv_kernel_size=3,
#                  max_pool_flag=True, max_pool_kernel_size=(2, 2, 2)):
#         super(Encoder, self).__init__()
#         self.max_pool = nn.MaxPool3d(kernel_size=max_pool_kernel_size, stride=2) if max_pool_flag else None
#         self.double_conv = DoubleConv(in_channels, out_channels, kernel_size=conv_kernel_size)

#     def forward(self, x):
#         if self.max_pool is not None:
#             x = self.max_pool(x)
#         x = self.double_conv(x)
#         return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super(Decoder, self).__init__()
        self.upsample = nn.ConvTranspose3d(2*out_channels, 2*out_channels, kernel_size, scale_factor, padding=1, output_padding=1)
        self.double_conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, encoder_features, x):
        x = self.upsample(x)
        x = torch.cat((encoder_features, x), dim=1)
        x = self.double_conv(x)
        return x


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DoubleConv, self).__init__()

        if in_channels < out_channels:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels // 2
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_conv(1, conv1_in_channels, conv1_out_channels, kernel_size)
        # conv2
        self.add_conv(2, conv2_in_channels, conv2_out_channels, kernel_size)


    def add_conv(self, pos, in_channels, out_channels, kernel_size):
        assert pos in [1, 2], 'pos must be either 1 or 2'

        self.add_module(f'conv{pos}', nn.Conv3d(in_channels, out_channels, kernel_size, padding=1))
        self.add_module(f'relu{pos}', nn.ReLU(inplace=True))
        self.add_module(f'norm{pos}', nn.BatchNorm3d(out_channels))


class AttentionBlock(nn.Module):
    def __init__(self, channel_l, channel_g, init_channel=64):
        super(AttentionBlock, self).__init__()
        self.W_x1 = nn.Conv3d(channel_l, channel_l, kernel_size=1)
        self.W_x2 = nn.Conv3d(channel_l, channel_g, kernel_size=int(channel_g/channel_l),
                              stride=int(channel_g/channel_l), padding=(channel_g//channel_l//2)-1)
        self.W_g1 = nn.Conv3d(init_channel, channel_l, kernel_size=int(channel_l/init_channel),
                              stride=int(channel_l/init_channel), padding=(channel_l//init_channel//2)-1)
        self.W_g2 = nn.Conv3d(channel_g, channel_g, kernel_size=1)
        self.relu = nn.ReLU()
        self.psi1 = nn.Conv3d(channel_l, out_channels=1, kernel_size=1)
        self.psi2 = nn.Conv3d(channel_g, out_channels=1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x_l, x_g, first_layer_f):
        # First Attention Operation
        first_layer_afterconv = self.W_g1(first_layer_f)
        xl_afterconv = self.W_x1(x_l)
        att_map_first = self.sig(self.psi1(self.relu(first_layer_afterconv + xl_afterconv)))
        xl_after_first_att = x_l * att_map_first

        # Second Attention Operation
        xg_afterconv = self.W_g2(x_g)
        xl_after_first_att_and_conv = self.W_x2(xl_after_first_att)
        att_map_second = self.sig(self.psi2(self.relu(xg_afterconv + xl_after_first_att_and_conv)))
        att_map_second_upsample = F.upsample(att_map_second, size=x_l.size()[2:], mode='trilinear')
        out = xl_after_first_att * att_map_second_upsample
        return out


if __name__ == '__main__':
    model = CSA_UNETR_decoder(1, 5, final_sigmoid_flag=True, init_channel_number=16).cuda()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    summary(model, (1, 96, 96, 96), batch_size=4)

