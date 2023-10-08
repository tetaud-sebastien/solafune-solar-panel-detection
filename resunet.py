from torch.nn import functional as F
import torch
from torch import nn

    


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, double_conv = True, norm='none',
                 activation='relu', pad_type='reflect'):
        """
        
        """
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zeros':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        self.double_conv = double_conv
        # initialize normalization
        norm_dim = out_channels
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=conv_padding, dilation=dilation,
                              bias=self.use_bias, padding_mode=pad_type)

        # self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
        #                       padding=1, dilation=1,
        #                       bias=self.use_bias, padding_mode=pad_type)



    def forward(self, xin):
        if self.pad:
            x = self.conv(self.pad(xin))
        else:
            x = self.conv(xin)
        
        # if self.double_conv:

        #     x = self.conv1(x)

        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class TransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='reflect',
                 activation='lrelu', norm='in', scale_factor=2):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, 
                                  pad_type=pad_type,
                                  activation=activation, norm=norm)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.conv2d(x)
        return x


def generator_transposed_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                              activation='relu',
                              norm='none', scale_factor=2):
    return TransposeConv2dLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride,
                                padding=padding, dilation=dilation, activation=activation, norm=norm,
                                scale_factor=scale_factor)


def generator_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',
                   norm='none'):
    return Conv2dBlock(input_channels=in_channels, output_dim=out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation,  activation=activation, norm=norm)



# class SimpleResidualBlock(nn.Module):

#     def __init__(self, num_channels):
#         super().__init__()
#         self.conv1 = Conv2dBlock(in_channels=num_channels, out_channels=, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',norm='bn')
#         self.conv2 = Conv2dBlock(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',norm='bn')

#     def forward(self, x):
#         print(x.shape)
#         out = self.conv1(x)
#         print(out.shape)

#         out = self.conv2(out)
#         print(out.shape)
#         return out + x 

class Encoder(nn.Module):
    def __init__(self, num_channels=3):
        super(Encoder, self).__init__()
        
        
        # Encoder
        self.conv0 = Conv2dBlock(in_channels=num_channels, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, activation='relu',norm='bn')

        self.conv1 = Conv2dBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, activation='relu',norm='bn')
        self.conv1_1 = Conv2dBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',norm='bn')

        self.conv2 = Conv2dBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1, activation='relu',norm='bn')
        self.conv2_1 = Conv2dBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',norm='bn')

        self.conv3 = Conv2dBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1, activation='relu',norm='bn')
        self.conv3_1 = Conv2dBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',norm='bn')

        self.conv4 = Conv2dBlock(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, dilation=1, activation='relu',norm='bn')
        self.conv4_1 = Conv2dBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',norm='bn')

        self.conv5 = Conv2dBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1, dilation=1, activation='relu',norm='bn')


    def forward(self, x):

        # Encoder
        
        X0 = self.conv0(x) # (224X224X3) -> (112X112X64)
        
        X1 = self.conv1(X0) # (112X112X64) -> (56X56X64)
        X1_1 = self.conv1_1(X1)
        X1_1 = X1 + X1_1
        

        X2 = self.conv2(X1_1) # (56X56X64) -> (28X28X128)
        X2_1 = self.conv2_1(X2)
        X2_1 = X2 + X2_1

        X3 = self.conv3(X2) # (28X28X128) -> (14X14X256)
        X3_1 = self.conv3_1(X3)
        X3_1 = X3 + X3_1

        X4 = self.conv4(X3_1) # (7X7X256) -> (7X7X512)
        X4_1 = self.conv4_1(X4)
        X4 = X4_1 + X4


        X5 = self.conv5(X4_1) # (7X7X512) -> (4x4x1024)
       

        return [X0, X1, X2, X3, X4, X5]


class SegDecoder(nn.Module):
    def __init__(self, num_channels, num_class):
        super(SegDecoder, self).__init__()
        
       
        
        # Decoder
        self.t_conv0 = TransposeConv2dLayer(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',norm='bn')

        self.t_conv1 = TransposeConv2dLayer(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',norm='bn',scale_factor=2)
        self.t_conv1_1 = TransposeConv2dLayer(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',norm='bn',scale_factor=1)


        self.t_conv2 = TransposeConv2dLayer(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',norm='bn')
        self.t_conv2_1 = TransposeConv2dLayer(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',norm='bn',scale_factor=1)

        self.t_conv3 = TransposeConv2dLayer(in_channels=384, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',norm='bn')
        self.t_conv3_1 = TransposeConv2dLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',norm='bn',scale_factor=1)

        
        self.t_conv4 = TransposeConv2dLayer(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',norm='bn')
        self.t_conv4_1 = TransposeConv2dLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',norm='bn',scale_factor=1)

        self.t_conv5 = TransposeConv2dLayer(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, activation='relu',norm='bn')
        self.out = Conv2dBlock(in_channels=64, out_channels=num_class, kernel_size=1, stride=1, padding=0, dilation=1, activation='sigmoid',norm='none')
        

    def forward(self, x):

        X0, X1, X2, X3, X4, X5 = x
        #Encoder
        t_X0 = self.t_conv0(X5)
        t_X0 = torch.cat((t_X0, X4), dim=1)

        t_X1 = self.t_conv1(t_X0)
        t_X1_1 = self.t_conv1_1(t_X1)
        t_X1_1 = t_X1_1 + t_X1
        t_X1 = torch.cat((t_X1_1, X3), dim=1)

        t_X2 = self.t_conv2(t_X1)
        t_X2_1 = self.t_conv2_1(t_X2)
        t_X2_1 = t_X2_1 + t_X2
        t_X2 = torch.cat((t_X2_1, X2), dim=1)

        t_X3 = self.t_conv3(t_X2)
        t_X3_3 = self.t_conv3_1(t_X3)
        t_X3_3 = t_X3_3 + t_X3
        t_X3 = torch.cat((t_X3_3, X1), dim=1)

        t_X4 = self.t_conv4(t_X3)
        t_X4_1 = self.t_conv4_1(t_X4)
        t_X4_1 = t_X4_1 + t_X4
        t_X4 = torch.cat((t_X4_1, X0), dim=1)

        t_X5 = self.t_conv5(t_X4)
        X = self.out(t_X5)

        return X



class Unet(nn.Module):
    def __init__(self, num_channels=3,num_class=1):
        super(Unet, self).__init__()
    
        
        self.encoder = Encoder()
        self.seg_decoder = SegDecoder(num_channels, num_class)
    

    def forward(self, x):

        encoded = self.encoder(x)
        seg_out = self.seg_decoder(encoded)
        
        return seg_out

if __name__== "__main__":


    x = torch.rand((1,3,64,64))
    net = Unet(num_channels=3,num_class=1)
    seg_out = net(x)
    print(seg_out.shape)



