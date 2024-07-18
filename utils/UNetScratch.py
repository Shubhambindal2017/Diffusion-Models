import torch
from torch import nn
from torchsummary import summary

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_connection=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.skip_connection=skip_connection
        self.in_channels = in_channels
        self.out_channels = out_channels

        if skip_connection and in_channels!=out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            ## In some other - noticed it got declared and use shortcut in forward

    def forward(self, x):
        outputConv1 = self.conv1(x)
        outputConv2 = self.conv2(outputConv1)
        if self.skip_connection:
            if self.in_channels==self.out_channels:
                output = x+outputConv2
            else:
                output = self.shortcut(x)+outputConv2
            return output / 1.414
        else:
            return outputConv2

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, downSampleScale, isOutputEncodingEmd=False):
        super().__init__()
        if isOutputEncodingEmd:
            self.model = nn.Sequential(nn.AvgPool2d(downSampleScale), nn.GELU())
        else:
            self.model = nn.Sequential(
                ResNetBlock(in_channels, out_channels),
                ResNetBlock(out_channels, out_channels),
                nn.MaxPool2d(downSampleScale),
            )

    def forward(self, x):
        return self.model(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, downSampleScale, isInputEncodingEmd=False):
        super().__init__()
        ## output size formula - (H-1)*S + K ; 
        ## where H - input size ; S - stride ; K - Kernel size
        ## if want to upsample by 4 - if input size - 1 then K - 4, S - 4 [no effect]
        ## if want to upsample by 2 - if input size - H (2 divisible) then K - 2, S - 2
        ## seems keeping kernel_size and stride - 2 make them upsample by 2; while by 4 - upsample 4
        if isInputEncodingEmd:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=downSampleScale, stride=downSampleScale),
                nn.GroupNorm(8, out_channels),
                nn.ReLU()
            )
        else:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=downSampleScale, stride=downSampleScale),
                ResNetBlock(out_channels, out_channels),
                ResNetBlock(out_channels, out_channels),    
            )

    def forward(self, x, skip):
        if skip is not None:
            #print(x.shape, skip.shape)
            x = torch.concat((x, skip), 1)
        return self.model(x)
        
class embedFCLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.model = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )
    def forward(self, x):
        x = x.view(-1, self.in_channels)
        return self.model(x)

class UNet(nn.Module):
    '''
    Basic UNet from scratch
    Make sure downsampleList and h are correspondingly fine 
        - i.e h after downsample should result in h-w of 1
          while num_channels at that output - will be the encoding embd size
    '''
    def __init__(self, in_channels, c_size, f_size, h, downsampleList, inChannels_outChannelsList):
        '''
        in_channels - num of channels in input
        c_size - context size length
        f_size - num of features from input
        h - height of input ; width of input should also be same
        downsampleList - list of downsampling scales
        inChannels_outChannelsList - should be of same length as downsampleList; 
                                        each element in this will be a tuple for that downsampleElement
                                        each tuple denotes (in_ch, out_ch)
        '''
        self.c_size = c_size
        self.in_channels = in_channels
        super().__init__()
        self.inital_conv = ResNetBlock(in_channels, f_size, skip_connection=True)
        
        self.downs = nn.ModuleList()
        for i in range(len(downsampleList)):
            isOutputEncodingEmd = False
            downSampleScale = downsampleList[i]
            in_ch, out_ch = inChannels_outChannelsList[i]
            if i==len(downsampleList)-1:
                isOutputEncodingEmd=True
            self.downs.append(DownSample(in_ch, out_ch, downSampleScale, isOutputEncodingEmd))

        self.contexts = nn.ModuleList()
        for i in range(len(downsampleList)-2, -1, -1):
            in_ch, out_ch = inChannels_outChannelsList[i]
            self.contexts.append(embedFCLayer(c_size, out_ch))

        self.timesteps = nn.ModuleList()
        for i in range(len(downsampleList)-2, -1, -1):
            in_ch, out_ch = inChannels_outChannelsList[i]
            self.timesteps.append(embedFCLayer(1, out_ch))

        self.ups = nn.ModuleList()
        for i in range(len(downsampleList)-1, -1, -1):
            upSampleScale = downsampleList[i]
            in_ch, out_ch = inChannels_outChannelsList[i]
            # multiplying by 2 because one will be from below to up - while other connection from left side - i.e output from upsample side
            if i == len(downsampleList)-1:
                self.ups.append(UpSample(out_ch, in_ch, upSampleScale, isInputEncodingEmd=True))
            else:
                self.ups.append(UpSample(2*out_ch, in_ch, upSampleScale))

        in_ch, out_ch = inChannels_outChannelsList[i]
        self.output = nn.Sequential(
            nn.Conv2d(2 * in_ch, in_ch, 3, 1, 1), # reduce number of feature maps 
            nn.GroupNorm(8, in_ch), # normalize
            nn.ReLU(),
            nn.Conv2d(in_ch, in_channels, 3, 1, 1), # map to same number of channels as input
        )

    def forward(self, x, t, c=None):
        '''
        x - input - size [batch, in_channels, h, w]
        t - timestamp - size [batch, timestamp_scaler i.e size of 1, 1, 1]
        c - context - size [batch, context i.e size of c_size, 1, 1]
        '''        
        if c is None:
            c = torch.zeros(x.shape[0], self.c_size).to(x.device)

        #print(f'x : shape : {x.shape}')
        x = self.inital_conv(x)
        #print(f'x : shape : {x.shape}')
        downSampleOutput = []
        ds = x
        for downSample in self.downs:
            ds = downSample(ds)
            #print(f'ds : shape : {ds.shape}')
            downSampleOutput.append(
                ds
            )
        encodingEmbdVector = downSampleOutput[-1]
        upSampleOutput = []
        us = encodingEmbdVector
        for i in range(len(self.ups)):
            upSample = self.ups[i]
            if i == 0:
                skip=None
                us = upSample(us, skip=skip)
            else:
                ## Not -1 actual meaning - instead just -1 because c and t is one less in list ; not for i=0
                skip = downSampleOutput[len(self.downs)-1-i]
                cEmbd = self.contexts[i-1](c).view(-1, skip.shape[1], 1, 1)
                tEmbd = self.timesteps[i-1](t).view(-1, skip.shape[1], 1, 1)
                assert skip.shape[1]==us.shape[1]
                #print(cEmbd.shape[0], x.shape[0], skip.shape[1])
                assert cEmbd.shape[0]==x.shape[0]
                us = upSample(cEmbd*us+tEmbd, skip=skip)
            #print(f'us : shape : {us.shape}')
            upSampleOutput.append(
                us
            )
        return self.output(torch.cat((upSampleOutput[-1], x), 1))

'''
in_channels = 3 
c_size = 5
f_size = 64
h = 16
downsampleList              =  [ 2, 2, 4]
inChannels_outChannelsList  =  [(f_size, f_size), (f_size, 2*f_size), (2*f_size, 2*f_size)]
uNetObj = UNet(in_channels, c_size, f_size, h, downsampleList, inChannels_outChannelsList)
output = uNetObj(torch.rand((8, 3, 16, 16)), torch.rand((8, 1, 1, 1)), torch.rand((8, 5, 1, 1)))
print(output.shape)
summary(uNetObj, [(3, 16, 16), (1, 1, 1), (5, 1, 1)])
#print(output)

from torchviz import make_dot

make_dot(output, params=dict(list(uNetObj.named_parameters()))).render("rnn_torchviz", format="png")
'''


'''
S - Size
n_feat - f_size
Input         - [B, 3, S, S]      i.e [B, 3, 16, 16]
conv-init     - [B, n_feat, S, S] i.e [B, 64, 16, 16]      
DS1           - [B, 2*n_feat, S/2, S/2] i.e [B, 128, 8, 8]     
DS2           - [B, 2*n_feat, S/4, S/4] i.e [B, 128, 4, 4]         
DS3           - [B, 2*n_feat, S/16, S/16] i.e [B, 128, 1, 1]     
EncodingVector - [B, 128, 1, 1]     
US1            - [B, 128, 4, 4]
US2            - [B, 128, 4, 4] + DS2 i.e [B, 2* (2*feat), 4, 4]-> [B, 2*feat, 8, 8] i.e [B, 128, 8, 8]
similarly for US3 - output - [B, 64, 16, 16]
'''