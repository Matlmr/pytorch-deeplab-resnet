import torch.nn as nn
import torch.nn.functional as F
#import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from copy import deepcopy
affine_par = True


def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation_)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):

    def __init__(self,dilation_series,padding_series,NoLabels):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(2048,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)


    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

# Pyramid Pooling Module
class PPM(nn.Module):
    def __init__(self,NoLabels):
        super(PPM,self).__init__()
        self.conv2d_list = nn.ModuleList()
        for i in (1,2,3,6):
            pool = nn.AdaptiveAvgPool2d(output_size=i)
            conv = nn.Conv2d(in_channels = 2048, out_channels = 512, kernel_size = 1)
            self.conv2d_list.append(nn.Sequential(pool,conv))
        self.conv2d = nn.Conv2d(in_channels = 4096, out_channels = NoLabels, kernel_size = 1)
        
    def forward(self,x):
        concat = x
        for i in range(len(self.conv2d_list)):
            level = F.upsample(input = self.conv2d_list[i](x), size = (x.size(2), x.size(3)), mode = 'bilinear')
            concat = torch.cat((concat, level), dim = 1)
        out = self.conv2d(concat)
        return out

class PSPModule(nn.Module):
    """
    Pyramid Scene Parsing module
    """
    def __init__(self, in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=1):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage_1(in_features, size) for size in sizes])
        self.bottleneck = self._make_stage_2(in_features * (len(sizes)//4 + 1), out_features)
        self.relu = nn.ReLU()
        self.final = nn.Conv2d(out_features, n_classes, kernel_size=1)

    def _make_stage_1(self, in_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_features, in_features//4, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(in_features//4, affine=affine_par)
        relu = nn.ReLU(inplace=True)

        return nn.Sequential(prior, conv, bn, relu)

    def _make_stage_2(self, in_features, out_features):
        conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features, affine=affine_par)
        relu = nn.ReLU(inplace=True)

        return nn.Sequential(conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages]
        priors.append(feats)
        bottle = self.relu(self.bottleneck(torch.cat(priors, 1)))
        out = self.final(bottle)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers,NoLabels, psp = False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation__ = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 4)
        if not psp:
            self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],NoLabels)
        else:
            #self.layer5 = PSPModule(n_classes=NoLabels)
            self.layer5 = PPM(NoLabels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation_=dilation__))

        return nn.Sequential(*layers)
    
    def _make_pred_layer(self,block, dilation_series, padding_series,NoLabels):
        return block(dilation_series,padding_series,NoLabels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x


class MS_Deeplab(nn.Module):
    def __init__(self,block,NoLabels, psp = False):
        super(MS_Deeplab,self).__init__()
        self.Scale = ResNet(block,[3, 4, 23, 3],NoLabels, psp)   #changed to fix #4
        self.psp = psp

    def forward(self,x):
        if not self.psp:
            input_size = x.size()[2]
            self.interp1 = nn.Upsample(size = (int(input_size*0.75)+1,int(input_size*0.75)+1),mode='bilinear')
            self.interp2 = nn.Upsample(size = (int(input_size*0.5)+1,int(input_size*0.5)+1),mode='bilinear')
            self.interp3 = nn.Upsample(size = (outS(input_size),outS(input_size)),mode='bilinear')
            out = []
            x2 = self.interp1(x)
            x3 = self.interp2(x)
            out.append(self.Scale(x))        # for original scale
            out.append(self.interp3(self.Scale(x2)))        # for 0.75x scale
            out.append(self.Scale(x3))        # for 0.5x scale


            x2Out_interp = out[1]
            x3Out_interp = self.interp3(out[2])
            temp1 = torch.max(out[0],x2Out_interp)
            out.append(torch.max(temp1,x3Out_interp))
            return out
        else:
            x = self.Scale(x)
            out = []
            out.append(x)
            return out

    def load_pretrained_ms(self, base_network, nInputChannels=3):
        flag = 0
        for container, container_ori in zip(self.Scale.modules(), base_network.modules()):
            for module, module_ori in zip(container.modules(), container_ori.modules()):
                #if isinstance(module, nn.Conv2d):
                    #assert(3 == 4)
                #    assert(True)
                #if isinstance(module_ori, nn.Conv2d):
                    #assert(5 == 6)
                #    assert(True)
                #if isinstance(module, nn.Conv2d) and isinstance(module_ori, nn.Conv2d):
                #    assert(7 == 8)
                if isinstance(module, nn.Conv2d) and isinstance(module_ori, nn.Conv2d):

                    if not flag and nInputChannels != 3:
                        module.weight[:, :3, :, :].data = deepcopy(module_ori.weight.data)
                        module.bias = deepcopy(module_ori.bias)
                        for i in range(3, int(module.weight.data.shape[1])):
                            module.weight[:, i, :, :].data = deepcopy(module_ori.weight[:, -1, :, :][:, np.newaxis, :, :].data)
                        flag = 1
                    elif module.weight.data.shape == module_ori.weight.data.shape:
                        print('Updating convolutional layer')
                        module.weight = deepcopy(module_ori.weight)
                        module.bias = deepcopy(module_ori.bias)
                    else:
                        print('Skipping Conv layer with size: {} and target size: {}'
                              .format(module.weight.data.shape, module_ori.weight.data.shape))
                elif isinstance(module, nn.BatchNorm2d) and isinstance(module_ori, nn.BatchNorm2d) \
                        and module.weight.data.shape == module_ori.weight.data.shape:
                    print('Updating batchnorm layer')
                    module.weight.data = deepcopy(module_ori.weight.data)
                    module.bias.data = deepcopy(module_ori.bias.data)

def Res_Deeplab(NoLabels=21, psp=False):
    model = MS_Deeplab(Bottleneck,NoLabels, psp)
    #model = PSP(Bottleneck,NoLabels)
    #model = ResNet(block,[3, 4, 23, 3],NoLabels)

    return model
