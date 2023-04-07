'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, in_channels=3, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(in_channels, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)
        
        self.inter_feature = {}
        self.inter_gradient = {}
        self.register_all_hooks()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
    def from_input_to_features(self, x, index):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        
        
        return out
        
    def from_features_to_output(self, x, index):
        out = F.avg_pool2d(F.relu(self.bn(x)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
    def make_hook(self, name, flag):
        if flag == 'forward':
            def hook(m, input, output):
                #print(name)
                self.inter_feature[name] = output
            return hook
        elif flag == 'backward':
            def hook(m, input, output):
                self.inter_gradient[name] = output
            return hook
        else:
            assert False
            
    def register_all_hooks(self):
        self.conv1.register_forward_hook(self.make_hook("Conv1_Conv1_Conv1_", 'forward'))

        lenth_each_dense_layer = [6,12,24,16]
        
        '''for i in range(1,5):
            if j in [lenth_each_dense_layer[i]]:'''
            
        for j in range(6):
            self.dense1[j].conv1.register_forward_hook(self.make_hook("dense1_"+str(j)+"_conv1_", 'forward'))
            self.dense1[j].conv2.register_forward_hook(self.make_hook("dense1_"+str(j)+"_conv2_", 'forward'))
        self.trans1.conv.register_forward_hook(self.make_hook("trans1"+"_conv_conv_", 'forward'))
             
        for j in range(12):
            self.dense2[j].conv1.register_forward_hook(self.make_hook("dense2_"+str(j)+"_conv1_", 'forward'))
            self.dense2[j].conv2.register_forward_hook(self.make_hook("dense2_"+str(j)+"_conv2_", 'forward'))
        self.trans2.conv.register_forward_hook(self.make_hook("trans2"+"_conv_conv_", 'forward'))

        for j in range(24):
            self.dense3[j].conv1.register_forward_hook(self.make_hook("dense3_"+str(j)+"_conv1_", 'forward'))
            self.dense3[j].conv2.register_forward_hook(self.make_hook("dense3_"+str(j)+"_conv2_", 'forward'))
        self.trans3.conv.register_forward_hook(self.make_hook("trans3"+"_conv_conv_", 'forward'))
        
        for j in range(16):
            self.dense4[j].conv1.register_forward_hook(self.make_hook("dense4_"+str(j)+"_conv1_", 'forward'))
            self.dense4[j].conv2.register_forward_hook(self.make_hook("dense4_"+str(j)+"_conv2_", 'forward'))
        
        
def DenseNet121(num_classes=10, in_channels=3):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, in_channels=in_channels, num_classes=num_classes)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

def test():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()