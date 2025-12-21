import torch
import torch.nn as nn
import torch.nn.functional as F


"""
resnet block is: 
f(x) + x:

x -> conv -> norm -> relu -> conv -> norm -> (shortcut(+)) -> relu

f(x) = 2 conv layers (with norm and relu)
x = identity shortcut
output = f(x) + x then relu

"""


#base
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        #x shortcut
        if stride !=1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + self.shortcut(x)
        out = F.relu(x)
            
        return out
        













