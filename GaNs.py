#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:54:01 2019

@author: sachin
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

batch_size = 64
image_size = 64

transform = transforms.Compose([transforms.Scale(image_size), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

dataset = dset.CIFAR10(root = "./data", download = True, transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size,
                                         shuffle = True, num_workers = 2)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
class G(nn.Module):
    
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channels = 100, out_channels = 512,
                                   kernel_size = 4,stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(num_features = 512),                
                nn.ReLU(True),
                
                nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 4,
                                   stride = 2, padding = 1, bias = False),
                nn.BatchNorm2d(num_features = 256),   
                nn.ReLU(True),       
                
                nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4,
                                   stride = 2, padding = 1, bias = False),
                nn.BatchNorm2d(num_features = 128),   
                nn.ReLU(True), 
                 
                nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4,
                                   stride = 2, padding = 1, bias = False),
                nn.BatchNorm2d(num_features = 64),   
                nn.ReLU(True),
                
                nn.ConvTranspose2d(in_channels = 64, out_channels = 3, kernel_size = 4,
                                   stride = 2, padding = 1, bias = False),
                nn.Tanh()
        )
         
    def forward(self, input):
        output = self.main(input)
        return output
    
netG = G()
netG.apply(weights_init)            

class D(nn.Module):
    
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 4,
                          stride = 2, padding = 1, bias = False),
                nn.LeakyReLU(negative_slope = 0.2, inplace = True),
                
                nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4,
                          stride = 2, padding = 1, bias = False),
                nn.BatchNorm2d(num_features = 128),          
                nn.LeakyReLU(negative_slope = 0.2, inplace = True),
                
                nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4,
                          stride = 2, padding = 1, bias = False),
                nn.BatchNorm2d(num_features = 256),          
                nn.LeakyReLU(negative_slope = 0.2, inplace = True),
                
                nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4,
                          stride = 2, padding = 1, bias = False),
                nn.BatchNorm2d(num_features = 512),          
                nn.LeakyReLU(negative_slope = 0.2, inplace = True),
                
                nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 4,
                          stride = 1, padding = 0, bias = False),
                nn.Sigmoid()          
                )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)
    
netD = D()
netD.apply(weights_init)

# Training Phase.

criterion = torch.nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.9, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.9, 0.999))

for epoch in range(0, 25):
    for i, data in enumerate(dataloader, 0):
        
        # Training Descriminator.
        netD.zero_grad()
        real, _ = data
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0]))
        output = netD(input)
        errD_real = criterion(output, target)
        
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = netD(fake.detach())
        errD_fake = criterion(output, target)
        
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()
        
        # Training Generator.
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = netD(fake)
        errG = criterion(output, target)
        errG.backward()
        optimizerG.step()
        
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.item(), errG.item()))
        if i % 100 == 0:
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True)
            fake = netG(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)
             
        
        
        
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        