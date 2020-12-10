# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 06:18:58 2020

@author: david
"""

import torch

class SLP(torch.nn.Module):
    "Neurona de una sola capa"
    
    def __init__(self, input_shape, output_shape, device):
        
        super(SLP, self).__init__()
        self.device = device
        self.input_shape = input_shape[0]
        self.hidden_shape = 40
        
        if str(self.device) == 'cuda':
            self.linear1 = torch.nn.Linear(self.input_shape, self.hidden_shape).cuda()
            self.out = torch.nn.Linear(self.hidden_shape, output_shape).cuda()
        else:
            self.linear1 = torch.nn.Linear(self.input_shape, self.hidden_shape)
            self.out = torch.nn.Linear(self.hidden_shape, output_shape)
        
    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.out(x)
        return x
    