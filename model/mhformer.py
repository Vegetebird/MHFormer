import torch
import torch.nn as nn
from model.module.trans import Transformer as Transformer_Encoder
from model.module.trans_hypothesis import Transformer

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        layers, channel, d_hid, length  = args.layers, args.channel, args.d_hid, args.frames
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints

        self.norm_1 = nn.LayerNorm(length)
        self.norm_2 = nn.LayerNorm(length)
        self.norm_3 = nn.LayerNorm(length)

        self.trans_auto_1 = Transformer_Encoder(4, length, length*2, length=2*self.num_joints_in, h=9)
        self.trans_auto_2 = Transformer_Encoder(4, length, length*2, length=2*self.num_joints_in, h=9)
        self.trans_auto_3 = Transformer_Encoder(4, length, length*2, length=2*self.num_joints_in, h=9)

        self.encoder_1 = nn.Sequential(nn.Conv1d(2*self.num_joints_in, channel, kernel_size=1))
        self.encoder_2 = nn.Sequential(nn.Conv1d(2*self.num_joints_in, channel, kernel_size=1))
        self.encoder_3 = nn.Sequential(nn.Conv1d(2*self.num_joints_in, channel, kernel_size=1))

        self.Transformer = Transformer(layers, channel*3, d_hid, length=length)
        
        self.fcn = nn.Sequential(
            nn.BatchNorm1d(channel*3, momentum=0.1),
            nn.Conv1d(channel*3, 3*self.num_joints_out, kernel_size=1)
        )

    def forward(self, x):
        x = x[:, :, :, :, 0].permute(0, 2, 3, 1).contiguous() 
        x = x.view(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1).contiguous()  

        ## MHG
        x_1 = x   + self.trans_auto_1(self.norm_1(x))
        x_2 = x_1 + self.trans_auto_2(self.norm_2(x_1)) 
        x_3 = x_2 + self.trans_auto_3(self.norm_3(x_2))
        
        ## Embedding
        x_1 = self.encoder_1(x_1)
        x_1 = x_1.permute(0, 2, 1).contiguous() 

        x_2 = self.encoder_2(x_2)
        x_2 = x_2.permute(0, 2, 1).contiguous()

        x_3 = self.encoder_3(x_3) 
        x_3 = x_3.permute(0, 2, 1).contiguous()

        ## SHR & CHI
        x = self.Transformer(x_1, x_2, x_3) 

        ## Head
        x = x.permute(0, 2, 1).contiguous() 
        x = self.fcn(x) 

        x = x.view(x.shape[0], self.num_joints_out, -1, x.shape[2]) 
        x = x.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1) 

        return x






