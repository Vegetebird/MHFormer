import torch
import torch.nn as nn
from einops import rearrange
from model.module.trans import Transformer as Transformer_s
from model.module.trans_hypothesis import Transformer

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.norm_1 = nn.LayerNorm(args.frames)
        self.norm_2 = nn.LayerNorm(args.frames)
        self.norm_3 = nn.LayerNorm(args.frames)

        self.trans_auto_1 = Transformer_s(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        self.trans_auto_2 = Transformer_s(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        self.trans_auto_3 = Transformer_s(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)

        self.encoder_1 = nn.Sequential(nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1))
        self.encoder_2 = nn.Sequential(nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1))
        self.encoder_3 = nn.Sequential(nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1))

        self.Transformer = Transformer(args.layers, args.channel*3, args.d_hid, length=args.frames)
        
        self.fcn = nn.Sequential(
            nn.BatchNorm1d(args.channel*3, momentum=0.1),
            nn.Conv1d(args.channel*3, 3*args.out_joints, kernel_size=1)
        )

    def forward(self, x):
        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()

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
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

        return x






