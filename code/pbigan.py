from collections import defaultdict
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import pprint
import argparse
from decoder import ConvDecoder, dconv_bn_relu, dconv_bn_relu3d_nodep,dconv_bn_relu3d_dep
from encoder import ConvEncoder, conv_ln_lrelu, conv_ln_lrelu3d
from torch.distributions.normal import Normal
from mmd import mmd
import flow
import pdb
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from resunet import Resnet_block
from visualize import Visualizer

def mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:1' if use_cuda else 'cpu')


class CUnetPBiGAN(nn.Module):
    # motivated by https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
    def __init__(self, in_channels, out_channels, latent_size, flow_depth, sample, device,dropout=False, logprob=False):
        super().__init__()
        dim = 64
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.device = device

        self.down1 = nn.Sequential(
            nn.Conv2d(1, dim, 5, 2, 2), 
            nn.LeakyReLU(0.2)
        )
        self.down2 = conv_ln_lrelu(dim, dim * 2)
        self.down3 = conv_ln_lrelu(dim * 2, dim * 4)
        self.down4 = conv_ln_lrelu(dim * 4, dim * 4)
        self.down5 = conv_ln_lrelu(dim * 4, dim * 8)
        self.down6 = conv_ln_lrelu(dim * 8, dim * 8)
        self.down_end = nn.Conv2d(dim * 8, latent_size, 4)
        
        self.down1_t = nn.Sequential(
            nn.Conv2d(in_channels, dim, 5, 2, 2), 
            nn.LeakyReLU(0.2)
        )
        self.down2_t = conv_ln_lrelu(dim, dim * 2)
        self.down3_t = conv_ln_lrelu(dim * 2, dim * 4)
        self.down4_t = conv_ln_lrelu(dim * 4, dim * 4)
        self.down5_t = conv_ln_lrelu(dim * 4, dim * 8)
        self.down6_t = conv_ln_lrelu(dim * 8, dim * 8)
        self.down_end_t = nn.Conv2d(dim * 8, latent_size, 4)

        self.enc_chunk = 2
            
        fc_out_size = latent_size * self.enc_chunk
     
        self.fc = nn.Sequential(
            nn.Linear(latent_size, fc_out_size),
            nn.LayerNorm(fc_out_size),
            nn.LeakyReLU(0.2),
            nn.Linear(fc_out_size, fc_out_size),
        )

        self.fc_t = nn.Sequential(
            nn.Linear(latent_size, fc_out_size),
            nn.LayerNorm(fc_out_size),
            nn.LeakyReLU(0.2),
            nn.Linear(fc_out_size, fc_out_size),
        )

        self.sample = sample
        
        # decoder
        self.l1 = nn.Sequential(
            nn.Linear(latent_size*2, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())

        self.up1 = dconv_bn_relu(dim * 8, dim * 8)
        self.up2 = dconv_bn_relu(dim * 8, dim * 4)
        self.up3 = dconv_bn_relu(dim * 4, dim * 4)
        self.up4 = dconv_bn_relu(dim * 4, dim * 2 )
        self.up5 = dconv_bn_relu(dim * 2, dim )
        self.up_end = nn.ConvTranspose2d(dim, self.out_channels, 5, 2,
        padding=2, output_padding=1)

        
    def forward(self, target, target_bpr, conditional,cond_bpr):
        
        x_t = self.down1_t(target)  # x_t: 32, 1, 256, 256
        x_t = self.down2_t(x_t) # x1_t: 32, 64, 128, 128
        x_t = self.down3_t(x_t) # x2_t: 32, 128, 64, 64
        x_t = self.down4_t(x_t) # x3_t:32, 256, 32, 32
        x_t = self.down5_t(x_t) # x4_t:32, 256, 16, 16
        x_t = self.down6_t(x_t) # x5_t: 32, 512, 8, 8
        x_t = self.down_end_t(x_t) # x6_t: 32, 512, 4, 4
                                       # x7_t: 32, 128, 1, 1
        
        # x_mask = x * mask
        x_c = self.down1(conditional)  # 32, 64, 64, 64
        x_c = self.down2(x_c) # 32, 128, 32, 32
        x_c = self.down3(x_c) # 32, 256, 16, 16
        x_c = self.down4(x_c) # 32, 512, 8, 8
        x_c = self.down5(x_c) # 32, 512, 4, 4
        x_c = self.down6(x_c)
        x_c = self.down_end(x_c) # 32, 128, 1, 1

        x7_t = x_t.view(target.shape[0], -1)
        fc_out_t = self.fc(x7_t).chunk(self.enc_chunk, dim = 1) # 32x384 -> [[32, 128], [32, 128], [32, 128]]

        x7_c = x_c.view(conditional.shape[0], -1)
        fc_out_c = self.fc(x7_c).chunk(self.enc_chunk, dim = 1) # 32x384 -> [[32, 128], [32, 128], [32, 128]]
        
        mu_t, logvar_t = fc_out_t[:2]
        mu_c, logvar_c = fc_out_c[:2]
        
        if self.sample:
            std = F.softplus(logvar_t)
            # print(mu_t, std)
            qz_x = Normal(mu_t, std)
            z_T = qz_x.rsample()
            
            std = F.softplus(logvar_c)
            qz_x_c = Normal(mu_c, std)
            z_C = qz_x_c.rsample()
            
        else:
            z_T = mu_t
            z_C = mu_c

       
        z_recon = torch.cat((z_C, z_T), dim=1)
        
        z_prior = torch.randn((z_C.shape[0], self.latent_size)).to(self.device)
        z_gen = torch.cat((z_C, z_prior), dim=1)
       
        z_recon =  F.normalize(z_recon)
        z_gen =  F.normalize(z_gen)
        x_gen = self.l1(z_gen)
        x_gen = x_gen.view(x_gen.shape[0], -1, 4, 4)
        x_gen = self.up1(x_gen)
        x_gen = self.up2(x_gen)
        x_gen = self.up3(x_gen)
        x_gen = self.up4(x_gen)
        x_gen = self.up5(x_gen)
        x_gen = self.up_end(x_gen)
        x_gen = torch.sigmoid(x_gen)

        x_recon = self.l1(z_recon)
        x_recon = x_recon.view(x_recon.shape[0], -1, 4, 4)
        x_recon = self.up1(x_recon)
        x_recon = self.up2(x_recon)
        x_recon = self.up3(x_recon)
        x_recon = self.up4(x_recon)
        x_recon = self.up5(x_recon)
        x_recon = self.up_end(x_recon)
        x_recon = torch.sigmoid(x_recon)

        return z_T, z_gen, x_gen, x_recon, mu_t, logvar_t

    def generate_img(self, x_c):
        x1_c = self.down1(x_c)  # 32, 64, 64, 64
        x2_c = self.down2(x1_c) # 32, 128, 32, 32
        x3_c = self.down3(x2_c) # 32, 256, 16, 16
        x4_c = self.down4(x3_c) # 32, 512, 8, 8
        x5_c = self.down5(x4_c) # 32, 512, 4, 4
        x6_c = self.down6(x5_c)
        x7_c = self.down_end(x6_c) # 32, 128, 1, 1


        x7_c = x7_c.view(x_c.shape[0], -1)
        fc_out_c = self.fc(x7_c).chunk(self.enc_chunk, dim = 1) # 32x384 -> [[32, 128], [32, 128], [32, 128]]
        
    
        mu_c, logvar_c = fc_out_c[:2]
        
        if self.sample:
        
            std = F.softplus(logvar_c)
            qz_x_c = Normal(mu_c, std)
            z_C = qz_x_c.rsample()
            
        else:                                                                                                                        
            z_C = mu_c

        z_prior = torch.randn((z_C.shape[0], self.latent_size)).to(self.device)
        z_gen = torch.cat((z_C, z_prior), dim=1)
        z_gen =  F.normalize(z_gen)

        x_gen = self.l1(z_gen)
        x_gen = x_gen.view(x_gen.shape[0], -1, 4, 4)
        x_gen = self.up1(x_gen)
        x_gen = self.up2(x_gen)
        x_gen = self.up3(x_gen)
        x_gen = self.up4(x_gen)
        x_gen = self.up5(x_gen)
        x_gen = self.up_end(x_gen)
        x_gen = torch.sigmoid(x_gen)

        return x_gen

class CResUnetPBiGAN(nn.Module):
    # motivated by https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
    def __init__(self, in_channels, latent_size, flow_depth, sample, device, dropout=False, logprob=False):
        super().__init__()
        dim = 32
        self.out_channels = 1
        self.latent_size = latent_size
        self.device = device

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, dim, 5, 2, 2), 
            nn.LeakyReLU(0.2)
        )
        self.down2 = Resnet_block(dim, dim , 2, dropout)
        self.down3 = Resnet_block(dim * 2, dim* 2, 2, dropout)
        self.down4 = Resnet_block(dim * 4, dim* 4, 2, dropout)
        self.down5 = Resnet_block(dim * 8, dim* 8, 1, dropout)
        self.down6 = Resnet_block(dim * 8, dim* 8, 2, dropout)
        self.down_end = nn.Conv2d(dim * 16, latent_size, 4)
        
        self.down1_t = nn.Sequential(
            nn.Conv2d(in_channels, dim, 5, 2, 2), 
            nn.LeakyReLU(0.2)
        )

        self.down2_t = Resnet_block(dim, dim , 2)
        self.down3_t = Resnet_block(dim * 2, dim* 2, 2)
        self.down4_t = Resnet_block(dim * 4, dim* 4, 2)
        self.down5_t = Resnet_block(dim * 8, dim* 8, 1)
        self.down6_t = Resnet_block(dim * 8, dim* 8, 2)
        self.down_end_t = nn.Conv2d(dim * 16, latent_size, 4)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_chunk = 2
            
        fc_out_size = latent_size * self.enc_chunk
     
        self.fc = nn.Sequential(
            nn.Linear(latent_size, fc_out_size),
            nn.LayerNorm(fc_out_size),
            nn.LeakyReLU(0.2),
            nn.Linear(fc_out_size, fc_out_size),
        )

        self.fc_t = nn.Sequential(
            nn.Linear(latent_size, fc_out_size),
            nn.LayerNorm(fc_out_size),
            nn.LeakyReLU(0.2),
            nn.Linear(fc_out_size, fc_out_size),
        )

        self.sample = sample
        
        # decoder
        self.l1 = nn.Sequential(
            nn.Linear(latent_size*2, dim * 16 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 16 * 4 * 4),
            nn.ReLU())

        self.upconv1 = nn.ConvTranspose2d(dim * 16, dim * 16, kernel_size=2, stride=2)
        self.up1 = Resnet_block(dim * 16, dim * 16, 0.5, dropout)
        self.upconv2 = nn.ConvTranspose2d(dim * 8, dim * 8, kernel_size=2, stride=2)
        self.up2 = Resnet_block(dim * 8, dim * 8, 1, dropout)
        self.upconv3 = nn.ConvTranspose2d(dim * 8, dim * 8, kernel_size=2, stride=2)
        self.up3 = Resnet_block(dim * 8, dim * 8, 0.5, dropout)
        self.upconv4 = nn.ConvTranspose2d(dim * 4, dim * 4, kernel_size=2, stride=2)
        self.up4 = Resnet_block(dim * 4, dim * 4, 0.5, dropout)
        self.upconv5 = nn.ConvTranspose2d(dim * 2, dim * 2, kernel_size=2, stride=2)
        self.up5 = Resnet_block(dim * 2, dim * 2, 0.5, dropout)
        self.up_end = nn.ConvTranspose2d(dim, self.out_channels, 5, 2,
        padding=2, output_padding=1)

        
    def forward(self, target, target_bpr, conditional,cond_bpr):
        
        x_t = self.down1_t(target)  # x_t: 32, 1, 256, 256
        x_t = self.down2_t(x_t) # x1_t: 32, 64, 128, 128
        x_t = self.maxpool(x_t)
        x_t = self.down3_t(x_t) # x2_t: 32, 128, 64, 64
        x_t = self.maxpool(x_t)
        x_t = self.down4_t(x_t) # x3_t:32, 256, 32, 32
        x_t = self.maxpool(x_t)
        x_t = self.down5_t(x_t) # x4_t:32, 256, 16, 16
        x_t = self.maxpool(x_t)
        x_t = self.down6_t(x_t) # x5_t: 32, 512, 8, 8
        x_t = self.maxpool(x_t)
        x_t = self.down_end_t(x_t) # x6_t: 32, 512, 4, 4
                                       # x7_t: 32, 128, 1, 1
        

        x_c = self.down1(conditional)  # torch.Size([32, 32, 128, 128])
        x_c = self.down2(x_c) # torch.Size([32, 64, 128, 128])
        x_c = self.maxpool(x_c) # torch.Size([32, 64, 64, 64])
        x_c = self.down3(x_c) # torch.Size([32, 128, 64, 64])
        x_c = self.maxpool(x_c) #torch.Size([32, 128, 32, 32])
        x_c = self.down4(x_c) # torch.Size([32, 256, 32, 32])
        x_c = self.maxpool(x_c) #torch.Size([32, 256, 16, 16])
        x_c = self.down5(x_c) # torch.Size([32, 256, 16, 16])
        x_c = self.maxpool(x_c) #torch.Size([32, 256, 8, 8])
        x_c = self.down6(x_c) #torch.Size([32, 512, 8, 8])
        x_c = self.maxpool(x_c)#torch.Size([32, 512, 4, 4])
        x_c = self.down_end(x_c) # 32, 128, 1, 1

        x7_t = x_t.view(target.shape[0], -1)
        fc_out_t = self.fc_t(x7_t).chunk(self.enc_chunk, dim = 1) # 32x384 -> [[32, 128], [32, 128], [32, 128]]

        x7_c = x_c.view(conditional.shape[0], -1)
        fc_out_c = self.fc(x7_c).chunk(self.enc_chunk, dim = 1) # 32x384 -> [[32, 128], [32, 128], [32, 128]]
        
        mu_t, logvar_t = fc_out_t[:2]
        mu_c, logvar_c = fc_out_c[:2]
        
        if self.sample:
            std = F.softplus(logvar_t)

            qz_x = Normal(mu_t, std)
            z_T = qz_x.rsample()
            
            std = F.softplus(logvar_c)
            qz_x_c = Normal(mu_c, std)
            z_C = qz_x_c.rsample()
            
        else:
            z_T = mu_t
            z_C = mu_c

 
        z_recon = torch.cat((z_C, z_T), dim=1)
        z_prior = torch.randn((z_C.shape[0], self.latent_size)).to(self.device)
        z_gen = torch.cat((z_C, z_prior), dim=1)
  
        z_recon =  F.normalize(z_recon)
        z_gen =  F.normalize(z_gen)

 

        x_gen = self.l1(z_gen)
        x_gen = x_gen.view(x_gen.shape[0], -1, 4, 4)
        x_gen = self.upconv1(x_gen) # torch.Size([16, 512, 8, 8])
        x_gen = self.up1(x_gen) #torch.Size([16, 256, 8, 8])
        x_gen = self.upconv2(x_gen)  # torch.Size([16, 256, 16, 16])
        x_gen = self.up2(x_gen)   # torch.Size([16, 256, 16, 16])
        x_gen = self.upconv3(x_gen)  # torch.Size([16, 256, 32, 32])
        x_gen = self.up3(x_gen)      # torch.Size([16, 128, 32, 32])
        x_gen = self.upconv4(x_gen)  # torch.Size([16, 128, 64, 64])
        x_gen = self.up4(x_gen)      # torch.Size([16, 64, 64, 64])
        x_gen = self.upconv5(x_gen)  # torch.Size([16, 64, 128, 128])
        x_gen = self.up5(x_gen)      # torch.Size([16, 32, 128, 128])     
        x_gen = self.up_end(x_gen)   # torch.Size([16, 1, 256, 256])
        x_gen = torch.sigmoid(x_gen)


        x_recon = self.l1(z_recon)
        x_recon = x_recon.view(x_recon.shape[0], -1, 4, 4)
        x_recon = self.upconv1(x_recon)
        x_recon = self.up1(x_recon)
        x_recon = self.upconv2(x_recon)
        x_recon = self.up2(x_recon)
        x_recon = self.upconv3(x_recon)
        x_recon = self.up3(x_recon)
        x_recon = self.upconv4(x_recon)
        x_recon = self.up4(x_recon)
        x_recon = self.upconv5(x_recon)
        x_recon = self.up5(x_recon)
        x_recon = self.up_end(x_recon)
        x_recon = torch.sigmoid(x_recon)

        return z_T, z_gen, x_gen, x_recon, mu_t, logvar_t

    def generate_img(self, x_c):
        x1_c = self.down1(x_c)  # 32, 64, 64, 64
        x2_c = self.down2(x1_c) # 32, 128, 32, 32
        x2_c = self.maxpool(x2_c)
        x3_c = self.down3(x2_c) # 32, 256, 16, 16
        x3_c = self.maxpool(x3_c)
        x4_c = self.down4(x3_c) # 32, 512, 8, 8
        x4_c = self.maxpool(x4_c)
        x5_c = self.down5(x4_c) # 32, 512, 4, 4
        x5_c = self.maxpool(x5_c)
        x6_c = self.down6(x5_c)
        x6_c = self.maxpool(x6_c)
        x7_c = self.down_end(x6_c) # 32, 128, 1, 1


        x7_c = x7_c.view(x_c.shape[0], -1)
        fc_out_c = self.fc(x7_c).chunk(self.enc_chunk, dim = 1) # 32x384 -> [[32, 128], [32, 128], [32, 128]]
        
    
        mu_c, logvar_c = fc_out_c[:2]
        
        if self.sample:
        
            std = F.softplus(logvar_c)
            qz_x_c = Normal(mu_c, std)
            z_C = qz_x_c.rsample()
            
        else:                                                                                                                        
            z_C = mu_c

        z_prior = torch.randn((z_C.shape[0], self.latent_size)).to(self.device)
        z_gen = torch.cat((z_C, z_prior), dim=1)
        z_gen =  F.normalize(z_gen)

        x_gen = self.l1(z_gen)
        x_gen = x_gen.view(x_gen.shape[0], -1, 4, 4)
        x_gen = self.upconv1(x_gen) # torch.Size([16, 512, 8, 8])
        x_gen = self.up1(x_gen) #torch.Size([16, 256, 8, 8])
        x_gen = self.upconv2(x_gen)  # torch.Size([16, 256, 16, 16])
        x_gen = self.up2(x_gen)   # torch.Size([16, 256, 16, 16])
        x_gen = self.upconv3(x_gen)  # torch.Size([16, 256, 32, 32])
        x_gen = self.up3(x_gen)      # torch.Size([16, 128, 32, 32])
        x_gen = self.upconv4(x_gen)  # torch.Size([16, 128, 64, 64])
        x_gen = self.up4(x_gen)      # torch.Size([16, 64, 64, 64])
        x_gen = self.upconv5(x_gen)  # torch.Size([16, 64, 128, 128])
        x_gen = self.up5(x_gen)      # torch.Size([16, 32, 128, 128])     
        x_gen = self.up_end(x_gen)   # torch.Size([16, 1, 256, 256])
        x_gen = torch.sigmoid(x_gen)

        return x_gen

class CUnetVAE(nn.Module):
    # motivated by https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
    def __init__(self, in_channels, latent_size, flow_depth, sample, device, logprob=False,  recon_loss = 'mse'):
        super().__init__()
        dim = 64
        self.out_channels = 1
        self.latent_size = latent_size
        self.ae_loss = recon_loss
        self.device = device

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, dim, 5, 2, 2), 
            nn.LeakyReLU(0.2)
        )
        self.down2 = conv_ln_lrelu(dim, dim * 2)
        self.down3 = conv_ln_lrelu(dim * 2, dim * 4)
        self.down4 = conv_ln_lrelu(dim * 4, dim * 4)
        self.down5 = conv_ln_lrelu(dim * 4, dim * 8)
        self.down6 = conv_ln_lrelu(dim * 8, dim * 8)
        self.down_end = nn.Conv2d(dim * 8, self.latent_size, 4)
        

        self.down1_t = nn.Sequential(
            nn.Conv2d(in_channels, dim, 5, 2, 2), 
            nn.LeakyReLU(0.2)
        )
        self.down2_t = conv_ln_lrelu(dim, dim * 2)
        self.down3_t = conv_ln_lrelu(dim * 2, dim * 4)
        self.down4_t = conv_ln_lrelu(dim * 4, dim * 4)
        self.down5_t = conv_ln_lrelu(dim * 4, dim * 8)
        self.down6_t = conv_ln_lrelu(dim * 8, dim * 8)
        self.down_end_t = nn.Conv2d(dim * 8, self.latent_size, 4)


        
        if flow_depth > 0:
            # IAF
            hidden_size = latent_size * 2
            flow_layers = [flow.InverseAutoregressiveFlow(
                latent_size, hidden_size, latent_size)
                for _ in range(flow_depth)]

            flow_layers.append(flow.Reverse(latent_size))
            self.q_z_flow = flow.FlowSequential(*flow_layers)
            # self.q_z_flow_c = flow.FlowSequential(*flow_layers)
            self.enc_chunk = 3
        else:
            self.q_z_flow = None
            self.enc_chunk = 2
            
        fc_out_size = latent_size * self.enc_chunk
        self.fc_t = nn.Sequential(
            nn.Linear(latent_size, fc_out_size),
            nn.LayerNorm(fc_out_size),
            nn.LeakyReLU(0.2),
            nn.Linear(fc_out_size, fc_out_size),
        )

 
        self.fc = nn.Sequential(
            nn.Linear(latent_size, fc_out_size),
            nn.LayerNorm(fc_out_size),
            nn.LeakyReLU(0.2),
            nn.Linear(fc_out_size, fc_out_size),
        )
        self.sample = sample
        
        # decoder
        self.l1 = nn.Sequential(
            nn.Linear(latent_size * 2, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())

        self.up1 = dconv_bn_relu(dim * 8, dim * 8)
        self.up2 = dconv_bn_relu(dim * 8, dim * 4)
        self.up3 = dconv_bn_relu(dim * 4, dim * 4)
        self.up4 = dconv_bn_relu(dim * 4, dim * 2 )
        self.up5 = dconv_bn_relu(dim * 2, dim )
        self.up_end = nn.ConvTranspose2d(dim, self.out_channels, 5, 2,
padding=2, output_padding=1)


        
    def forward(self, x_t, target_bpr, x_c,cond_bpr, ae):
        
        x1_t = self.down1_t(x_t)  # x_t: 32, 1, 256, 256
        x2_t = self.down2_t(x1_t) # x1_t: 32, 64, 128, 128
        x3_t = self.down3_t(x2_t) # x2_t: 32, 128, 64, 64
        x4_t = self.down4_t(x3_t) # x3_t:32, 256, 32, 32
        x5_t = self.down5_t(x4_t) # x4_t:32, 256, 16, 16
        x6_t = self.down6_t(x5_t) # x5_t: 32, 512, 8, 8
        x7_t = self.down_end_t(x6_t) # x6_t: 32, 512, 4, 4
                                       # x7_t: 32, 128, 1, 1
        
        # x_mask = x * mask
        x1_c = self.down1(x_c)  # 32, 64, 64, 64
        x2_c = self.down2(x1_c) # 32, 128, 32, 32
        x3_c = self.down3(x2_c) # 32, 256, 16, 16
        x4_c = self.down4(x3_c) # 32, 512, 8, 8
        x5_c = self.down5(x4_c) # 32, 512, 4, 4
        x6_c = self.down6(x5_c)
        x7_c = self.down_end(x6_c) # 32, 128, 1, 1

        x7_t = x7_t.view(x_t.shape[0], -1)
        fc_out_t = self.fc_t(x7_t).chunk(self.enc_chunk, dim = 1) # 32x384 -> [[32, 128], [32, 128], [32, 128]]

        x7_c = x7_c.view(x_c.shape[0], -1)
        fc_out_c = self.fc(x7_c).chunk(self.enc_chunk, dim = 1) # 32x384 -> [[32, 128], [32, 128], [32, 128]]
        
        mu_t, logvar_t = fc_out_t[:2]
        mu_c, logvar_c = fc_out_c[:2]
        
        if self.sample:
            
            std = F.softplus(logvar_t)
            # print(mu_t, std)
            qz_x = Normal(mu_t, std)
            z_T = qz_x.rsample()
            
            std = F.softplus(logvar_c)
            qz_x_c = Normal(mu_c, std)
            z_C = qz_x_c.rsample()
            
        else:                                                                                                                        
            z_T = mu_t
            z_C = mu_c

        if self.q_z_flow:
            z_T, _ = self.q_z_flow(z_T, context=fc_out_t[2])
            z_C, _ = self.q_z_flow(z_C, context=fc_out_c[2])
            

        z_gen = torch.cat((z_C, z_T), dim=1)


        x_gen = self.l1(z_gen)
        x_gen = x_gen.view(x_gen.shape[0], -1, 4, 4)
        x_gen = self.up1(x_gen)
        x_gen = self.up2(x_gen)
        x_gen = self.up3(x_gen)
        x_gen = self.up4(x_gen)
        x_gen = self.up5(x_gen)
        x_gen = self.up_end(x_gen)
        x_gen = torch.sigmoid(x_gen)

        
        return z_T, z_gen, x_gen, mu_t, logvar_t
    
    def generate_img(self, x_c):
        x1_c = self.down1(x_c)  # 32, 64, 64, 64
        x2_c = self.down2(x1_c) # 32, 128, 32, 32
        x3_c = self.down3(x2_c) # 32, 256, 16, 16
        x4_c = self.down4(x3_c) # 32, 512, 8, 8
        x5_c = self.down5(x4_c) # 32, 512, 4, 4
        x6_c = self.down6(x5_c)
        x7_c = self.down_end(x6_c) # 32, 128, 1, 1


        x7_c = x7_c.view(x_c.shape[0], -1)
        fc_out_c = self.fc(x7_c).chunk(self.enc_chunk, dim = 1) # 32x384 -> [[32, 128], [32, 128], [32, 128]]
        
    
        mu_c, logvar_c = fc_out_c[:2]
        
        if self.sample:
        
            std = F.softplus(logvar_c)
            qz_x_c = Normal(mu_c, std)
            z_C = qz_x_c.rsample()
            
        else:                                                                                                                        
            z_C = mu_c


        z_T = torch.randn((z_C.shape[0], self.latent_size)).to(self.device)
        z_gen = torch.cat((z_C, z_T), dim=1)

        x_gen = self.l1(z_gen)
        x_gen = x_gen.view(x_gen.shape[0], -1, 4, 4)
        x_gen = self.up1(x_gen)
        x_gen = self.up2(x_gen)
        x_gen = self.up3(x_gen)
        x_gen = self.up4(x_gen)
        x_gen = self.up5(x_gen)
        x_gen = self.up_end(x_gen)
        x_gen = torch.sigmoid(x_gen)

        return x_gen



class ConvCritic(nn.Module):
    def __init__(self, in_channels, latent_size):
        super().__init__()
        dim = 64
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            conv_ln_lrelu(dim * 8, dim * 8),
            nn.Conv2d(dim * 8, latent_size, 4))

        self.x_fc = nn.Linear(latent_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.latent_size = latent_size

    def forward(self, input, activation='wdist'): 
        x = self.conv(input)
        x = x.view(x.shape[0], -1)
        x = self.x_fc(x)
        if activation == 'sigmoid':
            x = self.sigmoid(x)
        elif activation == 'wdist':
            x = x.view(-1)
        return x

class ConvCritic_bigan(nn.Module):
    def __init__(self, in_channels, latent_size):
        super().__init__()
        dim = 64
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            conv_ln_lrelu(dim * 8, dim * 8),
            nn.Conv2d(dim * 8, latent_size, 4))

        self.conv1 = nn.Conv2d(3, dim, 5, 2, 2)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = conv_ln_lrelu(dim, dim * 2)
        self.conv3 = conv_ln_lrelu(dim * 2, dim * 4)
        self.conv4 = conv_ln_lrelu(dim * 4, dim * 4)
        self.conv5 = conv_ln_lrelu(dim * 4, dim * 8)
        self.conv6 = conv_ln_lrelu(dim * 8, dim * 8)
        self.conv7 = nn.Conv2d(dim * 8, latent_size, 4)

        embed_size = 64

        self.z_fc = nn.Sequential(
            nn.Linear(latent_size*3, embed_size),
            nn.LayerNorm(embed_size),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_size, embed_size),
        )

        self.x_fc = nn.Linear(latent_size, embed_size)

        self.xz_fc = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.LayerNorm(embed_size),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_size, 1),
        )
        self.x_fc1 = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.LayerNorm(embed_size),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_size, 1),
        )
        self.latent_size = latent_size

    def forward(self, input): 
        x, z = input #z:16x129
      
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.x_fc(x)
      
        z = self.z_fc(z)
        xz = torch.cat((x, z), 1)
        xz = self.xz_fc(xz)
        xz = xz.view(-1)
        return xz

class GradientPenalty:
    def __init__(self, critic, device, batch_size=64, gp_lambda=10):
        self.critic = critic
        self.gp_lambda = gp_lambda
        # Interpolation coefficient
        self.eps = torch.empty(batch_size, device=device)
        # For computing the gradient penalty
        self.ones = torch.ones(batch_size).to(device)
        # self.device = device

    def interpolate(self, real, fake):
        
        eps = self.eps.view([-1] + [1] * (len(real.shape) - 1))
        return (eps * real + (1 - eps) * fake).requires_grad_()

    def __call__(self, real, fake):
        
        real = real.detach() # detach target_img, z_T
        fake = fake.detach()
        self.eps.uniform_(0, 1)
        interp = self.interpolate(real, fake)
        a = self.critic(interp)
        # pdb.set_trace()
        grad_d = grad(a,
                      interp,
                      grad_outputs=self.ones,
                      create_graph=True)
        batch_size = real.shape[0]
        grad_d = grad_d[0].view(batch_size, -1)
        grad_penalty = ((grad_d.norm(dim=1) - 1)**2).mean() * self.gp_lambda
        return grad_penalty


class GradientPenalty_bigan:
    def __init__(self, critic, device, batch_size=64, gp_lambda=10):
        self.critic = critic
        self.gp_lambda = gp_lambda
        # Interpolation coefficient
        self.eps = torch.empty(batch_size, device=device)
        # For computing the gradient penalty
        self.ones = torch.ones(batch_size).to(device)
        # self.device = device

    def interpolate(self, real, fake):
        
        eps = self.eps.view([-1] + [1] * (len(real.shape) - 1))
        return (eps * real + (1 - eps) * fake).requires_grad_()

    def __call__(self, real, fake):
        
        real = [x.detach() for x in real] # detach target_img, z_T
        fake = [x.detach() for x in fake]
        self.eps.uniform_(0, 1)
        interp = [self.interpolate(a, b) for a, b in zip(real, fake)]
        a = self.critic(interp)
        # pdb.set_trace()
        grad_d = grad(a,
                      interp,
                      grad_outputs=self.ones,
                      create_graph=True)
        batch_size = real[0].shape[0]
        grad_d = torch.cat([g.view(batch_size, -1) for g in grad_d], 1)
        grad_penalty = ((grad_d.norm(dim=1) - 1)**2).mean() * self.gp_lambda
        return grad_penalty


