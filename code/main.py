from __future__ import print_function
import torch.nn as nn
import os, pickle
import sys

sys.path.append('./NCE')
sys.path.append('./models')
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
import argparse
#import socket
from torchvision import transforms, datasets

import pandas as pd

import numpy as np
from data_loader import abdomen_slice_bpr, abdomen_slice_pair, abdomen_blsa
from tqdm import tqdm
import sklearn.metrics as metrics
from collections import defaultdict
from visualize import Visualizer
from mmd import mmd
import torch.nn.functional as F
from pbigan import *
import pdb 
import plot
from PIL import Image
import yaml
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as psnr

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

L1_criterion = nn.L1Loss(reduction='mean')
L2_criterion = nn.MSELoss(reduction='mean')
smooth_l1_criterion = nn.SmoothL1Loss(reduction='mean')

def cal_recon_loss(x_t, x_gen, ae=None):
    # if ae == 'mse':
    l2 = L2_criterion(x_gen, x_t)

    l1 = L1_criterion(x_gen, x_t)
    # elif ae == 'smooth_l1':
    #     recon_loss = smooth_l1_criterion(x_gen, x_t)

    ssim_loss = 1 - ssim(x_gen, x_t, data_range=1, size_average=True)

    return l1, l2, ssim_loss

class Trainer(object):
    def __init__(self, cfig, device):
        self.device = device
        self.cfig = cfig
        self.csv_path = os.path.join(self.cfig['save_path'], 'csv')
       
        self.ae_weight = 0
        self.bce_criterion = nn.BCELoss()
            
        if cfig['resunet']:
            self.pbigan = CResUnetPBiGAN(cfig['in_channels'], latent_size = cfig['latent'], flow_depth = cfig['flow'], logprob=False, sample = cfig['sample'], device=self.device, dropout=cfig['dropout']).to(self.device)
        else:
            self.pbigan = CUnetPBiGAN(cfig['in_channels'], cfig['out_channels'], latent_size = cfig['latent'], flow_depth = cfig['flow'], logprob=False, sample = cfig['sample'], device=self.device, dropout=cfig['dropout']).to(self.device)
        
        self.critic = ConvCritic(cfig['in_channels'], cfig['latent']).to(self.device)
        
        self.lr = cfig['lr']
        
        self.optimizer = optim.Adam(
            self.pbigan.parameters(), lr=self.lr, betas=(.5, .9))
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, betas=(.5, .9))
        
        self.scheduler_pbigan = self.make_scheduler(self.optimizer, self.lr, cfig['min_lr'], cfig['max_epoch'])
        
        # self.scheduler_cls = self.make_scheduler(self.optimizer_cls, self.lr, cfig['min_lr'], cfig['max_epoch'])
        self.scheduler = self.cfig['schedule']
        
        self.train_loader, self.val_loader = self.data_loader()

        self.grad_penalty = GradientPenalty(self.critic, batch_size=cfig['batch_size'], device=self.device)
        
        self.vis = Visualizer(self.cfig['save_path'], self.cfig['batch_size'])
        
    def make_scheduler(self, optimizer, lr, min_lr, epochs, steps=10): # origial steps = 10
        if min_lr < 0:
            return None
        step_size = epochs // steps
        gamma = (min_lr / lr)**(10 / steps) # original 1 / steps
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)
        
    def adjust_learning_rate(self, optimizer, epoch, steps, gamma):
        if epoch in steps:
            self.lr *= gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr   


    def data_loader(self):
        
        
        if self.cfig['pair']:
            tr_dataset = abdomen_slice_pair(self.cfig,'train')
            val_dataset = abdomen_slice_pair(self.cfig, 'val')
        # assert (len(set(partition_IDs['train']) & set(partition_IDs['validation'])) == 0)
        else:
            tr_dataset = abdomen_slice_bpr(self.cfig,'train')
            val_dataset = abdomen_slice_bpr(self.cfig, 'val')
        # tt_dataset = abdomen_slice_bpr(self.cfig, 'tt')

        paramstrain = {'shuffle': True,
                           'num_workers': 4,
                           'batch_size': self.cfig['batch_size']}
        paramstest = {'shuffle': False,
                       'num_workers': 4,
                       'batch_size': self.cfig['batch_size']}


        # assert len(set(partition_IDs['train']) & set(partition_IDs['validation']) ) == 0
        self.len_train = len(tr_dataset)
        self.len_val = len(val_dataset)
        print ('len of train, val sets', len(tr_dataset), len(val_dataset))
        # print ('len of tt set', len(tt_dataset))
        training_generator = data.DataLoader(tr_dataset, drop_last=True, **paramstrain)

        val_generator = data.DataLoader(val_dataset, drop_last=False, **paramstest)
        # tt_generator = data.DataLoader(tt_dataset, **paramstest, drop_last=False)

        return training_generator, val_generator #, tt_generator


    def train(self):
        #TODO: try pretrain
        if self.cfig['pbigan_pretrain']:
            # a = input('before load')
            trained_model = torch.load(self.cfig['pbigan_pretrain'], map_location=self.device)
            # b = input('after load')
            self.pbigan.load_state_dict(trained_model['pbigan'])
            self.critic.load_state_dict(trained_model['critic'])

        n_critic = 5        
        for epoch in range(self.cfig['start_epoch'], self.cfig['max_epoch']):
            model_root = os.path.join(self.cfig['save_path'], 'models')
            if not os.path.exists(model_root):
                os.mkdir(model_root)

            model_pth = '%s/model_epoch_%04d.pth' % (model_root, epoch)
            
            self.loss_breakdown = defaultdict(float)
            self.train_epoch(epoch, n_critic, self.cfig['save_path'])
            if (epoch < 5) or (int(epoch % 10) == 9):
                plot.flush()
            plot.tick()
            self.val_epoch(epoch, 'val')
            for key in self.loss_breakdown.keys():
                if 'val' in key:
                    self.loss_breakdown[key] /= self.len_val
                elif 'lr' in key:
                    continue
                else:
                    self.loss_breakdown[key] /= self.len_train
            self.vis.plot_loss(epoch, self.loss_breakdown)

        
    def train_epoch(self, epoch, n_critic, save_path):
        
        for param_group in self.optimizer.param_groups:
            self.lr = param_group['lr']
        self.loss_breakdown['lr*100'] =  100 * self.lr
        
        if epoch >= self.cfig['ae_start']:
            self.ae_weight = self.cfig['ae'] #* (epoch - self.cfig['ae_start']) / (self.cfig['max_epoch'] - self.cfig['ae_start'])


        for batch_idx, data_tup in tqdm(enumerate(self.train_loader)):
            cond_img, target_img, _ = data_tup
            cond_img, target_img = cond_img.to(self.device), target_img.to(self.device)
            target_label, target_bpr, cond_bpr = None, None, None
           
            rec_img = self.train_nodgan(cond_img, target_img, target_label, target_bpr, cond_bpr, epoch, n_critic, save_path, batch_idx)
               

        if self.scheduler:
            self.scheduler_pbigan.step()
                

        model_dict = {
            'pbigan': self.pbigan.state_dict(),
            'critic': self.critic.state_dict(),
            'history': self.vis.history,
            'epoch': epoch,
            'args': self.cfig,
        }
        
        if self.cfig['save_interval'] > 0:
            model_dir = self.cfig['save_path'] + '/models'
            mkdir(self.cfig['save_path'] + '/models')
        torch.save(model_dict, self.cfig['save_path'] +  '/model.pth')
        if self.cfig['save_interval']  > 0 and (epoch + 1) % self.cfig['save_interval']  == 0:
            torch.save(model_dict, model_dir + "/{:04d}.pth".format(epoch))
   

    def train_nodgan(self, cond_img, target_img, target_label, target_bpr, cond_bpr, epoch, n_critic, save_path, batch_idx): #feat is the feature from clinical info
        self.pbigan.train()
        self.critic.train()
        
        self.pbigan.eval()
        valid = torch.ones((target_img.shape[0],1), requires_grad=False).to(device)
        fake = torch.zeros((target_img.shape[0],1), requires_grad=False).float().to(device)

        
        z_enc, z_gen, x_gen, x_recon, _, _  = self.pbigan(target_img, target_bpr, cond_img, cond_bpr)
        # if cond_img: keep cond_img background, if target_img: keep target_img background
        
        x_gen = x_gen.detach()
        x_recon = x_recon.detach()
        
        if self.cfig['dis_act'] == 'sigmoid':
            real_score = self.critic(target_img, activation='sigmoid')
            fake_score_gen = self.critic(x_gen, activation='sigmoid')
            fake_score_recon = self.critic(x_recon, activation='sigmoid')

            loss_dis_real = self.bce_criterion(real_score, valid)
            loss_dis_gen = self.bce_criterion(fake_score_gen, fake)
            loss_dis_recon = self.bce_criterion(fake_score_recon, fake)
            
            loss_discriminator = (loss_dis_real + loss_dis_gen + loss_dis_recon ) / 3

        # print(loss_discriminator)
        elif self.cfig['dis_act'] == 'wdist':
            real_score = self.critic(target_img).mean()
            fake_score_gen = self.critic(x_gen).mean()
            fake_score_recon = self.critic(x_recon).mean()
            w_dist1 = real_score - fake_score_gen
            w_dist2 = real_score - fake_score_recon
        
            #z_enc: 16x129
            loss_discriminator = (-w_dist1 - w_dist2 + self.grad_penalty(target_img,x_gen) + self.grad_penalty(target_img,x_recon))/2
        
        self.critic_optimizer.zero_grad()
        loss_discriminator.backward() # 
        self.critic_optimizer.step()
        self.critic.eval()
        self.pbigan.train()
       
        self.loss_breakdown['D'] += loss_discriminator.item()
        
        for p in self.critic.parameters():
            p.requires_grad_(False)

        z_enc, z_gen, x_gen, x_recon, mu_t, logvar_t  = self.pbigan(target_img, target_bpr, cond_img, cond_bpr)


        if self.cfig['dis_act'] == 'sigmoid':
            fake_score_gen = self.critic(x_gen, activation='sigmoid')
            fake_score_recon = self.critic(x_recon, activation='sigmoid')

            loss_generator_gen = self.bce_criterion(fake_score_gen, valid)
            loss_generator_recon = self.bce_criterion(fake_score_recon, valid)
            
            loss_generator = (loss_generator_gen + loss_generator_recon) / 2
            
        # print(loss_discriminator)
        elif self.cfig['dis_act'] == 'wdist':
            real_score = self.critic(target_img).mean()
            fake_score_gen = self.critic(x_gen).mean()
            fake_score_recon = self.critic(x_recon).mean()
            loss_generator = ((real_score - fake_score_gen - fake_score_recon)/3) * self.cfig['G_weight']

        
        # G_loss = real_score - fake_score

        kl_loss = 0.5 * torch.sum(-1 - logvar_t + torch.exp(logvar_t) + mu_t**2)
        self.loss_breakdown['kl'] += kl_loss.item()

        l1loss_gen, l2loss_gen, ssim_loss_gen = cal_recon_loss(target_img, x_gen, self.cfig['aeloss'])
        l1loss_recon, l2loss_recon, ssim_loss_recon = cal_recon_loss(target_img, x_recon, self.cfig['aeloss'])
        # print(l1loss_gen, l1loss_recon, ssim_loss_gen, ssim_loss_recon, loss_generator)
        recon_loss = 0
        if 'l1' in self.cfig['aeloss']:
            recon_loss += (l1loss_gen + l1loss_recon)
        if 'l2' in self.cfig['aeloss']:
            recon_loss += (l2loss_gen + l2loss_recon)
        if 'ssim_loss' in self.cfig['aeloss']:
            recon_loss += (ssim_loss_gen + ssim_loss_recon)

        recon_loss = recon_loss * self.ae_weight 

        loss =  recon_loss + kl_loss + loss_generator


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        self.loss_breakdown['G_loss'] += loss_generator.item()*x_gen.shape[0]
        self.loss_breakdown['recon_loss'] += (recon_loss.item()*x_gen.shape[0])
        self.loss_breakdown['mse_gen'] += l2loss_gen.item()*x_gen.shape[0]
        self.loss_breakdown['mse_recon'] += l2loss_recon.item()*x_gen.shape[0]
        self.loss_breakdown['l1_gen'] += l1loss_gen.item()*x_gen.shape[0]
        self.loss_breakdown['l1_recon'] += l1loss_recon.item()*x_gen.shape[0]
        self.loss_breakdown['ssim_gen'] += (1- ssim_loss_gen.item())*x_gen.shape[0]
        self.loss_breakdown['ssim_recon'] += (1- ssim_loss_recon.item())*x_gen.shape[0]
        self.loss_breakdown['total'] += loss.item()

        for p in self.critic.parameters():
            p.requires_grad_(True)
        
        self.vis.plot(epoch, target_img, cond_img, x_gen, target_label, None, batch_idx, phase='train',seg=False)
 
        return x_gen#, casePred_gen
            

    def val_epoch(self, epoch, phase = 'val'):
        self.pbigan.eval()


        if phase == 'val':
            loader = self.val_loader
        
        with torch.no_grad():
            for batch_idx, data_tup in tqdm(enumerate(loader)):
                

                cond_img, target_img, _ = data_tup
                
                cond_img, target_img = cond_img.to(self.device),target_img.to(self.device)
                target_label, target_bpr, cond_bpr = None, None, None
               
                # target_bpr, cond_bpr = None, None
                x_gen = self.pbigan.generate_img(cond_img)

                l1loss, l2loss, ssim_loss = cal_recon_loss(target_img, x_gen, self.cfig['aeloss'])
                recon_loss = 0
                if 'l1' in self.cfig['aeloss']:
                    recon_loss += l1loss
                if 'l2' in self.cfig['aeloss']:
                    recon_loss += l2loss
                if 'ssim_loss' in self.cfig['aeloss']:
                    recon_loss += ssim_loss
                recon_loss = recon_loss * self.ae_weight 

                self.loss_breakdown[phase + '_mse'] += l2loss.item()*x_gen.shape[0]
                self.loss_breakdown[phase + '_l1'] += l1loss.item()*x_gen.shape[0]
                self.loss_breakdown[phase + '_ssim'] += (1-ssim_loss.item())*x_gen.shape[0]
                
                predictions_bn = None
                self.vis.plot(epoch, target_img, cond_img, x_gen, target_label, predictions_bn, batch_idx, phase=phase, seg=False)

class Tester(object):
    def __init__(self, cfig, device, dataset='btcv'):
        print(device)
        self.device = device
        self.cfig = cfig
        self.dataset = dataset
        self.csv_path = os.path.join(self.cfig['save_path'], 'csv')

        self.ae_weight = 0

        self.pbigan = CUnetPBiGAN(cfig['in_channels'], cfig['out_channels'], latent_size = cfig['latent'], flow_depth = cfig['flow'], logprob=False, sample = cfig['sample'], device=self.device).to(self.device)
        
        self.vis = Visualizer(self.cfig['save_path'], self.cfig['batch_size'])

    def btcv_loader(self):
        train_dataset = abdomen_slice_pair(self.cfig, 'train')
        test_dataset = abdomen_slice_pair(self.cfig, 'test')
        dataset = test_dataset
        paramstest = {'shuffle': False,
                       'num_workers': 4,
                       'batch_size': self.cfig['batch_size']}

        test_generator = data.DataLoader(dataset, drop_last=False, **paramstest)
        return test_generator

    def blsa_loader(self):
        test_dataset = abdomen_blsa(self.cfig, 'test')
        dataset = test_dataset
        paramstest = {'shuffle': False,
                       'num_workers': 4,
                       'batch_size': self.cfig['batch_size']}

        test_generator = data.DataLoader(dataset, drop_last=False, **paramstest)
        return test_generator


    def test(self):

        trained_model = torch.load(self.cfig['pbigan_pretrain'], map_location=self.device)

        self.pbigan.load_state_dict(trained_model['pbigan'])
        self.outdir = self.cfig['save_path'] + '/generated_img_{}'.format(self.dataset)
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        if self.dataset == 'btcv' or self.dataset == 'imagevub2':
            btcv_test_loader = self.btcv_loader()
            self.test_epoch(btcv_test_loader, 'test')
        elif self.dataset == 'blsa':
            blsa_test_loader = self.blsa_loader()
            self.test_epoch(blsa_test_loader, 'test',)
        


    def test_epoch(self, loader, phase = 'test'):
        self.pbigan.eval()

        if self.dataset == 'btcv' or self.dataset == 'imagevub2':
            with open(os.path.join(self.cfig['save_path'], 'result2.txt'), 'w') as f:
                ssim_list, psnr_list = [], []
                with torch.no_grad():
                    for batch_idx, data_tup in tqdm(enumerate(loader)):
                        
                        
                        cond_img, target_img,target_filename = data_tup
                        
                        cond_img, target_img = cond_img.to(self.device),target_img.to(self.device)
                        target_label, target_bpr, cond_bpr = None, None, None
                    
                     
                        x_gen = self.pbigan.generate_img(cond_img)
                        
                        l1loss, l2loss, ssim_loss = cal_recon_loss(target_img, x_gen, self.cfig['aeloss'])
                        
                        
                        x_gen_np = x_gen.data.cpu().numpy().squeeze() * 255.
                        x_gen_np = x_gen_np.astype(np.uint8)
                        target_img_np = target_img.data.cpu().numpy().squeeze() * 255.
                        target_img_np = target_img_np.astype(np.uint8)
                        skimage_ssim = structural_similarity(x_gen_np, target_img_np, data_range=255)
                        skimage_psnr = psnr(x_gen_np, target_img_np, data_range=255)
                        ssim_list.append(skimage_ssim)
                        psnr_list.append(skimage_psnr)
                        print('[{}] save {} ssim {}'.format(batch_idx, target_filename[0], skimage_ssim))
                        predictions_bn = None
                        f.write('ssim:{}, {}\n'.format(skimage_ssim, target_filename[0]))
                
                        self.vis.plot(target_filename, target_img, cond_img, x_gen, target_label, predictions_bn, batch_idx, phase=phase, seg=False)
                f.close()
            ssim_mean = np.mean(ssim_list)
            psnr_mean = np.mean(psnr_list)
            print('ssim: {}'.format(ssim_mean))
            print('psnr: {}'.format(psnr_mean))

        
        else:
            with torch.no_grad():
                for batch_idx, data_tup in tqdm(enumerate(loader)):
                    cond_img, target_filename = data_tup 
                    cond_img = cond_img.to(self.device)
                    target_label, target_bpr, cond_bpr = None, None, None
                    try:
                        x_gen = self.pbigan.generate_img(cond_img)
                    except:
                        print(target_filename)
                        continue
                    x_gen_np = x_gen.data.cpu().numpy().squeeze() * 255.
                    x_gen_np = x_gen_np.astype(np.uint8)
                    x_gen_img = Image.fromarray(x_gen_np).convert('RGB')
    

if __name__ == '__main__':

    yaml_file = ''
    with open(yaml_file, 'r') as f:
        cfig = yaml.load(f)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(cfig, device)
    trainer.train()

    tester = Tester(cfig, device, 'btcv')
    tester.test()    