import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import flow
import pdb
import torch

def conv_ln_lrelu(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 5, 2, 2),
        nn.InstanceNorm2d(out_dim, affine=True),
        nn.LeakyReLU(0.2))

def conv_ln_lrelu3d(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim,5,2,2),
        nn.InstanceNorm3d(out_dim, affine=True),
        nn.LeakyReLU(0.2))

class ConvEncoder(nn.Module):
    def __init__(self, in_channel, latent_size, sample, flow_depth=2, logprob=False):
        super().__init__()

        if logprob:
            self.encode_func = self.encode_logprob
        else:
            self.encode_func = self.encode

        dim = 64
        self.ls = nn.Sequential(
            nn.Conv2d(in_channel, dim, 5, 2, 2), nn.LeakyReLU(0.2), 
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            # conv_ln_lrelu(dim * 4, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            conv_ln_lrelu(dim * 8, dim * 8),
            nn.Conv2d(dim * 8, latent_size, 4))


        if flow_depth > 0:
            # IAF
            hidden_size = latent_size * 2
            flow_layers = [flow.InverseAutoregressiveFlow(
                latent_size, hidden_size, latent_size)
                for _ in range(flow_depth)]

            flow_layers.append(flow.Reverse(latent_size))
            self.q_z_flow = flow.FlowSequential(*flow_layers)
            self.enc_chunk = 3
        else:
            self.q_z_flow = None
            self.enc_chunk = 2

        fc_out_size = latent_size * self.enc_chunk
        self.fc = nn.Sequential(
            nn.Linear(latent_size, fc_out_size),
            nn.LayerNorm(fc_out_size),
            nn.LeakyReLU(0.2),
            nn.Linear(fc_out_size, fc_out_size),
        )
        self.sample = sample

    def forward(self, input, k_samples=5):
        return self.encode_func(input, k_samples)


    def encode(self, input, feat = None):

        x = self.ls(input)

        x = x.view(input.shape[0], -1)
        
        fc_out = self.fc(x).chunk(self.enc_chunk, dim=1)
        mu, logvar = fc_out[:2]
        
        if feat is not None:
            assert mu.shape == feat.shape
            mu = torch.mean(torch.stack([mu, feat]), 0)
            logvar = torch.mean(torch.stack([logvar, feat]), 0)
        if self.sample:
            std = F.softplus(logvar)
            qz_x = Normal(mu, std)
            z = qz_x.rsample()
        else:
            z = mu
        if self.q_z_flow:
            z, _ = self.q_z_flow(z, context=fc_out[2])
        return z


class ConvEncoder64(nn.Module): # before 20201207
    def __init__(self, latent_size, flow_depth=2, logprob=False):
        super().__init__()

        if logprob:
            self.encode_func = self.encode_logprob
        else:
            self.encode_func = self.encode

        dim = 64
        self.ls = nn.Sequential(
            nn.Conv2d(3, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, latent_size, 4))
        
#         self.ls = nn.Sequential(nn.Conv2d(3, dim, 5, 2, 2), nn.LeakyReLU(0.2), conv_ln_lrelu(dim, dim * 2),conv_ln_lrelu(dim * 2, dim * 4),conv_ln_lrelu(dim * 4, dim * 4),conv_ln_lrelu(dim * 4, dim * 8),nn.Conv2d(dim * 8, latent_size, 4))

        if flow_depth > 0:
            # IAF
            hidden_size = latent_size * 2
            flow_layers = [flow.InverseAutoregressiveFlow(
                latent_size, hidden_size, latent_size)
                for _ in range(flow_depth)]

            flow_layers.append(flow.Reverse(latent_size))
            self.q_z_flow = flow.FlowSequential(*flow_layers)
            self.enc_chunk = 3
        else:
            self.q_z_flow = None
            self.enc_chunk = 2

        fc_out_size = latent_size * self.enc_chunk
        self.fc = nn.Sequential(
            nn.Linear(latent_size, fc_out_size),
            nn.LayerNorm(fc_out_size),
            nn.LeakyReLU(0.2),
            nn.Linear(fc_out_size, fc_out_size),
        )

    def forward(self, input, k_samples=5):
        return self.encode_func(input, k_samples)

    def encode_logprob(self, input, k_samples=5):
        x = self.ls(input)
        x = x.view(input.shape[0], -1)
        fc_out = self.fc(x).chunk(self.enc_chunk, dim=1)
        mu, logvar = fc_out[:2]
        std = F.softplus(logvar)
        qz_x = Normal(mu, std)
        z = qz_x.rsample([k_samples])
        log_q_z = qz_x.log_prob(z)
        if self.q_z_flow:
            z, log_q_z_flow = self.q_z_flow(z, context=fc_out[2])
            log_q_z = (log_q_z + log_q_z_flow).sum(-1)
        else:
            log_q_z = log_q_z.sum(-1)
        return z, log_q_z

    def encode(self, input, feat):

        x = self.ls(input)
        x = x.view(input.shape[0], -1)
        fc_out = self.fc(x).chunk(self.enc_chunk, dim=1)
        mu, logvar = fc_out[:2]
        if feat is not None:
            assert mu.shape == feat.shape
            mu = torch.mean(torch.stack([mu, feat]), 0)
            logvar = torch.mean(torch.stack([logvar, feat]), 0)
        std = F.softplus(logvar)
        qz_x = Normal(mu, std)
        z = qz_x.rsample()
        if self.q_z_flow:
            z, _ = self.q_z_flow(z, context=fc_out[2])
        return z



