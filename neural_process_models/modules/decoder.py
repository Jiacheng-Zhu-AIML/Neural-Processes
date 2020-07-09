'''
Implementation of decoder for neural process.

Steps:
1. concatenate the target_x and latent variables r_star and z
2. Then pass them input a MLP
3. According to deepmind imp, then split the hidden to get mu and sigma
    Maybe using reparamerization trick will break something here

Can be attended with cross-attention *Whether to put the cross-attention in this encoder?

From the deepmind implementation
decoder_output_sizes = [HIDDEN_SIZE]*2 + [2] => decoder_hidden_dim_list[-1] = 2
Here 2 comes from y_dim * 2

The operation on latent variables should be completed outside the decoder
'''

import torch
from torch import nn
import torch.nn.functional as F

from .mlp import MLP


class Decoder(nn.Module):
    def __init__(
            self,
            x_dim,
            y_dim,
            mid_hidden_dim_list,  # the dims of hidden starts of mlps
            latent_dim=32,  # the dim of last axis of sc and z..
            use_deterministic_path=True,  # whether use d_path or not will change the size of input
            use_lstm=False,
        ):
        super(Decoder, self).__init__()

        self.hidden_dim_list = mid_hidden_dim_list + [y_dim*2]

        if use_deterministic_path:
            self.decoder_input_dim = 2 * latent_dim + x_dim
        else:
            self.decoder_input_dim = latent_dim + x_dim

        if use_lstm:
            # self._decoder = LSTMBlock(hidden_dim_2, hidden_dim_2, batchnorm=batchnorm, dropout=dropout,
            #                           num_layers=n_decoder_layers)
            pass
        else:
            # self._decoder = BatchMLP(hidden_dim_2, hidden_dim_2, batchnorm=batchnorm, dropout=dropout,
            #                          num_layers=n_decoder_layers)
            self.decoder_mlp = MLP(input_size=self.decoder_input_dim, output_size_list=self.hidden_dim_list)
        # self._mean = nn.Linear(hidden_dim_2, y_dim)
        # self._std = nn.Linear(hidden_dim_2, y_dim)
        self._use_deterministic_path = use_deterministic_path
        # self._min_std = min_std
        # self._use_lvar = use_lvar

    def forward(self, r, z, target_x):
        # r:        (b, target_seq_len, latent_dim)
        # z:        (b, target_seq_len, latent_dim)
        # target_x: (b, target_seq_len, x_dim)

        # concatenate target_x and representation
        if self._use_deterministic_path:
            z = torch.cat([r, z], dim=-1)
            # z (b, target_seq_len, 2 * latent_dim )
        hidden_mu_sigma = torch.cat([z, target_x], dim=-1)
        # (b, target_len, 2 * latent_dim + x_dim)

        mu_sigma = self.decoder_mlp(hidden_mu_sigma)
        # (b, target_len, 2 * y_dim)

        # Get the mean and the variance ???
        # print('type(mu_sigma) =', type(mu_sigma))
        # print('mu_sigma.size() =', mu_sigma.size())
        # output_debug = mu_sigma.split(chunks=2, dim=-1)
        # print('output_debug[0].size() =', output_debug[0].size())

        mu, log_sigma = mu_sigma.chunk(chunks=2, dim=-1)
        # print('mu.size() =', mu.size())
        # print('log_sigma.size() =', log_sigma.size())
        # mu (b, target_len, y_dim)
        # sigma (b, target_len. y_dim)

        # Bound the variance
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        # Get the distibution
        dist = torch.distributions.Normal(mu, sigma)
        return dist, mu, sigma
