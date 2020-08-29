'''
Implementation of deterministic encoder for neural process.
Can be attended with cross-attention.
TODO: whether to put the cross-attention in this encoder?
'''

import torch
from torch import nn

from .mlp import MLP
from .attention import Attention


class DeterministicEncoder(nn.Module):
    def __init__(
        self,
        input_x_dim,
        input_y_dim,
        hidden_dim_list,  # the dims of hidden starts of mlps
        latent_dim=32,  # the dim of last axis of r..
        self_attention_type="dot",
        use_self_attn=True,
        attention_layers=2,
        use_lstm=False,
        cross_attention_type="dot",
        cross_attention_rep='mlp',
        attention_dropout=0,
    ):
        super().__init__()
        self.input_dim = input_x_dim + input_y_dim
        self.hidden_dim_list = hidden_dim_list
        self.hidden_dim = hidden_dim_list[-1]
        self.latent_dim = latent_dim
        self.use_self_attn = use_self_attn

        if latent_dim != hidden_dim_list[-1]:
            print('Warning, Check the dim of latent z and the dim of mlp last layer!')

        # NOTICE: In my paper, we seems to substitute the mlp with LSTM
        #  but we actually add a LSTM before the mlp
        if use_lstm:
            # self._encoder = LSTMBlock(input_dim, hidden_dim, batchnorm=batchnorm, dropout=dropout,
            #                           num_layers=n_encoder_layers)
            pass
        else:
            self.deter_encoder_mlp = MLP(input_size=self.input_dim, output_size_list=hidden_dim_list)
            # Output should be (b, seq_len, hidden_dim_list[-1])
            # output: (b, seq_len, hidden_dim)

        if use_self_attn:
            self._self_attention = Attention(
                self.latent_dim,
                self_attention_type,
                attention_layers,
                rep="identity",
            )
            pass
        self._cross_attention = Attention(
            self.latent_dim,
            cross_attention_type,
            x_dim=input_x_dim,
            attention_layers=attention_layers,
            mlp_hidden_dim_list=self.hidden_dim_list,
            rep=cross_attention_rep,  # TODO: fix later
        )


    def forward(self, context_x, context_y, target_x):
        # Concatenate x and y along the filter axes
        encoder_input = torch.cat([context_x, context_y], dim=-1)
        # encoder_input (b, seq_len, input_dim=input_x_dim + input_y_dim)
        # = (b, seq_len, input_dim)

        # Pass final axis through MLP
        hidden_r_i = self.deter_encoder_mlp(encoder_input)
        # hidden_r_i (b, seq_len, latent_dim)

        if self.use_self_attn:
            hidden_r_i = self._self_attention(hidden_r_i, hidden_r_i, hidden_r_i)
        else:
            hidden_r_i = hidden_r_i

        # Apply attention as mean aggregation
        # In the ANP paper, context_x and target_x are first passed by a mlp
        # In my paper, all x are first passed by lstm
        #
        h = self._cross_attention(context_x, hidden_r_i, target_x)
        # context_x     (b, seq_len, input_x_dim)           # Key
        # hidden_r_i    (b, seq_len, latent_dim)            # Value
        # target_x      (b, target_seq_len, input_x_dim)    # Query
        #
        return h        # (b, target_seq_len, latent_dim)
