'''
Implementation of MLP (machine learning process) stage of neutral process.
Done according to deepmind imp and that[*] implementation.

Fundamental building blocks:
-n hidden layers
-ReLU activation at each layer (except final layer)
**Pytorch will automatically operate on the last dim**

Structure:
input(B, seq_len, input_dim)
	V linear_1 [w_1 in (input_dim, hidden_size_1)] + ReLU
 variable(B, seq_len, hidden_size_1)
	V linear_2 [w_2 in (hidden_size_1, hidden_size_2)] + ReLU
 variable(B, seq_len, hidden_size_2)
	V ...
	...
	V last linear without ReLU
 variable(B, seq_len, output_size)
'''

from torch import nn


class MLP(nn.Module):
    '''
    Apply MLP to the final axis of a 3D tensor
    '''
    def __init__(self, input_size, output_size_list):
        '''
        Parameters:
        -input_size (int): number of dimensions for each point in each sequence.
        -output_size_list (list of ints): number of output dimensions for each layer.
        '''
        super().__init__()
        self.input_size = input_size  # e.g. 2
        self.output_size_list = output_size_list  # e.g. [128, 128, 128, 128]
        network_size_list = [input_size] + self.output_size_list  # e.g. [2, 128, 128, 128, 128]
        network_list = []

        # iteratively build neural network.
        for i in range(1, len(network_size_list) - 1):
            network_list.append(nn.Linear(network_size_list[i-1], network_size_list[i], bias=True))
            network_list.append(nn.ReLU())

        # Add final layer, create sequential container.
        network_list.append(nn.Linear(network_size_list[-2], network_size_list[-1]))
        self.mlp = nn.Sequential(*network_list)

    def forward(self, x):
        return self.mlp(x)
