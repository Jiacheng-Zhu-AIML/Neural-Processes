'''
Implementation of batch MLP.
Differs from vanilla MLP in that it reuses already defined MLPs.

Args:
        input: input tensor of shape [B,n,d_in].
        output_sizes: An iterable containing the output sizes of the MLP as defined
            in `basic.Linear`.
    Returns:
        tensor of shape [B,n,d_out] where d_out=output_size
'''

from torch import nn

from .np_block_relu_2d import NPBlockRelu2d


class BatchMLP(nn.Module):
    def __init__(
        self, input_size, output_size, num_layers=2, dropout=0, batchnorm=False
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.initial = NPBlockRelu2d(
            input_size, output_size, dropout=dropout, batchnorm=batchnorm
        )
        self.encoder = nn.Sequential(
            *[
                NPBlockRelu2d(
                    output_size, output_size, dropout=dropout, batchnorm=batchnorm
                )
                for _ in range(num_layers - 2)
            ]
        )
        self.final = nn.Linear(output_size, output_size)

    def forward(self, x):
        x = self.initial(x)
        x = self.encoder(x)
        return self.final(x)
