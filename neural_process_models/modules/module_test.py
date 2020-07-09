import torch

from latent_encoder import LatentEncoder
# from deterministic_encoder import DeterministicEncoder
# from decoder import Decoder


torch.manual_seed(0)

input_size = 2
output_list = [4, 4, 4, 4]
x = torch.randn(3, 5, 3)
y = torch.randn(3, 5, 2)
print('x.size() =', x.size())

encoder_test = LatentEncoder(input_x_dim=2, input_y_dim=3, hidden_dim_list=[4,4,4], latent_dim=4)

dist, mu, sigma = encoder_test(x, y)
print('dist =', dist)
print('mu = ', mu)
print('sigma = ', sigma)

# with torch.no_grad():
#     mlp_test = MLP(input_size=input_size, output_size_list=output_list)
#     print('mlp_test.mlp =', mlp_test.mlp)
#     my_output = mlp_test(x)
#
#
#     mlp_tt = BatchMLP(input_size=input_size, output_size=4, num_layers=4)
#     print('mlp_tt.initial =', mlp_tt.initial)
#     print('mlp_tt.encoder =', mlp_tt.encoder)
#     print('mlp_tt.final =', mlp_tt.final)
#
#     tt_output = mlp_tt(x)
#     print('my_output.size() =', my_output)
#     print('tt_output.size() =', tt_output)