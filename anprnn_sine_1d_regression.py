"""Script that utilizes an ANPRNN to regress points to a sine curve."""

import os
import sys
import torch
from matplotlib import pyplot as plt

# Provide access to modules in repo.
sys.path.insert(0, os.path.abspath('neural_process_models'))
sys.path.insert(0, os.path.abspath('misc'))

from neural_process_models.anp_rnn import ANP_RNN_Model
from misc.test_sin_regression.Sin_Wave_Data import sin_wave_data, plot_functions

data = sin_wave_data()

np_model = ANP_RNN_Model(x_dim=1,
                     y_dim=1,
                     mlp_hidden_size_list=[256, 256, 256, 256],
                     latent_dim=256,
                     use_rnn=True,
                     use_self_attention=True,
                     le_self_attention_type="dot",
                     de_self_attention_type="dot",
                     de_cross_attention_type="multihead",
                     use_deter_path=True)

optim = torch.optim.Adam(np_model.parameters(), lr=1e-4)

num_epochs = 15000
batch_size = 16

loss_list = []

for epoch in range(1, num_epochs + 1):
    print("step = " + str(epoch))

    np_model.train()

    plt.clf()
    optim.zero_grad()

    ctt_x, ctt_y, tgt_x, tgt_y = data.query(batch_size=batch_size,
                                            context_x_start=-6,
                                            context_x_end=6,
                                            context_x_num=200,
                                            target_x_start=-6,
                                            target_x_end=6,
                                            target_x_num=200)

    mu, sigma, log_p, kl, loss = np_model(ctt_x, ctt_y, tgt_x, tgt_y)

    # print('kl =', kl)
    print('loss = ', loss)
    # print('mu.size() =', mu.size())
    # print('sigma.size() =', sigma.size())

    # tgt_x_np = tgt_x[0, :, :].squeeze(-1).numpy()
    # print('tgt_x_np.shape =', tgt_x_np.shape)

    loss.backward()
    loss_list.append(loss.item())
    optim.step()

    np_model.eval()
    # plt.ion()
    # fig = plt.figure()
    plot_functions(tgt_x.numpy(),
                   tgt_y.numpy(),
                   ctt_x.numpy(),
                   ctt_y.numpy(),
                   mu.detach().numpy(),
                   sigma.detach().numpy())
    title_str = 'ANP-RNN Training at epoch ' + str(epoch)
    plt.title(title_str)
    if epoch % 250 == 0:
        plt.savefig(title_str)
    plt.pause(0.1)

# plt.ioff()
# plt.show() 

print(loss_list)