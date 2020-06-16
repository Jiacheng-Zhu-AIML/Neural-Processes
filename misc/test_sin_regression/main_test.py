import numpy as np
from matplotlib import pyplot as plt
import collections
from tqdm.auto import tqdm
import torch
from NPModel import NeuralProcessModel
from Sin_Wave_Data import sin_wave_data, plot_functions


if __name__ == '__main__':
    print(1)
    data = sin_wave_data()

    NP_model = NeuralProcessModel(   x_dim=1,
                                     y_dim=1,
                                     mlp_hidden_size_list=[256, 256, 256, 256],
                                     latent_dim=256,
                                     use_rnn=False,
                                     use_self_attention=True,
                                     use_deter_path=True)

    optim = torch.optim.Adam(NP_model.parameters(), lr=1e-4)

    epoch = 1000
    batch_size = 16

    for e in range(epoch):
        NP_model.train()
        print('step = ', e)
        plt.clf()
        ctt_x, ctt_y, tgt_x, tgt_y = data.query(batch_size=batch_size,
                                                context_x_start=-6,
                                                context_x_end=6,
                                                context_x_num=200,
                                                target_x_start=-6,
                                                target_x_end=6,
                                                target_x_num=200)
        # print('ctt_x.size() =', ctt_x.size())
        optim.zero_grad()

        mu, sigma, log_p, kl, loss = NP_model(ctt_x, ctt_y, tgt_x, tgt_y)

        # print('kl =', kl)
        print('loss = ', loss)
        # print('mu.size() =', mu.size())
        # print('sigma.size() =', sigma.size())

        # tgt_x_np = tgt_x[0, :, :].squeeze(-1).numpy()
        # print('tgt_x_np.shape =', tgt_x_np.shape)

        loss.backward()
        optim.step()

        NP_model.eval()
        plt.ion()
        # fig = plt.figure()
        plot_functions(tgt_x.numpy(),
                       tgt_y.numpy(),
                       ctt_x.numpy(),
                       ctt_y.numpy(),
                       mu.detach().numpy(),
                       sigma.detach().numpy())
        title_str = 'mid_largehidden_200_ANP_sins_' + str(e) +'_iter.png'
        plt.title(title_str)
        plt.savefig('image/' + title_str)
        plt.pause(0.00001)

    plt.ioff()
    plt.show()













