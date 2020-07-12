import torch
from matplotlib import pyplot as plt

from neural_process_models.attentive_neural_process import NeuralProcessModel
from misc.test_sin_regression.Sin_Wave_Data import sin_wave_data, plot_functions


if __name__ == '__main__':
    data = sin_wave_data()

    np_model = NeuralProcessModel(x_dim=1,
                                  y_dim=1,
                                  mlp_hidden_size_list=[256, 256, 256, 256],
                                  latent_dim=256,
                                  use_rnn=False,
                                  use_self_attention=True,
                                  use_deter_path=True)

    optim = torch.optim.Adam(np_model.parameters(), lr=1e-4)

    num_epochs = 1000
    batch_size = 16

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
        optim.step()

        np_model.eval()
        plt.ion()
        # fig = plt.figure()
        plot_functions(tgt_x.numpy(),
                       tgt_y.numpy(),
                       ctt_x.numpy(),
                       ctt_y.numpy(),
                       mu.detach().numpy(),
                       sigma.detach().numpy())
        title_str = 'Training at epoch ' + str(epoch)
        plt.title(title_str)
        # plt.savefig(title_str)
        plt.pause(0.1)

    plt.ioff()
    plt.show()