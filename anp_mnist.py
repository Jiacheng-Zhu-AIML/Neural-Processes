"""Script that applies ANP to the MNIST dataset."""

import os
import sys
import torch
from matplotlib import pyplot as plt
import mnist
import random

# Provide access to modules in repo.
sys.path.insert(0, os.path.abspath('neural_process_models'))

from neural_process_models.anp import ANP_Model


# Retrieve 10000 test data points from MNIST, prepare data data for ANP.

print("Retrieving and preparing data...")

x_data = torch.Tensor([float(i) for i in range(784)])
x_data = x_data.view(784, 1)

test_images = torch.from_numpy(mnist.test_images()).float()  # (10000 x 28 x 28)
test_images = test_images.view(10000, 784, 1)
test_labels = mnist.test_labels()

y_data = [torch.Tensor([]) for i in range(10)]	# separate data by digit

for i in range(test_images.size()[0]):
	y_data[test_labels[i]] = torch.cat((y_data[test_labels[i]],
		torch.unsqueeze(test_images[i], 0)), 0)

print("Retrieved and prepared data. Training...")


# Initialize model, hyperparameters

model = ANP_Model(x_dim=1,	# x_dim: pixel index (0-783)
			      y_dim=1,	# y_dim: pixel value (0-255)
			      mlp_hidden_size_list=[256, 256, 256, 256],
			      latent_dim=256,
			      use_rnn=False,
			      use_self_attention=False,
			      use_deter_path=True)

optim = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 1000
batch_size = 100


# Train model; display results

ctt_x = torch.Tensor([])
tgt_x = torch.Tensor([])
for i in range(batch_size):
	ctt_x = torch.cat((ctt_x, torch.unsqueeze(x_data, 0)), 0)
	tgt_x = torch.cat((tgt_x, torch.unsqueeze(x_data, 0)), 0)


for epoch in range(1, num_epochs + 1):
    print("step = " + str(epoch))

    model.train()

    plt.clf()
    optim.zero_grad()

    ctt_y = torch.Tensor([])
    tgt_y = torch.Tensor([])
    for digit in range(10):
    	num_to_sample = int(2 * batch_size / 10)

    	sample_indices = random.sample(range(y_data[digit].size()[0]), num_to_sample)

    	for idx in sample_indices[:int(batch_size / 10)]:
    		ctt_y = torch.cat((ctt_y, torch.unsqueeze(y_data[digit][idx], 0)), 0)

    	for idx in sample_indices[int(batch_size / 10):]:
    		tgt_y = torch.cat((tgt_y, torch.unsqueeze(y_data[digit][idx], 0)), 0)

    # ctt_x: (batch_size x 784 x 1), ctt_y: (batch_size x 784 x 1)
    # tgt_x: (batch_size x 784 x 1), tgt_y: (batch_size x 784 x 1)
    mu, sigma, log_p, kl, loss = model(ctt_x, ctt_y, tgt_x, tgt_y)

    # print('kl =', kl)
    print('loss = ', loss)
    # print('mu.size() =', mu.size())
    # print('sigma.size() =', sigma.size())

    # tgt_x_np = tgt_x[0, :, :].squeeze(-1).numpy()
    # print('tgt_x_np.shape =', tgt_x_np.shape)

    loss.backward()
    optim.step()

    model.eval()
    plt.ion()
    # fig = plt.figure()
    
    # Visualize first target image.
    pred_y = mu[0].view(28, 28).detach().numpy()
    print(min(mu[0].view(784).detach().numpy()))
    print(max(mu[0].view(784).detach().numpy()))

    plt.imshow(pred_y, cmap="gray", vmin=0, vmax=255)

    title_str = 'Training at epoch ' + str(epoch)
    plt.title(title_str)
    # plt.savefig(title_str)
    plt.pause(0.1)

plt.ioff()
plt.show()