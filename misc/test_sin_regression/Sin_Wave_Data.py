import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt
import math

'''
Done
'''

def plot_functions(target_x, target_y, context_x, context_y, pred_y, std):
    """Plots the predicted mean and variance and the context points.

  Args:
    target_x: An array of shape [B,num_targets,1] that contains the
        x values of the target points.
    target_y: An array of shape [B,num_targets,1] that contains the
        y values of the target points.
    context_x: An array of shape [B,num_contexts,1] that contains
        the x values of the context points.
    context_y: An array of shape [B,num_contexts,1] that contains
        the y values of the context points.
    pred_y: An array of shape [B,num_targets,1] that contains the
        predicted means of the y values at the target points in target_x.
    std: An array of shape [B,num_targets,1] that contains the
        predicted std dev of the y values at the target points in target_x.
      """
    # Plot everything
    plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
    plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
    plt.plot(context_x[0], context_y[0], 'ko', markersize=10)
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - std[0, :, 0],
        pred_y[0, :, 0] + std[0, :, 0],
        alpha=0.25,
        facecolor='blue',
        interpolate=True)

    # Make the plot pretty
    # plt.yticks([-6, 0, 6], fontsize=16)
    # plt.xticks([-6, 0, 6], fontsize=16)
    # plt.ylim([-6, 6])
    plt.grid('off')
    ax = plt.gca()




class sin_wave_data():
    def __init__(self,
                 sin1_amp=1,
                 sin1_freq=1,
                 sin1_phase=1,
                 sin2_amp=2,
                 sin2_freq=2,
                 sin2_phase=1,
                 noise=0.1):
        '''
        f(x) = A_1 sin(w_1 x + phi_1) + A_2 sin(w_2 x + phi_2)
        :param x_start:
        :param x_end:
        :param sin1_amp:
        :param sin1_freq:
        :param sin1_phase:
        :param sin2_amp:
        :param sin2_freq:
        :param sin2_phase_2:
        '''
        self.sin1_amp = sin1_amp
        self.sin1_freq = sin1_freq
        self.sin1_phase = sin1_phase
        self.sin2_amp = sin2_amp
        self.sin2_freq = sin2_freq
        self.sin2_phase = sin2_phase
        self.noise = noise

        def torch_function(tensor_x):
            tensor_y_1 = self.sin1_amp * torch.sin(tensor_x * self.sin1_freq + self.sin1_phase)
            tensor_y_2 = self.sin2_amp * torch.sin(tensor_x * self.sin2_freq + self.sin2_phase)
            tensor_y_noise = torch.randn(tensor_x.size()) * math.sqrt(self.noise)
            return tensor_y_1 + tensor_y_2 + tensor_y_noise

        self.tensor_function = torch_function



    def query(self,
              batch_size=8,
              context_x_start=-6,
              context_x_end=6,
              context_x_num=50,
              target_x_start=-6,
              target_x_end=6,
              target_x_num=50,
              **kwargs):
        '''
        return context_x, context_y, target_x, target_y
        :param batch_size:
        :param context_x_start:
        :param context_x_end:
        :param context_x_num:
        :param target_x_start:
        :param target_x_end:
        :param target_x_num:
        :return:
        '''
        context_x = torch.linspace(context_x_start, context_x_end, context_x_num)
        context_x = (context_x.repeat(batch_size, 1)).view(batch_size, -1, 1)
        context_y = self.tensor_function(context_x)

        target_x = torch.linspace(target_x_start, target_x_end, target_x_num)
        target_x = target_x.repeat(batch_size, 1).view(batch_size, -1, 1)
        target_y = self.tensor_function(target_x)
        # print('context_x.size() =', context_x.size())

        return context_x, context_y, target_x, target_y


if __name__ == '__main__':
    '''
    Use data class
    '''
    data= sin_wave_data()

    batch_size = 4
    ctt_x, ctt_y, tgt_x, tgt_y = data.query(batch_size=batch_size,
                                            context_x_start=-6,
                                            context_x_end=6,
                                            context_x_num=50,
                                            target_x_start=0,
                                            target_x_end=5,
                                            target_x_num=40)
    print('ctt_x.size() =', ctt_x.size())
    # Verify

    plt.ion()
    plt.clf()
    for i in range(100):
        ctt_x, ctt_y, tgt_x, tgt_y = data.query(batch_size=batch_size,
                                            context_x_start=-6,
                                            context_x_end=6,
                                            context_x_num=50,
                                            target_x_start=0,
                                            target_x_end=5,
                                            target_x_num=40)
        plt.clf()
        # pick one x-y-seq from context
        ctt_x_array = ctt_x[0, :, :].squeeze().numpy()
        ctt_y_array = ctt_y[0, :, :].squeeze().numpy()
        plt.plot(ctt_x_array, ctt_y_array, label='context')

        tgt_x_array = tgt_x[0, :, :].squeeze().numpy()
        tgt_y_array = tgt_y[0, :, :].squeeze().numpy()
        plt.plot(tgt_x_array, tgt_y_array, label='target')
        plt.legend()
        plt.pause(0.1)

    plt.ioff()
    plt.show()