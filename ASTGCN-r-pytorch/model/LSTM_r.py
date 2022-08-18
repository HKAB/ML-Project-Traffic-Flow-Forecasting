# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_submodule(nn.Module):

    def __init__(self, DEVICE, input_size, hidden_size, output_size):
        '''
        :param DEVICE:
        :param input_size:  Number of features
        :param hidden_size: Hidden size of LSTM
        :param output_size: Number of targets we 
                            want to predict (1 here since we only neef to predict the total flow)
        '''

        super(LSTM_submodule, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size)

        self.linear = nn.Linear(hidden_size, output_size)

        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''

        x = x.permute(0, 1, 3, 2) # (B, N_nodes, T_in, F_in)
        B, N, T, F = x.shape

        output, _ = self.lstm(x.reshape(B*N, T, F)) # (B, N_nodes, T_in, F_in) -> (B*N_nodes, T_in, hidden_size)

        output = self.linear(output) # (B*N_nodes, T_in, hidden_size) -> (B*N_nodes, T_in, 1)

        output = output.reshape(B, N, T)

        return output


def make_model(DEVICE, input_size, hidden_size, output_size):
    '''

    :param DEVICE:
    :param input_size:
    :param hidden_size:
    :param output_size:
    :return:
    '''

    model = LSTM_submodule(DEVICE, input_size, hidden_size, output_size)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model