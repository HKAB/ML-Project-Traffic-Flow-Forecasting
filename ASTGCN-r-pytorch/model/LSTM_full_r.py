# -*- coding:utf-8 -*-
import torch.nn as nn
import torch

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

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2)

        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

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

        output = nn.ReLU()(self.linear(self.dropout(output))) # (B*N_nodes, T_in, hidden_size) -> (B*N_nodes, T_in, 1)

        output = output.reshape(B, N, T)

        return output


class LSTM_full_submodule(nn.Module):

    def __init__(self, DEVICE, input_size, hidden_size, output_size):
        '''
        :param DEVICE:
        :param input_size:  Number of features
        :param hidden_size: Hidden size of LSTM
        :param output_size: Number of targets we 
                            want to predict (1 here since we only neef to predict the total flow)
        '''

        super(LSTM_full_submodule, self).__init__()

        self.h_model = LSTM_submodule(DEVICE, input_size, hidden_size, output_size)
        self.d_model = LSTM_submodule(DEVICE, input_size, hidden_size, output_size)
        self.w_model = LSTM_submodule(DEVICE, input_size, hidden_size, output_size)

        self.W_h = torch.zeros(output_size, requires_grad=True, device=DEVICE)
        self.W_d = torch.zeros(output_size, requires_grad=True, device=DEVICE)
        self.W_w = torch.zeros(output_size, requires_grad=True, device=DEVICE)

        nn.init.uniform_(self.W_h)
        nn.init.uniform_(self.W_d)
        nn.init.uniform_(self.W_w)

        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x_h, x_d, x_w):
        '''
        :param x_h, x_d, x_w: (B, N_nodes, F_in, T_in)
        '''
        h_pred = self.h_model(x_h) # (B, N_nodes, T_out)
        d_pred = self.d_model(x_d) # (B, N_nodes, T_out)
        w_pred = self.w_model(x_w) # (B, N_nodes, T_out)

        return self.W_h*h_pred + self.W_d*d_pred + self.W_w*w_pred


def make_model(DEVICE, input_size, hidden_size, output_size):
    '''

    :param DEVICE:
    :param input_size:
    :param hidden_size:
    :param output_size:
    :return:
    '''

    model = LSTM_full_submodule(DEVICE, input_size, hidden_size, output_size)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model