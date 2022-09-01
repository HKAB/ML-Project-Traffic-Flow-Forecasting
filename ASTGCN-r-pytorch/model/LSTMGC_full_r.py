from torch import nn
import numpy as np
from torch.autograd import Variable
import torch
import typing
from torch import nn
import torch.nn.functional as F


class GraphInfo:
    """Creates datastructure for graph from numpy array.
    Reference:
        https://keras.io/examples/timeseries/timeseries_traffic_forecasting/
    Args:
        adj_matrix: np.ndarray with shape `(num_edges, 3)`
        num_nodes: number of vertices in graph
    """
    def __init__(self, 
        adj_matrix: np.ndarray,
        num_nodes: int,
        device):
        self.num_nodes = num_nodes
        adj_matrix = torch.from_numpy(adj_matrix)
        self.edges = (adj_matrix[:, 0].to(device), adj_matrix[:, 1].to(device))
  

class GraphConv(nn.Module):
    """
    Input to the layer is a 4D tensor of shape (num_nodes, batch_size, input_seq_length, in_feat).
    """
    def __init__(
        self,
        DEVICE,
        in_feat,
        out_feat,
        graph_info: GraphInfo,
        aggregation_type="mean",
        combination_type="concat",
        activation: typing.Optional[str] = None
    ):
        super(GraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.weight = Variable(torch.nn.init.xavier_uniform_(torch.Tensor(in_feat, out_feat)),requires_grad=True).to(DEVICE)
        self.activation = activation
        self.DEVICE = DEVICE
    def aggregate(self, neighbour_representations: torch.Tensor):
        aggregation_func = {
            "sum": self.unsorted_segment_sum,
            "mean": self.unsorted_segment_mean,
            # "max": unsorted_segment_max,
        }.get(self.aggregation_type)
        
        num_nodes = self.graph_info.num_nodes
        current_node = self.graph_info.edges[0].to(torch.int64)
        
        # print(f'neighbour_representations {neighbour_representations}')
        if aggregation_func:
            return aggregation_func(
                neighbour_representations,
                current_node,
                num_nodes
            )

        raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")
    def unsorted_segment_sum(self, data, segment_ids, num_segments):
        """
        Reference: https://gist.github.com/bbrighttaer/207dc03b178bbd0fef8d1c0c1390d4be
        
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.
    
        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"
    
        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long().to(self.DEVICE)
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])
    
        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"
    
        shape = [num_segments] + list(data.shape[1:])
        tensor = torch.zeros(*shape).to(self.DEVICE).scatter_add(0, segment_ids, data.float()).to(self.DEVICE)
        tensor = tensor.type(data.dtype)
        return tensor
    def unsorted_segment_mean(self, data, segment_ids, num_segments):
        """
        Reference: https://pytorch-scatter.readthedocs.io/en/1.4.0/_modules/torch_scatter/mean.html#scatter_mean
        
        Computes the mean along segments of a tensor. Analogous to tf.unsorted_segment_mean.
    
        :param data: A tensor whose segments are to be summed and.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"
    
        out = self.unsorted_segment_sum(data, segment_ids, num_segments)
        count = self.unsorted_segment_sum(torch.ones_like(data),segment_ids, num_segments)
        return out / count.clamp(min=1)
    def gather_(self, data , segment_ids):
        # print(f'data={data.shape}')
        # print(segment_ids.max())
        # assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long().to(self.DEVICE)
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])
        output = torch.gather(data, 0, segment_ids)
        return output
    def compute_nodes_representation(self, features: torch.Tensor):
        """Computes each node's representation.

        The nodes' representations are obtained by multiplying the features tensor with
        `self.weight`. Note that
        `self.weight` has shape `(in_feat, out_feat)`.

        Args:
            features: Tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

        Returns:
            A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
        """
        return torch.matmul(features.float(), self.weight.float()) 
    def compute_aggregated_messages(self, features: torch.Tensor):
        # print(features.shape, self.graph_info.edges[1].shape)
        idx = self.graph_info.edges[1].to(torch.int64)
        neighbour_representations = self.gather_(features.float(), idx)
        aggregated_messages = self.aggregate(neighbour_representations)
        return torch.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation: torch.Tensor, aggregated_messages: torch.Tensor):
        if self.combination_type == "concat":
            h = torch.concat([nodes_representation, aggregated_messages], dim=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        return self.activation(h)
        
    def forward(self, x):
        """Forward pass.
        Args:
            features: tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

        Returns:
            A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
        """
        nodes_representation = self.compute_nodes_representation(x)
        aggregated_messages = self.compute_aggregated_messages(x)
        return self.update(nodes_representation, aggregated_messages)


class LSTMGC_submodule(nn.Module):
    """Layer comprising a convolution layer followed by LSTM and dense layers."""

    def __init__(
        self,
        DEVICE,
        in_feat,
        out_feat,
        hidden_feat: int,
        input_seq_len: int,
        output_seq_len: int, 
        graph_info: GraphInfo,
        graph_conv_params: typing.Optional[dict] = None,
    ):
        super(LSTMGC_submodule, self).__init__()

        # graph conv layer
        if graph_conv_params is None:
            graph_conv_params = {
                "aggregation_type": "mean",
                "combination_type": "concat",
                "activation": None,
            }
        self.graph_conv = GraphConv(in_feat, hidden_feat, graph_info, **graph_conv_params)

        self.lstm = nn.LSTM(input_size=hidden_feat*2, hidden_size=hidden_feat, num_layers=2,batch_first=True)
        self.dense = nn.Linear(hidden_feat, out_feat)

        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.out_feat = out_feat

        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, inputs):
        """Forward pass.

        Args:
            inputs: torch.Tensor of shape `(batch_size, input_seq_len, num_nodes, in_feat)`
            inputs: torch.Tensor of shape `(batch_size, num_nodes, in_feat, input_seq_len)`

        Returns:
            A tensor of shape `(batch_size, input_seq_len, num_nodes)`.
        """

        # convert shape to  (num_nodes, batch_size, input_seq_len, in_feat)
        # inputs = torch.unsqueeze(inputs, 3)
        # inputs = inputs.permute(2, 0, 1, 3)
        inputs = inputs.permute(1, 0, 3, 2)
        # print(inputs.shape)
        
        gcn_out = self.graph_conv(
            inputs
        )  # gcn_out has shape: (num_nodes, batch_size, input_seq_len, hidden_feat*2)
        shape = gcn_out.shape
        num_nodes, batch_size, input_seq_len, out_feat = (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )
        # print(f'gcn_out = {gcn_out.shape}')
        # LSTM takes only 3D tensors as input
        gcn_out = torch.reshape(gcn_out, (batch_size * num_nodes, input_seq_len, out_feat))
        # print(f'gcn_out = {gcn_out.shape}')
        lstm_out, (_, _) = self.lstm(gcn_out)  # lstm_out has shape: (batch_size * num_nodes, input_seq_len, hidden_feat)
        # print(f'lstm_out = {lstm_out.shape}')
        dense_output = self.dense(lstm_out)  # dense_output has shape: (batch_size * num_nodes, input_seq_len, out_feat)
        # print(f'dense_output = {dense_output.shape}')
        # NOTE: RESHAPE
        output = torch.reshape(dense_output, (num_nodes, batch_size, self.input_seq_len, self.out_feat))
        output = output.permute(1, 2, 0, 3).squeeze(dim=-1) # Tensor of shape (batch_size, input_seq_len, num_nodes)
        output = output[:, self.input_seq_len-self.output_seq_len:,:]

        output = output.permute(0, 2, 1) # (batch_size, num_nodes, input_seq_len)
        # print(f'out = {output.shape}')
        return output

class LSTMGC_full_submodule(nn.Module):
    def __init__(   self, DEVICE, in_channels, out_feat, 
                    hidden_feat, len_input, 
                    num_for_predict, graph_info, graph_conv_params):
        super().__init__()

        self.h_model = LSTMGC_submodule(DEVICE, in_channels, out_feat, hidden_feat, len_input, num_for_predict, graph_info, graph_conv_params)
        self.d_model = LSTMGC_submodule(DEVICE, in_channels, out_feat, hidden_feat, len_input, num_for_predict, graph_info, graph_conv_params)
        self.w_model = LSTMGC_submodule(DEVICE, in_channels, out_feat, hidden_feat, len_input, num_for_predict, graph_info, graph_conv_params)

        self.W_h = torch.zeros(num_for_predict, requires_grad=True, device=DEVICE)
        self.W_d = torch.zeros(num_for_predict, requires_grad=True, device=DEVICE)
        self.W_w = torch.zeros(num_for_predict, requires_grad=True, device=DEVICE)

        nn.init.uniform_(self.W_h)
        nn.init.uniform_(self.W_d)
        nn.init.uniform_(self.W_w)

        self.to(DEVICE)
    def forward(self, x_h, x_d, x_w):
        '''
        :param x_h, x_d, x_w: (B, N_nodes, F_in, T_in)
        '''
        h_pred = self.h_model(x_h) # (B, N_nodes, T_out)
        d_pred = self.d_model(x_d) # (B, N_nodes, T_out)
        w_pred = self.w_model(x_w) # (B, N_nodes, T_out)

        return self.W_h*h_pred + self.W_d*d_pred + self.W_w*w_pred


def make_model(DEVICE, in_channels, out_feat, hidden_feat, adj_mx, num_for_predict, len_input, num_of_vertices):
    '''

    :param DEVICE:
    :param input_size:
    :param hidden_size:
    :param output_size:
    :return:
    '''

    graph_info = GraphInfo(adj_mx, num_of_vertices, DEVICE)
    graph_conv_params = {
        "aggregation_type": "mean",
        "combination_type": "concat",
        "activation": F.relu,
    }
    model = LSTMGC_full_submodule(  DEVICE, in_channels, out_feat, 
                                    hidden_feat, len_input, 
                                    num_for_predict, graph_info, graph_conv_params)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model