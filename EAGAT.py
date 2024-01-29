import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv, GCNConv, GATConv
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.typing import Adj, NoneType, OptPairTensor, OptTensor, Size
from torch import Tensor
from typing import Optional, Tuple, Union
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn import norm
# from torch_geometric.utils import is_sparse, is_torch_sparse_tensor


class EAGAT(GATConv):
    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, heads: int = 1, concat: bool = True, negative_slope: float = 0.2, dropout: float = 0, add_self_loops: bool = True, edge_dim: Optional[int] = None, fill_value: Union[float, Tensor, str] = 'mean', bias: bool = True, **kwargs):
        super().__init__(in_channels, out_channels, heads, concat, negative_slope, dropout, add_self_loops, edge_dim, fill_value, bias, **kwargs)
        self.edge_linear = nn.Sequential(
            nn.Linear(out_channels, 1),
            norm.LayerNorm(1),
            nn.LeakyReLU()
        )
        self.update_x_j_linear = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            norm.LayerNorm(out_channels)
        )
    
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None, size: Size = None, return_attention_weights: NoneType = None) -> Tensor:
        H, C = self.heads, self.out_channels  # heads 1, output channels 5

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)  # node 3, head 1, outputs 5 
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        # x_src [3, 1, 2] self.att_src [1, heads, out_channels] alpha_src [3, 1]
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst) # node attention 系数

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        
        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha_j = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha_j, size=size, edge_attr=edge_attr)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out
        # return super().forward(x, edge_index, edge_attr, size, return_attention_weights)

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor, edge_attr: OptTensor, index: Tensor, ptr: OptTensor, size_i: int) -> Tensor:

        alpha = alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return alpha
        # return super().edge_update(alpha_j, alpha_i, edge_attr, index, ptr, size_i)

    def message(self, x_i: Tensor, x_j: Tensor, alpha: Tensor, edge_index_i: Tensor, edge_index_j: Tensor, edge_attr: Tensor) -> Tensor:
        edge_attr = self.lin_edge(edge_attr.view(-1, 1))
        edge_alpha = softmax(self.edge_linear(x_i.squeeze(1) + edge_attr), edge_index_i)
        edge_update = (edge_alpha * edge_attr).unsqueeze(1)
        update_x_j = self.update_x_j_linear(x_j + edge_update)
        return alpha.unsqueeze(-1) * update_x_j
    

if __name__ == '__main__':
    edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1],
                           [2, 0]], dtype=torch.long)
    x = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float)
    edge_attr = torch.tensor([[10], [15], [20], [25], [30]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index.t().contiguous())

    gat = EAGAT(3, 5, edge_dim=1)
    out = gat(x, edge_index.t().contiguous(), edge_attr)