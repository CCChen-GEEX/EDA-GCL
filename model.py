import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch
from torch.nn import Linear, BatchNorm1d, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class GINConvWithEdgeWeight(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GINConvWithEdgeWeight, self).__init__(aggr='add')  # "add" aggregation
        self.mlp = torch.nn.Sequential(
            Linear(in_channels, out_channels),
            BatchNorm1d(out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), device=x.device)

        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        # x_j: [E, out_channels]
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        # MLP after aggregation
        return self.mlp(aggr_out)


class GraphEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 num_layers: int = 1, drop_p: float = 0.0):
        super(GraphEncoder, self).__init__()

        assert num_layers >= 1
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop_p)
        self.conv = nn.ModuleList()
        self.batch_norm = nn.ModuleList()

        for i in range(num_layers):
            if num_layers == 1:
                self.conv.append(GINConvWithEdgeWeight(in_channels, out_channels))
                bn = nn.BatchNorm1d(out_channels)
            else:
                if i == 0:
                    self.conv.append(GINConvWithEdgeWeight(in_channels, 2 * out_channels))
                    bn = nn.BatchNorm1d(2 * out_channels)
                elif i == num_layers - 1:
                    self.conv.append(GINConvWithEdgeWeight(2 * out_channels, out_channels))
                    bn = nn.BatchNorm1d(out_channels)
                else:
                    self.conv.append(GINConvWithEdgeWeight(2 * out_channels, 2 * out_channels))
                    bn = nn.BatchNorm1d(2 * out_channels)
            self.batch_norm.append(bn)

        self.activation = activation

    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers):
            x = self.conv[i](x, edge_index, edge_weight)
            x = self.batch_norm[i](x)
            x = self.dropout(x)
            x = self.activation(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, num_layers: int = 1, drop_p: float = 0.0):
        super(Encoder, self).__init__()

        assert num_layers >= 1
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop_p)
        self.conv = nn.ModuleList()

        for i in range(num_layers):
            if num_layers == 1:
                self.conv.append(base_model(in_channels, out_channels))
            else:
                if i == 0:
                    self.conv.append(base_model(in_channels, 2 * out_channels))
                elif i == num_layers - 1:
                    self.conv.append(base_model(2 * out_channels, out_channels))
                else:
                    self.conv.append(base_model(2 * out_channels, 2 * out_channels))

        self.activation = activation

    def forward(self, x, edge_index, edge_weight):
        for i in range(self.num_layers):
            x = self.conv[i](x, edge_index, edge_weight)
            x = self.dropout(x)
            x = self.activation(x)
        return x


class TrainModel(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(TrainModel, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    
    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_weight)


class AdversarialModel(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 num_edge_hidden: int, tau: float = 0.5, drop_p: float = 0.0):
        super(AdversarialModel, self).__init__()
        
        self.encoder: Encoder = encoder
        self.tau: float = tau
        
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        
        self.mlp_edge_model = nn.Sequential(
            nn.Linear(num_hidden * 2, num_hidden),
            nn.Dropout(drop_p),
            nn.ReLU(),
            nn.Linear(num_hidden, num_edge_hidden),
            nn.Dropout(drop_p),
            nn.ReLU(),
            nn.Linear(num_edge_hidden, 1)
        )
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
        
    def filter_upper_edges(self, edges):
        u, v = edges[0], edges[1]
        mask = u < v
        filtered_u, filtered_v = u[mask], v[mask]
        result = torch.stack([filtered_u, filtered_v], dim=0)
        return result
    
    def forward(self, x, edge_index, edge_weight):
        z = self.encoder(x, edge_index, edge_weight)

        upper_edges = self.filter_upper_edges(edge_index)
        lower_edges = torch.stack([upper_edges[1], upper_edges[0]], dim=0)

        upper_edge_fea = torch.cat([z[upper_edges[0]], z[upper_edges[1]]], dim=1)
        lower_edge_fea = torch.cat([z[lower_edges[0]], z[lower_edges[1]]], dim=1)

        upper_edge_logits = self.mlp_edge_model(upper_edge_fea)
        lower_edge_logits = self.mlp_edge_model(lower_edge_fea)

        return {
        'upper_edge_logits': upper_edge_logits,
        'lower_edge_logits': lower_edge_logits,
        'upper_edge_fea': upper_edge_fea,
        'lower_edge_fea': lower_edge_fea
        }  