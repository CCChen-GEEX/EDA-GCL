import argparse
import os.path as osp
import random
from time import perf_counter as t
import os
from utils import set_everything, get_dataset

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborSampler

from model import Encoder, TrainModel, AdversarialModel
from eval import label_classification
import math

# Set deterministic algorithms for reproducibility
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def generate_aug_edge_weight(edge_info, temperature=1.0, bias=0.0001):
    """Generate augmented edge weights for adversarial learning."""
    device = edge_info['upper_edge_logits'].device
    logits_shape = edge_info['upper_edge_logits'].size()

    # Upper edge weight
    eps_upper = (bias - (1 - bias)) * torch.rand(logits_shape) + (1 - bias)
    gate_upper = torch.log(eps_upper) - torch.log(1 - eps_upper)
    gate_upper = gate_upper.to(device)
    upper_gate_inputs = (gate_upper + edge_info['upper_edge_logits']) / temperature
    upper_edge_weight = torch.sigmoid(upper_gate_inputs).squeeze()

    # Lower edge weight
    eps_lower = (bias - (1 - bias)) * torch.rand(logits_shape) + (1 - bias)
    gate_lower = torch.log(eps_lower) - torch.log(1 - eps_lower)
    gate_lower = gate_lower.to(device)
    lower_gate_inputs = (gate_lower + edge_info['lower_edge_logits']) / temperature
    lower_edge_weight = torch.sigmoid(lower_gate_inputs).squeeze()
    
    # Adversarial regularization
    reg = F.l1_loss(upper_edge_weight, lower_edge_weight)

    # Edge thresholding
    noise = torch.randn_like(upper_edge_weight) * 1e-7  
    upper_edge_weight_with_noise = upper_edge_weight + noise
    lower_edge_weight_with_noise = lower_edge_weight + noise
    
    mask = upper_edge_weight_with_noise > lower_edge_weight_with_noise
    upper_edge_weight = torch.where(mask, upper_edge_weight, torch.tensor(0.).to(device))
    lower_edge_weight = torch.where(~mask, lower_edge_weight, torch.tensor(0.).to(device))

    return upper_edge_weight, lower_edge_weight, reg


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora_lcc')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--learning_rate_train', type=float, default=0.001)
    parser.add_argument('--learning_rate_adv', type=float, default=0.0005)
    parser.add_argument('--num_hidden', type=int, default=1024)
    parser.add_argument('--num_proj_hidden', type=int, default=1024)
    parser.add_argument('--num_edge_hidden', type=int, default=64)
    parser.add_argument('--activation', type=str, default='prelu')
    parser.add_argument('--base_model', type=str, default='GCNConv', choices=['GCNConv'])
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--wd_train', type=float, default=1e-5)
    parser.add_argument('--wd_adv', type=float, default=1e-5)
    parser.add_argument('--reg_lambda', type=float, default=0.5)
    parser.add_argument('--batch_root_size', type=int, default=400)
    
    args = parser.parse_args()

    # Validate GPU ID and set device
    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    # Set random seed for reproducibility
    set_everything(args.seed)

    # Load hyperparameters from arguments
    lr_train = args.learning_rate_train
    lr_adv = args.learning_rate_adv
    num_hidden = args.num_hidden
    num_proj_hidden = args.num_proj_hidden
    num_edge_hidden = args.num_edge_hidden
    activation = ({
        'relu': F.relu, 
        'prelu': nn.PReLU(), 
        'rrelu': nn.RReLU(), 
        'leakyrelu': nn.LeakyReLU(), 
        'gelu': nn.GELU()
    })[args.activation]
    base_model = ({'GCNConv': GCNConv})[args.base_model]
    num_layers = args.num_layers

    tau = args.tau
    num_epochs = args.num_epochs
    wd_train = args.wd_train
    wd_adv = args.wd_adv
    reg_lambda = args.reg_lambda

    def filter_upper_edges(edges):
        """Filter edges to keep only upper triangular edges (u < v)."""
        u, v = edges[0], edges[1]
        mask = u < v
        filtered_u, filtered_v = u[mask], v[mask]
        result = torch.stack([filtered_u, filtered_v], dim=0)
        return result
    
    # Load dataset
    dataset = get_dataset('./datasets/', args.dataset)
    data = dataset[0]

    # Convert edge index to undirected graph
    data.edge_index = to_undirected(data.edge_index)
    num_features = data.x.shape[1]

    # Set device and move data to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Initialize encoder model
    encoder = Encoder(
        num_features, 
        num_hidden, 
        activation,
        base_model=base_model, 
        num_layers=num_layers
    ).to(device)
    
    # Train model for minimizing loss
    train_model = TrainModel(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer_train = torch.optim.Adam(
        train_model.parameters(), 
        lr=lr_train, 
        weight_decay=wd_train
    )
    
    # Adversarial model for maximizing loss
    adv_model = AdversarialModel(
        encoder, 
        num_hidden, 
        num_proj_hidden, 
        num_edge_hidden, 
        tau
    ).to(device)
    optimizer_adv = torch.optim.Adam(
        adv_model.parameters(), 
        lr=lr_adv, 
        weight_decay=wd_adv
    )
    
    # Initialize subgraph sampler
    subgraph_sampler = NeighborSampler(
        data.edge_index, 
        sizes=[5]*num_layers, 
        num_nodes=data.x.shape[0]
    )

    # Setup logging directory and file
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = osp.join(log_dir, f"run_homo_results.txt")

    # Write arguments to log file
    with open(log_file, "a") as f:
        f.write("########################################\n")
        f.write(str(vars(args)) + "\n")    

    # Start training timer
    start = t()
    prev = start
    
    # Training loop: contrastive learning on two undirected adversarial graphs
    for epoch in range(1, num_epochs + 1):
        # Sample root nodes for subgraph
        root_size = args.batch_root_size
        root_nodes = torch.randperm(data.x.shape[0])[:root_size]
        batch_size, n_id, adjs = subgraph_sampler.sample(root_nodes)

        # Extract subgraph data
        x_sub = data.x[n_id].to(device)
        edge_index_sub = adjs[-1].edge_index.to(device)
        edge_index_sub = to_undirected(edge_index_sub)
        
        # Split edges into upper and lower triangular parts
        upper_edges = filter_upper_edges(edge_index_sub)
        lower_edges = torch.stack([upper_edges[1], upper_edges[0]], dim=0)
        
        edge_index_sub = torch.cat([upper_edges, lower_edges], dim=1)
        
        # ==================== Adversarial Model Training ====================
        adv_model.train()
        adv_model.zero_grad()
        train_model.eval()
        
        # Generate adversarial views
        egde_info_adv = adv_model(
            x_sub, 
            edge_index_sub, 
            torch.ones(edge_index_sub.shape[1]).to(device)
        )
        
        upper_edge_weight, lower_edge_weight, reg = generate_aug_edge_weight(egde_info_adv)
        
        z_1 = train_model(
            x_sub, 
            edge_index_sub, 
            torch.cat([upper_edge_weight, upper_edge_weight], dim=0)
        )  # Attacked view 1
        z_2 = train_model(
            x_sub, 
            edge_index_sub, 
            torch.cat([lower_edge_weight, lower_edge_weight], dim=0)
        )  # Attacked view 2
        
        view_loss = train_model.loss(z_1, z_2)
        edge_loss = F.mse_loss(
            egde_info_adv['upper_edge_fea'], 
            egde_info_adv['lower_edge_fea']
        )
        loss = view_loss + edge_loss - reg_lambda * reg
        
        (-loss).backward()
        optimizer_adv.step()
            
        # ==================== Train Model Training ====================
        train_model.train()
        train_model.zero_grad()
        adv_model.eval()
        
        egde_info_train = adv_model(
            x_sub, 
            edge_index_sub, 
            torch.ones(edge_index_sub.shape[1]).to(device)
        )
        upper_edge_weight, lower_edge_weight, _ = generate_aug_edge_weight(egde_info_train)

        z_1 = train_model(
            x_sub, 
            edge_index_sub, 
            torch.cat([upper_edge_weight, upper_edge_weight], dim=0)
        )  # Attacked view 1
        z_2 = train_model(
            x_sub, 
            edge_index_sub, 
            torch.cat([lower_edge_weight, lower_edge_weight], dim=0)
        )  # Attacked view 2
        
        model_loss = train_model.loss(z_1, z_2)
        
        model_loss.backward()
        optimizer_train.step()

        # Log training progress
        now = t()
        print(
            f'(T) | Epoch={epoch:03d}, loss={model_loss:.4f}, '
            f'this epoch {now - prev:.4f}, total {now - start:.4f}'
        )
        prev = now

    # ==================== Evaluation ====================
    with torch.no_grad():
        emb = train_model(
            data.x, 
            data.edge_index, 
            torch.ones(data.num_edges).to(device)
        )
        
    micro_f1_mean, micro_f1_std, macro_f1_mean, macro_f1_std, acc_mean, acc_std = \
        label_classification(emb, data, args.dataset, ratio=0.1)
    
    formatted_result = (
        f"micro_f1: {micro_f1_mean:.2f}±{micro_f1_std:.2f}, "
        f"macro_f1: {macro_f1_mean:.2f}±{macro_f1_std:.2f}, "
        f"acc: {acc_mean:.2f}±{acc_std:.2f}"
    )

    # Write results to log file
    with open(log_file, 'a') as f:
        f.write('epoch: ' + str(epoch) + '\n')
        f.write(formatted_result + '\n')
    print('-----------------')