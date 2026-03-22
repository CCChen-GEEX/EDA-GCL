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
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

from model import Encoder, TrainModel, AdversarialModel
from eval import label_classification

import math

# Set deterministic algorithms for reproducibility
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def generate_aug_edge_weight(edge_info, temperature=1.0, bias=0.0001):
    """
    Generate augmented edge weights for adversarial views.
    
    Args:
        edge_info: Dictionary containing edge logits and features
        temperature: Temperature parameter for sigmoid scaling
        bias: Bias parameter for epsilon sampling
    
    Returns:
        upper_edge_weight: Weights for upper edges
        lower_edge_weight: Weights for lower edges
        reg: Adversarial regularization loss
    """
    device = edge_info['upper_edge_logits'].device
    logits_shape = edge_info['upper_edge_logits'].size()

    # ========== Upper Edge Weight ==========
    eps_upper = (bias - (1 - bias)) * torch.rand(logits_shape) + (1 - bias)
    gate_upper = torch.log(eps_upper) - torch.log(1 - eps_upper)
    gate_upper = gate_upper.to(device)
    upper_gate_inputs = (gate_upper + edge_info['upper_edge_logits']) / temperature
    upper_edge_weight = torch.sigmoid(upper_gate_inputs).squeeze()

    # ========== Lower Edge Weight ==========
    eps_lower = (bias - (1 - bias)) * torch.rand(logits_shape) + (1 - bias)
    gate_lower = torch.log(eps_lower) - torch.log(1 - eps_lower)
    gate_lower = gate_lower.to(device)
    lower_gate_inputs = (gate_lower + edge_info['lower_edge_logits']) / temperature
    lower_edge_weight = torch.sigmoid(lower_gate_inputs).squeeze()
    
    # ========== Adversarial Regularization ==========
    reg = F.l1_loss(upper_edge_weight, lower_edge_weight)

    # ========== Edge Thresholding ==========
    noise = torch.randn_like(upper_edge_weight) * 1e-7  
    upper_edge_weight_with_noise = upper_edge_weight + noise
    lower_edge_weight_with_noise = lower_edge_weight + noise
    
    mask = upper_edge_weight_with_noise > lower_edge_weight_with_noise
    upper_edge_weight = torch.where(mask, upper_edge_weight, torch.tensor(0.).to(device))
    lower_edge_weight = torch.where(~mask, lower_edge_weight, torch.tensor(0.).to(device))

    return upper_edge_weight, lower_edge_weight, reg


if __name__ == '__main__':
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
    parser.add_argument('--adv_lambda', type=float, default=1.0)
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)
    set_everything(args.seed)

    # ========== Hyperparameters ==========
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
    adv_lambda = args.adv_lambda

    # ========== Helper Functions ==========
    def filter_upper_edges(edges):
        """Filter edges to keep only upper triangular (u < v)."""
        u, v = edges[0], edges[1]
        mask = u < v
        filtered_u, filtered_v = u[mask], v[mask]
        result = torch.stack([filtered_u, filtered_v], dim=0)
        return result
    
    # ========== Dataset Loading ==========
    dataset = get_dataset('./datasets/', args.dataset)
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)
    num_features = data.x.shape[1]
    
    upper_edges = filter_upper_edges(data.edge_index)
    lower_edges = torch.stack([upper_edges[1], upper_edges[0]], dim=0)
    data.edge_index = torch.cat([upper_edges, lower_edges], dim=1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # ========== Model Initialization ==========
    encoder = Encoder(
        num_features, num_hidden, activation,
        base_model=base_model, 
        num_layers=num_layers
    ).to(device)
    
    # Train model for minimizing loss
    train_model = TrainModel(
        encoder, num_hidden, num_proj_hidden, tau
    ).to(device)
    optimizer_train = torch.optim.Adam(
        train_model.parameters(), 
        lr=lr_train, 
        weight_decay=wd_train
    )
    
    # Adversarial model for maximizing loss
    adv_model = AdversarialModel(
        encoder, num_hidden, num_proj_hidden, num_edge_hidden, tau
    ).to(device)
    optimizer_adv = torch.optim.Adam(
        adv_model.parameters(), 
        lr=lr_adv, 
        weight_decay=wd_adv
    )

    # ========== Logging Setup ==========
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = osp.join(log_dir, f"run_homo_results.txt")

    with open(log_file, "a") as f:
        f.write("########################################\n")
        f.write(str(vars(args)) + "\n")    

    # ========== Training Loop ==========
    start = t()
    prev = start
    
    # Contrastive learning between two undirected adversarial graphs
    for epoch in range(1, num_epochs + 1):
        
        # ----- Adversarial View Generation -----
        adv_model.train()
        adv_model.zero_grad()
        train_model.eval()
        
        egde_info_adv = adv_model(
            data.x, 
            data.edge_index, 
            torch.ones(data.num_edges).to(device)
        )
        
        upper_edge_weight, lower_edge_weight, reg = generate_aug_edge_weight(egde_info_adv)
        
        z_1 = train_model(
            data.x, 
            data.edge_index, 
            torch.cat([upper_edge_weight, upper_edge_weight], dim=0)
        )  # Attacked view 1
        
        z_2 = train_model(
            data.x, 
            data.edge_index, 
            torch.cat([lower_edge_weight, lower_edge_weight], dim=0)
        )  # Attacked view 2
        
        view_loss = train_model.loss(z_1, z_2)
        edge_loss = F.mse_loss(
            egde_info_adv['upper_edge_fea'], 
            egde_info_adv['lower_edge_fea']
        )
        loss = view_loss + adv_lambda * edge_loss - reg_lambda * reg
        
        (-loss).backward()
        optimizer_adv.step()
        
        # ----- Train Model Optimization -----
        train_model.train()
        train_model.zero_grad()
        adv_model.eval()
        
        egde_info_train = adv_model(
            data.x, 
            data.edge_index, 
            torch.ones(data.num_edges).to(device)
        )
        
        upper_edge_weight, lower_edge_weight, _ = generate_aug_edge_weight(egde_info_train)

        z_1 = train_model(
            data.x, 
            data.edge_index, 
            torch.cat([upper_edge_weight, upper_edge_weight], dim=0)
        )  # Attacked view 1
        
        z_2 = train_model(
            data.x, 
            data.edge_index, 
            torch.cat([lower_edge_weight, lower_edge_weight], dim=0)
        )  # Attacked view 2
        
        model_loss = train_model.loss(z_1, z_2)
        model_loss.backward()
        optimizer_train.step()

        now = t()
        print(
            f'(T) | Epoch={epoch:03d}, loss={model_loss:.4f}, '
            f'this epoch {now - prev:.4f}, total {now - start:.4f}'
        )
        prev = now

    # ========== Evaluation ==========
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

    with open(log_file, 'a') as f:
        f.write('epoch: ' + str(epoch) + '\n')
        f.write(formatted_result + '\n')
    
    print('-----------------')