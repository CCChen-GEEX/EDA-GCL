# EDA-GCL
![image text](https://github.com/CCChen-GEEX/EDA-GCL/blob/main/overview.png "The pipeline of EDA-GCL")

🔥**EDA-GCL**: "Edge Self-Adversarial Augmentation Enhances Graph Contrastive Learning Against Neighborhood Inconsistency". This repository contains the official PyTorch implementation of our work.

## 🚀 About

**EDA-GCL** is an edge self-adversarial augmentation framework for graph contrastive learning that enhances robustness against neighborhood inconsistency by maximizing bidirectional edge feature discrepancies. The framework employs an alternating min-max optimization paradigm with linear computational complexity, demonstrating superior effectiveness on both homophilic and heterophilic graphs under various noise scenarios.

## Usage

Train and evaluate the model for heterophilous graphs by executing
```
sh script/train_hete.sh
```
Train and evaluate the model for homophilous graphs by executing
```
sh script/train_homo.sh
```
Performance may vary with different `torch-geometric` version. Also provide the script `search_hyper.sh` for searching new hyperparameter combinations.
```
sh script/search_hyper.sh
```

## Requirements
