# EDA-GCL
![image text](https://github.com/CCChen-GEEX/EDA-GCL/blob/main/overview.png "The pipeline of EDA-GCL")

🔥**EDA-GCL**: "Edge Self-Adversarial Augmentation Enhances Graph Contrastive Learning Against Neighborhood Inconsistency". This repository contains the official PyTorch implementation of our work.

## 🚀 About

**EDA-GCL** is an edge self-adversarial augmentation framework for graph contrastive learning that enhances robustness against neighborhood inconsistency by maximizing bidirectional edge feature discrepancies.

## Usage

Train and evaluate the model for heterophilous graphs by executing
```
sh script/train_hete.sh
```
Train and evaluate the model for homophilous graphs by executing
```
sh script/train_homo.sh
```
Performance results are subject to variation based on the specific `torch-geometric` version utilized. Additionally, we supply `search_hyper.sh` to enable efficient exploration of new hyperparameter configurations.
```
sh script/search_hyper.sh
```

## Requirements
- torch 2.0.0+cu118
- torch-geometric 2.6.1
- PyYAML 6.0.2
- numpy 1.26.4
- scikit-learn 1.6.1
- deeprobust 0.2.11

Install all dependencies using
```
pip install -r requirements.txt
```
## Citation

Please cite our paper if you use the code:

```
@inproceedings{chen2026edge,
  title={Edge Self-Adversarial Augmentation Enhances Graph Contrastive Learning Against Neighborhood Inconsistency},
  author={Chen, Chunchun and Wei, Xing and Yang, Jiayi and Wang, Chenrun and Fu, Yiwei and Zhang, Yuxing and Sun, Xin and Fan, Rui and Ye, Wei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={24},
  pages={20005--20013},
  year={2026}
}
```
## 🗨️ Contacts

For more details about our article, feel free to reach out to us. We are here to provide support and answer any questions you may have. 

- **Email**: Send us your inquiries at [c2chen@tongji.edu.cn].
