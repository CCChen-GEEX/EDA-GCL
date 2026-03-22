#############################################
dataset=cora_lcc
seed=123
learning_rate_train=0.001
learning_rate_adv=0.0005
num_hidden=1024
num_proj_hidden=1024
num_edge_hidden=64
activation=prelu
base_model=GCNConv
num_layers=2
tau=0.4
num_epochs=200
reg_lambda=0.5

python train_homo.py \
    --dataset $dataset \
    --seed $seed \
    --learning_rate_train $learning_rate_train \
    --learning_rate_adv $learning_rate_adv \
    --num_hidden $num_hidden \
    --num_proj_hidden $num_proj_hidden \
    --num_edge_hidden $num_edge_hidden \
    --activation $activation \
    --base_model $base_model \
    --num_layers $num_layers \
    --tau $tau \
    --num_epochs $num_epochs \
    --reg_lambda $reg_lambda


#############################################
dataset=citeseer_lcc
seed=123
learning_rate_train=0.001
learning_rate_adv=0.0005
num_hidden=128
num_proj_hidden=128
num_edge_hidden=128
activation=prelu
base_model=GCNConv
num_layers=2
tau=0.9
num_epochs=150
reg_lambda=0.9

python train_homo.py \
    --dataset $dataset \
    --seed $seed \
    --learning_rate_train $learning_rate_train \
    --learning_rate_adv $learning_rate_adv \
    --num_hidden $num_hidden \
    --num_proj_hidden $num_proj_hidden \
    --num_edge_hidden $num_edge_hidden \
    --activation $activation \
    --base_model $base_model \
    --num_layers $num_layers \
    --tau $tau \
    --num_epochs $num_epochs \
    --reg_lambda $reg_lambda


############################################
dataset=PubMed
seed=123
learning_rate_train=0.001
learning_rate_adv=1e-5
num_hidden=512
num_proj_hidden=512
num_edge_hidden=128
activation=prelu
base_model=GCNConv
num_layers=2
tau=0.7
num_epochs=400
reg_lambda=0.5

python train_homo_sub.py \
    --dataset $dataset \
    --seed $seed \
    --learning_rate_train $learning_rate_train \
    --learning_rate_adv $learning_rate_adv \
    --num_hidden $num_hidden \
    --num_proj_hidden $num_proj_hidden \
    --num_edge_hidden $num_edge_hidden \
    --activation $activation \
    --base_model $base_model \
    --num_layers $num_layers \
    --tau $tau \
    --num_epochs $num_epochs \
    --reg_lambda $reg_lambda


#############################################
dataset=AmazonP
seed=123
learning_rate_train=0.001
learning_rate_adv=5e-05
num_hidden=256
num_proj_hidden=256
num_edge_hidden=64
activation=rrelu
base_model=GCNConv
num_layers=2
tau=0.3
num_epochs=1900
reg_lambda=0.3

python train_homo_sub.py \
    --dataset $dataset \
    --seed $seed \
    --learning_rate_train $learning_rate_train \
    --learning_rate_adv $learning_rate_adv \
    --num_hidden $num_hidden \
    --num_proj_hidden $num_hidden \
    --num_edge_hidden $num_edge_hidden \
    --activation $activation \
    --base_model $base_model \
    --num_layers $num_layers \
    --tau $tau \
    --num_epochs $num_epochs \
    --reg_lambda $reg_lambda


#############################################
dataset=AmazonC
seed=123
learning_rate_train=0.0005
learning_rate_adv=0.00001
num_hidden=512
num_proj_hidden=512
num_edge_hidden=128
activation=prelu
base_model=GCNConv
num_layers=2
tau=0.4
num_epochs=1000
reg_lambda=0.3

python train_homo_sub.py \
    --dataset $dataset \
    --seed $seed \
    --learning_rate_train $learning_rate_train \
    --learning_rate_adv $learning_rate_adv \
    --num_hidden $num_hidden \
    --num_proj_hidden $num_hidden \
    --num_edge_hidden $num_edge_hidden \
    --activation $activation \
    --base_model $base_model \
    --num_layers $num_layers \
    --tau $tau \
    --num_epochs $num_epochs \
    --reg_lambda $reg_lambda


#############################################
dataset=CoauthorC
seed=123
learning_rate_train=0.001
learning_rate_adv=0.0005
num_hidden=128
num_proj_hidden=128
num_edge_hidden=32
activation=rrelu
base_model=GCNConv
num_layers=2
tau=0.4
num_epochs=1200
reg_lambda=0.5

python train_homo_sub.py \
    --dataset $dataset \
    --seed $seed \
    --learning_rate_train $learning_rate_train \
    --learning_rate_adv $learning_rate_adv \
    --num_hidden $num_hidden \
    --num_proj_hidden $num_hidden \
    --num_edge_hidden $num_edge_hidden \
    --activation $activation \
    --base_model $base_model \
    --num_layers $num_layers \
    --tau $tau \
    --num_epochs $num_epochs \
    --reg_lambda $reg_lambda


#############################################
dataset=CoauthorP
seed=123
learning_rate_train=0.0005
learning_rate_adv=5e-05
num_hidden=128
num_proj_hidden=128
num_edge_hidden=32
activation=prelu
base_model=GCNConv
num_layers=2
tau=0.4
num_epochs=500
reg_lambda=0.5

python train_homo_sub.py \
    --dataset $dataset \
    --seed $seed \
    --learning_rate_train $learning_rate_train \
    --learning_rate_adv $learning_rate_adv \
    --num_hidden $num_hidden \
    --num_proj_hidden $num_hidden \
    --num_edge_hidden $num_edge_hidden \
    --activation $activation \
    --base_model $base_model \
    --num_layers $num_layers \
    --tau $tau \
    --num_epochs $num_epochs \
    --reg_lambda $reg_lambda