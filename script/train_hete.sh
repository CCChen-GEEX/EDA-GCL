#############################################
dataset=chameleon
seed=123
learning_rate_train=0.001
learning_rate_adv=1e-05
num_hidden=512
num_proj_hidden=512
num_edge_hidden=32
activation=leakyrelu
base_model=GCNConv
num_layers=2
tau=0.3
num_epochs=600
reg_lambda=0.2

python train_hete.py \
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
dataset=squirrel
seed=123
learning_rate_train=0.001
learning_rate_adv=1e-05
num_hidden=512
num_proj_hidden=512
num_edge_hidden=64
activation=relu
base_model=GCNConv
num_layers=2
tau=0.4
num_epochs=400
reg_lambda=0.1

python train_hete.py \
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
dataset=Actor
seed=123
learning_rate_train=0.001
learning_rate_adv=1e-05
num_hidden=128
num_proj_hidden=128
num_edge_hidden=32
activation=prelu
base_model=GCNConv
num_layers=2
tau=0.9
num_epochs=150
reg_lambda=0.7

python train_hete.py \
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
dataset=Texas
seed=123
learning_rate_train=0.0005
learning_rate_adv=0.0005
num_hidden=128
num_proj_hidden=128
num_edge_hidden=64
activation=gelu
base_model=GCNConv
num_layers=2
tau=0.3
num_epochs=100
reg_lambda=0.6

python train_hete.py \
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
dataset=Wisconsin
seed=123
learning_rate_train=0.001
learning_rate_adv=0.0005
num_hidden=128
num_proj_hidden=128
num_edge_hidden=64
activation=gelu
base_model=GCNConv
num_layers=2
tau=0.9
num_epochs=400
reg_lambda=0.5

python train_hete.py \
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