dataset=chameleon
seed=123
base_model=GCNConv
num_layers=2

for num_epochs in 100 200 300 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000;do
for activation in prelu relu rrelu gelu leakyrelu;do
for num_hidden in 128 256 512 ;do
for num_edge_hidden in 32 64 128;do
for learning_rate_train in 0.01 0.005 0.001 0.0005;do
for learning_rate_adv in 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001;do
for reg_lambda in 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9 1.0 3 5 10;do
for tau in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;do

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

done
done
done
done
done
done
done
done


#############################################
dataset=cora_lcc
seed=123
base_model=GCNConv
num_layers=2

for num_epochs in 100 200 300 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000;do
for activation in prelu relu rrelu gelu leakyrelu;do
for num_hidden in 128 256 512 ;do
for num_edge_hidden in 32 64 128;do
for learning_rate_train in 0.01 0.005 0.001 0.0005;do
for learning_rate_adv in 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001;do
for reg_lambda in 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9 1.0 3 5 10;do
for tau in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;do

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
    
done
done
done
done
done
done
done
done


#############################################
dataset=AmazonP
seed=123
base_model=GCNConv
num_layers=2

for num_epochs in 100 200 300 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000;do
for activation in prelu relu rrelu gelu leakyrelu;do
for num_hidden in 128 256 512 ;do
for num_edge_hidden in 32 64 128;do
for learning_rate_train in 0.01 0.005 0.001 0.0005;do
for learning_rate_adv in 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001;do
for reg_lambda in 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9 1.0 3 5 10;do
for tau in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;do

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
    
done
done
done
done
done
done
done
done