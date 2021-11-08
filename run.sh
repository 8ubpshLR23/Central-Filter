
# --------------------------vgg_16_bn----------------------------------
python main_win.py \
--arch vgg_16_bn \
--resume [pre-trained model dir] \
--compress_rate [0.3]*5+[0.5]*3+[0.8]*4 \
--num_workers 1 \
--epochs 1 \
--lr 0.001 \
--lr_decay_step 5 \
--save_id 1 \
--weight_decay 0.005 \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--ablation_id 0


# --------------------------resnet_56----------------------------------

python main_win.py \
--arch resnet_56 \
--resume [pre-trained model dir] \
--compress_rate [0.]+[0.2,0.]*2+[0.6,0.]*7+[0.7,0.]*9+[0.8,0.]*9 \
--num_workers 1 \
--epochs 1 \
--lr 0.001 \
--lr_decay_step 5 \
--save_id 1 \
--weight_decay 0.005 \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--ablation_id 0


# --------------------------densenet_40----------------------------------

python main_win.py \
--arch densenet_40 \
--resume [pre-trained model dir] \
--compress_rate [0.]+[0.5]*12+[0.3]+[0.4]*12+[0.3]+[0.4]*9+[0.]*3 \
--num_workers 1 \
--epochs 1 \
--lr 0.001 \
--lr_decay_step 5 \
--save_id 1 \
--weight_decay 0.005 \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--ablation_id 0


# --------------------------googlenet----------------------------------

python main_win.py \
--arch googlenet \
--resume [pre-trained model dir] \
--compress_rate [0.2]+[0.7]*15+[0.8]*9+[0.,0.4,0.] \
--num_workers 2 \
--epochs 1 \
--lr 0.001 \
--save_id 1 \
--weight_decay 0. \
--data_dir [dataset dir] \
--dataset CIFAR10 \
--ablation_id 0

python main_win.py \
--arch googlenet \
--from_scratch True \
--resume finally_pruned_model/googlenet_1.pt \
--num_workers 2 \
--epochs 30 \
--lr 0.01 \
--lr_decay_step 5,15 \
--save_id 1 \
--weight_decay 0.005 \
--data_dir [dataset dir] \
--dataset CIFAR10


# ----------------------------resnet_50--------------------------------

python main_win.py \
--arch resnet_50 \
--resume [pre-trained model dir] \
--data_dir [dataset dir] \
--dataset ImageNet \
--compress_rate [0.]+[0.1,0.1,0.2]*1+[0.5,0.5,0.2]*2+[0.1,0.1,0.2]*1+[0.5,0.5,0.2]*3+[0.1,0.1,0.2]*1+[0.5,0.5,0.2]*5+[0.1,0.1,0.1]+[0.2,0.2,0.1]*2 \
--num_workers 8 \
--batch_size 64 \
--epochs 2 \
--lr_decay_step 1 \
--lr 0.001 \
--save_id 1 \
--weight_decay 0. \
--input_size 224 \
--start_cov 0 

python main_win.py \
--arch resnet_50 \
--from_scratch True \
--resume finally_pruned_model/resnet_50_1.pt \
--num_workers 8 \
--epochs 30 \
--lr 0.001 \
--lr_decay_step 5,15 \
--save_id 2 \
--batch_size 64 \
--weight_decay 0.0005 \
--input_size 224 \
--data_dir [dataset dir] \
--dataset ImageNet 