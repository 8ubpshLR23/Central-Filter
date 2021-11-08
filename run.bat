
rem --------------------------vgg_16_bn----------------------------------

@echo off
start cmd /c ^
"cd /D [code dir]  ^
& [python.exe dir]/python.exe main_win.py ^
--arch vgg_16_bn ^
--resume [pre-trained model dir] ^
--compress_rate [0.3]*5+[0.5]*3+[0.8]*4 ^
--num_workers 1 ^
--epochs 30 ^
--lr 0.001 ^
--lr_decay_step 5 ^
--save_id 1 ^
--weight_decay 0.005 ^
--dataset CIFAR10 ^
--data_dir [dataset dir] ^
& pause"

rem --------------------------resnet_56----------------------------------

@echo off
start cmd /c ^
"cd /D [code dir]  ^
& [python.exe dir]/python.exe main_win.py ^
--arch resnet_56 ^
--resume [pre-trained model dir] ^
--compress_rate [0.]+[0.2,0.]*2+[0.6,0.]*7+[0.7,0.]*9+[0.8,0.]*9 ^
--num_workers 1 ^
--epochs 30 ^
--lr 0.001 ^
--lr_decay_step 5 ^
--save_id 1 ^
--weight_decay 0.005 ^
--dataset CIFAR10 ^
--data_dir [dataset dir] ^
& pause"

rem -------------------------densenet_40-----------------------------------

@echo off
start cmd /c ^
"cd /D [code dir]  ^
& [python.exe dir]/python.exe main_win.py ^
--arch densenet_40 ^
--resume [pre-trained model dir] ^
--compress_rate [0.]+[0.5]*12+[0.3]+[0.4]*12+[0.3]+[0.4]*9+[0.]*3 ^
--num_workers 1 ^
--epochs 30 ^
--lr 0.001 ^
--lr_decay_step 5 ^
--save_id 2 ^
--weight_decay 0.005 ^
--dataset CIFAR10 ^
--data_dir [dataset dir] ^
& pause"


rem ----------------------------resnet_50--------------------------------

@echo off
start cmd /c ^
"cd /D [code dir]  ^
& [python.exe dir]/python.exe main_win.py ^
--arch resnet_50 ^
--resume [pre-trained model dir] ^
--compress_rate [0.]+[0.1,0.1,0.2]*1+[0.5,0.5,0.2]*2+[0.1,0.1,0.2]*1+[0.5,0.5,0.2]*3+[0.1,0.1,0.2]*1+[0.5,0.5,0.2]*5+[0.1,0.1,0.1]+[0.2,0.2,0.1]*2 ^
--num_workers 4 ^
--epochs 2 ^
--lr 0.001 ^
--lr_decay_step 1 ^
--save_id 1 ^
--batch_size 64 ^
--weight_decay 0. ^
--input_size 224 ^
--start_cov 0 ^
--dataset ImageNet ^
--data_dir [dataset dir] ^
& [python.exe dir]/python.exe main_win.py ^
--arch resnet_50 ^
--from_scratch True ^
--resume finally_pruned_model/resnet_50_1.pt ^
--num_workers 4 ^
--epochs 40 ^
--lr 0.001 ^
--lr_decay_step 5,20 ^
--save_id 1 ^
--batch_size 64 ^
--weight_decay 0.0005 ^
--input_size 224 ^
--dataset ImageNet ^
--data_dir [dataset dir] ^
& pause"

rem ----------------------------googlenet--------------------------------

@echo off
start cmd /c ^
"cd /D [code dir]  ^
& [python.exe dir]/python.exe main_win.py ^
--arch googlenet ^
--resume [pre-trained model dir] ^
--compress_rate [0.2]+[0.8]*24+[0.,0.4,0.] ^
--num_workers 1 ^
--epochs 1 ^
--lr 0.001 ^
--lr_decay_step 5 ^
--save_id 1 ^
--weight_decay 0. ^
--dataset CIFAR10 ^
--data_dir [dataset dir] ^
& [python.exe dir]/python.exe main_win.py ^
--arch googlenet ^
--from_scratch True ^
--resume finally_pruned_model/googlenet_1.pt ^
--num_workers 2 ^
--epochs 30 ^
--lr 0.01 ^
--lr_decay_step 5,15 ^
--save_id 1 ^
--weight_decay 0.005 ^
--data_dir [dataset dir] ^
--dataset CIFAR10 ^
& pause"
