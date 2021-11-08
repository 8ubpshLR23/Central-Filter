
rem --------------------------vgg_16_bn----------------------------------

rem @echo off
rem set root=C:\...\
rem set pojname=CF
rem set arch=vgg_16_bn  
rem start cmd /c ^
rem "cd /D %root%%pojname%  ^
rem & [python.exe dir]/python.exe main_win.py ^
rem --arch vgg_16_bn ^
rem --resume [pre-trained model dir] ^
rem --compress_rate [0.3]*5+[0.5]*3+[0.8]*4 ^
rem --num_workers 1 ^
rem --epochs 1 ^
rem --job_dir %root%%pojname% ^
rem --lr 0.001 ^
rem --lr_decay_step 5 ^
rem --save_id 1 ^
rem --weight_decay 0.005 ^
rem --dataset CIFAR10 ^
rem --data_dir G:/data ^
rem & pause"

rem --------------------------resnet_56----------------------------------

rem @echo off
rem set root=C:\...\
rem set pojname=CF
rem set arch=resnet_56  
rem start cmd /c ^
rem "cd /D %root%%pojname%  ^
rem & [python.exe dir]/python.exe main_win.py ^
rem --arch resnet_56 ^
rem --resume [pre-trained model dir] ^
rem --compress_rate [0.]+[0.2,0.]*2+[0.6,0.]*7+[0.7,0.]*9+[0.8,0.]*9 ^
rem --num_workers 1 ^
rem --epochs 1 ^
rem --job_dir %root%%pojname% ^
rem --lr 0.001 ^
rem --lr_decay_step 5 ^
rem --save_id 1 ^
rem --weight_decay 0.005 ^
rem --dataset CIFAR10 ^
rem --data_dir G:\data ^
rem & pause"

rem -------------------------densenet_40-----------------------------------

rem @echo off
rem set root=C:\...\
rem set pojname=CF
rem set arch=densenet_40  
rem start cmd /c ^
rem "cd /D %root%%pojname%  ^
rem & [python.exe dir]/python.exe main_win.py ^
rem --arch densenet_40 ^
rem --resume [pre-trained model dir] ^
rem --compress_rate [0.]+[0.5]*12+[0.3]+[0.4]*12+[0.3]+[0.4]*9+[0.]*3 ^
rem --num_workers 1 ^
rem --epochs 1 ^
rem --job_dir %root%%pojname% ^
rem --lr 0.001 ^
rem --lr_decay_step 5 ^
rem --save_id 2 ^
rem --weight_decay 0.005 ^
rem --dataset CIFAR10 ^
rem --data_dir G:\data ^
rem & pause"


rem ----------------------------resnet_50--------------------------------

rem @echo off
rem set root=C:\...\
rem set pojname=CF
rem set arch=resnet_50  
rem start cmd /c ^
rem "cd /D %root%%pojname%  ^
rem & [python.exe dir]/python.exe main_win.py ^
rem --arch resnet_50 ^
rem --resume [pre-trained model dir] ^
rem --compress_rate [0.]+[0.1,0.1,0.2]*1+[0.5,0.5,0.2]*2+[0.1,0.1,0.2]*1+[0.5,0.5,0.2]*3+[0.1,0.1,0.2]*1+[0.5,0.5,0.2]*5+[0.1,0.1,0.1]+[0.2,0.2,0.1]*2 ^
rem --num_workers 4 ^
rem --epochs 1 ^
rem --job_dir %root%%pojname% ^
rem --lr 0.001 ^
rem --lr_decay_step 1 ^
rem --save_id 1 ^
rem --batch_size 64 ^
rem --weight_decay 0. ^
rem --input_size 224 ^
rem --start_cov 0 ^
rem --dataset ImageNet ^
rem --data_dir G:\ImageNet ^
rem & [python.exe dir]/python.exe main_win.py ^
rem --arch resnet_50 ^
rem --from_scratch True ^
rem --resume finally_pruned_model/resnet_50_1.pt ^
rem --num_workers 4 ^
rem --epochs 30 ^
rem --job_dir %root%%pojname% ^
rem --lr 0.001 ^
rem --lr_decay_step 5,15 ^
rem --save_id 1 ^
rem --batch_size 64 ^
rem --weight_decay 0.0005 ^
rem --input_size 224 ^
rem --dataset ImageNet ^
rem --data_dir G:\ImageNet ^
rem & pause"

rem ----------------------------googlenet--------------------------------

rem @echo off
rem set root=C:\...\
rem set pojname=CF
rem set arch=googlenet  
rem start cmd /c ^
rem "cd /D %root%%pojname%  ^
rem & [python.exe dir]/python.exe main_win.py ^
rem --arch googlenet ^
rem --resume [pre-trained model dir] ^
rem --compress_rate [0.2]+[0.8]*24+[0.,0.4,0.] ^
rem --num_workers 1 ^
rem --epochs 1 ^
rem --job_dir %root%%pojname% ^
rem --lr 0.001 ^
rem --lr_decay_step 5 ^
rem --save_id 1 ^
rem --weight_decay 0. ^
rem --dataset CIFAR10 ^
rem --data_dir G:\data ^
rem & [python.exe dir]/python.exe main_win.py ^
rem --arch googlenet ^
rem --from_scratch True ^
rem --resume finally_pruned_model/googlenet_1.pt ^
rem --num_workers 2 ^
rem --epochs 30 ^
rem --job_dir %root%%pojname% ^
rem --lr 0.01 ^
rem --lr_decay_step 5,15 ^
rem --save_id 1 ^
rem --weight_decay 0.005 ^
rem --data_dir G:\data ^
rem --dataset CIFAR10 ^
rem & pause"
