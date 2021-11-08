

@echo off
set root=C:\Users\huxf\Desktop\dyztmp\
set pojname=CF
set arch=vgg_16_bn
start cmd /c ^
"cd /D %root%%pojname%  ^
& %root%dyzenv\Scripts\python.exe rank.py ^
--arch %arch% ^
--resume ../pre_train_model/CIFAR-10/%arch%.pt ^
--num_workers 1 ^
--job_dir %root%%pojname% ^
--image_num 1 ^
--dataset CIFAR10 ^
--data_dir G:\data ^
--calc_dis_mtx True ^
--batch_size 1 ^
& pause"