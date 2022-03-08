# **Network Compression via Central Filter**



## Environments

The code has been tested in the following environments:

- Python 3.8
- PyTorch 1.8.1
- cuda 10.2
- torchsummary, torchvision, thop

Both windows and linux are available.

## Pre-trained Models

**CIFAR-10:**

[Vgg-16](https://drive.google.com/open?id=1i3ifLh70y1nb8d4mazNzyC4I27jQcHrE) | [ResNet56](https://drive.google.com/open?id=1f1iSGvYFjSKIvzTko4fXFCbS-8dw556T) |  [DenseNet-40](https://drive.google.com/open?id=12rInJ0YpGwZd_k76jctQwrfzPubsfrZH) | [GoogLeNet](https://drive.google.com/open?id=1rYMazSyMbWwkCGCLvofNKwl58W6mmg5c)

**ImageNet:**

[ResNet50](https://drive.google.com/open?id=1OYpVB84BMU0y-KU7PdEPhbHwODmFvPbB)

## Running Code

The experiment is divided into two steps. We have provided the calculated data and can skip the first step.

### Similarity Matrix Generation

```bash
@echo off
@rem for windows
start cmd /c ^
"cd /D [code dir]  ^
& [python.exe dir]\python.exe rank.py ^
--arch [model arch name] ^
--resume [pre-trained model dir] ^
--num_workers [worker numbers] ^
--image_num [batch numbers] ^
--batch_size [batch size] ^
--dataset [CIFAR10 or ImageNet] ^
--data_dir [data dir] ^
--calc_dis_mtx True ^
& pause"
```

```sh
# for linux
python rank.py \
--arch [model arch name] \
--resume [pre-trained model dir] \
--num_workers [worker numbers] \
--image_num [batch numbers] \
--batch_size [batch size] \
--dataset [CIFAR10 or ImageNet] \
--data_dir [data dir] \
--calc_dis_mtx True
```

### Model Training

The experimental results and related configurations covered in this paper are as follows.

**1. VGGNet**

|   Architecture   | Compression Rate                        |    Params    |     Flops      | Accuracy |
| :--------------: | :----------------------------------- | :----------: | :------------: | :------: |
| VGG-16(Baseline) |                                      | 14.98M(0.0%) | 313.73M(0.0%)  |  93.96%  |
|      VGG-16      | [0.3]+[0.2]*4+[0.3]*2+[0.4]+[0.85]*4 | 2.45M(83.6%) | 124.10M(60.4%) |  93.67%  |
|      VGG-16      | [0.3]*5+[0.5]*3+[0.8]*4              | 2.18M(85.4%) | 91.54M(70.8%)  |  93.06%  |
|      VGG-16      | [0.3]*2+[0.45]*3+[0.6]*3+[0.85]*4    | 1.51M(89.9%) | 65.92M(79.0%)  |  92.49%  |

```sh
python main_win.py \
--arch vgg_16_bn \
--resume [pre-trained model dir] \
--compress_rate [0.3]*2+[0.45]*3+[0.6]*3+[0.85]*4 \
--num_workers [worker numbers] \
--epochs 30 \
--lr 0.001 \
--lr_decay_step 5 \
--save_id 1 \
--weight_decay 0.005 \
--data_dir [dataset dir] \
--dataset CIFAR10 
```

**2. ResNet-56**

|    Architecture     | Compression Rate                                    |    Params    |     Flops     | Accuracy |
| :-----------------: | :----------------------------------------------- | :----------: | :-----------: | :------: |
| ResNet-56(Baseline) |                                                  | 0.85M(0.0%)  | 125.49M(0.0%) |  93.26%  |
|      ResNet-56      | [0.]+[0.2,0.]*9+[0.3,0.]*9+[0.4,0.]*9            | 0.53M(37.6%) | 86.11M(31.4%) |  93.64%  |
|      ResNet-56      | [0.]+[0.3,0.]*9+[0.4,0.]*9+[0.5,0.]*9            | 0.45M(47.1%) | 75.7M(39.7%)  |  93.59%  |
|      ResNet-56      | [0.]+[0.2,0.]*2+[0.6,0.]*7+[0.7,0.]*9+[0.8,0.]*9 | 0.19M(77.6%) | 40.0M(68.1%)  |  92.19%  |

```sh
python main_win.py \
--arch resnet_56 \
--resume [pre-trained model dir] \
--compress_rate [0.]+[0.2,0.]*2+[0.6,0.]*7+[0.7,0.]*9+[0.8,0.]*9 \
--num_workers [worker numbers] \
--epochs 30 \
--lr 0.001 \
--lr_decay_step 5 \
--save_id 1 \
--weight_decay 0.005 \
--data_dir [dataset dir] \
--dataset CIFAR10 
```

**3.DenseNet-40**

|     Architecture      | Compression Rate                                     |    Params    |     Flops      | Accuracy |
| :-------------------: | :------------------------------------------------ | :----------: | :------------: | :------: |
| DenseNet-40(Baseline) |                                                   | 1.04M(0.0%)  | 282.00M(0.0%)  |  94.81%  |
|      DenseNet-40      | [0.]+[0.3]*12+[0.1]+[0.3]*12+[0.1]+[0.3]*8+[0.]*4 | 0.67M(35.6%) | 165.38M(41.4%) |  94.33%  |
|      DenseNet-40      | [0.]+[0.5]*12+[0.3]+[0.4]*12+[0.3]+[0.4]*9+[0.]*3 | 0.46M(55.8%) | 109.40M(61.3%) |  93.71%  |

```sh
# for linux
python main_win.py \
--arch densenet_40 \
--resume [pre-trained model dir] \
--compress_rate [0.]+[0.5]*12+[0.3]+[0.4]*12+[0.3]+[0.4]*9+[0.]*3 \
--num_workers [worker numbers] \
--epochs 30 \
--lr 0.001 \
--lr_decay_step 5 \
--save_id 1 \
--weight_decay 0.005 \
--data_dir [dataset dir] \
--dataset CIFAR10 
```

**4. GoogLeNet**

|    Architecture     | Compression Rate                      |    Params    |    Flops     | Accuracy |
| :-----------------: | :--------------------------------- | :----------: | :----------: | :------: |
| GoogLeNet(Baseline) |                                    | 6.15M(0.0%)  | 1520M(0.0%)  |  95.05%  |
|      GoogLeNet      | [0.2]+[0.7]*15+[0.8]*9+[0.,0.4,0.] | 2.73M(55.6%) | 0.56B(63.2%) |  94.70%  |
|      GoogLeNet      | [0.2]+[0.9]*24+[0.,0.4,0.]         | 2.17M(64.7%) | 0.37B(75.7%) |  94.13%  |

```sh

python main_win.py \
--arch googlenet \
--resume [pre-trained model dir] \
--compress_rate [0.2]+[0.9]*24+[0.,0.4,0.] \
--num_workers [worker numbers] \
--epochs 1 \
--lr 0.001 \
--save_id 1 \
--weight_decay 0. \
--data_dir [dataset dir] \
--dataset CIFAR10

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
```

**4. ResNet-50**

|    Architecture     | Compression Rate                                                |    Params     |    Flops     | Top-1 Accuracy | Top-5 Accuracy |
| :-----------------: | :----------------------------------------------------------- | :-----------: | :----------: | :------------: | -------------- |
| ResNet-50(baseline) |                                                              | 25.55M(0.0%)  | 4.11B(0.0%)  |     76.15%     | 92.87%         |
|      ResNet-50      | [0.]+[0.1,0.1,0.2]*1+[0.5,0.5,0.2]*2+[0.1,0.1,0.2]*1+[0.5,0.5,0.2]*3+[0.1,0.1,0.2]*1+[0.5,0.5,0.2]*5+[0.1,0.1,0.1]+[0.2,0.2,0.1]*2 | 16.08M(36.9%) | 2.13B(47.9%) |     75.08%     | 92.30%         |
|      ResNet-50      | [0.]+[0.1,0.1,0.4]*1+[0.7,0.7,0.4]*2+[0.2,0.2,0.4]*1+[0.7,0.7,0.4]*3+[0.2,0.2,0.3]*1+[0.7,0.7,0.3]*5+[0.1,0.1,0.1]+[0.2,0.3,0.1]*2 | 13.73M(46.2%) | 1.50B(63.5%) |     73.43%     | 91.57%         |
|      ResNet-50      | [0.]+[0.2,0.2,0.65]*1+[0.75,0.75,0.65]*2+[0.15,0.15,0.65]*1+[0.75,0.75,0.65]*3+[0.15,0.15,0.65]*1+[0.75,0.75,0.65]*5+[0.15,0.15,0.35]+[0.5,0.5,0.35]*2 | 8.10M(68.2%)  | 0.98B(76.2%) |     70.26%     | 89.82%         |

```sh
python main_win.py \
--arch resnet_50 \
--resume [pre-trained model dir] \
--data_dir [dataset dir] \
--dataset ImageNet \
--compress_rate [0.]+[0.1,0.1,0.4]*1+[0.7,0.7,0.4]*2+[0.2,0.2,0.4]*1+[0.7,0.7,0.4]*3+[0.2,0.2,0.3]*1+[0.7,0.7,0.3]*5+[0.1,0.1,0.1]+[0.2,0.3,0.1]*2 \
--num_workers [worker numbers] \
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
--epochs 40 \
--lr 0.001 \
--lr_decay_step 5,20 \
--save_id 2 \
--batch_size 64 \
--weight_decay 0.0005 \
--input_size 224 \
--data_dir [dataset dir] \
--dataset ImageNet 
```

