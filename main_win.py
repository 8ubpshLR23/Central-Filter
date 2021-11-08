import torch.nn as nn
import numpy as np
import torch
import math
import argparse
import torch.optim as optim
import time
import os
import copy
import pickle
import random
from PIL import Image
from copy import deepcopy
from collections import OrderedDict
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import datasets, models, transforms

from utils import *
from models import *
from mask import *
from rank import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 & ImageNet Pruning')

parser.add_argument(
	'--data_dir',    
	default='./data',    
	type=str,   
	metavar='DIR',                 
	help='path to dataset')
parser.add_argument(
	'--dataset',     
	default='CIFAR10',   
	type=str,   
	choices=('CIFAR10','ImageNet'),
	help='dataset')
parser.add_argument(
	'--num_workers', 
	default=0,           
	type=int,   
	metavar='N',                   
	help='number of data loading workers (default: 0)')
parser.add_argument(
    '--epochs',
    type=int,
    default=15,
    help='The num of epochs to train.')
parser.add_argument(
	'--lr',         
	default=0.01,        
	type=float,                                
	help='initial learning rate')
parser.add_argument(
    '--lr_decay_step',
    default='5,10',
    type=str,
    metavar='LR',
    help='learning rate decay step')
parser.add_argument(
    '--resume',
    type=str,
    default=None,
    metavar='PATH',
    help='load the model from the specified checkpoint')
parser.add_argument(
	'--batch_size', 
	default=128, 
	type=int,
    metavar='N',
    help='mini-batch size')
parser.add_argument(
	'--momentum', 
	default=0.9, 
	type=float, 
	metavar='M',
    help='momentum')
parser.add_argument(
	'--weight_decay', 
	default=0., 
	type=float,
    metavar='W', 
    help='weight decay',
    dest='weight_decay')
parser.add_argument(
	'--gpu', 
	default='0', 
	type=int,
    help='GPU id to use.')
parser.add_argument(
    '--job_dir',
    type=str,
    default='',
    help='The directory where the summaries will be stored.')
parser.add_argument(
    '--compress_rate',
    type=str,
    default=None,
    help='compress rate of each conv')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('resnet_50','vgg_16_bn','resnet_56','densenet_40','googlenet'),
    help='The architecture to prune')
parser.add_argument(
    '--start_cov',
    type=int,
    default=-1,
    help='The num of conv to start prune')
parser.add_argument(
    '--input_size',
    type=int,
    default=32,
    help='The num of input size')
parser.add_argument(
    '--save_id',
    type=int,
    default=0,
    help='save_id')
parser.add_argument(
    '--ablation_id',
    type=int,
    default=0,
    help='for ablation study')
parser.add_argument(
    '--from_scratch',
    type=bool,
    default=False,
    help='train from_scratch')


args           = None
lr_decay_step  = None
logger         = None
compress_rate  = None
trainloader    = None
testloader     = None
criterion      = None
device         = None
model          = None
mask           = None
best_acc       = 0.
best_accs      = []
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def init():
	global args,lr_decay_step,logger,compress_rate,trainloader,testloader,criterion,device,model,mask,best_acc,best_accs
	args = parser.parse_args()
	lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
	logger = get_logger(os.path.join(args.job_dir, 'log/log'))
	compress_rate = format_compress_rate(args.compress_rate)
	trainloader,testloader = load_data(data_name = args.dataset, data_dir = args.data_dir, batch_size = args.batch_size, num_workers = args.num_workers)
	criterion = nn.CrossEntropyLoss()
	device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
	model = eval(args.arch)().to(device)
	mask = eval('mask_'+args.arch)(model=model, job_dir=args.job_dir, device=device)
	best_acc = 0.  # best test accuracy

	if args.from_scratch is False and args.arch == 'resnet_50' and os.path.exists('temp/compress_rate_resnet50'):
		with open('temp/compress_rate_resnet50', 'rb') as f:
			last_compress_rate = pickle.load(f)
			if args.start_cov > 1: compress_rate = last_compress_rate[:(args.start_cov-1)*3+1] + compress_rate[(args.start_cov-1)*3+1:]
			# if args.start_cov < 16: compress_rate =  compress_rate[:(args.start_cov-1)*3+1] + last_compress_rate[(args.start_cov-1)*3+1:]

	logger.info('args:{}'.format(args))
	pass

def train(epoch,model,cov_id,trainloader,optimizer,criterion,mask = None):
	losses = AverageMeter('Loss', ':.4f')
	top1 = AverageMeter('Acc@1', ':.2f')
	top5 = AverageMeter('Acc@5', ':.2f')

	model.train()
	since = time.time()
	_since = time.time()
	for i, (inputs,labels) in enumerate(trainloader, 0):

		if i > 1 : break

		inputs = inputs.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		if mask is not None : mask.grad_mask(cov_id)

		acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
		losses.update(loss.item(), inputs.size(0))
		top1.update(acc1[0], inputs.size(0))
		top5.update(acc5[0], inputs.size(0))

		if i!=0 and i%2000 == 0:
			_end = time.time()
			logger.info('epoch[{}]({}/{}) Loss: {:.4f} Acc@1: {:.2f} Acc@5: {:.2f} time: {:.4f}'.format(epoch,i,int(1280000/args.batch_size),losses.avg,top1.avg,top5.avg,_end - _since))
			_since = time.time()

	end = time.time()
	logger.info('train    Loss: {:.4f} Acc@1: {:.2f} Acc@5: {:.2f} time: {:.4f}'.format(losses.avg,top1.avg,top5.avg,end - since))

def validate(epoch,model,cov_id,testloader,criterion,save = True):
	losses = AverageMeter('Loss', ':.4f')
	top1 = AverageMeter('Acc@1', ':.2f')
	top5 = AverageMeter('Acc@5', ':.2f')

	model.eval()
	with torch.no_grad():
		since = time.time()
		for i, data in enumerate(testloader, 0):
			if i > 10 : break
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)

			outputs = model(inputs)
			loss = criterion(outputs, labels)

			acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
			losses.update(loss.item(), inputs.size(0))
			top1.update(acc1[0], inputs.size(0))
			top5.update(acc5[0], inputs.size(0))

		end = time.time()
		logger.info('validate Loss: {:.4f} Acc@1: {:.2f} Acc@5: {:.2f} time: {:.4f}'.format(losses.avg,top1.avg,top5.avg,end - since))

	global best_acc
	if save and best_acc < top1.avg:
		best_acc = top1.avg
		state = {
			'state_dict': model.state_dict(),
			'best_prec1': best_acc,
			'epoch': epoch,
			# 'scheduler':scheduler.state_dict(),
			# 'optimizer': optimizer.state_dict() 
		}
		if not os.path.isdir(args.job_dir + '/pruned_checkpoint'):
			os.makedirs(args.job_dir + '/pruned_checkpoint')
		cov_name = '_cov' + str(cov_id)
		if cov_id == -1: cov_name = ''
		torch.save(state,args.job_dir + '/pruned_checkpoint/'+args.arch+cov_name + '.pt')
		logger.info('storing checkpoint:'+'/pruned_checkpoint/'+args.arch+cov_name + '.pt')

	return top1.avg,top5.avg

def iter_vgg(layers = 13):

	cfg_cov = [0,1,3,4,6,7,8,10,11,12,14,15,16]
	ranks = []
	optimizer = None 
	scheduler = None

	for cov_id in range(layers-2,-1,-1):

		logger.info("===> pruning layer {}".format(cov_id))
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)
		
		if cov_id == layers-2: #start_cov
			pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
			logger.info('loading checkpoint:' + args.resume)
			model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))

		else :
			pruned_checkpoint = torch.load(args.job_dir + "/pruned_checkpoint/"+args.arch+"_cov" + str(cov_id+1) + '.pt', map_location=device)
			logger.info('loading checkpoint:' + "/pruned_checkpoint/"+args.arch+"_cov" + str(cov_id+1) + '.pt')
			model.load_state_dict(pruned_checkpoint['state_dict'])

		conv_name = 'features.conv' + str(cfg_cov[cov_id]) + '.weight'
		state_dict_conv = model.state_dict()[conv_name]
		rank,delegate_G = ger_rank_one_ablation_study(model,args.arch,cov_id,compress_rate[cov_id],trainloader,device,type = args.ablation_id)

		conv_name = 'features.conv' + str(cfg_cov[cov_id + 1]) + '.weight'
		state_dict_conv = model.state_dict()[conv_name]

		ranks.insert(0,rank)
		mask.layer_mask(cov_id,param_per_cov=4, arch=args.arch)

		if args.ablation_id < 1:
			before_acc1,_ = validate(0,model,0,testloader,criterion,save = False)
			incs = adjust_filter(conv_name,delegate_G,model = model)

			after_acc1,_=validate(0,model,0,testloader,criterion,save = False)

			if after_acc1<before_acc1:
				adjust_filter_repeal(conv_name,incs,model = model)
				validate(0,model,0,testloader,criterion,save = False)

		_train(cov_id)

		logger.info("===> conv_id {} bestacc {:.4f}".format(cov_id,best_accs[-1]))

	logger.info(best_accs)
	logger.info([len(x) for x in ranks])

	finally_state_dict = torch.load(args.job_dir + "/pruned_checkpoint/"+args.arch+"_cov" + str(0) + '.pt', map_location=device)
	rst_model = vgg_16_bn(_state_dict(finally_state_dict['state_dict']),ranks)
	logger.info(rst_model)
	flops,params = model_size(rst_model,args.input_size,device)
	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}
	if not os.path.isdir(args.job_dir + '/finally_pruned_model'):
		os.makedirs(args.job_dir + '/finally_pruned_model')
	torch.save(state,args.job_dir + '/finally_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'/finally_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('finally model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def iter_resnet_56():

	ranks = []
	layers = 55

	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	for layer_id in range(3,0,-1):
		for block_id in range(8,-1,-1):

			logger.info("===> pruning layer_id {} block_id {}".format(layer_id,block_id))

			if layer_id == 3 and block_id == 8: 
				pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
				logger.info('loading checkpoint:' + args.resume)
				model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))
			else :
				pruned_checkpoint = torch.load(args.job_dir + "/pruned_checkpoint/"+args.arch+"_cov" + str(_id) + '.pt', map_location=device)
				logger.info('loading checkpoint:' + "/pruned_checkpoint/"+args.arch+"_cov" + str(_id) + '.pt')
				model.load_state_dict(pruned_checkpoint['state_dict'])

			tmp_ranks = []
			for cov_id in range(1,3):
				_id = (layer_id-1)*18 + block_id * 2 + cov_id 
				conv_name = 'layer'+str(layer_id)+'.'+str(block_id)+'.conv'+str(cov_id)+'.weight'
				state_dict_conv = model.state_dict()[conv_name]
				rank = []
				rank,delegate_G = ger_rank_one_ablation_study(model,args.arch,_id,compress_rate[_id],trainloader,device,type = args.ablation_id)
				mask.layer_mask(_id,param_per_cov=3, arch=args.arch)

				if cov_id == 1 and args.ablation_id < 1:
					conv_name = get_conv_name(args.arch,_id+1) 	
					before_acc1,_ = validate(0,model,0,testloader,criterion,save = False)					
					incs = adjust_filter(conv_name,delegate_G,model = model)
					after_acc1,_ = validate(0,model,0,testloader,criterion,save = False)
					if after_acc1<before_acc1:
						adjust_filter_repeal(conv_name,incs,model = model)
						validate(0,model,0,testloader,criterion,save = False)
				tmp_ranks.append(rank)
			ranks = tmp_ranks + ranks

			_train(_id)

			logger.info("===> layer_id {} block_id {} bestacc {:.4f}".format(layer_id,block_id,best_accs[-1]))

	#--------------------------------------------the first layer------------------------------------#

	pruned_checkpoint = torch.load(args.job_dir + "/pruned_checkpoint/"+args.arch+"_cov" + str(2) + '.pt', map_location=device)
	logger.info('loading checkpoint:' + "/pruned_checkpoint/"+args.arch+"_cov" + str(2) + '.pt')
	model.load_state_dict(pruned_checkpoint['state_dict'])
	rank,delegate_G = ger_rank_one_ablation_study(model,args.arch,0,compress_rate[0],trainloader,device,type = args.ablation_id)
	mask.layer_mask(0,param_per_cov=3, arch=args.arch)
	ranks.insert(0,rank)

	conv_name = 'layer1.0.conv1.weight'
	if args.ablation_id < 1: adjust_filter(conv_name,delegate_G,model = model)
	if compress_rate[0] > 0.:
		_train(0)
	else:
		_train_last(0)
	logger.info("===> the first layer conv1 bestacc {:.4f}".format(best_accs[-1]))
	#------------------------------------------------------------------------------------------------#


	#---------------------------------------------save model-----------------------------------------#

	logger.info(best_accs)
	logger.info([len(x) for x in ranks])

	finally_state_dict = torch.load(args.job_dir + "/pruned_checkpoint/"+args.arch+"_cov" + str(0) + '.pt', map_location=device)
	rst_model = resnet_56(compress_rate = compress_rate,oristate_dict = _state_dict(finally_state_dict['state_dict']),ranks=ranks)

	logger.info(rst_model)
	flops,params = model_size(rst_model,args.input_size,device)

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}

	if not os.path.isdir(args.job_dir + '/finally_pruned_model'):
		os.makedirs(args.job_dir + '/finally_pruned_model')
	torch.save(state,args.job_dir + '/finally_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'/finally_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('finally model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def iter_resnet_50():
	ranks = []
	layers = 49
	stage_repeat = [3, 4, 6, 3]

	_map = [0]
	cnt = 1
	for x in stage_repeat:
		for y in range(x):
			_map.append(cnt)
			_map.append(cnt+1)
			_map.append(cnt+2)
			if y == 0: _map.append(cnt+2)
			cnt += 3
    #---------------------------------------------get start point-----------------------------------------#
    # mask_resume
	layer_start_id = 1
	block_start_id = 0
	start_id = 0
	mask_resume = None
	if args.start_cov >= 1:
		start_cov = args.start_cov
		mask_resume = os.path.join(args.job_dir, 'mask/mask_resnet_50')

		cnt = 0
		flag = True
		for layer_id in range(1,4+1):
			if flag is False: break
			for block_id in range(stage_repeat[layer_id-1]):
				cnt += 1
				if cnt == start_cov:
					layer_start_id = layer_id
					block_start_id = block_id
					flag = False
					break
				start_id += 3
				if block_id == 0: start_id += 1 
    #---------------------------------------------loading checkpoint-----------------------------------------#

	pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
	logger.info('loading checkpoint:' + args.resume)
	# print(pct.keys())
	model.load_state_dict(pruned_checkpoint)

	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	#--------------------------------------------------------------------------------------------------------#
	conv_names = []
	for name, module in model.named_modules():
		if isinstance(module,nn.Conv2d):
			conv_names.append(name+'.weight')
	
	if args.start_cov < 1:
		rank,delegate_G = ger_rank_one(model,args.arch,0,compress_rate[0],trainloader,device)
	#-----------------------------------loading ranks form checkpoint----------------------------------------#
	utils.logger.info('loading ranks form checkpoint...')
	prefix = "rank_conv/"+args.arch+"/rank_conv"
	subfix = ".npy"
	rank = np.load(prefix + str(_map[0]) + subfix)
	utils.logger.info('loading '+prefix + str(_map[0]) + subfix)
	ranks.append(rank)
	_id = 0
	last_union_rank = None
	for layer_id in range(1,layer_start_id+1):
		for block_id in range(0,stage_repeat[layer_id-1]):
			if layer_id >= layer_start_id and block_id >= block_start_id: break
			for cov_id in range(1,4):
				# print(layer_id,block_id,cov_id)
				_id += 1
				rank = np.load(prefix + str(_map[_id]) + subfix)
				utils.logger.info('loading '+prefix + str(_map[_id]) + subfix)
				ranks.append(rank)
				if block_id == 0 and cov_id == 3:
					_id += 1
					rank = np.load(prefix + str(_map[_id]) + subfix)
					utils.logger.info('loading '+prefix + str(_map[_id]) + subfix)
					last_union_rank = rank
					ranks.append(rank)

	#-------------------------------------------the cov ids per block----------------------------------------#
	block_3th_index = []
	cnt = 0
	for _,x in enumerate(stage_repeat):
		tmp = []
		for i in range(x):
			cnt += 1
			tmp.append(cnt*3)
		block_3th_index.append(tmp)
	#-------------------------------------------pruning------------------------------------------------------#
	_id = start_id
	for layer_id in range(layer_start_id,4+1): #toddo

		#-----------------------------union_rank per block-----------------#
		logger.info("===> calc the union_rank of the 3-th Conv per Block")
		if block_start_id > 0:
			union_rank = last_union_rank
		else :
			tmp_cprate = compress_rate[block_3th_index[layer_id-1][0]] # 
			N = stage_out_channel[int(block_3th_index[layer_id-1][0]/3)]
			idx_cprate = 0.

			if tmp_cprate > 0.:
				for i in range(99,-10,-10):
					if i < 0. : i = 1
					idx_cprate = i/100
					utils.logger.info('idx_cprate: {:.4f}'.format(idx_cprate))
					union_rank = set()
					for i,x in enumerate(block_3th_index[layer_id-1]):
						rank,_ = ger_rank_one(model,args.arch,x,idx_cprate,trainloader,device)
						union_rank = union_rank|set(rank)
					union_rank = list(union_rank)
					rst_cprate = (len(union_rank))/N
					# print(idx_cprate,tmp_cprate,rst_cprate)
					if rst_cprate < (1.-tmp_cprate): 
						continue
					else :
						union_rank = union_rank[0:int(N*(1-tmp_cprate))]
						break
			else :
				rank,_ = ger_rank_one(model,args.arch,block_3th_index[layer_id-1][0],tmp_cprate,trainloader,device)
				union_rank = rank
			for i,x in enumerate(block_3th_index[layer_id-1]):
				if compress_rate[x] > 0. : compress_rate[x] = tmp_cprate

		with open('temp/compress_rate_resnet50', "wb") as f:
			# print('>>>>> ',compress_rate)
			pickle.dump(compress_rate, f)

		#---------------------------------------------------------------------#

		for block_id in range(block_start_id,stage_repeat[layer_id-1]):
			logger.info("===> pruning layer_id {} block_id {}".format(layer_id,block_id))
			block_start_id = 0 # reset
			if _id != 0:
				pruned_checkpoint = torch.load(args.job_dir + "/pruned_checkpoint/"+args.arch+"_cov" + str(_id) + '.pt', map_location=device)
				logger.info('loading checkpoint:' + "/pruned_checkpoint/"+args.arch+"_cov" + str(_id) + '.pt')
				model.load_state_dict(pruned_checkpoint['state_dict'])

			delegate_Gs = []
			cov_ids = []

			for cov_id in range(1,4):
				_id += 1
				conv_name = 'layer'+str(layer_id)+'.'+str(block_id)+'.conv'+str(cov_id)+'.weight'
				state_dict_conv = model.state_dict()[conv_name]
				if cov_id == 3:
					rank = union_rank
					block_3th_delegate = get_delegate_from_rank(model,args.arch,_map[_id],compress_rate[_map[_id]],trainloader,device,rank)
				else :
					rank,delegate_G = ger_rank_one(model,args.arch,_map[_id],compress_rate[_map[_id]],trainloader,device)
				mask.layer_mask(_id,param_per_cov=3, arch=args.arch)
				delegate_Gs.append(delegate_G)
				cov_ids.append(_id)
				if block_id == 0 and cov_id == 3:
					_id += 1
					mask.layer_mask(_id,param_per_cov=3, arch=args.arch)
					ranks.append(rank)
					cov_ids.append(_id)
				ranks.append(rank)

			#----------------------------adjusting weights-------------------#

			validate(0,model,0,testloader,criterion,save = False)
			cnt = 0
			for x in cov_ids:
				if cnt < 2:
					conv_name = conv_names[x+1]
					adjust_filter(conv_name,delegate_Gs[cnt],model = model)
				cnt += 1

			if layer_id == 4 and block_id == 2:
				pass
			else :
				conv_name = conv_names[cov_ids[-1]+1]
				adjust_filter(conv_name,block_3th_delegate,model = model)
			validate(0,model,0,testloader,criterion,save = False)

			#---------------------------------------------------------------#

			_train(_id)

			logger.info("===> layer_id {} block_id {} bestacc {:.4f}".format(layer_id,block_id,best_accs[-1]))


	#---------------------------------------------save model-----------------------------------------#

	with open('temp/compress_rate_resnet50', "wb") as f:
		pickle.dump(compress_rate, f)

	logger.info(best_accs)
	logger.info(compress_rate)
	logger.info([len(x) for x in ranks])

	finally_state_dict = torch.load(args.job_dir + "/pruned_checkpoint/"+args.arch+'_cov52.pt', map_location=device)
	rst_model = resnet_50(compress_rate=compress_rate,oristate_dict = _state_dict(finally_state_dict['state_dict']),ranks = ranks)
	logger.info(rst_model)
	# input_size = 224
	flops,params = model_size(rst_model,args.input_size,device)

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}
	if not os.path.isdir(args.job_dir + '/finally_pruned_model'):
		os.makedirs(args.job_dir + '/finally_pruned_model')
	torch.save(state,args.job_dir + '/finally_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'/finally_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('finally model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def iter_densenet_40():
	ranks = []
	delegate_Gs = []
	splits = []
	num_inplanes = 24

	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
	logger.info('loading checkpoint:' + args.resume)
	model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))

	#---------------------------------------------#
	conv_names = []
	for name, module in model.named_modules():
		if isinstance(module,nn.Conv2d):
			conv_names.append(name+'.weight')
	#---------------------------------------------#

	rank,delegate_G = ger_rank_one(model,args.arch,0,compress_rate[0],trainloader,device)
	ranks.append(rank) #the first layer
	mask.layer_mask(0,param_per_cov=3, arch=args.arch)

	validate(0,model,0,testloader,criterion,save = False)
	adjust_filter(conv_names[1],delegate_G)
	validate(0,model,0,testloader,criterion,save = False)

	_train(0)
	logger.info("===> the first layer conv1 bestacc {:.4f}".format(best_accs[-1]))

	_id = 1
	delegate_Gs.append(delegate_G)
	splits.append(num_inplanes)

	#------------------------------pruning-------------------------------#

	for dense_id in range(1,4):
		for block_id in range(0,12):

			logger.info("===> pruning dense_id {} block_id {}".format(dense_id,block_id))
			
			pruned_checkpoint = torch.load(args.job_dir + "/pruned_checkpoint/"+args.arch+"_cov" + str(_id-1) + '.pt', map_location=device)
			logger.info('loading checkpoint:' + "/pruned_checkpoint/"+args.arch+"_cov" + str(_id-1) + '.pt')
			model.load_state_dict(pruned_checkpoint['state_dict'])

			conv_name = 'dense' + str(dense_id) +'.' + str(block_id) + '.conv1.weight'
			state_dict_conv = model.state_dict()[conv_name]
			rank,delegate_G = ger_rank_one(model,args.arch,_id,compress_rate[_id],trainloader,device)
			mask.layer_mask(_id,param_per_cov=3, arch=args.arch)
			ranks.append(rank)
			delegate_Gs.append(delegate_G)
			splits.append(splits[-1]+len(state_dict_conv))
			if compress_rate[_id] > 0. and _id < 38:
				before_acc1,_ = validate(0,model,0,testloader,criterion,save = False)					
				incs = adjust_filter_densenet40(conv_names[_id+1],delegate_Gs,splits)
				after_acc1,_ = validate(0,model,0,testloader,criterion,save = False)
				if after_acc1<before_acc1:
					adjust_filter_densenet40_repeal(conv_names[_id+1],incs)
					validate(0,model,0,testloader,criterion,save = False)

			_train(_id)
			_id += 1
			logger.info("===> dense_id {} block_id {} best_acc {:.4f}".format(dense_id,block_id,best_accs[-1]))

		#trans1.conv1.weight
		if dense_id < 3:
			logger.info("===> pruning trans {}".format(dense_id))
			conv_name = 'trans' + str(dense_id) +'.conv1.weight'
			state_dict_conv = model.state_dict()[conv_name]
			rank,delegate_G = ger_rank_one(model,args.arch,_id,compress_rate[_id],trainloader,device)
			mask.layer_mask(_id,param_per_cov=3, arch=args.arch)
			ranks.append(rank)

			before_acc1,_ = validate(0,model,0,testloader,criterion,save = False)
			incs = adjust_filter(conv_names[_id+1] ,delegate_G,model = model)
			after_acc1,_ = validate(0,model,0,testloader,criterion,save = False)
			if after_acc1<before_acc1:
				adjust_filter_repeal(conv_names[_id+1],incs,model = model)
				validate(0,model,0,testloader,criterion,save = False)

			_train(_id)
			logger.info("===> trans {} best_acc {:.4f}".format(dense_id,best_accs[-1]))
			_id += 1
			splits = []
			delegate_Gs = []
			splits.append(len(state_dict_conv))
			delegate_Gs.append(delegate_G)

	#---------------------------------------------save model-----------------------------------------#

	logger.info(best_accs)
	logger.info([len(x) for x in ranks])

	finally_state_dict = torch.load(args.job_dir + "/pruned_checkpoint/"+args.arch+'_cov38.pt', map_location=device)
	rst_model = densenet_40(compress_rate = compress_rate,oristate_dict = _state_dict(finally_state_dict['state_dict']),ranks = ranks).to(device)
	flops,params = model_size(rst_model,args.input_size,device)
	logger.info(rst_model)
	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}
	if not os.path.isdir(args.job_dir + '/finally_pruned_model'):
		os.makedirs(args.job_dir + '/finally_pruned_model')
	torch.save(state,args.job_dir + '/finally_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'/finally_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('finally model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def iter_googlenet():
	ranks = []
	inceptions = ['a3','b3','a4','b4','c4','d4','e4','a5','b5']
	rank_save_id  = ['_3','_5_1','_5_2']
	branch_subfix = ['.branch1x1.0.weight','.branch3x3.0.weight','.branch5x5.0.weight','.branch_pool.1.weight']
	num_inplanes = 192
	delegate_Gs = []
	filters = [
			[192],
            [64, 128, 32, 32],
            [128, 192, 96, 64],
            [192, 208, 48, 64],
            [160, 224, 64, 64],
            [128, 256, 64, 64],
            [112, 288, 64, 64],
            [256, 320, 128, 128],
            [256, 320, 128, 128],
            [384, 384, 128, 128]
        ]

	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
	logger.info('loading checkpoint:' + args.resume)
	model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))


	rank,delegate_G = ger_rank_one(model,args.arch,0,compress_rate[0],trainloader,device)
	ranks.append([rank])
	delegate_Gs.append(delegate_G)

	mask.layer_mask(0,param_per_cov=28, arch=args.arch)

	if args.ablation_id < 1: 

		validate(0,model,0,testloader,criterion,save = False)

		for x in branch_subfix:
			name = 'inception_'+inceptions[0]+x
			adjust_filters(name,delegate_Gs,filters[0],model = model)

	after_acc1,_ = validate(0,model,0,testloader,criterion,save = False)

	_train(0)
	if args.epochs == 0:
		best_accs.append(round(after_acc1.item(),4))

	logger.info("===> the first layer conv1 bestacc {:.4f}".format(best_accs[-1]))

	cnt = 1
	for i,inception_id in enumerate(inceptions):

		i += 1
		_rank = []
		delegate_Gs = []

		pruned_checkpoint = torch.load(args.job_dir + "/pruned_checkpoint/"+args.arch+"_cov" + str(i-1) + '.pt', map_location=device)
		logger.info('loading checkpoint:' + "/pruned_checkpoint/"+args.arch+"_cov" + str(i-1) + '.pt')
		model.load_state_dict(pruned_checkpoint['state_dict'])

		delegate_Gs.append([])
		_branch_subfix = ['.branch3x3.3.weight','.branch5x5.3.weight','.branch5x5.6.weight']
		for k in range(3):
			conv_name = 'inception_' + str(inception_id) + _branch_subfix[k]
			state_dict_conv = model.state_dict()[conv_name]

			rank,delegate_G = ger_rank_one_ablation_study(model,args.arch,(i-1)*3+k+1,compress_rate[cnt],trainloader,device,type = args.ablation_id,save_id=str(i)+rank_save_id[k])
			if k==1 and args.ablation_id < 1: 
				adjust_filter('inception_'+inception_id+_branch_subfix[2],delegate_G,model = model)
			_rank.append(rank)
			if k != 1: delegate_Gs.append(delegate_G)
			cnt += 1
		mask.layer_mask(i,param_per_cov=28, arch=args.arch)
		ranks.append(_rank)
		delegate_Gs.append([])

		if i <= 8 and args.ablation_id < 1: 
			validate(0,model,0,testloader,criterion,save = False)
			for j,x in enumerate(branch_subfix):
				name = 'inception_'+inceptions[i]+x
				adjust_filters(name,delegate_Gs,filters[i],model = model)
		after_acc1,_ = validate(0,model,0,testloader,criterion,save = False)

		_train(i)

		if args.epochs == 0:
			best_accs.append(round(after_acc1.item(),4))

		logger.info("===> inception_id {} best_acc {:.4f}".format(i,best_accs[-1]))


	#---------------------------------------------save model-----------------------------------------#

	logger.info(best_accs)
	logger.info(compress_rate)

	finally_state_dict = torch.load(args.job_dir + "/pruned_checkpoint/"+args.arch+'_cov9.pt', map_location=device)
	rst_model =  googlenet(compress_rate = compress_rate,oristate_dict = _state_dict(finally_state_dict['state_dict']),ranks = ranks).to(device)
	flops,params = model_size(rst_model,args.input_size,device)
	logger.info(rst_model)

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}
	if not os.path.isdir(args.job_dir + '/finally_pruned_model'):
		os.makedirs(args.job_dir + '/finally_pruned_model')
	torch.save(state,args.job_dir + '/finally_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'/finally_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('finally model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def _train_last(i):
	global best_acc,best_accs
	lr=args.lr*0.1**len(lr_decay_step)
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,weight_decay = args.weight_decay)
	for epoch in range(0, args.epochs):
		logger.info('epoch {} learning_rate {} '.format(epoch,lr))
		train(epoch, model,i,trainloader,optimizer,criterion,mask) #,mask
		validate(epoch,model,i,testloader,criterion)
	best_accs.append(round(best_acc.item(),4))
	best_acc=0.

def _train(i):
	global best_acc,best_accs
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)
	for epoch in range(0, args.epochs):
		logger.info('epoch {} learning_rate {} '.format(epoch,optimizer.param_groups[0]['lr']))
		# print('[')
		# validate(0,model,0,testloader,criterion,save = False)
		# print(']')
		train(epoch, model,i,trainloader,optimizer,criterion,mask) #,mask
		scheduler.step()
		validate(epoch,model,i,testloader,criterion)
		# print(best_acc)
	if args.epochs > 0 and best_acc > 0.:
		best_accs.append(round(best_acc.item(),4))
	else:
		best_accs.append(0.)
	best_acc=0.

def train_resnet_50_from_scratch():

	ranks = []

	pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
	logger.info('loading checkpoint:' + args.resume)
	model = resnet_50().to(device)
	model.load_state_dict(pruned_checkpoint['state_dict'])

	#-----------------------------------------------------------------------------#
	stage_repeat = [3, 4, 6, 3]
	_map = [0]
	cnt = 1
	for x in stage_repeat:
		for y in range(x):
			_map.append(cnt)
			_map.append(cnt+1)
			_map.append(cnt+2)
			if y == 0: _map.append(cnt+2)
			cnt += 3
	#-----------------------------------------------------------------------------#

	utils.logger.info('loading ranks form checkpoint...')
	prefix = "rank_conv/"+args.arch+"/rank_conv"
	subfix = ".npy"
	rank = np.load(prefix + str(_map[0]) + subfix)
	utils.logger.info('loading '+prefix + str(_map[0]) + subfix)
	ranks.append(rank)
	_id = 0
	last_union_rank = None
	for layer_id in range(1,4+1):
		for block_id in range(0,stage_repeat[layer_id-1]):
			for cov_id in range(1,4):
				_id += 1
				rank = np.load(prefix + str(_map[_id]) + subfix)
				utils.logger.info('loading '+prefix + str(_map[_id]) + subfix)
				ranks.append(rank)
				if block_id == 0 and cov_id == 3:
					_id += 1
					rank = np.load(prefix + str(_map[_id]) + subfix)
					utils.logger.info('loading '+prefix + str(_map[_id]) + subfix)
					last_union_rank = rank
					ranks.append(rank)
	print([len(x) for x in ranks])
	#-----------------------------------------------------------------------------#

	validate(0,model,0,testloader,criterion,save = False)

	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	for epoch in range(0, args.epochs):
		logger.info('epoch {} learning_rate {} '.format(epoch,optimizer.param_groups[0]['lr']))
		train(epoch,model,0,trainloader,optimizer,criterion,mask) #,mask
		scheduler.step()
		validate(epoch,model,-1,testloader,criterion)

	rst_model = resnet_50(compress_rate=compress_rate,oristate_dict = _state_dict(model.state_dict()),ranks = ranks).to(device)
	logger.info(rst_model)

	flops,params = model_size(rst_model,args.input_size,device)

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': round(best_acc.item(),4),
		'compress_rate': compress_rate
	}
	if not os.path.isdir(args.job_dir + '/finally_pruned_model'):
		os.makedirs(args.job_dir + '/finally_pruned_model')
	torch.save(state,args.job_dir + '/finally_pruned_model/'+args.arch+'_'+str(args.save_id)+'_fs.pt')
	logger.info('storing pruned_model:'+'/finally_pruned_model/'+args.arch+'_'+str(args.save_id)+'_fs.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('finally model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def train_from_scratch():
	pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
	logger.info('loading checkpoint:' + args.resume)
	# print(pruned_checkpoint.keys())
	compress_rate = pruned_checkpoint['compress_rate']

	# print(compress_rate)

	model = eval(args.arch)(compress_rate=compress_rate).to(device)

	# print(model)
	model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))

	validate(0,model,0,testloader,criterion,save = False)

	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	for epoch in range(0, args.epochs):
		logger.info('epoch {} learning_rate {} '.format(epoch,optimizer.param_groups[0]['lr']))
		train(epoch,model,0,trainloader,optimizer,criterion) #,mask
		scheduler.step()
		validate(epoch,model,-1,testloader,criterion)

	flops,params = model_size(model,args.input_size,device)

	best_model = torch.load(args.job_dir + "/pruned_checkpoint/"+args.arch+'.pt', map_location=device)
	rst_model = eval(args.arch)(compress_rate=compress_rate).to(device)
	rst_model.load_state_dict(_state_dict(best_model['state_dict']))

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': round(best_acc.item(),4),
		'compress_rate': compress_rate
	}
	if not os.path.isdir(args.job_dir + '/finally_pruned_model'):
		os.makedirs(args.job_dir + '/finally_pruned_model')
	torch.save(state,args.job_dir + '/finally_pruned_model/'+args.arch+'_'+str(args.save_id)+'_fs.pt')
	logger.info('storing pruned_model:'+'/finally_pruned_model/'+args.arch+'_'+str(args.save_id)+'_fs.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('finally model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))
	pass

def adjust_filter(conv_name,delegate_G,model = None):
	# print('++ ',conv_name,len(delegate_G))
	logger.info("adjusting filter: "+conv_name)
	if len(delegate_G) < 1: return 
	incs = []
	state_dict_conv = model.state_dict()[conv_name]
	for i,x in enumerate(state_dict_conv):
		n,h,w = x.size()
		inc = []
		for j in range(n):
			_sum = 0.
			for k in delegate_G[j]:
				_sum += x[k] 
			x[j] += _sum
			inc.append(_sum)
		incs.append(inc)
	return incs

def adjust_filter_repeal(conv_name,incs,model = None):
	logger.info("undo: "+conv_name)
	state_dict_conv = model.state_dict()[conv_name]
	for i,x in enumerate(state_dict_conv):
		n,h,w = x.size()
		for j in range(n):
			x[j] -= incs[i][j]
	pass

def adjust_filter_densenet40(conv_name,delegate_Gs,splits):
	logger.info("adjusting filter: "+conv_name)
	state_dict_conv = model.state_dict()[conv_name]
	incs = []
	for i,x in enumerate(state_dict_conv):
		n,h,w = x.size()
		cnt   = 0
		base  = 0 
		inc = []
		for j in range(n):
			_sum = 0.
			if j >= splits[cnt]:
				cnt += 1
				base = j
			if len(delegate_Gs[cnt]) < 1: 
				inc.append(_sum)
				continue
			
			for k in delegate_Gs[cnt][j-base]:
				_sum += x[k+base]
			x[j] += _sum
			inc.append(_sum)
		incs.append(inc)
	return incs

def adjust_filter_densenet40_repeal(conv_name,incs):
	logger.info("undo: "+conv_name)
	state_dict_conv = model.state_dict()[conv_name]
	for i,x in enumerate(state_dict_conv):
		n,h,w = x.size()
		cnt   = 0
		base  = 0 
		for j in range(n):
			x[j] -= incs[i][j]

def adjust_filters(conv_name,delegate_Gs,splits,model = None):
	# print('++ ',conv_name,[len(x) for x in delegate_Gs],splits)
	logger.info("adjusting filter: "+conv_name)
	state_dict_conv = model.state_dict()[conv_name]

	_sum = 0
	_splits = []
	for i,x in enumerate(splits):
		_sum += x
		_splits.append(_sum)
	
	for i,x in enumerate(state_dict_conv):
		n,h,w = x.size()
		cnt   = 0
		base  = 0 
		for j in range(n):
			if j >= _splits[cnt]:
				cnt += 1
				base = j
			# print(j,cnt,base,_splits[cnt])
			if len(delegate_Gs[cnt]) < 1: continue
			for k in delegate_Gs[cnt][j-base]:
				x[j] += x[k+base]

def get_conv_name(arch,i):
	if arch == 'densenet_40':
		dense_id = 1
		block_id = 0
		if i == 0 :
			return 'conv1.weight'
		elif i>=1 and i<=12:
			dense_id = 1
			block_id = i - 1
		elif i == 13:
			return 'trans1.conv1.weight'
		elif i>=14 and i<=25:
			dense_id = 2
			block_id = i - 14
		elif i == 26:
			return 'trans2.conv1.weight'
		elif i>=27 and i<=37:
			dense_id = 3
			block_id = i - 27
		return 'dense'+str(dense_id)+'.'+str(block_id)+'.conv1.weight'
	elif arch == 'resnet_56':
		layer_id = 1
		block_id = 0
		cov_id   = 1
		if i%18 == 0:
			layer_id = int(i/18)
		else :
			layer_id = int(i/18) + 1
		t = i - (layer_id-1)*18-1
		block_id = int(t/2)
		if t%2==0:
			cov_id = 1
		else :
			cov_id = 2
		return 'layer'+str(layer_id)+'.'+str(block_id)+'.conv'+str(cov_id)+'.weight'
	return ''

def _state_dict(state_dict):
	rst = []
	for n, p in state_dict.items():
		if "total_ops" not in n and "total_params" not in n:
			rst.append((n.replace('module.', ''), p))
	rst = dict(rst)
	return rst

if __name__ == '__main__':

	init()

	if args.from_scratch is True:
		# if args.arch == 'resnet_50':
		# 	train_resnet_50_from_scratch()
		# else :
		# 	train_from_scratch()
		train_from_scratch()
	else :
		if args.arch == 'vgg_16_bn':
			iter_vgg(13)
		elif args.arch == 'resnet_56':
			iter_resnet_56()
		elif args.arch == 'resnet_50':
			iter_resnet_50()
		elif args.arch == 'densenet_40':
			iter_densenet_40()
		elif args.arch == 'googlenet':
			iter_googlenet()



