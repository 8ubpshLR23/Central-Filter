import os
import numpy as np
import utils
import math
import random
import torch
import argparse

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
    '--image_num',
    type=int,
    default=1,
    help='The num of images to calc closeness centrality.')
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
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('resnet_50','vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet'),
    help='The architecture to prune')
parser.add_argument(
    '--input_size',
    type=int,
    default=32,
    help='The num of input size')
parser.add_argument(
    '--layer_id',
    type=int,
    default=-1,
    help='The layer to calc')
parser.add_argument(
    '--compress_rate',
    type=str,
    default=None,
    help='compress rate of each conv')
parser.add_argument(
    '--calc_dis_mtx',
    type=bool,
    default=False,
    help='Whether to calculate the distance matrix?')
parser.add_argument(
    '--similarity_analysis',
    type=bool,
    default=False,
    help='similarity_analysis')

args           = None
logger         = None
trainloader    = None
trainloader    = None
device         = None
model          = None
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
compress_rate  = None
model_layers   = None

# for runing on wins
def init():
	global args,logger,trainloader,testloader,device,model,compress_rate,model_layers
	args = parser.parse_args()
	logger = get_logger(os.path.join(args.job_dir, 'log/log2'))
	trainloader,testloader = load_data(data_name = args.dataset, data_dir = args.data_dir, batch_size = args.batch_size, num_workers = args.num_workers)
	device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
	compress_rate = format_compress_rate(args.compress_rate)

	np.seterr(divide='ignore',invalid='ignore')

	# model = eval(args.arch)().to(device)
	if compress_rate is not None:
		model = eval(args.arch)(compress_rate = compress_rate).to(device)
	else :
		model = eval(args.arch)().to(device)

	pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
	logger.info('loading checkpoint:' + args.resume)

	if args.arch == 'resnet_50':
		model.load_state_dict(pruned_checkpoint)
	else :
		model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))

	model_layers = {'vgg_16_bn':13,
					'resnet_56':55,
					'googlenet':1+27,
					'densenet_40':39,
					'resnet_50':50}
	logger.info('args:{}'.format(args))

def _state_dict(state_dict):
	rst = []
	for n, p in state_dict.items():
		if "total_ops" not in n and "total_params" not in n:
			rst.append((n.replace('module.', ''), p))
	rst = dict(rst)
	return rst

#closeness centrality 
def get_closeness_2(Graph,N):
	close = []
	for i in range(N):
		sum = 0.0
		for _,w in Graph[i]:
			# sum += w #
			sum += 1024 - w 
		if sum <= 0: 
			close.append(0)
			# print(i,len(Graph[i]),0,999)
			continue
		# print(i,len(Graph[i]),sum,len(Graph[i])/sum)
		close.append(len(Graph[i])/sum)
	return close

def in_scc(F,rots,x):
	rt = find(F,x)
	if rt in rots:
		return True
	else :
		return False

def del_filter(G,closeness,F,N,cdel = -1,pruned_filters = None):
	oris = set(range(N))
	savs = set()
	dels = set()
	rots = set() 
	ind = np.argsort(closeness)
	cnt = 0 
	delegate_G = [[] for x in range(N)] # 卷积核替代图

	# print('1  ',closeness)
	# print('2  ',ind)

	for i in range(N-1,-1,-1):
		if closeness[ind[i]] > 0:
			if ind[i] not in dels and ind[i] not in savs and not in_scc(F,rots,ind[i]):
				savs.add(ind[i])
				rots.add(find(F,ind[i]))
			for _,w in G[ind[i]]:
				if _ not in dels and _ not in savs: 
					cnt += 1
					if cdel > 0 and cnt > cdel: 
						return list(oris-dels),delegate_G
					dels.add(_)
					if pruned_filters is not None :
						if _ in pruned_filters:
							delegate_G[ind[i]].append(_)
					else :
						delegate_G[ind[i]].append(_)
					
	return list(oris-dels),delegate_G

def del_filter_reverse(closeness,cdel = -1):
	ind = np.argsort(closeness)
	return ind[cdel-1:-1],None

def del_filter_random(closeness,cdel = -1):
	ind = list(range(len(closeness)))
	return random.sample(ind,len(ind)-cdel),None

def find(F,x):
	if F[x] == x: 
		return x
	else:
		F[x] = find(F,F[x])
	return F[x]

def merge(F,x,y):
	if find(F,x) != find(F,y): F[find(F,y)] = find(F,x)

def get_dels(G,N,m):
	dels = set()
	sccs = set()
	F = list(range(N + 1))

	for i in range(N):
		for j,w in enumerate(G[i]):
			if  w > m and i != j:
				dels.add(i)
				dels.add(j)
				merge(F,i,j)
	del_num = len(dels)
	for i in dels:
		sccs.add(find(F,i))
	scc_num = len(sccs)
	return del_num,scc_num,F

def get_threshold(G,ratio,N):
	X = math.ceil(N * ratio) # 要删去的个数
	eps = 1e-7
	l = np.array(G).min()-eps
	r = np.array(G).max()+eps
	rst = r #
	F = None

	while l<r:
		m = (l+r)/2
		del_num,scc_num,F = get_dels(G,N,m)
		if del_num - scc_num < X:
			r = m-eps
		elif del_num - scc_num > X:
			rst = l
			l = m+eps
		else :
			return m,del_num,scc_num,F
	del_num,scc_num,F = get_dels(G,N,rst)
	return rst,del_num,scc_num,F

def get_rank():
	layers = model_layers[args.arch]
	if args.calc_dis_mtx: 
		if args.layer_id != -1:
			get_distance_matrix_one(layers,args.layer_id)
		else :
			get_distance_matrix(layers)

def ger_rank_one(model,arch,layer_id,ratio,trainloader,device,calc_dis_mtx = False,save_id = None):

	if calc_dis_mtx: get_distance_matrix_one(model,arch,layer_id,trainloader,device)
	if save_id is None : save_id = layer_id

	D =  np.load('rank_conv/'+arch+'/dis_mtx'+str(layer_id)+'.npy')
	utils.logger.info('loading rank_conv/'+arch+'/dis_mtx' + str(layer_id) + '.npy')
	num = len(D)
	G = []
	delegate_G = []

	# print(D[13])

	if ratio > 0.:
		threshold,del_num,scc_num,F = get_threshold(D,ratio,num)
		for i in range(num):
			edges = []
			for j,w in enumerate(D[i]):
				if w > threshold and i != j:
					edges.append((j,w))
			G.append(edges)
		closeness = get_closeness_2(G,num)
		# closeness = get_degree(G,num)
		rank,delegate_G = del_filter(G,closeness,F,num,cdel = math.ceil(num*ratio)) # 
	else :
		rank = list(range(num))
		delegate_G = []

	if not os.path.isdir('rank_conv/' + arch):
		os.makedirs('rank_conv/' + arch)

	np.save('rank_conv/'+arch+'/rank_conv' + str(save_id) + '.npy', rank)
	utils.logger.info('storing rank_conv/'+arch+'/rank_conv' + str(save_id) + '.npy')

	return rank,delegate_G

def ger_rank_one_ablation_study(model,arch,layer_id,ratio,trainloader,device,type = 0,calc_dis_mtx = False,save_id = None):

	if calc_dis_mtx: get_distance_matrix_one(model,arch,layer_id,trainloader,device)
	if save_id is None : save_id = layer_id

	D =  np.load('rank_conv/'+arch+'/dis_mtx'+str(layer_id)+'.npy')
	utils.logger.info('loading rank_conv/'+arch+'/dis_mtx' + str(layer_id) + '.npy')
	num = len(D)
	G = []
	delegate_G = []


	if ratio > 0.:
		threshold,del_num,scc_num,F = get_threshold(D,ratio,num)
		for i in range(num):
			edges = []
			for j,w in enumerate(D[i]):
				if w > threshold and i != j:
					edges.append((j,w))
			G.append(edges)
		closeness = get_closeness_2(G,num)
		if type == 0 or type == 3:
			rank,delegate_G = del_filter(G,closeness,F,num,cdel = math.ceil(num*ratio)) # 
		elif type == 1:
			rank,delegate_G = del_filter_reverse(closeness,cdel = math.ceil(num*ratio))
		elif type == 2:
			rank,delegate_G = del_filter_random(closeness,cdel = math.ceil(num*ratio))
	else :
		rank = list(range(num))
		delegate_G = []

	if not os.path.isdir('rank_conv/' + arch):
		os.makedirs('rank_conv/' + arch)

	np.save('rank_conv/'+arch+'/rank_conv' + str(save_id) + '.npy', rank)
	utils.logger.info('storing rank_conv/'+arch+'/rank_conv' + str(save_id) + '.npy')

	return rank,delegate_G

def get_delegate_from_rank(model,arch,layer_id,ratio,trainloader,device,rank,save_id = None):

	if save_id is None : save_id = layer_id

	D =  np.load('rank_conv/'+arch+'/dis_mtx'+str(layer_id)+'.npy')
	utils.logger.info('loading rank_conv/'+arch+'/dis_mtx' + str(layer_id) + '.npy')
	num = len(D)
	G = []
	delegate_G = []

	pruned_filters = set(list(range(num))) - set(rank)

	if ratio > 0.:
		threshold,del_num,scc_num,F = get_threshold(D,ratio,num)
		for i in range(num):
			edges = []
			for j,w in enumerate(D[i]):
				if w > threshold and i != j:
					edges.append((j,w))
			G.append(edges)
		closeness = get_closeness_2(G,num)
		_,delegate_G = del_filter(G,closeness,F,num,cdel = math.ceil(num*ratio),pruned_filters=pruned_filters) # 
	else :
		_ = list(range(num))

	np.save('rank_conv/'+arch+'/rank_conv' + str(save_id) + '.npy', rank)
	utils.logger.info('storing rank_conv/'+arch+'/rank_conv' + str(save_id) + '.npy')
	
	return delegate_G

def inference():
    total = 0
    limit = args.image_num

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            if i >= limit:
               break

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            # total += outputs.size(0)

def pearson(image1, image2):
    X = np.vstack([image1, image2])
    return np.corrcoef(X)[0][1]

def get_distance_matrix(layers = -1):
	assert layers > 0
	conv_outs = []

	class LayerActivations:
		features = None
		def __init__(self, feature, start = -1, size = -1):
			self.hook = feature.register_forward_hook(self.hook_fn)
			self.total = 0
			self.distance_matrix = None
			self.start = start
			self.size = size
		def hook_fn(self, module, input, output):
			a = output.shape[0]
			b = output.shape[1]
			self.total += a
			f,e = 0,b
			if self.start != -1 :
				f,e = self.start,self.start+self.size
			else :
				self.start = 0

			D = [[] for x in range(e-f)]


			for i in range(f,e):
				for j in range(f,e):
					if i == j : 
						D[i-self.start].append(1.*a)
						continue
					_sum = 0.
					for k in range(a):
						h,w = output[k,i,:,:].detach().size()
						x = output[k,i,:,:].detach().view(-1, h*w)
						y = output[k,j,:,:].detach().view(-1, h*w)

						if (len(set(x[0].tolist())) == 1 and len(set(y[0].tolist())) == 1) or (x.sum() == 0. and y.sum() == 0.):
							_sum += 1
						elif x.sum() == 0. or y.sum() == 0.:
							_sum += -1
						else :
							_sum += pearson(x.cpu(),y.cpu())
					D[i-self.start].append(_sum)
			if self.distance_matrix is None:
				self.distance_matrix = D
			else :
				self.distance_matrix = list(np.add(self.distance_matrix,D))
			self.features = output.cpu()
			
		def remove(self):
			self.hook.remove()

	if args.arch  == 'vgg_16_bn':
		# cfg = [0,3,7,10,14,17,20,24,27,30,34,37,40]
		cfg = [0,1,3,4,6,7,8,10,11,12,14,15,16]
		for i in cfg:#12,13,1
			conv_outs.append(eval('model.features.relu'+str(i)))
	elif args.arch == 'resnet_56':
		conv_outs.append(model.relu) #conv1
		for i in range(1,4):
			name = 'model.layer' + str(i)
			for _,x in enumerate(eval(name)) :
				conv_outs.append(x.relu1)
				conv_outs.append(x.relu2)
	elif args.arch == 'resnet_50':
		# conv_outs.append(model.relu)
		conv_outs.append(model.maxpool)
		for i in range(1,5):
			name = 'model.layer' + str(i)
			for _,x in enumerate(eval(name)) :
				conv_outs.append(x.relu1)
				conv_outs.append(x.relu2)
				conv_outs.append(x.relu3)
		pass
	elif args.arch == 'densenet_40':
		conv_outs.append(model.conv1)

		for i in range(1,4):
			name = 'model.dense' + str(i)
			for _,x in enumerate(eval(name)):
				conv_outs.append(x.conv1)
			if i <= 2:
				conv_outs.append(eval('model.trans'+str(i)+'.conv1'))
	
	elif args.arch == 'googlenet':
		cfg = ['a3','b3','a4','b4','c4','d4','e4','a5','b5']
		conv_outs.append(model.pre_layers[2])
		for i,x in enumerate(cfg):
			name = 'model.inception_' + x
			conv_outs.append(eval(name + '.branch3x3')[5])
			conv_outs.append(eval(name + '.branch5x5')[5])
			conv_outs.append(eval(name + '.branch5x5')[8])

	if not os.path.isdir('rank_conv/' + args.arch):
		os.makedirs('rank_conv/' + args.arch)

	feature_num = len(conv_outs)
	
	for i,feature in enumerate(conv_outs): 
		print('--->',i)
		if args.arch == 'densenet_40':
			conv_out = LayerActivations(feature)
		else :
			conv_out = LayerActivations(feature)
		model.eval()
		inference()
		conv_out.remove()
		# conv_out2.remove()
		
		np.save('rank_conv/'+args.arch+'/dis_mtx' + str(i) + '.npy', torch.tensor(conv_out.distance_matrix)/conv_out.total)
		utils.logger.info('storing rank_conv/'+args.arch+'/dis_mtx' + str(i) + '.npy')

def get_distance_matrix_one(layers = -1,layer_id = -1):
	assert layers > 0
	assert args.layer_id != -1
	conv_outs = []

	class LayerActivations:
		features = None
		def __init__(self, feature, start = -1, size = -1):
			self.hook = feature.register_forward_hook(self.hook_fn)
			self.total = 0
			self.distance_matrix = None
			self.start = start
			self.size = size
		def hook_fn(self, module, input, output):
			a = output.shape[0]
			b = output.shape[1]
			self.total += a
			f,e = 0,b
			if self.start != -1 :
				f,e = self.start,self.start+self.size
			else :
				self.start = 0

			D = [[] for x in range(e-f)]

			# print(a,b,f,e)

			for i in range(f,e):
				print('----->',i)
				for j in range(f,e):
					if i == j : 
						D[i-self.start].append(1.*a)
						continue
					_sum = 0.
					for k in range(a):
						# .detach()  not for gradient calc, save memory 
						# torch.cuda.empty_cache()
						h,w = output[k,i,:,:].detach().size()
						x = output[k,i,:,:].detach().view(-1, h*w)
						y = output[k,j,:,:].detach().view(-1, h*w)
						# if x.sum() == 0 or y.sum() == 0 or len(set(x)) == 1 or len(set(y)) == 1:
						# 	_sum += -1
						# if k == 0 :
						# 	print('---> ',i,j,k,x.sum(),y.sum(),len(set(x[0].tolist())),len(set(y[0].tolist())))
						if (len(set(x[0].tolist())) == 1 and len(set(y[0].tolist())) == 1) or (x.sum() == 0. and y.sum() == 0.):
							_sum += 1
						elif x.sum() == 0. or y.sum() == 0.:
							_sum += -1
						else :
							_sum += pearson(x.cpu(),y.cpu())
					D[i-self.start].append(_sum)
			# print([len(x) for x in D])
			if self.distance_matrix is None:
				self.distance_matrix = D
			else :
				self.distance_matrix = list(np.add(self.distance_matrix,D))
			self.features = output.cpu()
			
		def remove(self):
			self.hook.remove()

	if args.arch   == 'vgg_16_bn':
		pass
	elif args.arch == 'resnet_56':
		pass
	elif args.arch == 'resnet_50':
		# conv_outs.append(model.relu)
		if layer_id == 0:
			conv_outs.append(model.maxpool)
		else :
			cnt = 0
			flag = True
			for i in range(1,5):
				name = 'model.layer' + str(i)
				# print(name)
				# print(cnt,layer_id)
				if flag is False: break
				for _,x in enumerate(eval(name)) :
					# print(_,x)
					cnt += 1
					if cnt == layer_id : 
						conv_outs.append(x.relu1)
						flag = False
						break
					cnt += 1
					if cnt == layer_id : 
						conv_outs.append(x.relu2)
						flag = False
						break
					cnt += 1
					if cnt == layer_id : 
						# print(233333)
						conv_outs.append(x.relu3)
						flag = False
						break
		pass
	elif args.arch == 'densenet_40':
		pass
	elif args.arch == 'googlenet':
		pass

	print(conv_outs)
	feature_num = len(conv_outs)
	for i,feature in enumerate(conv_outs): 
		print('--->',i,feature)
		if args.arch == 'densenet_40':
			conv_out = LayerActivations(feature,_index[i][0],_index[i][1])
		else :
			conv_out = LayerActivations(feature)
		model.eval()
		inference()
		conv_out.remove()
		if not os.path.isdir('rank_conv/' + args.arch):
			os.makedirs('rank_conv/' + args.arch)
		np.save('rank_conv/'+args.arch+'/dis_mtx' + str(layer_id) + '.npy', torch.tensor(conv_out.distance_matrix)/conv_out.total)
		utils.logger.info('storing rank_conv/'+args.arch+'/dis_mtx' + str(layer_id) + '.npy')

if __name__ == '__main__':

	init()
	get_rank()

