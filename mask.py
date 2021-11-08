import torch
import numpy as np
import pickle
import utils

class mask_vgg_16_bn:
    def __init__(self, model=None, job_dir='',device=None):
        self.model = model
        # self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=4, arch="vgg_16_bn"):
        params = self.model.parameters()
        prefix = "rank_conv/"+arch+"/rank_conv"
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            # resume=self.job_dir+'/mask'
            resume='mask/mask_vgg_16_bn'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == (cov_id + 1) * param_per_cov:
                break
            if index == cov_id * param_per_cov:
                f, c, w, h = item.size()
                rank = np.load(prefix + str(cov_id) + subfix)
                utils.logger.info('loading '+prefix + str(cov_id) + subfix)

                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in rank:
                    zeros[i, 0, 0, 0] = 1.
                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]

            if index > cov_id * param_per_cov and index < (cov_id + 1) * param_per_cov :
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        mask_keys = self.mask.keys()
        for index, item in enumerate(params):
            # if index == (cov_id + 1) * self.param_per_cov:
            #     break
            if index < cov_id * self.param_per_cov:
                continue
            # if index%4==0:
            #     print('--->',cov_id,index, len(item))
            if index in mask_keys:
                # print('==',index)
                item.data = item.data * self.mask[index]#prune certain weight

class mask_resnet_56:
    def __init__(self, model=None, job_dir='',device=None):
        self.model = model
        # self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=3,  arch="resnet_56"):
        params = self.model.parameters()
        prefix = "rank_conv/"+arch+"/rank_conv"
        subfix = ".npy"


        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume='mask/mask_resnet_56'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == (cov_id + 1)*param_per_cov:
                break

            if index == cov_id * param_per_cov:
                f, c, w, h = item.size()
                rank = np.load(prefix + str(cov_id) + subfix)
                utils.logger.info('loading '+prefix + str(cov_id) + subfix)
                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in rank:
                    zeros[i, 0, 0, 0] = 1.
                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]

            elif index > cov_id*param_per_cov and index < (cov_id + 1)*param_per_cov:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index].to(self.device)

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        if len(self.mask.keys()) < 1 : return 
        params = self.model.parameters()
        mask_keys = self.mask.keys()
        for index, item in enumerate(params):
            if index == (cov_id + 1)*self.param_per_cov:
                break
            # if index < cov_id * self.param_per_cov:
            #     continue
            if index in mask_keys:
                item.data = item.data * self.mask[index].to(self.device)#prune certain weight

class mask_densenet_40:
    def __init__(self, model=None, job_dir='',device=None):
        self.model = model
        self.job_dir=job_dir
        self.device=device
        self.mask = {}

    def layer_mask(self, cov_id, resume=None, param_per_cov=3,  arch="densenet_40"):
        params = self.model.parameters()
        prefix = "rank_conv/"+arch+"/rank_conv"
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume='mask/mask_densenet_40'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == (cov_id + 1) * param_per_cov:
                break
            if index == cov_id  * param_per_cov:
                f, c, w, h = item.size()
                rank = np.load(prefix + str(cov_id) + subfix)
                utils.logger.info('loading '+prefix + str(cov_id) + subfix)

                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in rank:
                    zeros[i, 0, 0, 0] = 1.
                self.mask[index] = zeros  # covolutional weight
                # print(rank)
                # print(torch.squeeze(self.mask[index]))
                item.data = item.data * self.mask[index]

            # prune BN's parameter
            if index > cov_id * param_per_cov and index < (cov_id + 1) * param_per_cov :
                if cov_id>=1 and cov_id!=13 and cov_id!=26:
                    self.mask[index] = torch.cat([self.mask[index-param_per_cov], torch.squeeze(zeros)], 0).to(self.device)
                else:
                    self.mask[index] = torch.squeeze(zeros).to(self.device)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        if len(self.mask.keys()) < 1 : return 
        params = self.model.parameters()
        mask_keys = self.mask.keys()
        for index, item in enumerate(params):
            if index == (cov_id + 1)*self.param_per_cov:
                break
            if index in mask_keys:
                item.data = item.data * self.mask[index].to(self.device)#prune certain weight

class mask_googlenet:
    def __init__(self, model=None, job_dir='',device=None):
        self.model = model
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    # per inception
    def layer_mask(self, inception_id, resume=None, param_per_cov=28,  arch="googlenet"):
        params = self.model.parameters()
        prefix = "rank_conv/"+arch+"/rank_conv"
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume='mask/mask_googlenet'

        self.param_per_cov=param_per_cov


        for index, item in enumerate(params):

            if index == inception_id * param_per_cov + 4:
                break
            if (index == 0 and inception_id == 0) \
                or index == (inception_id - 1) * param_per_cov + 4 + 8 \
                or index == (inception_id - 1) * param_per_cov + 4 + 16 or \
                index == (inception_id - 1) * param_per_cov + 4 + 20 :
                if index == 0 and inception_id == 0:
                    rank = np.load(prefix + str(inception_id)+ subfix)
                    utils.logger.info('loading '+prefix + str(inception_id) + subfix)
                elif index == (inception_id - 1) * param_per_cov + 4 + 8:
                    rank = np.load(prefix + str(inception_id)+'_'+'3' + subfix)
                    utils.logger.info('loading '+prefix + str(inception_id)+'_'+'3' + subfix)
                elif index == (inception_id - 1) * param_per_cov + 4 + 16:
                    rank = np.load(prefix + str(inception_id)+'_'+'5_1' + subfix)
                    utils.logger.info('loading '+prefix + str(inception_id)+'_'+'5_1' + subfix)
                elif index == (inception_id - 1) * param_per_cov + 4 + 20:
                    rank = np.load(prefix + str(inception_id)+'_'+'5_2' + subfix)
                    utils.logger.info('loading '+prefix + str(inception_id)+'_'+'5_2' + subfix)

                f, c, w, h = item.size()
                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in rank:
                    zeros[i, 0, 0, 0] = 1.
                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]

            elif (index >= 1 and index <= 3 and inception_id == 0) \
                or (index > (inception_id - 1) * param_per_cov + 4 + 8 and index < (inception_id - 1) * param_per_cov + 4 + 8 + 4) \
                or (index > (inception_id - 1) * param_per_cov + 4 + 16 and index < (inception_id - 1) * param_per_cov + 4 + 16 + 4)\
                or (index > (inception_id - 1) * param_per_cov + 4 + 20 and index < (inception_id - 1) * param_per_cov + 4 + 20 + 4) :
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index]
            else :
                continue

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, inception_id):
        if len(self.mask.keys()) < 1 : return 
        params = self.model.parameters()
        mask_keys = self.mask.keys()
        for index, item in enumerate(params):
            if index == inception_id * self.param_per_cov + 4: 
                break
            if index in mask_keys:
                item.data = item.data * self.mask[index].to(self.device)#prune certain weight

class mask_resnet_50:
    def __init__(self, model=None, job_dir='',device=None):
        self.model = model
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=3, arch="resnet_50"):
        params = self.model.parameters()
        prefix = "rank_conv/"+arch+"/rank_conv"
        subfix = ".npy"
        downsample = [4,14,27,46]
        _map   = [0,1,2,3,3,4,5,6,7,8,9,10,11,12,12,13,14,15,16,17,18,19,20,21, 
                22,23,24,24, 25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,42,
                43,44,45,46,47,48]

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume='mask/mask_resnet_50'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == (cov_id + 1) * param_per_cov:
                break

            if index == cov_id * param_per_cov:
                f, c, w, h = item.size()
                rank = np.load(prefix + str(_map[cov_id]) + subfix)
                utils.logger.info('loading '+prefix + str(_map[cov_id]) + subfix)
                zeros = torch.zeros(f, 1, 1, 1).to(self.device) #.cuda(self.device[0])#.to(self.device)
                for i in rank:
                    zeros[i, 0, 0, 0] = 1.
                self.mask[index] = zeros  # covolutional weight
                
                item.data = item.data * self.mask[index]

            elif index > cov_id * param_per_cov and index < (cov_id + 1)* param_per_cov:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        if len(self.mask.keys()) < 1 : return 
        params = self.model.parameters()
        mask_keys = self.mask.keys()
        for index, item in enumerate(params):
            if index == (cov_id + 1)*self.param_per_cov:
                break
            # if index < cov_id * self.param_per_cov:
            #     continue
            if index in mask_keys:
                item.data = item.data * self.mask[index].to(self.device)#prune certain weight