import logging

from models.simple import Discriminator
from utils.utils import th

logger = logging.getLogger('logger')
from torch.nn.functional import log_softmax
from shutil import copyfile

import math
import torch
import random
import numpy as np
import os
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch import autograd



class Helper:
    def __init__(self, current_time, params, name):
        self.current_time = current_time
        self.target_model = None
        self.local_model = None
        self.dataset_size = 0
        self.train_dataset = None
        self.test_dataset = None
        self.poisoned_data = None
        self.test_data_poison = None
        self.writer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.params = params
        self.name = name
        self.best_loss = math.inf
        self.folder_path = f'saved_models/model_{self.name}_{current_time}'

        self.lr = self.params.get('lr', None)
        self.decay = self.params.get('decay', None)
        self.momentum = self.params.get('momentum', None)
        self.epochs = self.params.get('epochs', None)
        self.is_save = self.params.get('save_model', False)
        self.log_interval = self.params.get('log_interval', 1000)
        self.batch_size = self.params.get('batch_size', None)
        self.test_batch_size = self.params.get('test_batch_size', None)
        self.optimizer = self.params.get('optimizer', None)
        self.scheduler = self.params.get('scheduler', False)
        self.resumed_model = self.params.get('resumed_model', False)

        self.poisoning_proportion = self.params.get('poisoning_proportion', False)
        self.backdoor = self.params.get('backdoor', False)
        self.poison_number = self.params.get('poison_number', 8)
        self.log = self.params.get('log', True)
        self.tb = self.params.get('tb', True)
        self.random = self.params.get('random', True)
        self.alpha = self.params.get('alpha', 1)

        self.data = self.params.get('data', 'cifar')
        self.scale_threshold = self.params.get('scale_threshold', 1)
        self.normalize = self.params.get('normalize', 'none')

        self.losses = self.params.get('losses', 'normal')

        self.start_epoch = 1
        self.fixed_model = None
        self.ALL_TASKS =  ['backdoor', 'normal', 'latent_fixed', 'latent', 'ewc', 'nc_adv',
                           'nc', 'mask_norm']
        self.defense = self.params.get('defense', False)

        self.poison_images = self.params.get('poison_images', False)
        self.poison_images_test = self.params.get('poison_images_test', False)
        self.transform_train = self.params.get('transform_train', False)

        if self.log:
            try:
                os.mkdir(self.folder_path)
            except FileExistsError:
                logger.info('Folder already exists')
        else:
            self.folder_path = None

        self.gan = self.params.get('gan', False)
        if self.gan:
            self.discriminator = Discriminator()
            self.discriminator = self.discriminator.to(self.device)
            self.discriminator_optim = optim.SGD(self.discriminator.parameters(), lr=self.lr,
                                                        weight_decay=self.decay, momentum=self.momentum)

        self.nc = True if 'nc_adv' in self.params.get('losses', list()) else False
        self.mixed = None
        self.mixed_optim = None

        # if not self.params.get('environment_name', False):
        #     self.params['environment_name'] = self.name

        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path

    def save_model(self, model=None, epoch=0, val_loss=0):

        if self.params['save_model'] and self.log:
            # save_model
            logger.info("saving model")
            model_name = '{0}/model_last.pt.tar'.format(self.params['folder_path'])
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,
                          'lr': self.params['lr']}
            self.save_checkpoint(saved_dict, False, model_name)
            if epoch in self.params.get('save_on_epochs', []):
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params['save_model']:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    @staticmethod
    def norm(parameters, max_norm):
        total_norm = 0
        for p in parameters:
            torch.sum(torch.pow(p))
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in parameters:
            p.grad.data.mul_(clip_coef)

    # def compute_rdp(self):
    #     from compute_dp_sgd_privacy import apply_dp_sgd_analysis
    #
    #     N = self.dataset_size
    #     logger.info(f'Dataset size: {N}. Computing RDP guarantees.')
    #     q = self.params['batch_size'] / N  # q - the sampling ratio.
    #
    #     orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
    #               list(range(5, 64)) + [128, 256, 512])
    #
    #     steps = int(math.ceil(self.params['epochs'] * N / self.params['batch_size']))
    #
    #     apply_dp_sgd_analysis(q, self.params['z'], steps, orders, 1e-6)

    @staticmethod
    def clip_grad(parameters, max_norm, norm_type=2):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
        return total_norm

    @staticmethod
    def clip_grad_scale_by_layer_norm(parameters, max_norm, norm_type=2):
        parameters = list(filter(lambda p: p.grad is not None, parameters))

        total_norm_weight = 0
        norm_weight = dict()
        for i, p in enumerate(parameters):
            param_norm = p.data.norm(norm_type)
            norm_weight[i] = param_norm.item()
            total_norm_weight += param_norm.item() ** norm_type
        total_norm_weight = total_norm_weight ** (1. / norm_type)

        total_norm = 0
        norm_grad = dict()
        for i, p in enumerate(parameters):
            param_norm = p.grad.data.norm(norm_type)
            norm_grad[i] = param_norm.item()
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for i, p in enumerate(parameters):
                if norm_grad[i] < 1e-3:
                    continue
                scale = norm_weight[i] / total_norm_weight
                p.grad.data.mul_(math.sqrt(max_norm) * scale / norm_grad[i])
        # print(total_norm)
        # total_norm = 0
        # norm_grad = dict()
        # for i, p in enumerate(parameters):
        #     param_norm = p.grad.data.norm (norm_type)
        #     norm_grad[i] = param_norm
        #     total_norm += param_norm.item() ** norm_type
        # total_norm = total_norm ** (1. / norm_type)
        # print(total_norm)
        return total_norm



    def get_grad_vec(self, model):
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            size += layer.view(-1).shape[0]
        if self.device.type == 'cpu':
            sum_var = torch.FloatTensor(size).fill_(0)
        else:
            sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            sum_var[size:size + layer.view(-1).shape[0]] = (layer.grad).view(-1)
            size += layer.view(-1).shape[0]

        return sum_var

    def get_optimizer(self, model):
        if self.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=self.lr,
                                  weight_decay=self.decay, momentum=self.momentum)
        elif self.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.decay)
        else:
            raise ValueError(f'No optimizer: {self.optimizer}')

        return optimizer

    def check_resume_training(self, model, lr=False):
        from models.resnet import ResNet18

        if self.resumed_model:
            logger.info('Resuming training...')
            loaded_params = torch.load(f"saved_models/{self.resumed_model}")
            model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']
            if lr:
                self.lr = loaded_params.get('lr', self.lr)

            self.fixed_model = ResNet18()
            self.fixed_model.to(self.device)
            self.fixed_model.load_state_dict(loaded_params['state_dict'])

            logger.warning(f"Loaded parameters from saved model: LR is"
                        f" {self.lr} and current epoch is {self.start_epoch}")

    def flush_writer(self):
        if self.writer:
            self.writer.flush()

    def plot(self, x, y, name):
        if self.writer is not None:
            self.writer.add_scalar(tag=name, scalar_value=y, global_step=x)
            self.flush_writer()
        else:
            return False

    @staticmethod
    def fix_random(seed=0):
        # logger.warning('Setting random seed for reproducible results.')
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        return True

    @staticmethod
    def copy_grad(model: nn.Module):
        grads = list()
        for name, params in model.named_parameters():
            if not params.requires_grad:
                a=1
                # print(name)
            else:
                grads.append(params.grad.clone().detach())
        model.zero_grad()
        return grads

    @staticmethod
    def combine_grads(model: nn.Module, grads_dict, scales_dict, tasks):
        for t in tasks:
            for i, params in enumerate(model.parameters()):
                params.grad += scales_dict[t]*grads_dict[t][i]

        return True

    def compute_normal_loss(self, model, criterion, inputs, labels, grads=True, **kwargs):
        outputs, outputs_latent = model(inputs)
        loss = criterion(outputs, labels).mean()
        if grads:
            loss.backward()
            grads = self.copy_grad(model)

        return loss, grads

    def compute_backdoor_loss(self, model, criterion, inputs_back, normal_labels, bck_labels, grads=True):
        outputs, outputs_latent = model(inputs_back)
        loss = criterion(outputs, bck_labels).mean()
        # loss = torch.topk(loss[bck_labels != normal_labels], 3, largest=False)[0]
        # loss = loss.sum()/normal_labels.shape[0]

        if grads:
            loss.backward()
            grads = self.copy_grad(model)

        return loss, grads

    def compute_latent_cosine_similarity(self, model, fixed_model, inputs, grads=True, **kwargs):
        with torch.no_grad():
            _, fixed_latent = fixed_model(inputs)
        _, latent = model(inputs)
        loss = -torch.cosine_similarity(latent, fixed_latent).mean() + 1
        if grads:
            loss.backward()
            grads = self.copy_grad(model)

        return loss, grads

    def compute_latent_fixed_loss(self, model, fixed_model, inputs, grads=True, **kwargs):
        if not fixed_model:
            return torch.tensor(0.0), 0.0
        with torch.no_grad():
            _, fixed_latent = fixed_model(inputs)
        _, latent = model(inputs)
        loss = torch.norm(latent-fixed_latent, dim=1).mean()
        if grads:
            loss.backward()
            grads = self.copy_grad(model)

        return loss, grads

    def compute_latent_loss(self, model, inputs, inputs_back, grads=True, **kwargs):
        _, latent = model(inputs)
        _, latent_bck = model(inputs_back)
        loss = torch.norm(latent-latent_bck)
        if grads:
            loss.backward()
            grads = self.copy_grad(model)

        return loss, grads

    def compute_losses(self, tasks, model, criterion, inputs, inputs_back,
                       labels, labels_back, fixed_model, compute_grad=True):
        grads = {}
        loss_data = {}
        # if not compute_grad:
        #     tasks = ['normal', 'backdoor', 'latent_fixed', 'latent']
        for t in tasks:

            if t == 'normal':
                loss_data[t], grads[t] = self.compute_normal_loss(model, criterion, inputs, labels, grads=compute_grad)
            elif t == 'backdoor':
                loss_data[t], grads[t] = self.compute_backdoor_loss(model, criterion, inputs_back, labels, labels_back,
                                                                    grads=compute_grad)
            elif t == 'latent_fixed':
                loss_data[t], grads[t] = self.compute_latent_fixed_loss(model, fixed_model, inputs, grads=compute_grad)
            elif t == 'latent':
                loss_data[t], grads[t] = self.compute_latent_loss(model, inputs, inputs_back, grads=compute_grad)
            elif t == 'ewc':
                loss_data[t], grads[t] = self.ewc_loss(model, grads=compute_grad)
            elif t == 'mask_norm':
                loss_data[t], grads[t] = self.norm_loss(model, grads=compute_grad)
            elif t == 'nc':
                loss_data[t], grads[t] = self.compute_normal_loss(model,  criterion, inputs, labels_back,
                                                                  grads=compute_grad)
            # elif t == 'nc_adv':
            #     loss_data[t], grads[t] = self.compute_normal_loss(model,  criterion, inputs, labels,
            #                                                       grads=compute_grad)

        return loss_data, grads


    def norm_loss(self, model, grads=False, p=1):
        if p==1:
            norm = torch.sum(th(model.mask))
        else:
            norm = torch.norm(th(model.mask))

        if grads:
            norm.backward()
            grads = self.copy_grad(model)

        return norm, grads



    # cos = torch.cosine_similarity(normal_grad, bck_grad, dim=0)
    # norm_norm = torch.norm(normal_grad)
    # bck_norm = torch.norm(bck_grad)
    # print(cos.item())

    def estimate_fisher(self, model, data_loader, sample_size):
        # sample loglikelihoods from the dataset.
        loglikelihoods = []
        for x, y in data_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            loglikelihoods.append(
                F.log_softmax(model(x)[0], dim=1)[range(self.batch_size), y]
            )
            if len(loglikelihoods) >= sample_size // self.batch_size:
                break
        # estimate the fisher information of the parameters.
        loglikelihoods = torch.cat(loglikelihoods).unbind()
        loglikelihood_grads = zip(*[autograd.grad(
            l, model.parameters(),
            retain_graph=(i < len(loglikelihoods))
        ) for i, l in enumerate(loglikelihoods, 1)])
        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
        param_names = [
            n.replace('.', '__') for n, p in model.named_parameters()
        ]
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, model, fisher):
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            model.register_buffer('{}_mean'.format(n), p.data.clone())
            model.register_buffer('{}_fisher'
                                 .format(n), fisher[n].data.clone())

    def ewc_loss(self, model, grads=True):
        try:
            losses = []
            for n, p in model.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(model, '{}_mean'.format(n))
                fisher = getattr(model, '{}_fisher'.format(n))
                # wrap mean and fisher in variables.
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p - mean) ** 2).sum())
            loss = (model.lamda / 2) * sum(losses)
            if grads:
                loss.backward()
                grads = self.copy_grad(model)
                return loss, grads
            else:
                return loss, grads

        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            print('exception')
            return torch.zeros(1).to(self.device), grads