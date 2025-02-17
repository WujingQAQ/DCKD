import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from tensorboardX import SummaryWriter
from yacs.config import CfgNode as CN
from ._base import Distiller

# Configuration
CFG = CN()
CFG.DCKD = CN()
CFG.DCKD.TEMPERATURE = 4.0
CFG.DCKD.INIT_SIGMA = 1.0
CFG.DCKD.MIN_SIGMA = 0.1
CFG.DCKD.LAMBDA_MIN = 0.0
CFG.DCKD.LAMBDA_MAX = 1.0
CFG.DCKD.K = 0.1
CFG.DCKD.X0 = 50
CFG.DCKD.MOMENTUM = 0.9
CFG.DCKD.MOMENTUM_KD = 0.9
CFG.DCKD.DELTA = 0.075

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DistillationOrientedTrainer(torch.optim.Optimizer):
    """Distillation-Oriented Trainer implementation"""
    def __init__(self, params, lr=1e-2, momentum=0.9, momentum_kd=0.9, dampening=0, weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if momentum_kd < 0.0:
            raise ValueError(f"Invalid momentum kd value: {momentum_kd}")
        
        defaults = dict(lr=lr, momentum=momentum, momentum_kd=momentum_kd,
                       dampening=dampening, weight_decay=weight_decay)
        self.kd_grad_buffer = []
        self.kd_grad_params = []
        self.kd_momentum_buffer = []
        super(DistillationOrientedTrainer, self).__init__(params, defaults)

    @torch.no_grad()
    def step_kd(self, closure=None):
        """Performs a single optimization step for KD."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_kd_buffer_list = []
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if 'momentum_kd_buffer' not in state:
                        momentum_kd_buffer_list.append(None)
                    else:
                        momentum_kd_buffer_list.append(state['momentum_kd_buffer'])
                    
        self.kd_momentum_buffer = momentum_kd_buffer_list
        self.kd_grad_buffer = d_p_list
        self.kd_grad_params = params_with_grad
        return loss

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            momentum_kd = group['momentum_kd']
            dampening = group['dampening']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
            
            # Update parameters
            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                if weight_decay != 0:
                    d_p = d_p.add(param, alpha=weight_decay)
                
                buf = momentum_buffer_list[i]
                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                d_p = buf
                param.add_(d_p, alpha=-lr)

            # Update momentum buffers
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

class DCKD(Distiller):
    """Dynamic Curriculum Knowledge Distillation with Adaptive Weight Strategy
    
    Implements both adaptive weight strategy and curriculum learning based on 
    homoscedastic uncertainty for knowledge distillation.
    """
    
    def __init__(self, student, teacher, cfg):
        super(DCKD, self).__init__(student, teacher)
        self.temperature = cfg.DCKD.TEMPERATURE
        self.init_sigma = cfg.DCKD.INIT_SIGMA
        self.min_sigma = cfg.DCKD.MIN_SIGMA
        
        # Initialize uncertainty parameters
        self.log_sigma1 = nn.Parameter(torch.ones(1) * np.log(self.init_sigma))
        self.log_sigma2 = nn.Parameter(torch.ones(1) * np.log(self.init_sigma))
        self.log_sigma3 = nn.Parameter(torch.ones(1) * np.log(self.init_sigma))
        
        # Feature adaptation layers
        self.adaptation_layers = self._build_adaptation_layers()
        
        # Training tracking
        self.register_buffer('current_epoch', torch.tensor(0))
        self.ce_loss_values = []
        self.kd_loss_values = []
        
        # Initialize optimizer
        self.optimizer = self._init_optimizer(cfg)
        
    def _init_optimizer(self, cfg):
        """Initialize the Distillation-Oriented Trainer optimizer"""
        return DistillationOrientedTrainer(
            self.get_learnable_parameters(),
            lr=cfg.SOLVER.LR,
            momentum=cfg.DCKD.MOMENTUM - cfg.DCKD.DELTA,
            momentum_kd=cfg.DCKD.MOMENTUM + cfg.DCKD.DELTA,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )

    def _build_adaptation_layers(self):
        """Build feature adaptation layers"""
        layers = nn.ModuleDict()
        s_channels = self.student.get_channel_nums()
        t_channels = self.teacher.get_channel_nums()
        
        for s_ch, t_ch, idx in zip(s_channels, t_channels, range(len(s_channels))):
            layers[f'adapt_{idx}'] = nn.Sequential(
                nn.Conv2d(s_ch, t_ch, 1, bias=False),
                nn.BatchNorm2d(t_ch),
                nn.ReLU(inplace=True)
            )
        return layers

    def compute_sigma(self, log_sigma):
        """Compute σ value with constraints"""
        sigma = torch.exp(log_sigma)
        return torch.clamp(sigma, min=self.min_sigma)

    def forward_train(self, image, target, epoch, **kwargs):
        """Forward propagation for training"""
        # Get student and teacher outputs
        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)
        
        # Compute σ values
        sigma1 = self.compute_sigma(self.log_sigma1)  # CE loss
        sigma2 = self.compute_sigma(self.log_sigma2)  # KD loss
        sigma3 = self.compute_sigma(self.log_sigma3)  # Feature loss
        
        # Calculate losses
        loss_ce = F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss(logits_student, logits_teacher)
        loss_feat = self.feature_loss(features_student, features_teacher)
        
        # Calculate weighted losses with curriculum learning
        lambda_curr = self.compute_curriculum_lambda(epoch)
        weighted_loss_ce = loss_ce / (2 * sigma1**2)
        weighted_loss_kd = loss_kd / (2 * sigma2**2) * lambda_curr
        weighted_loss_feat = loss_feat / (2 * sigma3**2) * lambda_curr
        
        # Regularization term
        reg_term = torch.log(sigma1) + torch.log(sigma2) + torch.log(sigma3)
        
        # Total loss
        total_loss = weighted_loss_ce + weighted_loss_kd + weighted_loss_feat + reg_term
        
        # Record losses
        self.ce_loss_values.append(loss_ce.item())
        self.kd_loss_values.append(loss_kd.item())
        
        losses_dict = {
            "loss_ce": weighted_loss_ce.item(),
            "loss_kd": weighted_loss_kd.item(),
            "loss_feat": weighted_loss_feat.item(),
            "loss_reg": reg_term.item(),
            "total_loss": total_loss,
            "sigma1": sigma1.item(),
            "sigma2": sigma2.item(),
            "sigma3": sigma3.item(),
            "lambda": lambda_curr,
            "raw_ce": loss_ce.item(),
            "raw_kd": loss_kd.item(),
            "raw_feat": loss_feat.item()
        }
        
        return logits_student, losses_dict

    def kd_loss(self, logits_student, logits_teacher):
        """Calculate knowledge distillation loss"""
        log_pred_student = F.log_softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean') * (self.temperature ** 2)
        return loss_kd

    def feature_loss(self, s_features, t_features):
        """Calculate feature distillation loss"""
        feat_loss = 0
        for idx, (s_feat, t_feat) in enumerate(zip(s_features, t_features)):
            s_feat = self.adaptation_layers[f'adapt_{idx}'](s_feat)
            feat_loss += F.mse_loss(s_feat, t_feat.detach())
        return feat_loss

    def compute_curriculum_lambda(self, epoch):
        """Compute curriculum learning lambda based on sigmoid function"""
        return self.lambda_min + (self.lambda_max - self.lambda_min) / (
            1 + np.exp(-self.k * (epoch - self.x0))
        )

    def get_learnable_parameters(self):
        """Return all learnable parameters"""
        return (
            list(self.student.parameters()) + 
            list(self.adaptation_layers.parameters()) + 
            [self.log_sigma1, self.log_sigma2, self.log_sigma3]
        ) 