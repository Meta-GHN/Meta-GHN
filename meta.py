import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch import optim
from learner import Learner, NoiseFilter
from utils import f1, f1_cuda


class Meta(nn.Module):
    def __init__(self, args, full_graph_adj_spt, full_graph_adj_qry, config_encoder, config_filter, config_classifier):
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.way
        self.k_spt = args.shot
        self.k_qry = args.qry
        self.setsz = args.setsz
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.full_graph_adj_spt = full_graph_adj_spt
        self.full_graph_adj_qry = full_graph_adj_qry

        self.encoder = Learner(config_encoder)
        self.filter = NoiseFilter(config_filter)
        self.classifer = Learner(config_classifier)

        self.alpha_list = nn.ParameterList([])
        self.alpha_list.extend(self.encoder.parameters())
        self.ind_encoder = len(self.alpha_list)
        self.alpha_list.extend(self.filter.parameters())
        self.ind_filter = len(self.alpha_list)
        self.alpha_list.extend(self.classifer.parameters())

        self.meta_optim = optim.Adam(self.alpha_list, lr=self.meta_lr)


    def clip_grad_by_norm_(self, grad, max_norm):
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def compress_set(self, feas, z_dim, num_set, graph_adj, varList):
        centers = feas.mean(2).unsqueeze(-2).repeat_interleave(self.setsz, dim=-2)
        diff = feas - centers
        diff = torch.cat((feas, diff), -1)
        diff = diff.view([self.n_way*num_set*(self.setsz), 2*z_dim])
        diff_score = self.filter(graph_adj, diff, vars = varList)
        diff_score = F.softmax(diff_score.view([self.n_way, num_set, self.setsz, 1]), dim = -2)
        #print(diff_score)
        feas = (feas * diff_score).sum(-2)
        return feas

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        task_num = self.task_num
        querysz = self.n_way * self.k_qry
        #querysz = self.k_qry

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        
        for i in range(task_num):
            feas = self.encoder(x_spt[i], vars=self.alpha_list[:self.ind_encoder], bn_training=True)
            z_dim = feas.size()[1]
            feas = feas.view([self.n_way, self.k_spt, self.setsz, z_dim])
            feas = self.compress_set(feas, z_dim, self.k_spt, self.full_graph_adj_spt, self.alpha_list[self.ind_encoder:self.ind_filter])
            logits = self.classifer(feas.view([self.n_way * self.k_spt, z_dim]), vars=self.alpha_list[self.ind_filter:], bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.alpha_list)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.alpha_list)))

            with torch.no_grad():
                feas = self.encoder(x_qry[i], vars=self.alpha_list[:self.ind_encoder], bn_training=True)
                z_dim = feas.size()[1]
                feas = feas.view([self.n_way, self.k_qry, self.setsz, z_dim])
                feas = self.compress_set(feas, z_dim, self.k_qry, self.full_graph_adj_qry, self.alpha_list[self.ind_encoder:self.ind_filter])
                logits_q = self.classifer(feas.view([self.n_way * self.k_qry, z_dim]), vars=self.alpha_list[self.ind_filter:], bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            with torch.no_grad():
                feas = self.encoder(x_qry[i], vars=fast_weights[:self.ind_encoder], bn_training=True)
                feas = feas.view([self.n_way, self.k_qry, self.setsz, z_dim])  
                feas = self.compress_set(feas, z_dim, self.k_qry, self.full_graph_adj_qry, fast_weights[self.ind_encoder:self.ind_filter])         
                logits_q = self.classifer(feas.view([self.n_way * self.k_qry, z_dim]), vars=fast_weights[self.ind_filter:], bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                feas = self.encoder(x_spt[i], vars=fast_weights[:self.ind_encoder], bn_training=True)
                feas = feas.view([self.n_way, self.k_spt, self.setsz, z_dim])                
                feas = self.compress_set(feas, z_dim, self.k_spt, self.full_graph_adj_spt, fast_weights[self.ind_encoder:self.ind_filter]) 
                logits = self.classifer(feas.view([self.n_way * self.k_spt, z_dim]), fast_weights[self.ind_filter:], bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))                
                feas = self.encoder(x_qry[i], vars=fast_weights[:self.ind_encoder], bn_training=True)
                feas = feas.view([self.n_way, self.k_qry, self.setsz, z_dim])
                feas = self.compress_set(feas, z_dim, self.k_qry, self.full_graph_adj_qry, fast_weights[self.ind_encoder:self.ind_filter]) 
                logits_q = self.classifer(feas.view([self.n_way * self.k_qry, z_dim]), vars=fast_weights[self.ind_filter:], bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        loss_q = losses_q[-1] / task_num
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num)

        self.encoder.vars = self.alpha_list[:self.ind_encoder]
        self.filter.vars = self.alpha_list[self.ind_encoder:self.ind_filter]
        self.classifer.vars = self.alpha_list[self.ind_filter:]

        return accs
        

    def forward_full(self, x_spt, y_spt, x_qry, y_qry, cuda):
        task_num = 1#self.task_num
        querysz = self.n_way * self.k_qry
        #querysz = self.k_qry

        losses_q = [0 for _ in range(self.update_step_test + 1)]
        f1s = [0 for _ in range(self.update_step_test + 1)]
        corrects = [0 for _ in range(self.update_step_test + 1)]

        for i in range(task_num):
            feas = self.encoder(x_spt[i], vars=self.alpha_list[:self.ind_encoder], bn_training=True)            
            logits = self.classifer(feas, vars=self.alpha_list[self.ind_filter:], bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            beta_list = nn.ParameterList([])
            beta_list.extend(self.alpha_list[:self.ind_encoder])
            beta_list.extend(self.alpha_list[self.ind_filter::])
            grad = torch.autograd.grad(loss, beta_list)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, beta_list)))

            with torch.no_grad():
                feas = self.encoder(x_qry[i], vars=self.alpha_list[:self.ind_encoder], bn_training=True)            
                logits_q = self.classifer(feas, vars=self.alpha_list[self.ind_filter:], bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct
                if cuda:
                    result = f1_cuda(logits_q, y_qry[i])
                else:
                    result = f1(logits_q, y_qry[i])
                f1s[0] = f1s[0] + result

            with torch.no_grad():
                feas = self.encoder(x_qry[i], vars=fast_weights[:self.ind_encoder], bn_training=True)            
                logits_q = self.classifer(feas, vars=fast_weights[self.ind_encoder:], bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct
                if cuda:
                    result = f1_cuda(logits_q, y_qry[i])
                else:
                    result = f1(logits_q, y_qry[i])
                f1s[1] = f1s[1] + result
                

            for k in range(1, self.update_step_test):
                feas = self.encoder(x_spt[i], vars=fast_weights[:self.ind_encoder], bn_training=True)            
                logits = self.classifer(feas, vars=fast_weights[self.ind_encoder:], bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                feas = self.encoder(x_qry[i], vars=fast_weights[:self.ind_encoder], bn_training=True)            
                logits_q = self.classifer(feas, vars=fast_weights[self.ind_encoder:], bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct
                    if cuda:
                        result = f1_cuda(logits_q, y_qry[i])
                    else:
                        result = f1(logits_q, y_qry[i])
                    f1s[k + 1] = f1s[k + 1] + result
                    

        loss_q = losses_q[-1] / task_num
        accs = np.array(corrects) / (1.0*querysz * task_num)
        f1s = np.array(f1s) / (1.0*task_num)
        return accs, f1s

    def parameters(self):
        return self.alpha_list