import torch
from torch import nn
from torch.nn import functional as F
from layers import GraphConvolution, GraphConv, GraphAttentionLayer

class Learner(nn.Module):
    def __init__(self, config):
        super(Learner, self).__init__()
        self.config = config
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'convt2d':
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
            elif name is 'linear':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'bn':
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d', 'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'
            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'
            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError
        return info

    def forward(self, x, vars=None, bn_training=True):
        if vars is None:
            vars = self.vars
        idx = 0
        bn_idx = 0
        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name is 'flatten':
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            else:
                raise NotImplementedError
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        return x

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars

class NoiseFilter(nn.Module):
    def __init__(self, config):
        super(NoiseFilter, self).__init__()
        
        self.vars = nn.ParameterList()
        self.graph_conv = []
        self.graph_att = []
        self.config = config
        
        for i, (name, param) in enumerate(self.config):
            
            if name is 'Linear':
                
                w = nn.Parameter(torch.ones(param[1], param[0]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
            
            '''if name is 'GraphConv':
                # param: in_dim, hidden_dim
                w = nn.Parameter(torch.Tensor(param[0], param[1]))
                torch.nn.init.xavier_uniform_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                self.graph_conv.append(GraphConv(param[0], param[1]))'''

            if name is 'GraphConv':
                # param: in_dim, hidden_dim
                w = nn.Parameter(torch.Tensor(param[0], param[1]))
                torch.nn.init.xavier_uniform_(w)
                self.vars.append(w)
                #self.vars.append(nn.Parameter(torch.zeros(param[1])))
                self.graph_conv.append(GraphConv(param[0], param[1]))

            if name is 'GraphATT':
                # param: in_dim, hidden_dim
                w = nn.Parameter(torch.Tensor(param[0], param[1]))
                torch.nn.init.xavier_uniform_(w)
                self.vars.append(w)
                b = nn.Parameter(torch.Tensor(2*param[1],1))
                torch.nn.init.xavier_uniform_(b, gain=1.414)
                self.vars.append(b)
                self.graph_att.append(GraphAttentionLayer(param[0], param[1]))


    def forward(self, adj, h, vars = None):
        # For undirected graphs, in_degree is the same as
        # out_degree.

        if vars is None:
            vars = self.vars

        idx = 0 
        idx_gcn = 0
        idx_att = 0

        for name, param in self.config:
            '''if name is 'GraphConv':
                w, b = vars[idx], vars[idx + 1]
                conv = self.graph_conv[idx_gcn]
                
                h = conv(adj, h, w, b)

                idx += 2 
                idx_gcn += 1'''

            if name is 'GraphConv':
                w = vars[idx]
                conv = self.graph_conv[idx_gcn]
                
                h = conv(adj, h, w)

                idx += 1 
                idx_gcn += 1
                        
            if name is 'Linear':
                w, b = vars[idx], vars[idx + 1]
                h = F.linear(h, w, b)
                idx += 2

            if name is 'GraphATT':
                w, b = vars[idx], vars[idx + 1]
                conv = self.graph_att[idx_att]
                
                h = conv(adj, h, w, b)

                idx += 2 
                idx_att += 1
       
        return h
            
    def zero_grad(self, vars=None):

        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars