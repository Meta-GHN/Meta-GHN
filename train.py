import torch
import numpy as np
import argparse

from itertools import combinations
from utils import *
from meta import Meta



def main(args):
    n_way = args.way
    k_shot = args.shot
    m_query = args.qry
    setsz = args.setsz

    meta_test_num = args.step
    meta_valid_num = args.step

    dataset = args.dataset

    normalize_features = True
    if dataset == 'ogbn':
        normalize_features = False

    adj, features, labels, degrees, class_list_train, class_list_valid, class_list_test, id_by_class = load_data(dataset, args, normalize_features)
    features = sgc_precompute(features, adj, args.degree)

    full_graph_adj_spt = full_graph_generate_large(n_way, k_shot, setsz)
    full_graph_adj_qry = full_graph_generate_large(n_way, m_query, setsz)


    if args.cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        full_graph_adj_spt = full_graph_adj_spt.cuda()
        full_graph_adj_qry = full_graph_adj_qry.cuda()

    valid_pool = [task_generator(id_by_class, class_list_valid, n_way, k_shot, m_query, is_train=False) for i in range(meta_valid_num)]
    test_pool = [task_generator(id_by_class, class_list_test, n_way, k_shot, m_query, is_train=False) for i in range(meta_test_num)]

    config_encoder = [
        ('linear', [args.hidden, features.size(1)]),
    ]

    config_filter = [
        ('GraphATT', [2*args.hidden, 1])
    ]

    config_classifier = [
        ('linear', [n_way, args.hidden])
    ]

    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    

    maml = Meta(args, full_graph_adj_spt, full_graph_adj_qry, config_encoder, config_filter, config_classifier).to(device)


    for j in range(args.epoch):

        x_spt, y_spt, x_qry, y_qry = \
            task_generator_maml_v2(features, labels, args.task_num, id_by_class, class_list_train, n_way, k_shot, m_query, setsz, args.cuda)

        accs = maml.forward(x_spt, y_spt, x_qry, y_qry)


        if (j+1) % args.verbose == 0:
            torch.save(maml.state_dict(), 'maml_'+str(n_way)+'_'+str(k_shot)+'_'+str(dataset)+'_'+str(args.noise)+'_'+str(args.cp)+'.pkl')


            #### Validation
            meta_test_acc = []
            meta_test_f1 = []
            for k in range(meta_test_num):
                model_meta_trained = Meta(args, full_graph_adj_spt, full_graph_adj_qry, config_encoder, config_filter, config_classifier).to(device)
                model_meta_trained.load_state_dict(torch.load('maml_'+str(n_way)+'_'+str(k_shot)+'_'+str(dataset)+'_'+str(args.noise)+'_'+str(args.cp)+'.pkl'))
                model_meta_trained.eval() 

                id_support, id_query, class_selected = valid_pool[k]
                x_spt, y_spt, x_qry, y_qry = [], [], [], []

                x_spt.append(features[id_support])    
                x_qry.append(features[id_query])
                    
                if args.cuda:
                    y_spt.append(torch.LongTensor([class_selected.index(i) for i in labels[id_support]]).cuda())
                    y_qry.append(torch.LongTensor([class_selected.index(i) for i in labels[id_query]]).cuda())
                else:
                    y_spt.append(torch.LongTensor([class_selected.index(i) for i in labels[id_support]]))
                    y_qry.append(torch.LongTensor([class_selected.index(i) for i in labels[id_query]]))

                accs,f1s = model_meta_trained.forward_full(x_spt, y_spt, x_qry, y_qry, args.cuda)
                meta_test_acc.append(accs[-1])
                meta_test_f1.append(f1s[-1])
            print("Epoch: {}, Meta-Valid_Accuracy: {}, Meta-Valid_F1: {}".format(j+1, np.array(meta_test_acc).mean(axis=0),
                                                                np.array(meta_test_f1).mean(axis=0)))

            #### Test
            meta_test_acc = []
            meta_test_f1 = []
            for k in range(meta_test_num):
                model_meta_trained = Meta(args, full_graph_adj_spt, full_graph_adj_qry, config_encoder, config_filter, config_classifier).to(device)
                model_meta_trained.load_state_dict(torch.load('maml_'+str(n_way)+'_'+str(k_shot)+'_'+str(dataset)+'_'+str(args.noise)+'_'+str(args.cp)+'.pkl'))
                model_meta_trained.eval() 

                id_support, id_query, class_selected = test_pool[k]
                x_spt, y_spt, x_qry, y_qry = [], [], [], []

                x_spt.append(features[id_support])    
                x_qry.append(features[id_query])
                    
                if args.cuda:
                    y_spt.append(torch.LongTensor([class_selected.index(i) for i in labels[id_support]]).cuda())
                    y_qry.append(torch.LongTensor([class_selected.index(i) for i in labels[id_query]]).cuda())
                else:
                    y_spt.append(torch.LongTensor([class_selected.index(i) for i in labels[id_support]]))
                    y_qry.append(torch.LongTensor([class_selected.index(i) for i in labels[id_query]]))

                accs,f1s = model_meta_trained.forward_full(x_spt, y_spt, x_qry, y_qry, args.cuda)
                meta_test_acc.append(accs[-1])
                meta_test_f1.append(f1s[-1])
            print("Epoch: {}, Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(j+1, np.array(meta_test_acc).mean(axis=0),
                                                                np.array(meta_test_f1).mean(axis=0)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--use_cuda', action='store_true', help='Disables CUDA training.')
    argparser.add_argument('--epoch', type=int, help='epoch number', default=50001)
    argparser.add_argument('--verbose', type=int, help='verbose epoch number', default=1000)
    argparser.add_argument('--way', type=int, help='n way', default=5)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.003)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.5)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=5)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
    argparser.add_argument('--shot', type=int, help='k shot for support set', default=3)
    argparser.add_argument('--qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--setsz', type=int, help='size of each set', default=4)
    argparser.add_argument('--hidden', type=int, help='Number of hidden units', default=16)

    argparser.add_argument('--dataset', default='ogbn', help='Dataset:Amazon_eletronics/dblp/ogbn')
    argparser.add_argument('--normalization', type=str, default='AugNormAdj',
                           help='Normalization method for the adjacency matrix.')
    argparser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    argparser.add_argument('--degree', type=int, default=2, help='degree of the approximation.')
    argparser.add_argument('--step', type=int, default=100, help='How many times to random select node to test')

    argparser.add_argument('--noise', type=int, default=0, help='0: symmetric_noise, 1: asymmetric_noise')
    argparser.add_argument('--cp', type=float, default=0.5, help='corruption noise type')

    args = argparser.parse_args()
    print(args)

    args.cuda = args.use_cuda and torch.cuda.is_available()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    main(args)