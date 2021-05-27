import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score

valid_num_dic = {'Amazon_clothing': 17, 'Amazon_eletronics': 36, 'dblp': 27, 'ogbn': 12}

def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)

def flip_labels_C(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C

def sgc_precompute(features, adj, degree):
    for i in range(degree):
        features = torch.spmm(adj, features)
    return features


def load_data(dataset_source, args, normalize_features = False):
    n1s = []
    n2s = []
    for line in open("data/{}_network".format(dataset_source)):
        n1, n2 = line.strip().split('\t')
        n1s.append(int(n1))
        n2s.append(int(n2))

    num_nodes = max(max(n1s),max(n2s)) + 1
    adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                 shape=(num_nodes, num_nodes))


    data_train = sio.loadmat("data/{}_train.mat".format(dataset_source))
    train_class = list(set(data_train["Label"].reshape((1,len(data_train["Label"])))[0]))
    

    data_test = sio.loadmat("data/{}_test.mat".format(dataset_source))
    class_list_test = list(set(data_test["Label"].reshape((1,len(data_test["Label"])))[0]))


    labels = np.zeros((num_nodes,1))
    labels[data_train['Index']] = data_train["Label"]
    labels[data_test['Index']] = data_test["Label"]

    features = np.zeros((num_nodes,data_train["Attributes"].shape[1]))
    
    if normalize_features:
        features[data_train['Index']] = normalize(data_train["Attributes"]).toarray()
        features[data_test['Index']] = normalize(data_test["Attributes"]).toarray()
        print('normalize_features')
    else:
        features[data_train['Index']] = data_train["Attributes"].toarray()
        features[data_test['Index']] = data_test["Attributes"].toarray()
        

    class_list = []
    for cla in labels:
        if cla[0] not in class_list:
            class_list.append(cla[0])  # unsorted

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(labels):
        id_by_class[cla[0]].append(id)

    train_class_map = {}
    for i in range(len(train_class)):
        train_class_map[train_class[i]] = i

    if args.noise == 0:
        C = uniform_mix_C(args.cp, len(train_class))
    elif args.noise == 1:
        C = flip_labels_C(args.cp, len(train_class))

    node_count = 0
    node_corrupt = 0
    for c in train_class:
        for index in id_by_class[c]:
            original = labels[index][0]
            labels[index][0] = train_class[np.random.choice(len(train_class), p=C[train_class_map[c]])]
            if original != labels[index][0]:
                node_corrupt += 1
                
            node_count += 1
    print('Total number of nodes', int(node_count), 'corrupted nodes', int(node_corrupt))

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(labels):
        id_by_class[cla[0]].append(id)

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    degree = np.sum(adj, axis=1)
    degree = torch.FloatTensor(degree)


    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    class_list_valid = random.sample(train_class, valid_num_dic[dataset_source])

    class_list_train = list(set(train_class).difference(set(class_list_valid)))

    return adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class 



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average='weighted')
    return f1

def f1_cuda(output, labels):
    output = output.max(1)[1]
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, output, average='weighted')
    return micro

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def task_generator_v2(id_by_class, class_list, n_way, k_set, m_query, m_shot, is_train=True):

    # sample class indices
    class_selected = random.sample(class_list, n_way)
    id_support = []
    id_query = []
    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_set * m_shot + m_query)
        id_support.extend(temp[:k_set * m_shot])
        id_query.extend(temp[k_set * m_shot:])

    return np.array(id_support), np.array(id_query), class_selected


def task_generator(id_by_class, class_list, n_way, k_shot, m_query, is_train=True):

    # sample class indices
    class_selected = random.sample(class_list, n_way)
    id_support = []
    id_query = []
    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])

    return np.array(id_support), np.array(id_query), class_selected

def task_generator_maml_v2(features, labels, task_num, id_by_class, class_list, n_way, k_set, m_query, m_shot, cuda):

    x_spt, y_spt, x_qry, y_qry = [], [], [], []


    for i in range(task_num):
        # sample class indices
        class_selected = random.sample(class_list, n_way)

        # sample support examples
        id_support = []
        id_query = []
        labels_support = []
        labels_query = []
        for cla in class_selected:
            selectd_support = []
            for j in range(k_set):
                temp = random.sample(id_by_class[cla], m_shot)
                id_support.extend(temp)
                selectd_support.extend(temp)

            id_remain = list(set(id_by_class[cla]).difference(set(selectd_support)))
            for j in range(m_query):
                temp = random.sample(id_remain, m_shot)
                id_query.extend(temp)

            labels_support += [cla] * k_set
            labels_query += [cla] * m_query

        x_spt.append(features[id_support])    
        x_qry.append(features[id_query])
        

        if cuda:
            y_spt.append(torch.LongTensor([class_selected.index(i) for i in labels_support]).cuda())
            y_qry.append(torch.LongTensor([class_selected.index(i) for i in labels_query]).cuda())
        else:
            y_spt.append(torch.LongTensor([class_selected.index(i) for i in labels_support]))
            y_qry.append(torch.LongTensor([class_selected.index(i) for i in labels_query]))

    return x_spt, y_spt, x_qry, y_qry

def task_generator_maml(features, labels, task_num, id_by_class, class_list, n_way, k_shot, m_query, cuda):

    x_spt, y_spt, x_qry, y_qry = [], [], [], []

    for i in range(task_num):
        # sample class indices
        class_selected = random.sample(class_list, n_way)

        # sample support examples
        id_support = []
        id_query = []
        for cla in class_selected:
            temp = random.sample(id_by_class[cla], k_shot + m_query)
            id_support.extend(temp[:k_shot])
            id_query.extend(temp[k_shot:])


        x_spt.append(features[id_support])    
        x_qry.append(features[id_query])
            
        if cuda:
            y_spt.append(torch.LongTensor([class_selected.index(i) for i in labels[id_support]]).cuda())
            y_qry.append(torch.LongTensor([class_selected.index(i) for i in labels[id_query]]).cuda())
        else:
            y_spt.append(torch.LongTensor([class_selected.index(i) for i in labels[id_support]]))
            y_qry.append(torch.LongTensor([class_selected.index(i) for i in labels[id_query]]))

    return x_spt, y_spt, x_qry, y_qry


def full_graph_generate_large(n_way, k_shot, setsz):
    x1 = []
    x2 = []
    for w in range(n_way):
        for k in range(k_shot):
            for s1 in range(setsz):
                for s2 in range(setsz):
                    x1.append(w * (k_shot * setsz) + k * setsz + s1)
                    x2.append(w * (k_shot * setsz) + k * setsz + s2)

    num_nodes = max(max(x1),max(x2)) + 1
    adj = sp.coo_matrix((np.ones(len(x1)), (x1, x2)),
                                 shape=(num_nodes, num_nodes))

    adj = normalize_adj(adj)

    return torch.FloatTensor(adj.todense())