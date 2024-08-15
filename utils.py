import torch
import torch.nn as nn
import pickle
import random
import numpy as np
import gc

def initialize_model(model):
    """
    Initialize Model Weights
    """
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            # He initialization works well with ReLU activation (Conv layers usually followed by a ReLU)
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            # Xavier initialization for linear layers
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.MultiheadAttention):
            # Initialize MultiheadAttention to small values to prevent early saturation (e.g., softmax)
            # A small constant can work, but let's try with normal as an alternative here
            nn.init.normal_(module.in_proj_weight, std=0.02)  # 'in_proj_weight' is part of MultiheadAttention
            if module.in_proj_bias is not None:
                nn.init.constant_(module.in_proj_bias, 0)


def load_feature(feature):
    path = 'data/final_' + feature + '_data1.pkl'
    with open(path, 'rb') as f:
        train, test = pickle.load(f)
    return train, test

def load_all_features(device):
    X_onto_train, X_onto_test_ = load_feature("onto")
    X_onto_test = X_onto_test_.to(device)
    del X_onto_test_
    gc.collect()
    print("onto data have been loaded!")
    X_prot_train, X_prot_test_ = load_feature("prot")
    X_prot_test = X_prot_test_.to(device)
    del X_prot_test_
    gc.collect()
    print("prot data have been loaded!")
    X_esm_train, X_esm_test_ = load_feature("esm")
    X_esm_test = X_esm_test_.to(device)
    del X_esm_test_
    gc.collect()
    print("esm data have been loaded!")
    y_train_, y_test_ = load_feature("label")
    y_test = y_test_.to(device)
    del y_test_
    gc.collect()
    y_train = y_train_.to(device)
    del y_train_
    gc.collect()
    print("label data have been loaded!")

    print(f"train set shape is: {y_train.shape}")
    print(f"test set shape is {y_test.shape}")

    return X_onto_train, X_prot_train, X_esm_train, y_train, X_onto_test, X_prot_test, X_esm_test, y_test

def save_fpr_tpr(path, fpr, tpr, auc):
    with open(path, 'wb') as f:
        pickle.dump({
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc
        }, f)
        print("fpr_tpr saved")

def save_pre_rec(path, pre, rec, aupr):
    with open(path, 'wb') as f:
        pickle.dump({
            'pre': pre,
            'rec': rec,
            'aupr': aupr
        }, f)
        print("pre_rec saved")

class MySampler(torch.utils.data.sampler.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DefaultConfig(object):

    save_fpr_tpr_path = 'final_ComLMEss_fpr_tpr.pkl'
    save_pre_rec_path = 'final_ComLMEss_pre_rec.pkl'

    seed = 323
    kfold = 5
    patience = 20
    lr = 0.0001
    batch_szie = 128
    dropout = 0.3
    filter = 64
    activation = 'relu'
    optimizer = 'Adam'
    kernel_size_onto = 5
    kernel_size_prot = 5
    kernel_size_esm = 5
    T = 5
