import gc
import os, random, torch, pickle
import numpy as np
from torch import nn
from model.ComLMEss import ComLMEss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from evaluation import compute_metrics, best_acc_thr
from utils import initialize_model, load_all_features, save_pre_rec, save_fpr_tpr, set_seed, MySampler, DefaultConfig




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
config = DefaultConfig()


class ProteinDataset(Dataset):
    def __init__(self, onto_data, prot_data, esm_data, labels):
        self.onto_data = onto_data
        self.prot_data = prot_data
        self.esm_data = esm_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        onto = self.onto_data[idx]
        prot = self.prot_data[idx]
        esm = self.esm_data[idx]
        label = self.labels[idx]
        return onto, prot, esm, label



def train_epoch(model, train_iter, optimizer, loss, scheduler):
    # Model on train mode
    model.train()

    all_trues = []
    all_scores = []
    losses, sample_num = 0.0, 0

    for batch_idx, (onto1, prot1, esm1, y1) in enumerate(train_iter):
        sample_num += y1.size(0)

        onto, prot, esm, y = onto1.to(device), prot1.to(device), esm1.to(device), y1.to(device)
        del onto1, prot1, esm1, y1
        gc.collect()

        # compute output
        output = model(onto, prot, esm).view(-1)

        # calculate and record loss
        loss_batch = loss(output, y)
        losses += loss_batch.item()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

        output = nn.Sigmoid()(output)

        all_trues.append(y.data.cpu().numpy())
        all_scores.append(output.data.cpu().numpy())

    scheduler.step()

    all_trues = np.concatenate(all_trues, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    acc, f1, pre, rec, mcc, AUC, AUPR, _, _,_,_ = compute_metrics(all_trues, all_scores)

    return losses / sample_num, acc, f1, pre, rec, mcc, AUC, AUPR


def eval_epoch(model,eval_iter, loss):
    # Model on eval mode
    model.eval()

    all_trues = []
    all_scores = []
    losses, sample_num = 0.0, 0
    for batch_idx, (onto1, prot1, esm1, y1) in enumerate(eval_iter):
        sample_num += y1.size(0)
        onto, prot, esm, y = onto1.to(device), prot1.to(device), esm1.to(device), y1.to(device)
        del onto1, prot1, esm1, y1
        gc.collect()

        # Create vaiables
        with torch.no_grad():
            onto_var = torch.autograd.Variable(onto.float()).to(device)
            prot_var = torch.autograd.Variable(prot.float()).to(device)
            esm_var = torch.autograd.Variable(esm.float()).to(device)
            y_var = torch.autograd.Variable(y.float()).to(device)

        # compute output
        output = model(onto_var, prot_var, esm_var).view(-1)


        # compute loss and record loss
        loss_batch = loss(output, y_var)
        losses += loss_batch.item()

        output = nn.Sigmoid()(output)

        all_trues.append(y_var.data.cpu().numpy())
        all_scores.append(output.data.cpu().numpy())

    all_trues = np.concatenate(all_trues, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    acc, f1, pre, rec, mcc, AUC, AUPR, fpr, tpr, p, r  = compute_metrics(all_trues, all_scores)

    return losses / sample_num, acc, f1, pre, rec, mcc, AUC, AUPR, fpr, tpr, p, r, all_scores, all_trues

def train(model, X_onto_train, X_onto_test,X_prot_train, X_prot_test,
          X_esm_train, X_esm_test, y_train, y_test, save, epoch_num, batch_size, lr, optimizer_name, T):

    train_dataset = ProteinDataset(X_onto_train, X_prot_train, X_esm_train, y_train)
    test_dataset = ProteinDataset(X_onto_test, X_prot_test, X_esm_test, y_test)
    del X_prot_train, X_esm_train,  X_onto_test, X_prot_test, X_esm_test, y_test
    gc.collect()

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # Optimizer
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0001)
    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=T, eta_min=0.000001)


    # Train and validation using 5-fold cross validation
    val_auprs, test_auprs = [], []
    val_aucs, test_aucs = [], []
    test_trues, kfold_test_scores = [], []
    kfold = config.kfold
    patience = config.patience
    skf = StratifiedKFold(n_splits=kfold, random_state=config.seed, shuffle=True)
    for i, (train_index, val_index) in enumerate(skf.split(X_onto_train.cpu(), y_train.cpu())):
        print(f'\nStart training CV fold {i + 1}:')
        train_sampler, val_sampler = MySampler(train_index), MySampler(val_index)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False,)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False,)

        initialize_model(model)
        count = 0
        best_val_aupr, best_test_aupr = .0, .0
        best_val_auc, best_test_auc = .0, .0
        best_test_scores = []
        best_model = model

        num_ess = torch.sum(y_train[train_index] == 1)
        num_noness = torch.sum(y_train[train_index] == 0)
        pos_weight = float(num_noness / num_ess) if num_ess > 0 else 1.0
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight])).to(device)

        for epoch in tqdm(range(epoch_num)):
            train_loss, train_acc, train_f1, train_pre, train_rec, train_mcc, train_auc, train_aupr = train_epoch(
                model=model,
                train_iter=train_loader,
                optimizer=optimizer,
                loss=loss,
                scheduler=scheduler
            )

            val_loss, val_acc, val_f1, val_pre, val_rec, val_mcc, val_auc, val_aupr, val_fpr, val_tpr, val_p, val_r, val_scores, _ = eval_epoch(
                model=model,
                eval_iter=val_loader,
                loss=loss,
            )

            test_loss, acc, f1, pre, rec, mcc, auc, aupr, fpr, tpr, p, r, test_scores, test_trues = eval_epoch(
                model=model,
                eval_iter=test_loader,
                loss=loss,
            )


            res = '\t'.join([
                '\nEpoch [%d/%d]' % (epoch + 1, epoch_num),
                '\nTraining set',
                'loss:%0.5f' % train_loss,
                'accuracy:%0.6f' % train_acc,
                'f-score:%0.6f' % train_f1,
                'precision:%0.6f' % train_pre,
                'recall:%0.6f' % train_rec,
                'mcc:%0.6f' % train_mcc,
                'auc:%0.6f' % train_auc,
                'aupr:%0.6f' % train_aupr,
                '\nValidation set',
                'loss:%0.5f' % val_loss,
                'accuracy:%0.6f' % val_acc,
                'f-score:%0.6f' % val_f1,
                'precision:%0.6f' % val_pre,
                'recall:%0.6f' % val_rec,
                'mcc:%0.6f' % val_mcc,
                'auc:%0.6f' % val_auc,
                'aupr:%0.6f' % val_aupr,
                '\nTesting set',
                'loss:%0.5f' % test_loss,
                'accuracy:%0.6f' % acc,
                'f-score:%0.6f' % f1,
                'precision:%0.6f' % pre,
                'recall:%0.6f' % rec,
                'mcc:%0.6f' % mcc,
                'auc:%0.6f' % auc,
                'aupr:%0.6f' % aupr,
            ])
            print(res)

            if val_auc > best_val_auc:
                count = 0
                best_model = model
                best_val_auc = val_auc
                best_val_aupr = val_aupr

                best_test_auc = auc
                best_test_aupr = aupr

                best_test_scores = test_scores

                print("!!!Get better model with valid AUC:{:.6f}. ".format(val_auc))

            else:
                count += 1
                if count >= patience:
                    torch.save(best_model, os.path.join(save,'model_{}_{:.3f}_{:.3f}.pkl'.format(i + 1, best_test_auc,best_test_aupr)))
                    print(f'Fold {i + 1} training done!!!\n')
                    break

        val_auprs.append(best_val_aupr)
        test_auprs.append(best_test_aupr)
        val_aucs.append(best_val_auc)
        test_aucs.append(best_test_auc)
        kfold_test_scores.append(best_test_scores)


    for i, (test_auc, test_aupr) in enumerate(zip(test_aucs, test_auprs)):
        print('Fold {}: test AUC:{:.6f}   test AUPR:{:.6f}.'.format(i + 1, test_auc, test_aupr))

    final_test_scores = np.sum(np.array(kfold_test_scores), axis=0) / kfold

    best_acc_threshold, best_acc = best_acc_thr(test_trues, final_test_scores)
    print('The best acc threshold is {:.2f} with the best acc({:.3f}).'.format(best_acc_threshold, best_acc))

    (final_acc, final_f1, final_pre, final_rec,
     final_mcc, final_AUC, final_AUPR, final_fpr, final_tpr, final_p, final_r) = compute_metrics(test_trues, final_test_scores, best_acc_threshold)


    save_fpr_tpr(config.save_fpr_tpr_path, final_fpr, final_tpr, final_AUC)
    save_pre_rec(config.save_pre_rec_path, final_p, final_r, final_AUPR)

    return final_acc, final_f1, final_pre, final_rec, final_mcc, final_AUC, final_AUPR, final_fpr, final_tpr


def train_models(X_onto_train_, y_train_, X_onto_test_, y_test_,
                 X_prot_train_, X_prot_test_, X_esm_train_, X_esm_test_,
                 Model, epoch_num, batch_size, lr, save, optimizer, T):

    X_onto_train = X_onto_train_.float()
    del X_onto_train_
    gc.collect()
    X_prot_train = X_prot_train_.float()
    del X_prot_train_
    gc.collect()
    X_esm_train = X_esm_train_.float()
    del X_esm_train_
    gc.collect()
    X_esm_test = X_esm_test_.float()
    del X_esm_test_
    gc.collect()
    X_prot_test = X_prot_test_.float()
    del X_prot_test_
    gc.collect()
    X_onto_test = X_onto_test_.float()
    del X_onto_test_
    gc.collect()
    y_train = y_train_.float()
    del y_train_
    gc.collect()
    y_test = y_test_.float()
    del y_test_
    gc.collect()

    model = Model.to(device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)

    acc, f1, pre, rec, mcc, auc, aupr, _, _ = train(model, X_onto_train, X_onto_test,
          X_prot_train, X_prot_test, X_esm_train, X_esm_test, y_train, y_test, save, epoch_num, batch_size, lr, optimizer, T)


    return acc, f1, pre, rec, mcc, auc, aupr


if __name__ == '__main__':



    set_seed(config['seed'])

    X_onto_train, X_prot_train, X_esm_train, y_train, X_onto_test, X_prot_test, X_esm_test, y_test = load_all_features(device)

    # model
    model = ComLMEss(kernel_size_onto=config.kernel_size_onto,
                     kernel_size_prot=config.kernel_size_prot,
                     kernel_size_esm=config.kernel_size_esm,
                     dropout=config.dropout,
                     num_filters=config.filter,
                     activation=config.activation)


    # param
    threshold, epoch_num, batch_size, lr, optimizer, T = 0.5, 500, config.batch_szie, config.lr, config.optimizer, config.T
    path_dir = './saved_models'
    result_file = 'results.csv'
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    print('Models save in ' + path_dir)

    acc, f1, pre, rec, mcc, auc, aupr = train_models(X_onto_train, y_train, X_onto_test, y_test, X_prot_train, X_prot_test,
                 X_esm_train, X_esm_test, model, epoch_num, batch_size, lr, path_dir, optimizer, T)


