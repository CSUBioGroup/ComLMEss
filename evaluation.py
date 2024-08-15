from sklearn import metrics


def compute_metrics(all_trues, all_scores, threshold=0.5):
    all_preds = (all_scores >= threshold)

    acc = metrics.accuracy_score(all_trues, all_preds)
    pre = metrics.precision_score(all_trues, all_preds)
    rec = metrics.recall_score(all_trues, all_preds)
    f1 = metrics.f1_score(all_trues, all_preds)
    mcc = metrics.matthews_corrcoef(all_trues, all_preds)
    fpr, tpr, _ = metrics.roc_curve(all_trues, all_scores)
    AUC = metrics.auc(fpr, tpr)
    p, r, _ = metrics.precision_recall_curve(all_trues, all_scores)
    AUPR = metrics.auc(r, p)

    return acc, f1, pre, rec, mcc, AUC, AUPR, fpr, tpr, p, r


def best_acc_thr(y_true, y_score):
    """ Calculate the best threshold with acc """
    best_thr = 0.5
    best_acc = 0

    for thr in range(1,100):
        thr /= 100
        acc, f1, pre, rec, mcc, AUC, AUPR, _, _, _, _ = compute_metrics(y_true, y_score, thr)

        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    return best_thr, best_acc