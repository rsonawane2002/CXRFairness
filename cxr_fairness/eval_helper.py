import torch
import numpy as np
from cxr_fairness.lib import misc
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
from cxr_fairness.data import Constants
from cxr_fairness.utils import compute_opt_thres

def predict_on_set(algorithm, loader, device, add_fields = ('sex', 'race')):
    preds, targets, paths, adds = [], [], [], {i:[] for i in add_fields}

    with torch.no_grad():
        for x, y, meta in loader:
            x = misc.to_device(x, device)
            algorithm.eval()
            logits = algorithm.predict(x)

            targets += y.detach().cpu().numpy().tolist()
            paths += meta['path']
            for j in add_fields:
                adds[j] += meta[j]

            if y.ndim == 1 or y.shape[1] == 1: # multiclass
                preds_list = torch.nn.Softmax(dim = 1)(logits)[:, 1].detach().cpu().numpy().tolist()
            else: # multilabel
                preds_list = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            if isinstance(preds_list, list):
                preds += preds_list
            else:
                preds += [preds_list]
    return np.array(preds), np.array(targets), np.array(paths), adds

def eval_metrics(algorithm, loader, device, protected_attr = None):
    preds, targets, paths, adds = predict_on_set(
        algorithm, loader, device,
        add_fields=((protected_attr,) if protected_attr is not None else ())
    )
    # mean AUC
    if targets.ndim == 2:  # multitask
        mean_auc_list = []
        optimal_thress = {}
        for i in range(targets.shape[1]):
            if len(np.unique(targets[:, i])) < 2:
                print(f"[WARNING] Skipping AUC for task {Constants.take_labels[i]}: only one class present.")
                continue
            auc = roc_auc_score(targets[:, i], preds[:, i])
            mean_auc_list.append(auc)
            optimal_thress[Constants.take_labels[i]] = compute_opt_thres(targets[:, i], preds[:, i])
        mean_auc = np.mean(mean_auc_list) if mean_auc_list else float("nan")
        pred_df = pd.DataFrame({
            **{'path': paths},
            **{label: preds[:, c] for c, label in enumerate(Constants.take_labels)}
        })
        # Binarize for classification metrics (use threshold 0.5 for now)
        bin_preds = (preds >= 0.5).astype(int)
    else:
        if len(np.unique(targets)) < 2:
            print("[WARNING] Only one class in targets. Skipping AUC.")
            mean_auc = float("nan")
            optimal_thress = None
        else:
            mean_auc = roc_auc_score(targets, preds)
            optimal_thress = compute_opt_thres(targets, preds)
        pred_df = pd.DataFrame({'path': paths, 'pred': preds})
        bin_preds = (preds >= 0.5).astype(int)

    # Compute precision/recall/f1 safely
    try:
        precision = precision_score(targets, bin_preds, average='weighted', zero_division=0)
        recall = recall_score(targets, bin_preds, average='weighted', zero_division=0)
        f1 = f1_score(targets, bin_preds, average='weighted', zero_division=0)
    except ValueError as e:
        print(f"[WARNING] Error computing classification metrics: {e}")
        precision, recall, f1 = float("nan"), float("nan"), float("nan")

    # compute worst AUC by group
    aucs = []
    if protected_attr is not None:
        adds[protected_attr] = np.array(adds[protected_attr])
        unique_groups = np.unique(adds[protected_attr])
        for grp in unique_groups:
            mask = adds[protected_attr] == grp
            if np.sum(mask) == 0:
                continue
            if targets.ndim == 2:
                group_auc_list = []
                for i in range(targets.shape[1]):
                    if len(np.unique(targets[:, i][mask])) < 2:
                        continue
                    group_auc_list.append(roc_auc_score(targets[:, i][mask], preds[:, i][mask]))
                if group_auc_list:
                    aucs.append(np.mean(group_auc_list))
            else:
                if len(np.unique(targets[mask])) < 2:
                    continue
                aucs.append(roc_auc_score(targets[mask], preds[mask]))

    return {
        'roc': mean_auc,
        'worst_roc': mean_auc if not aucs else min(aucs),
        'roc_gap': 0. if not aucs else max(aucs) - min(aucs),
        'optimal_thres': optimal_thress,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }, pred_df
   
   

