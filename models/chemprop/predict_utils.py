from deepchem.data import Dataset
from deepchem.models import Model
from typing import List, Callable, Tuple

import numpy as np
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, confusion_matrix, auc, roc_auc_score

def _get_model_prediction(model, dataset: Dataset, agg_fn:str):
    """
    wrapper to get prediction from sklearn models, deepchem models, and model ensembles (as list)
    """
    def predict(model, dataset):
        if isinstance(model, Model):
            pred = model.predict(dataset)
        else:
            pred = model.predict_proba(dataset.X)
        pred = [x[1] for x in pred]
        return pred

    if isinstance(model, List):
        predictions = np.empty((len(model), dataset.y.shape[0]))
        if isinstance(model[0], Tuple):
            model = [x[0] for x in model]
        for ensemble_id, submodel in enumerate(model):
            internal_pred = predict(submodel, dataset)
            predictions[ensemble_id] = internal_pred
        
        if agg_fn == 'mean':
            pred = np.mean(predictions, axis=0)
            uncertainty = np.quantile(predictions, 0.75, axis = 0) - np.quantile(predictions, 0.25, axis = 0)
        elif agg_fn == 'median':
            pred = np.median(predictions, axis=0)
            uncertainty = np.std(predictions, axis=0)
        else:
            print('Not Implemented')
            return None
        return pred, uncertainty
    else:
        pred = predict(model, dataset)
        return pred
    
def _get_best_thresholds(
                        model,
                        dataset,
                        test_indices: List,
                        threshold_choices:List = [round(x*0.10, 2) for x in range(3, 8)], 
                        verbose=True):

    nSplit = len(test_indices)
    mcc_at_thresh = np.zeros((len(threshold_choices), nSplit))
    dataset_predictions = []

    for idx, test_idx in enumerate(test_indices):
        print(f'\tFold {idx} test result..')
        test_set = dataset.select(test_idx)
        pred = _get_model_prediction(model, test_set, agg_fn=np.median)
        dataset_predictions.extend(pred)

        for tidx, threshold in enumerate(threshold_choices):
            mcc_at_thresh[tidx][idx] = matthews_corrcoef(test_set.y, [int(x>=threshold) for x in pred])

    avg_mcc_at_thresh = np.mean(mcc_at_thresh, axis=1)
    best_threshold_idx = np.argmax(avg_mcc_at_thresh)
    if verbose:
        print("best threshold", threshold_choices[best_threshold_idx])
        print("average MCC @best threshold: ", avg_mcc_at_thresh[best_threshold_idx])
    return threshold_choices[best_threshold_idx], avg_mcc_at_thresh[best_threshold_idx], dataset_predictions

def check_result(model, dataset, agg_fn=np.median, thresh=None):
    print("is model an ensemble: ", isinstance(model, List))
    pred = _get_model_prediction(model, dataset, agg_fn=agg_fn)

    # auprc = auc(recall, precision)
    # print("AUPRC: ", auprc)
    precision, recall, thresholds = precision_recall_curve(dataset.y, pred)
    if thresh is not None:
        median_best_thresh = thresh
    elif not isinstance(model, List):
        best_PreRec_idx = np.argmax([x*y for x,y in zip(precision, recall)])
        median_best_thresh = thresholds[best_PreRec_idx]
    else:
        median_best_thresh = 0.5

    mcc50 = matthews_corrcoef(dataset.y, [int(x>=0.5) for x in pred])
    mccBest = matthews_corrcoef(dataset.y, [int(x>=median_best_thresh) for x in pred])
    roc_auc = roc_auc_score(dataset.y, pred)
    auprc = auc(recall, precision)
    conf_matrix = confusion_matrix(dataset.y, [int(x>=median_best_thresh) for x in pred])

    print("MCC score @0.5: ", mcc50)
    print(f"Best threshold MCC score @{median_best_thresh}: ", mccBest)
    print(f"confusion matrix @{median_best_thresh}: ", conf_matrix)
    print("ROC-AUC score: ", roc_auc)
    print("AUPRC score: ", auprc)
    return mcc50, mccBest, conf_matrix, roc_auc, auprc