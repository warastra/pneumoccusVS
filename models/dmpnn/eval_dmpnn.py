import pandas as pd
from rdkit import Chem

from deepchem.feat import DMPNNFeaturizer
from deepchem.data import NumpyDataset
from deepchem.models.torch_models import DMPNNModel
import torch

from predict_utils import _get_best_thresholds, check_result, _get_model_prediction
from losses import dcFocalLoss

import os, argparse, glob
from datetime import datetime
import numpy as np      
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_auc_score, auc, matthews_corrcoef, precision_score, recall_score




if __name__ == '__main__':
    print("start time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=r'..\processed_datasets\10uM_FP_clustered__resistant_pneumococcus_augmented_dataset_v7.csv', type=str, help='dataset file path')
    parser.add_argument('--cluster_id_colname', default='cluster_id',  type=str, help='name of the column containing cluster ID')
    parser.add_argument('--model_dirs', default='_',  type=str, help='filename prefix of the saved model')
    parser.add_argument('--random_seed', default=32,  type=int, help='torch random seed')
    parser.add_argument('--global_features', action='store_true',  type=str, help='whether to concatenate RDKIT molecular descriptors')
    parser.add_argument('--loss_fn', default='cross_entropy', nargs='?',  type=str, choices=['cross_entropy', 'focal'], help='choice of algorithms')
    parser.add_argument('--focal_alpha', default=0.25,  type=int, help='focal loss alpha, only applicable if loss_fn == "focal"')
    parser.add_argument('--focal_gamma', default=2,  type=int, help='focal loss gamma, only applicable if loss_fn == "focal"')
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--data_limit', default=None, type=int)
    parser.add_argument('--output_path', default=None, type=str)
    args = parser.parse_args()
    print("### run arguments ###")
    print(args)
    print("#####################")

    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    test_df = pd.read_csv(args.data_path)
    if args.data_limit is not None:
        test_df = test_df.iloc[:args.data_limit]
    # test_df = df[df[args.cluster_id_colname]==-1]
    
    if args.algo == 'dmpnn':
        mpnn_featurizer = DMPNNFeaturizer()
        global_feat_size = 0
    elif args.algo == 'chemprop':
        mpnn_featurizer = DMPNNFeaturizer(features_generators=['rdkit_desc_normalized']) # with molecular-level features
        global_feat_size = 200

    mols = [Chem.MolFromSmiles(smiles) for smiles in test_df["Smiles"]]
    features = mpnn_featurizer.featurize(mols)

    test_dataset = NumpyDataset(
        X=features, y=None, 
        ids=test_df["Smiles"]
    )

    if args.loss_fn == 'focal':
        loss_fn = dcFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        loss_fn=None

    model_files = glob.glob(os.path.join('model_garden', f'{args.model_dirs}*'))
    model_ensembles = []
    print("inference start time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    for model_dir in model_files:
        model = DMPNNModel(mode='classification', 
                                # learning_rate=5e-5,
                                n_classes=2, 
                                batch_size=256, 
                                # enc_dropout_p=0.4, 
                                ffn_dropout_p=0.3, 
                                ffn_hidden=1600,
                                depth=6,
                                enc_activation='gelu',
                                ffn_activation='gelu',
                                bias=True,
                                aggregation='sum',
                                global_features_size=global_feat_size,
                                # model_dir='model_garden/{args.algo}_{args.loss_fn}_s{args.random_seed}'
                                model_dir=model_dir
                            )
        model.restore()
        model_ensembles.append(model)
    
    
    # print('#### Model Architecture ####')
    # print(model)
    # print("loss fn: ", loss_fn or 'cross_entropy')
    # print('############################')
    
    cp_pred, cp_uncertainty = _get_model_prediction(model_ensembles, test_dataset, agg_fn='median')
    print("inference end time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print(args.model_dirs)
    print('\tMCC @0.5: ', matthews_corrcoef(test_df['final_activity_label'], [x>0.5 for x in cp_pred]))
    print('\tprecision: ', precision_score(test_df['final_activity_label'], [x>0.5 for x in cp_pred]))
    print('\trecall: ', recall_score(test_df['final_activity_label'], [x>0.5 for x in cp_pred]))
    p_precision, p_recall, thresholds = precision_recall_curve(test_df['final_activity_label'], [x for x in cp_pred])
    print('\tAUPRC: ', auc(p_recall, p_precision))

    
    if args.output_path is not None:
        test_df['cp_pred'] = cp_pred
        test_df['cp_uncertainty'] = cp_uncertainty
        test_df.to_csv(args.output_path, index=False)
    
    print("end time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # best_threshold, best_mcc, dataset_prediction = _get_best_thresholds(
    #                                             model,
    #                                             dataset=mpnn_dataset,
    #                                             test_indices=[train_idx],
    #                                         )

    # fold_test_mcc, test_bestMCC, conf_matrix, rocauc, auprc = 
    # check_result(model_ensembles, 
    #             test_dataset, 
    #             agg_fn=np.median, 
    #             thresh=0.5)
