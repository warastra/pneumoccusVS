import pandas as pd
from rdkit import Chem

from deepchem.feat import DMPNNFeaturizer
from deepchem.data import NumpyDataset
from deepchem.models.torch_models import DMPNNModel
import torch

from predict_utils import _get_best_thresholds, check_result
from losses import dcFocalLoss
from sklearn.model_selection import train_test_split

import os, argparse, pickle
import numpy as np      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=r'..\processed_datasets\10uM_FP_clustered__resistant_pneumococcus_augmented_dataset_v7.csv', type=str, help='dataset file path')
    parser.add_argument('--cluster_id_colname', default='cluster_id',  type=str, help='name of the column containing cluster ID')
    parser.add_argument('--model_savename_prefix', default='_',  type=str, help='filename prefix of the saved model')
    parser.add_argument('--with_validation', action='store_true', help='train/test batch size')
    parser.add_argument('--val_cluster_ids', default=[0], nargs='+',type=int, help='cluster ids for validation set')
    parser.add_argument('--chosen_cluster_ids', default=[0, 2, 3, 12, 14, 25], nargs='+',type=int, help='cluster ids for validation set')
    parser.add_argument('--chosen_cluster', action='store_true', help='is using chosen cluster')
    parser.add_argument('--random_seed', default=32,  type=int, help='torch random seed')
    parser.add_argument('--nEpochs', default=10,  type=int, help='torch random seed')
    parser.add_argument('--batch_size', default=128,  type=int, help='torch random seed')
    parser.add_argument('--global_features', action='store_true',  type=str, help='whether to concatenate RDKIT molecular descriptors')
    parser.add_argument('--loss_fn', default='cross_entropy', nargs='?',  type=str, choices=['cross_entropy', 'focal'], help='choice of algorithms')
    parser.add_argument('--focal_alpha', default=25,  type=int, help='focal loss alpha as percentage, only applicable if loss_fn == "focal"')
    parser.add_argument('--focal_gamma', default=2,  type=int, help='focal loss gamma, only applicable if loss_fn == "focal"')
    parser.add_argument('--is_ensemble', action='store_true', help='whether to build ensemble models')
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--split_method', default='LDMO', choices=['random', 'LDMO'], help='method to split validation and training set, one of LDMO or random')
    args = parser.parse_args()
    print("### run arguments ###")
    print(args)
    print("#####################")

    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    df = pd.read_csv(args.data_path)
    
    if args.global_features :
        mpnn_featurizer = DMPNNFeaturizer(features_generators=['rdkit_desc_normalized']) # with molecular-level features
        global_feat_size = 200
        algo='global'
    else:
        mpnn_featurizer = DMPNNFeaturizer()
        global_feat_size = 0
        algo='noRDKITfeat'
    

    mols = [Chem.MolFromSmiles(smiles) for smiles in df["Smiles"]]
    features = mpnn_featurizer.featurize(mols)

    mpnn_dataset = NumpyDataset(
        X=features, y=df["final_activity_label"], 
        ids=df["Smiles"]
    )

    if args.loss_fn == 'focal':
        loss_fn = dcFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        loss_fn=None

    if args.train_all:
        test_fold = []
        df.loc[df[args.cluster_id_colname]==-1, args.cluster_id_colname] = 26
        train_val_df = df
    else:
        test_fold = [-1]
        test_idx = df[df[args.cluster_id_colname].isin(test_fold)].index.values
        train_val_df = df[~df.index.isin(test_idx)]
        test_dataset = mpnn_dataset.select(test_idx)

    if not args.with_validation:
        args.val_cluster_ids = []
    
    elif args.chosen_cluster:
        args.val_cluster_ids = [x for idx, x in enumerate(args.chosen_cluster_ids) if idx in args.val_cluster_ids]

    val_cluster_id = '_'.join([str(x) for x in args.val_cluster_ids])
    

    if args.with_validation:
        # val_idx = df[df[args.cluster_id_colname].isin(args.val_cluster_ids)].index.values
        val_idx = train_val_df[train_val_df[args.cluster_id_colname].isin(args.val_cluster_ids)].index.values
        if args.split_method == 'random':
            strat_idx = train_val_df['final_activity_label'].values
            tSize = len(val_idx)
            train_idx, val_idx = train_test_split(
                                    train_val_df.index.values, 
                                    stratify=strat_idx,
                                    test_size = tSize,
                                    random_state = args.val_cluster_ids[0]
                                )
        else:
            train_idx = [x for x in train_val_df.index.values if x not in val_idx]
        
        val_dataset = mpnn_dataset.select(val_idx)
        print('val_dataset label distribution: ', np.mean(val_dataset.y), ' of ', len(val_dataset.y))
    else:
        train_idx = train_val_df.index.values
    
    train_dataset = mpnn_dataset.select(train_idx)

    model = DMPNNModel(mode='classification', 
                            # learning_rate=5e-5,
                            n_classes=2, 
                            batch_size=args.batch_size, 
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
                            model_dir=os.path.join('model_garden', rf'{args.model_savename_prefix}{algo}_{args.loss_fn}_s{args.random_seed}')
                            )
    
    
    print('#### Model Architecture ####')
    print(model)
    print("loss fn: ", loss_fn or 'cross_entropy')
    print('############################')
    
    model.fit(train_dataset, loss=loss_fn, nb_epoch=args.nEpochs)

    # best_threshold, best_mcc, dataset_prediction = _get_best_thresholds(
    #                                             model,
    #                                             dataset=mpnn_dataset,
    #                                             test_indices=[train_idx],
    #                                         )

    # fold_test_mcc, test_bestMCC, conf_matrix, rocauc, auprc = 
    check_result(model, 
                test_dataset, 
                agg_fn=np.median, 
                thresh=0.5)
