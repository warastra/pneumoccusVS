import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

import os, argparse
import numpy as np
from typing import Any
from tqdm import tqdm
from functools import partial

from dataset import MoleculeDataset, smilesCollate
from lightning_model import FFN, FFNMolFormer, FFNMolFormer_Global
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'


MolFormerXL = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
MolFormerXL_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=r'../data/10uM_FP_clustered__resistant_pneumococcus_augmented_dataset_v7.csv', type=str, help='dataset file path')
    parser.add_argument('--cluster_id_colname', default='cluster_id',  type=str, help='name of the column containing cluster ID')
    parser.add_argument('--model_savename', default='GlobalMolFormer',  type=str, help='filename prefix of the saved model')
    parser.add_argument('--nEpochs', default=5,  type=int, help='NUmber of Epochs')
    parser.add_argument('--batch_size', default=256,  type=int, help='train/test batch size')
    parser.add_argument('--with_validation', action='store_true', help='train/test batch size')
    parser.add_argument('--val_cluster_ids', default=[0], nargs='+',type=int, help='cluster ids for validation set')
    parser.add_argument('--chosen_cluster_ids', default=[0, 2, 3, 12, 14, 25], nargs='+',type=int, help='cluster ids for validation set')
    parser.add_argument('--chosen_cluster', action='store_true', help='is using chosen cluster')
    parser.add_argument('--loss_fn', default='cross_entropy',  type=str, choices=['cross_entropy', 'focal'], help='loss function')
    parser.add_argument('--focal_alpha', default=0.25,  type=int, help='focal loss alpha, only applicable if loss_fn == "focal"')
    parser.add_argument('--focal_gamma', default=2,  type=int, help='focal loss gamma, only applicable if loss_fn == "focal"')
    parser.add_argument('--is_frozen', action='store_true', help='is MolFormer model weight frozen')
    parser.add_argument('--with_global_features', action='store_true', help="whether to concat RDKIT normalized molecular features to transformer output")
    parser.add_argument('--random_seed', default=32,  type=int, help='torch random seed')
    parser.add_argument('--dev_run', action='store_true', help='run a debug test')
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--num_workers', default=4,  type=int, help='number of dataloader workers')
    parser.add_argument('--train_all', action='store_true', help='is MolFormer model weight frozen')
    parser.add_argument('--eval', action='store_true', help='is predicting dataset')
    parser.add_argument('--split_method', default='LDMO', choices=['random', 'LDMO'], help='method to split validation and training set, one of LDMO or random')

    args = parser.parse_args()
    print("### run arguments ###")
    print(args)
    print("#####################")

    df = pd.read_csv(args.data_path)
    smiles_collater = partial(smilesCollate, tokenizer=MolFormerXL_tokenizer, max_len=800, get_global_features=args.with_global_features)
    seed_everything(args.random_seed)
    # torch.autograd.detect_anomaly(True)

    if args.with_global_features:
        model_fn = FFNMolFormer_Global
        ffn_hidden_dim = 1600
    else:
        model_fn = FFNMolFormer
        ffn_hidden_dim = 1536
    
    model = model_fn(n_layers=6, hidden_dim=ffn_hidden_dim, is_frozen=args.is_frozen, dropout=0.4)
    if args.train_all:
        test_fold = []
        args.chosen_cluster_ids = args.chosen_cluster_ids + [-1]
        train_val_df = df
    else:
        test_fold = [-1]
        test_idx = df[df[args.cluster_id_colname].isin(test_fold)].index.values
        train_val_df = df[~df.index.isin(test_idx)]

        test_dataset = MoleculeDataset(df.loc[test_idx]['Smiles'].values, df.loc[test_idx]['final_activity_label'].values)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=smiles_collater)

    if not args.with_validation:
        args.val_cluster_ids = []
    
    elif args.chosen_cluster:
        args.val_cluster_ids = [x for idx, x in enumerate(args.chosen_cluster_ids) if idx in args.val_cluster_ids]
    
    # print('#### Model Architecture ####')
    # print(model)
    # print('trainable params count: ', count_parameters(model))
    # print('############################')

    val_clusters_str = '_'.join([str(x) for x in args.val_cluster_ids])
    checkpoint_callback = ModelCheckpoint(
                            # dirpath=f"lightning_checkpoint\{args.model_savename}", 
                            filename=f"{args.model_savename}"+"_{epoch}",
                            save_top_k=1, 
                            save_on_train_epoch_end=True,
                            monitor='val_auprc', 
                            mode='max',
                            verbose=True)
    trainer = Trainer(
                default_root_dir="lightning_checkpoint", 
                max_epochs=args.nEpochs, 
                callbacks=[checkpoint_callback], 
                fast_dev_run=args.dev_run)
    
    print('training start...')
    if args.with_validation:
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
        
        val_dataset = MoleculeDataset(df.loc[val_idx]['Smiles'].values, df.loc[val_idx]['final_activity_label'].values)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=smiles_collater, num_workers=8)
    else:
        val_loader = test_loader
        train_idx = train_val_df.index.values

    print('Setting up train loader..')
    train_dataset = MoleculeDataset(df.loc[train_idx]['Smiles'].values, df.loc[train_idx]['final_activity_label'].values)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=smiles_collater, num_workers=8)
    print('Done\n')

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint_path)
    print('training finished.')
    trainer.save_checkpoint(os.path.join("model_checkpoints", f"{args.model_savename}.ckpt"))
    
    if not args.train_all:
        trainer.validate(model, dataloaders=val_loader)
    else:
        trainer.validate(model, dataloaders=test_loader)

