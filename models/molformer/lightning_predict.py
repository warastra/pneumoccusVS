import pandas as pd
import torch
import torch.nn as nn
import glob

from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

import os, argparse
import numpy as np
from typing import Any
from tqdm import tqdm
from functools import partial
from tqdm import tqdm

from dataset import MoleculeDataset, smilesCollate
from lightning_model import FFN, FFNMolFormer, FFNMolFormer_Global
from lightning.pytorch import Trainer, seed_everything
from deepchem.feat import RDKitDescriptors
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_auc_score, auc, matthews_corrcoef, precision_score, recall_score


device = 'cuda' if torch.cuda.is_available() else 'cpu'


MolFormerXL = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
MolFormerXL_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

def predictSmilesCollate(smiles, tokenizer, max_len=800, get_global_features=False):
    smiles_inputs = tokenizer(smiles, return_tensors="pt", padding='max_length', max_length=max_len)
    if get_global_features:
        featurizer = RDKitDescriptors(use_bcut2d=False, is_normalized=True)
        global_features = torch.tensor(featurizer.featurize(smiles)).to(torch.float32)
        global_features = torch.nan_to_num(global_features)
        return smiles_inputs, global_features
    
    return smiles_inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=r'../data/pneumococcus_conformer_dataset.csv', type=str, help='dataset file path')
    parser.add_argument('--batch_size', default=256,  type=int, help='train/test batch size')
    parser.add_argument('--with_global_features', action='store_true', help="whether to concat RDKIT normalized molecular features to transformer output")
    parser.add_argument('--random_seed', default=32,  type=int, help='torch random seed')
    parser.add_argument('--dev_run', action='store_true', help='run a debug test')
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--num_workers', default=4,  type=int, help='number of dataloader workers')
    parser.add_argument('--agg_fn', type=str, default='median', choices=['mean', 'median'], help='ensemble aggregation function')
    parser.add_argument('--output_path', default=None, type=str)
    args = parser.parse_args()
    print("### run arguments ###")
    print(args)
    print("#####################")

    df = pd.read_csv(args.data_path)
    smiles_collater = partial(predictSmilesCollate, tokenizer=MolFormerXL_tokenizer, max_len=800, get_global_features=args.with_global_features)
    seed_everything(args.random_seed)
    # torch.autograd.detect_anomaly(True)

    test_dataset = MoleculeDataset(df['Smiles'].values, label=None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                             collate_fn=smiles_collater, num_workers=args.num_workers)

    
    if args.with_global_features:
        model_fn = FFNMolFormer_Global
        ffn_hidden_dim = 1600
    else:
        model_fn = FFNMolFormer
        ffn_hidden_dim = 1536

    model_files = glob.glob(rf'{args.checkpoint_path}*')
    with torch.no_grad():
        ensemble_y_scores = []
        for model_path in tqdm(model_files):
            model = model_fn.load_from_checkpoint(model_path).to(device='cuda')
            model.eval()
            y_scores = []
            for batch in tqdm(test_loader, leave=False):
                batch = batch.to(device='cuda')
                output = model(**batch)
                # print(output.shape)
                y_scores.append(output.cpu())
            y_scores = torch.cat(y_scores, dim = 0)
            ensemble_y_scores.append(y_scores[:,1])
        
        ensemble_y_scores = torch.vstack(ensemble_y_scores)
        pred_prob = 1 / (1 + torch.exp(-ensemble_y_scores))
        if args.agg_fn == 'median':
            agg_y_scores = torch.median(pred_prob, dim=0).values
            uncertainty = torch.quantile(pred_prob, torch.tensor([0.25, 0.75]), dim=0)
            uncertainty = uncertainty[1,:] - uncertainty[0,:]
        elif args.agg_fn == 'mean':
            agg_y_scores = torch.mean(pred_prob, dim=0)
            # pred_prob = 1 / (1 + torch.exp(-agg_y_scores))
            uncertainty = torch.std(pred_prob, dim=0)
        else:
            print('please input a valid aggregation')

    df['prediction'] = agg_y_scores.detach().numpy()
    df['uncertainty'] = uncertainty.detach().numpy()
    print('\tMCC @top1%: ', matthews_corrcoef(df['final_activity_label'], [x>0.5 for x in df['prediction']]))
    print('\tprecision: ', precision_score(df['final_activity_label'], [x>0.5 for x in df['prediction']]))
    print('\trecall: ', recall_score(df['final_activity_label'], [x>0.5 for x in df['prediction']]))
    p_precision, p_recall, thresholds = precision_recall_curve(df['final_activity_label'], df['prediction'])
    print('\tAUPRC: ', auc(p_recall, p_precision))

    if args.output_path is not None:
        df[['Smiles', 'cluster_id', 'prediction', 'uncertainty']].to_csv(args.output_path, index=False)
    

