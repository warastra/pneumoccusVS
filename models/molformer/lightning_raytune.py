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
from pathlib import Path

from losses import WeightedFocalLoss
from train_test import train, eval
from dataset import MoleculeDataset, smilesCollate
from lightning_raytune_utils import train_fn
from lightning_model import FFNMolFormer
import ray
from ray import tune
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from ray.train.torch import TorchTrainer


device = 'cuda' if torch.cuda.is_available() else 'cpu'


MolFormerXL = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
MolFormerXL_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

def main(args):
    df = pd.read_csv(args.data_path)
    test_fold = -1
    smiles_collater = partial(smilesCollate, tokenizer=MolFormerXL_tokenizer, max_len=800)

    if not args.with_validation:
        args.val_cluster_ids = []
    elif args.chosen_cluster:
        args.val_cluster_ids = [args.chosen_cluster_ids[args.val_cluster_ids]]
    # val_clusters_str = '_'.join([str(x) for x in args.val_cluster_ids])

    train_dataset = MoleculeDataset(df[~df[args.cluster_id_colname].isin([test_fold]+ args.val_cluster_ids)]['Smiles'].values, 
                                        df[~df[args.cluster_id_colname].isin([test_fold]+ args.val_cluster_ids)]['final_activity_label'].values)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=smiles_collater, num_workers=2)


    val_dataset = MoleculeDataset(df[df[args.cluster_id_colname].isin(args.val_cluster_ids)]['Smiles'].values, 
                                df[df[args.cluster_id_colname].isin(args.val_cluster_ids)]['final_activity_label'].values)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=smiles_collater, num_workers=2)

    
    # test_dataset = MoleculeDataset(df[df[args.cluster_id_colname]==test_fold]['Smiles'].values, 
    #                             df[df[args.cluster_id_colname]==test_fold]['final_activity_label'].values)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=smiles_collater)

    if args.loss_fn=="focal":
        criterion = WeightedFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, device=device)
    else:
        criterion = nn.CrossEntropyLoss(reduction = 'none')

    raytune_config = {
            # 'is_frozen':args.is_frozen,
            'hidden_dim':tune.choice([768*(2**x) for x in range(0,3)]),
            # 'hidden_dim':tune.choice([768, 1536]),
            'n_layers':tune.choice([x*2 for x in range(1,4)]),
            # 'activation':tune.choice([nn.ReLU(), nn.GELU(), nn.SiLU()]),
            # 'activation':nn.GELU()
            # 'dropout':tune.choice([x*0.2 for x in range(0,3)])
            'dropout':tune.choice([0.2, 0.4])
    }
    # raytune_config = {
    #         # 'is_frozen':args.is_frozen,
    #         'hidden_dim':tune.grid_search([768]),
    #         'n_layers':tune.grid_search([x*2 for x in range(1,4)]),
    #         # 'activation':tune.choice([nn.ReLU(), nn.GELU(), nn.SiLU()]),
    #         # 'activation':nn.GELU()
    #         'dropout':tune.grid_search([0.4])
    # }
    scheduler = ASHAScheduler(
        metric="val_auprc",
        mode="max",
        max_t=args.nEpochs,
        grace_period=1,
        reduction_factor=2,
    )
    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 8, "GPU": 1}
    )
    run_config = RunConfig(
                            name='MolFormer_htune',
                            storage_path="~/WSL_DigiChem/Pneumococcus/molformer/ray_result",
                            log_to_file='molformer_ray.log',
                            checkpoint_config=CheckpointConfig(
                                num_to_keep=2,
                                checkpoint_score_attribute="val_auprc",
                                checkpoint_score_order="max"
                            )
    )
    print('hyperparam tuning start...')
    ray_trainer = TorchTrainer(
        partial(train_fn, 
                train_loader=train_loader,
                val_loader = val_loader,
                ),
        scaling_config=scaling_config,
        # run_config=run_config,
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": raytune_config},
        tune_config=tune.TuneConfig(num_samples=args.num_samples, scheduler=scheduler),
        # tune_config=tune.TuneConfig(scheduler=scheduler),
        run_config=run_config
    )
    resultGrid = tuner.fit()
    print('hyperparam tuning Done')

    best_trial = resultGrid.get_best_result(metric='val_auprc', mode='max')
    # best_trial = result.get_best_trial("auprc", "max", "last")
    # best_logdir = result.best_logdir("auprc", "max")
    print(f"Best trial config: {best_trial.config}")
    # print(f"Best trial logdir:  {best_logdir}")
    # print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    # print(f"Best trial final validation auprc: {best_trial.last_result['auprc']}")
    print(f"Best trial final validation loss: {best_trial.metrics['val_loss']}")
    print(f"Best trial final validation auprc: {best_trial.metrics['val_auprc']}")



    # best_trained_model = FFNMolFormer(
    #                     base=MolFormerXL, 
    #                     is_frozen=True,
    #                     hidden_dim=best_trial.config['hidden_dim'],
    #                     n_layers=best_trial.config['n_layers'],
    #                     dropout=best_trial.config['dropout'],
    #                     activation=nn.GELU()
    #                 )
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if args.gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    # # best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
    # best_checkpoint = best_trial.checkpoint
    # with best_checkpoint.as_directory() as checkpoint_dir:
    #     data_path = Path(checkpoint_dir) / "data.pkl"
    #     with open(data_path, "rb") as fp:
    #         best_checkpoint_data = pickle.load(fp)

    #     best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
    #     test_results, predictions = raytune_test(best_trained_model, test_loader, device, criterion=criterion)
    #     print("Best trial test set eval: ", test_results)
    
    # print("results dataframe for val cide {val_clusters_str}: ")
    # # print(result.results_df)
    # print(resultGrid.get_dataframe())
    # print("#####################")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=r'../data/10uM_FP_clustered__resistant_pneumococcus_augmented_dataset_v7.csv', type=str, help='dataset file path')
    parser.add_argument('--cluster_id_colname', default='cluster_id',  type=str, help='name of the column containing cluster ID')
    parser.add_argument('--model_savename', default='PneumococcusMolFormer',  type=str, help='filename prefix of the saved model')
    parser.add_argument('--nEpochs', default=10,  type=int, help='NUmber of Epochs')
    parser.add_argument('--batch_size', default=32,  type=int, help='train/test batch size')
    parser.add_argument('--with_validation', action='store_true', help='train/test batch size')
    parser.add_argument('--val_cluster_ids', default=0, nargs='?',type=int, help='cluster ids for validation set')
    parser.add_argument('--chosen_cluster_ids', default=[0, 2, 3, 12, 14, 25], nargs='+',type=int, help='cluster ids for validation set')
    parser.add_argument('--chosen_cluster', action='store_true', help='is using chosen cluster')
    parser.add_argument('--loss_fn', default='cross_entropy',  type=str, choices=['cross_entropy', 'focal'], help='loss function')
    parser.add_argument('--focal_alpha', default=0.25,  type=int, help='focal loss alpha, only applicable if loss_fn == "focal"')
    parser.add_argument('--focal_gamma', default=2,  type=int, help='focal loss gamma, only applicable if loss_fn == "focal"')
    parser.add_argument('--is_frozen', action='store_true', help='is MolFormer model weight frozen')
    parser.add_argument('--random_seed', default=32,  type=int, help='torch random seed')
    parser.add_argument('--num_samples', default=20,  type=int, help="number of sampled hyperparam configs")
    parser.add_argument('--gpus_per_trial', default=1,  type=int, help="number of GPUs per trial")
    args = parser.parse_args()
    print("### run arguments ###")
    print(args)
    print("#####################")
    torch.manual_seed(args.random_seed)
    torch.set_float32_matmul_precision('medium')
    main(args)

    

    

    
    