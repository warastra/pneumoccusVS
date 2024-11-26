from transformers import AutoModel, AutoTokenizer
from lightning_model import FFN, FFNMolFormer
from mcts_utils import mcts, mcts_rollout, mf_scoring_function
import pandas as pd
import numpy as np
import argparse
from functools import partial
from dataset import MoleculeDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=r'../pred_result/drug_repurposing_hub__tuned_Molformer_result.csv', type=str, help='dataset file path')
    parser.add_argument('--pred_colname', default='prediction', type=str, help='column name containing prediction score')
    # parser.add_argument('--batch_size', default=256,  type=int, help='train/test batch size')
    parser.add_argument('--dev_run', action='store_true', help='run a debug test')
    parser.add_argument('--checkpoint_path', default=None, type=str)
    # parser.add_argument('--num_workers', default=4,  type=int, help='number of dataloader workers')
    parser.add_argument('--min_mscore', default=0.5,  type=float, help='minimum prediction score for molecules to be searched')
    parser.add_argument('--min_rscore', default=0.2,  type=float, help='minimum prediction score for substructure to be accepted as rationale')
    parser.add_argument('--rollout', default=10,  type=int, help='The number of MCTS rollouts to perform')
    parser.add_argument('--max_atoms', default=20,  type=float, help='The maximum number of atoms allowed in an extracted rationale')
    parser.add_argument('--min_atoms', default=8,  type=float, help='The minimum number of atoms allowed in an extracted rationale')
    parser.add_argument('--num_rationales_to_keep', default=5,  type=float, help='number of rationales to keep for each molecule')
    parser.add_argument('--output_path', default=None, type=str)
    args = parser.parse_args()
    print("### run arguments ###")


    filename = args.data_path.split('/')[-1][:-4]
    df = pd.read_csv(args.data_path)
    df = df[df[args.pred_colname]>=args.min_mscore].reset_index(drop=True)
    # torch.autograd.detect_anomaly(True)
    
    model_files = glob.glob(rf'{args.checkpoint_path}*')
    trained_models = [FFNMolFormer.load_from_checkpoint(path).to(device='cuda') for path in model_files]
    scoring_func = partial(mf_scoring_function, trainedMF=trained_models)

    ### Running Monte Carlo Tree Search ###
    property_for_interpretation = 'antibiotic'
    results_df = {"smiles": [], property_for_interpretation: []}

    for i in range(args.num_rationales_to_keep):
        results_df[f"rationale_{i}"] = []
        results_df[f"rationale_{i}_score"] = []

    for i, smiles in tqdm(enumerate(df.Smiles.values.tolist()), leave=False):
        # print([smiles])
        # score = scoring_func([smiles])[0]
        # if score > args.min_mscore:
        rationales = mcts(
            smiles=smiles,
            scoring_function=scoring_func,
            n_rollout=args.rollout,
            max_atoms=args.max_atoms,
            prop_delta=args.min_rscore,
            min_atoms=args.min_atoms,
            # c_puct=c_puct,
        )
        # else:
        #     rationales = []

        results_df["smiles"].append(smiles)
        results_df[property_for_interpretation].append(df[args.pred_colname].iloc[i])

        if len(rationales) == 0:
            for i in range(args.num_rationales_to_keep):
                results_df[f"rationale_{i}"].append(None)
                results_df[f"rationale_{i}_score"].append(None)
        else:
            min_size = min(len(x.atoms) for x in rationales)
            min_rationales = [x for x in rationales if len(x.atoms) == min_size]
            rats = sorted(min_rationales, key=lambda x: x.P, reverse=True)

            for i in range(args.num_rationales_to_keep):
                if i < len(rats):
                    results_df[f"rationale_{i}"].append(rats[i].smiles)
                    results_df[f"rationale_{i}_score"].append(rats[i].P)
                else:
                    results_df[f"rationale_{i}"].append(None)
                    results_df[f"rationale_{i}_score"].append(None) 
    
    ResDF = pd.DataFrame(results_df)
    ResDF.to_csv(f'model_outputs/mcts_{filename}.csv', index=False)