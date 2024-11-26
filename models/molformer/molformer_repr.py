from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import torch
import argparse
from functools import partial
from dataset import MoleculeDataset
from torch.utils.data import DataLoader

MolFormerXL_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
base = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)

def predictSmilesCollate(smiles, tokenizer, max_len=800):
    smiles_inputs = tokenizer(smiles, return_tensors="pt", padding='max_length', max_length=max_len)
    return smiles_inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=r'../data/wildset_drug_repurposing_hub.csv', type=str, help='dataset file path')
    parser.add_argument('--batch_size', default=256,  type=int, help='train/test batch size')
    parser.add_argument('--dev_run', action='store_true', help='run a debug test')
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--num_workers', default=4,  type=int, help='number of dataloader workers')
    parser.add_argument('--output_path', default=None, type=str)
    args = parser.parse_args()
    print("### run arguments ###")

    filename = args.data_path.split('/')[-1][:-4]
    df = pd.read_csv(args.data_path)
    smiles_collater = partial(predictSmilesCollate, tokenizer=MolFormerXL_tokenizer, max_len=800, get_global_features=args.with_global_features)
    # torch.autograd.detect_anomaly(True)

    test_dataset = MoleculeDataset(df['Smiles'].values, label=None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                             collate_fn=smiles_collater, num_workers=args.num_workers)

    with torch.no_grad():
        MFRepr = []
        for batch in test_loader:
            tmpMFRepr = base(**batch).pooler_output
            MFRepr.append(tmpMFRepr.cpu())
        
        MFRepr = torch.cat(MFRepr, dim = 0)
        MFRepr = MFRepr.numpy()
        if args.output_path is None:
            output_path = f'model_outputs/MFRepr_{filename}.npy'
        else:
            output_path = args.output_path
        np.save(output_path, MFRepr)
    