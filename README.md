# pneumoccusVS
A study on the best models for predicting Antibiotics that could be effective against drug-resistant Streptococcus pneumoniae

## Data Compilation
Dataset for this study is compiled from ChEMBL and PubChem. Details on how the data are processed, filtered, and formatted can be found in `data_prep`

## Models
Random Forest, ChemProp, and MolFormer were trained on this dataset. Details on model training, hyperparameter tuning, and evaluation can be found under `models`.
Trained models are stored in `model_garden`.

### ChemProp
The code mainly utilise DeepChem's implementation of ChemProp, the main script for training ChemProp is `run_chemprop.py` ans can be run with bash command
```
python run_chemprop.py --data_path "processed_datasets/10uM_FP_clustered__resistant_pneumococcus_augmented_dataset_v7.csv" --cluster_id_colname cluster_id --val_cluster_ids 0 --algo chemprop --model_savename_prefix ChemPropCE_seed1val0 --random_seed 1 --with_validation --chosen_cluster
```

for prediction, the script `eval_chemprop.py` can be run with bash command
```
python eval_chemprop.py --data_path "processed_datasets/10uM_FP_clustered__resistant_pneumococcus_augmented_dataset_v7.csv" --model_dirs ChemPropCE_seed1 --algo chemprop
```

### MolFormer
IBM's MolFormer-XL hosted on HuggingFace is used here through the `transformer` package and wrapped in a pytorch-lightning module.
The main script to train the model is `Lightning_MF_run.py`. It can be run with bash command
```
python Lightning_MF_run.py --data_path "../data/10uM_FP_clustered__resistant_pneumococcus_augmented_dataset_v7.csv" --cluster_id_colname cluster_id --nEpochs 5 --model_savename TunedCEMolFormerSeed5${PBS_ARRAY_INDEX} --random_seed 5 --is_frozen --chosen_cluster --val_cluster_ids 0 --batch_size 128 --with_validation --split_method LDMO
```
for prediction, the script `lightning_predict.py` can be run with bash command
```
python lightning_predict.py --output_path pred_result/top_algo_holdout_set_output_mf_template.csv --num_workers 4 --checkpoint_path model_checkpoints/TunedCEMolFormerSeed3Val --data_path ../../processed_datasets/top_algo_holdout_set_output_v7.csv
```
