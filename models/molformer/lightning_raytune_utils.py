import torch
import torch.nn as nn
from lightning_model import FFNMolFormer
import lightning as L
from transformers import AutoModel, AutoTokenizer
from train_test import train as train_model, eval

from pathlib import Path
from typing import Dict, Any, List
import tempfile

# from ray import train
# from ray.train import Checkpoint, get_checkpoint
# import ray.cloudpickle as pickle
import ray.train.lightning
from ray.train.torch import TorchTrainer

MolFormerXL = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
MolFormerXL_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
                                                      
def train_fn(config:Dict, train_loader, val_loader=None, nEpochs=10):
    model = FFNMolFormer( 
                            is_frozen=True,
                            hidden_dim=config['hidden_dim'],
                            n_layers=config['n_layers'],
                            dropout=config['dropout'],
                        #  activation=nn.GELU(),
                        #  loss_fn=criterion
                        )
    trainer = L.Trainer(
        max_epochs=nEpochs,
        devices="auto",
        accelerator="auto",
        strategy=ray.train.lightning.RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        callbacks=[ray.train.lightning.RayTrainReportCallback()],
        # [1a] Optionally, disable the default checkpointing behavior
        # in favor of the `RayTrainReportCallback` above.
        enable_checkpointing=True,
    )

    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# def test_fn(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu", criterion=nn.CrossEntropyLoss(reduction = 'none')):
#     trainer.test
#     test_result = eval(model, device, test_loader, criterion=criterion)
#     report = {"loss": test_result['loss'], "auprc": test_result['AUPRC'], "mcc50":test_result['mcc50']}
#     return report, test_result['prediction']