from lightning_model import FFNMolFormer
import lightning as L
from transformers import AutoModel, AutoTokenizer

from pathlib import Path
from typing import Dict, Any, List
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

