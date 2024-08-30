import torch 
import torch.nn as nn 
from typing import Any, Mapping
import lightning as L
# from torchmetrics.classification import PrecisionRecallCurve
# from torchmetrics.functional import auc
from torcheval.metrics.functional import binary_auprc
from torchmetrics.functional.classification import binary_matthews_corrcoef
from transformers import AutoModel, AutoTokenizer
# from losses import WeightedFocalLoss

# torch.autograd.detect_anomaly(True)

class FFN(nn.Module):
    def __init__(
            self,
            input_dim = 768,
            hidden_dim = 1024,
            output_dim = 2,
            n_layers:int = 1,
            dropout:float = 0.3,
            activation:nn.modules.activation = nn.GELU(),
    ):
        super(FFN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = activation

        if n_layers == 1:
            self.linears: Any = [nn.Linear(input_dim, output_dim)]

        else:
            self.linears = [nn.Linear(input_dim, hidden_dim)] + [
                nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)
            ] + [nn.Linear(hidden_dim, output_dim)]

        self.linears = nn.ModuleList(self.linears)
        dropout_layer = nn.Dropout(dropout)
        self.dropout_p = nn.ModuleList([dropout_layer for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_layers == 1:
            return self.dropout_p[0](self.activation(self.linears[0](x)))
        else:
            # x = self.dropout_p[0](x)    # no activation at input
            for i in range(self.n_layers - 1):
                x = self.dropout_p[i](self.activation(self.linears[i](x)))
            return self.linears[-1](x)
        
class FFNMolFormer(L.LightningModule):
    def __init__(
                self,
                is_frozen=True,
                hidden_dim = 1024,
                output_dim = 2,
                n_layers:int = 1,
                dropout:float = 0.3,
                # activation:nn.modules.activation = nn.GELU(),
                # loss_fn = nn.CrossEntropyLoss(reduction = 'mean')
            ):
        super(FFNMolFormer, self).__init__()
        self.base = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
        self.ffn = FFN(
                input_dim=768,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                n_layers=n_layers,
                dropout=dropout,
                activation=nn.GELU()
            )
        self.loss_fn = nn.CrossEntropyLoss(reduction = 'mean')
        self.save_hyperparameters()
        self.y_true = []
        self.eval_y_scores = []
        self.eval_loss = []

        # Freeze MolFormer params
        if is_frozen:
            for param in self.base.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask).pooler_output
        outputs = self.ffn(outputs)
        return outputs
    
    def training_step(self, batch) -> torch.Tensor:
        X, labels = batch
        pred = self.forward(**X)
        y = labels.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        # is_valid = ~torch.isnan(torch.sum(y, 1))
        #Loss matrix
        loss = self.loss_fn(pred.double(), y)
        #loss matrix after removing null target
        # loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.dtype))
        # loss = torch.sum(loss_mat)/torch.sum(is_valid)
        auprc = binary_auprc(pred[:,1], y[:,1])
        self.log("train_auprc", auprc)
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=0.0001, weight_decay=0.1)
        return optim
    
    def validation_step(self, val_batch) -> torch.Tensor:
        X, labels = val_batch
        pred = self.forward(**X)
        y = labels.view(pred.shape).to(torch.float64)

        self.y_true.append(labels.view(pred.shape))
        self.eval_y_scores.append(pred)

        #Whether y is non-null or not.
        is_valid = ~torch.isnan(torch.sum(y, 1))
        #Loss matrix
        loss_mat = self.loss_fn(pred.to(torch.float32), y)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.dtype))
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        self.eval_loss.append(loss)
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        self.log("val_loss", avg_loss, sync_dist=True)
        y_true = torch.cat(self.y_true, dim = 0)
        y_scores = torch.cat(self.eval_y_scores, dim = 0)
        auprc = binary_auprc(y_scores[:,1], y_true[:,1])
        self.log("val_auprc", auprc, sync_dist=True)
        mcc = binary_matthews_corrcoef(y_scores[:,1], y_true[:,1], threshold=0.5)
        self.log("val_mcc50", mcc, sync_dist=True)
        
        self.eval_loss.clear()
        self.y_true.clear()
        self.eval_y_scores.clear()
    
class FFNMolFormer_Global(FFNMolFormer):
    def __init__(
                self, 
                is_frozen=True,
                hidden_dim = 1024,
                output_dim = 2,
                n_layers:int = 1,
                dropout:float = 0.3,
                # loss_fn = 'cross_entropy'
                # activation:nn.modules.activation = nn.GELU(),
                # loss_fn = nn.CrossEntropyLoss(reduction = 'mean')
            ):
        super(FFNMolFormer_Global, self).__init__(
                is_frozen, hidden_dim, output_dim, n_layers, dropout,
            )
        # self.base = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
        self.ffn = FFN(
                input_dim=768+200,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                n_layers=n_layers,
                dropout=dropout,
                activation=nn.GELU()
            )
        # if loss_fn=="focal":
        #     self.loss_fn = WeightedFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, device=device)
        # else:
        # self.loss_fn = nn.CrossEntropyLoss(reduction = 'mean')
        # self.save_hyperparameters()
        # self.y_true = []
        # self.eval_y_scores = []
        # self.eval_loss = []
        

        # Freeze MolFormer params
        if is_frozen:
            for param in self.base.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, global_features):
        outputs = self.base(input_ids, attention_mask).pooler_output
        outputs = torch.cat((outputs, global_features), dim=1)
        outputs = nn.functional.layer_norm(outputs, normalized_shape=outputs.shape)
        outputs = self.ffn(outputs)
        return outputs
    
    # def training_step(self, batch) -> torch.Tensor:
    #     X, global_features, labels = batch
    #     mf_output = self.base(X.input_ids, X.attention_mask).pooler_output
    #     mf_output = torch.cat((mf_output, global_features), dim=1)
    #     mf_output = nn.functional.layer_norm(mf_output, normalized_shape=mf_output.shape)
    #     pred = self.ffn(mf_output)
    #     y = labels.view(pred.shape).to(torch.float32)

    #     #Whether y is non-null or not.
    #     # is_valid = ~torch.isnan(torch.sum(y, 1))
    #     #Loss matrix
    #     loss = self.loss_fn(pred.to(torch.float32), y)
    #     print(loss, loss.type())
    #     # #loss matrix after removing null target
    #     # loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.dtype).to(device))
    #     # loss = torch.sum(loss_mat)/torch.sum(is_valid)
    #     return loss
    
    # def configure_optimizers(self):
    #     optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=0.0001, weight_decay=0.1)
    #     return optim
    
    # def validation_step(self, val_batch) -> torch.Tensor:
    #     X, global_features, labels = val_batch
    #     mf_output = self.base(X.input_ids, X.attention_mask).pooler_output
    #     mf_output = torch.cat((mf_output, global_features), dim=1)
    #     mf_output = nn.functional.layer_norm(mf_output, normalized_shape=mf_output.shape)
    #     pred = self.ffn(mf_output)
    #     y = labels.view(pred.shape).to(torch.float32)

    #     self.y_true.append(labels.view(pred.shape))
    #     self.eval_y_scores.append(pred)

    #     #Whether y is non-null or not.
    #     is_valid = ~torch.isnan(torch.sum(y, 1))
    #     #Loss matrix
    #     loss_mat = self.loss_fn(pred.to(torch.float32), y)
    #     #loss matrix after removing null target
    #     loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.dtype))
    #     loss = torch.sum(loss_mat)/torch.sum(is_valid)
    #     self.eval_loss.append(loss)
    #     # self.log("val_loss: ", loss)

    #     # # pr_curve = PrecisionRecallCurve(num_classes=2)  
    #     # # precision, recall, thresholds = pr_curve(pred, y)
    #     # # auprc = auc(recall, precision)
    #     # auprc = binary_auprc(pred[:,1], y[:,1])
    #     # self.log("val_auprc: ", auprc)
    
    # def on_validation_epoch_end(self):
    #     avg_loss = torch.stack(self.eval_loss).mean()
    #     self.log("val_loss", avg_loss)
    #     y_true = torch.cat(self.y_true, dim = 0)
    #     y_scores = torch.cat(self.eval_y_scores, dim = 0)
    #     auprc = binary_auprc(y_scores[:,1], y_true[:,1])
    #     self.log("val_auprc", auprc)
        
    #     self.eval_loss.clear()
    #     self.y_true.clear()
    #     self.eval_y_scores.clear()