import torch

class WeightedFocalLoss(torch.nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2, device='cuda', reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = torch.tensor(gamma).to(device)
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        at = self.alpha.gather(0, torch.argmax(targets, dim=1).type(torch.long))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        if self.reduction=='mean':
            F_loss = F_loss / inputs.shape[0]
        return F_loss