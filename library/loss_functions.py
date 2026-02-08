import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # Tensor of weights (e.g. for class imbalance)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: Logits [Batch, Num_Classes]
        targets: Class Indices [Batch]
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class QuantileLoss(nn.Module):
    """
    Quantile Loss (Pinball Loss) for Probabilistic Forecasting.
    Loss = max(q * (y - y_pred), (q - 1) * (y - y_pred))
    Input:
        preds: (B, num_quantiles) or (B, num_quantiles, ...)
        target: (B, 1) or (B, ...) - Target is same for all quantiles
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        # Ensure target matches batch dim
        # Reshape target to (B, 1) if needed to broadcast
        if target.dim() == 1:
            target = target.view(-1, 1)
            
        loss = 0
        for i, q in enumerate(self.quantiles):
            error = target - preds[:, i:i+1] 
            loss += torch.max((q-1)*error, q*error)
            
        return torch.mean(loss)

class HuberLoss(nn.Module):
    """
    Robust Regression Loss (MSE near 0, MAE far from 0).
    """
    def __init__(self, delta=1.0):
        super().__init__()
        self.loss_fn = nn.HuberLoss(delta=delta)
    
    def forward(self, pred, target):
        return self.loss_fn(pred, target)

class DirectionalLoss(nn.Module):
    """
    Loss = (1 - lambda) * MSE + lambda * DirectionalPenalty
    DirectionalPenalty = Mean(Relu(-sign(y_true) * sign(y_pred))) 
    Or simply: Mean(1 if sign mismatch else 0) * Magnitude?
    
    Standard approach:
    Loss = MSE + lambda * (1 if signs differ else 0) * |y_true - y_pred|
    """
    def __init__(self, lambda_dir=0.5, base_loss=nn.MSELoss()):
        super().__init__()
        self.lambda_dir = lambda_dir
        self.base_loss = base_loss
        
    def forward(self, pred, target):
        # Base Loss (Magnitude)
        base = self.base_loss(pred, target)
        
        # Directional Penalty
        # sign(pred) * sign(target) < 0 implies mismatch
        # We want to be differentiable. 
        # Tanh is a soft sign.
        # But let's use hard sign for logic, magnitude for gradient?
        # A common simple directional loss:
        # Loss = weight * MSE - (1-weight) * Correlation?
        
        # Here: Penalty proportional to error magnitude when signs differ.
        true_sign = torch.sign(target)
        pred_sign = torch.sign(pred)
        
        # Mismatch mask: where signs differ
        # sign * sign = -1 if differ, 1 if same (ignoring 0)
        mismatch = (true_sign * pred_sign) < 0
        
        # Penalty: |y - y_hat| where mismatch
        penalty = torch.mean(torch.abs(target - pred)[mismatch])
        if torch.isnan(penalty): penalty = 0.0 # if no mismatch
        
        return (1 - self.lambda_dir) * base + self.lambda_dir * penalty
