import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedLoss(nn.Module):
    def __init__(self, pos_weight=0.5, neg_weight=0.5):
        super(BalancedLoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        
    def forward(self, pred, target):
        pos_mask = (target > 0.5).float()
        neg_mask = (target <= 0.5).float()
        
        pos_num = pos_mask.sum().item()
        neg_num = neg_mask.sum().item()
        
        if pos_num == 0 or neg_num == 0:
            return F.mse_loss(pred, target)
        
        pos_loss = F.mse_loss(pred * pos_mask, target * pos_mask, reduction='sum') / pos_num
        neg_loss = F.mse_loss(pred * neg_mask, target * neg_mask, reduction='sum') / neg_num

        loss = self.pos_weight * pos_loss + self.neg_weight * neg_loss
        
        return loss