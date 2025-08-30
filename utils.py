# utils.py
# utils.py
import torch
import torch.nn as nn
import numpy as np

# PyTorch-based CCCLoss for Training (Back Prop Available)
class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, y_pred, y_true):
        mean_pred, mean_true = torch.mean(y_pred), torch.mean(y_true)   
        covariance = torch.mean((y_pred - mean_pred) * (y_true - mean_true))

        var_pred, var_true = torch.var(y_pred, correction=1), torch.var(y_true, correction=1)
        
        numerator = 2 * covariance
        denominator = var_true + var_pred + (mean_true - mean_pred)**2
        CCC = numerator / denominator
        
        return 1.0 - CCC # CCC Loss

# NumPy-based Evaluation Function for Validation/Evaluation
#[Source] https://github.com/AudioVisualEmotionChallenge/AVEC2018/blob/master/scripts_CES/calc_scores.py
def calculate_metrics_numpy(y_pred, y_true):
    # Computes the metrics CCC, PCC, and RMSE between the sequences x and y
    
    mean_pred, mean_true = np.nanmean(y_pred), np.nanmean(y_true)
    covariance = np.nanmean((y_pred - mean_pred) * (y_true - mean_true))
    
    # Sample Variance (ddof=1)
    var_pred, var_true = np.nanvar(y_pred, ddof=1), np.nanvar(y_true, ddof=1)
    
    numerator = 2 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    CCC = numerator / denominator
    
    std_pred, std_true = np.sqrt(var_pred), np.sqrt(var_true)
    PCC = covariance / (std_pred * std_true)
    RMSE = np.sqrt(np.nanmean((y_pred - y_true)**2))
    
    return CCC, PCC, RMSE