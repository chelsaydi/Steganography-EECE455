import torch
import torch.nn.functional as F

def psnr(mse, max_val=1.0):
    return 10 * torch.log10((max_val ** 2) / (mse + 1e-8))

def mse(img1, img2):
    return F.mse_loss(img1, img2)

def ber(bits_true, bits_pred_probs, threshold=0.5):
    pred = (bits_pred_probs >= threshold).float()
    return (pred != bits_true).float().mean()
