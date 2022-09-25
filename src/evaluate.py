import torch
from torchvision.utils import save_image
from .dice_score import dicescore, sensitivity, specificity
from pathlib import Path
import numpy as np


def evaluate(net, dataloader, device, image_path : Path):
    net.eval()
    dice_score  = 0
    sensitivity_score = 0
    specificity_score = 0

    with torch.no_grad():
        for image, mask_true, path_name in dataloader : 
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.cpu().numpy()
            
            # predict the mask
            mask_pred = net(image).detach().cpu()
            mask_pred = torch.zeros(mask_pred.shape).scatter(1, mask_pred.argmax(1).unsqueeze(1), 1).numpy().astype(np.uint8)
            
            dice_score += dicescore(mask_pred, mask_true)
            sensitivity_score += sensitivity(mask_pred, mask_true)
            specificity_score += specificity(mask_pred, mask_true)

            for pred, path in zip(mask_pred, path_name) : 
                save_image(torch.FloatTensor(pred), image_path.joinpath(f"{path}.png"))
    dice_score = dice_score / len(dataloader)
    sensitivity_score = sensitivity_score / len(dataloader)
    specificity_score = specificity_score / len(dataloader)

    return dice_score, sensitivity_score, specificity_score