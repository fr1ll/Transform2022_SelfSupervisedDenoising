"""Training utilities for blind-trace denoising."""

from __future__ import annotations

from typing import Iterable
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm


Batch = tuple[Tensor, Tensor, Tensor]


def n2v_train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    data_loader: Iterable[Batch] | DataLoader[Batch],
    device: torch.device,
    *,
    use_amp: bool = False,
    scaler: GradScaler | None = None,
) -> tuple[float, float]:
    """ Blind-spot network training function
    
    Parameters
    ----------
    model : torch model
        Neural network
    criterion : torch criterion
        Loss function 
    optimizer : torch optimizer
        Network optimiser
    data_loader : torch dataloader
        Premade data loader with training data batches
    device : torch device
        Device where training will occur (e.g., CPU or GPU)
    
    Returns
    -------
        loss : float
            Training loss across full dataset (i.e., all batches)
        accuracy : float
            Training RMSE accuracy across full dataset (i.e., all batches) 
    """
    
    model.train()
    accuracy = 0.0
    loss = 0.0

    for dl in tqdm(data_loader):
        # Load batch of data from data loader 
        X, y, mask = dl[0].to(device, non_blocking=True), dl[1].to(device, non_blocking=True), dl[2].to(device, non_blocking=True)

        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast(device_type='cuda'):
                yprob = model(X)
                ls = criterion(yprob * (1 - mask), y * (1 - mask))
            scaler.scale(ls).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Predict the denoised image based on current network weights
            yprob = model(X)
            # Compute loss function only at masked locations and backpropogate it
            ls = criterion(yprob * (1 - mask), y * (1 - mask))
            ls.backward()
            optimizer.step()

        # Retain training metrics (compute RMSE in torch)
        loss += float(ls.item())
        with torch.no_grad():
            rmse = torch.sqrt(torch.mean(((yprob.detach() - y) * (1 - mask)) ** 2)).item()
        accuracy += rmse
        
    # Divide cumulative training metrics by number of batches for training
    loss /= len(data_loader)
    accuracy /= len(data_loader)

    return float(loss), float(accuracy)


def n2v_evaluate(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable[Batch] | DataLoader[Batch],
    device: torch.device,
    *,
    use_amp: bool = False,
) -> tuple[float, float]:
    """ Blind-spot network evaluation function
    
    Parameters
    ----------
    model : torch model
        Neural network
    criterion : torch criterion
        Loss function 
    data_loader : torch dataloader
        Premade data loader with training data batches
    device : torch device
        Device where network computation will occur (e.g., CPU or GPU)
    
    Returns
    -------
        loss : float
            Validation loss across full dataset (i.e., all batches)
        accuracy : float
            Validation RMSE accuracy across full dataset (i.e., all batches) 
    """
    
    model.eval()
    accuracy = 0.0
    loss = 0.0

    for dl in tqdm(data_loader):
        
        # Load batch of data from data loader 
        X, y, mask = dl[0].to(device, non_blocking=True), dl[1].to(device, non_blocking=True), dl[2].to(device, non_blocking=True)
        
        with torch.no_grad():
            if use_amp:
                with autocast(device_type='cuda'):
                    yprob = model(X)
                    ls = criterion(yprob * (1 - mask), y * (1 - mask))
            else:
                yprob = model(X)
                ls = criterion(yprob * (1 - mask), y * (1 - mask))
            rmse = torch.sqrt(torch.mean(((yprob - y) * (1 - mask)) ** 2)).item()

        loss += float(ls.item())
        accuracy += rmse
        
    # Divide cumulative training metrics by number of batches for training
    loss /= len(data_loader)  
    accuracy /= len(data_loader)  

    return float(loss), float(accuracy)

