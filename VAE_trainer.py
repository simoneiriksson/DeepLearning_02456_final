from typing import List, Set, Dict, Tuple, Optional, Any
from utils.utils import cprint, StatusString
from collections import defaultdict
import torch, torch.nn as nn
import numpy as np

def VAE_trainer(models, validation_data, training_data, params, vi, train_loader, device, validation_loader, print_every=1, logfile=None):
    vae = models[0].to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    
    ######### VAE Training #########
    cprint("VAE Training", logfile)

    num_epochs = params['num_epochs']

    print_every = 1

    for epoch in range(num_epochs):
        training_epoch_data = defaultdict(list)
        _ = vae.train()
        for x, _ in train_loader:
            # batchwise normalization. Only to be used if imagewise normalization has been ocmmented out.
            # x = batch_normalize_images(x)
            x = x.to(device)
            # perform a forward pass through the model and compute the ELBO
            loss, diagnostics, outputs = vi(vae, x)
            optimizer.zero_grad()
            loss.backward()
            _ = nn.utils.clip_grad_norm_(vae.parameters(), 10_000)
            optimizer.step()
            for k, v in diagnostics.items():
                training_epoch_data[k] += list(v.cpu().data.numpy())

        for k, v in training_epoch_data.items():
            training_data[k] += [np.mean(training_epoch_data[k])]

        with torch.no_grad():
            _ = vae.eval()
            
            validation_epoch_data = defaultdict(list)
            
            for x, _ in validation_loader:
                # batchwise normalization. Only to be used if imagewise normalization has been ocmmented out.
                # x = batch_normalize_images(x)
                x = x.to(device)
                
                loss, diagnostics, outputs = vi(vae, x)
                
                for k, v in diagnostics.items():
                    validation_epoch_data[k] += list(v.cpu().data.numpy())
            
            for k, v in diagnostics.items():
                validation_data[k] += [np.mean(validation_epoch_data[k])]
                
            if epoch % print_every == 0:
                cprint(f"epoch: {epoch}/{num_epochs}", logfile)
                train_string = StatusString("training", training_epoch_data)
                evalString = StatusString("evaluation", validation_epoch_data)
                cprint(train_string, logfile)
                cprint(evalString, logfile)

        vae.update_()
        vi.update_vi()
    return validation_data, training_data, params, vae