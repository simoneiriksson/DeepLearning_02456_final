from typing import List, Set, Dict, Tuple, Optional, Any
from utils.utils import cprint, StatusString, DiscStatusString
from collections import defaultdict
import torch, torch.nn as nn
import numpy as np

def VAEGAN_trainer(models, validation_data, training_data, params, vi, train_loader, device, validation_loader, print_every=1, logfile=None):
    VAE = models[0].to(device)
    DISCmodel = models[1].to(device)

    VAE_optimizer = torch.optim.Adam(VAE.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    DISCmodel_optimizer = torch.optim.Adam(DISCmodel.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    
    ######### VAE Training #########
    cprint("VAE Training", logfile)

    num_epochs = params['num_epochs']

    print_every = 1

    for epoch in range(num_epochs):
        training_epoch_data = defaultdict(list)
        disc_training_epoch_data = defaultdict(list)
        disc_data = defaultdict(list)

        _ = VAE.train()
        _ = DISCmodel.train()
        for x, _ in train_loader:
            x = x.to(device)
            losses_mean, losses, outputs = vi(VAE, DISCmodel, x)

            # unfolding losses:
            image_loss = losses_mean['image_loss']
            kl_div = losses_mean['kl_div']
            disc_loss = losses_mean['disc_loss']
            disc_repr_loss = losses_mean['disc_repr_loss']

            loss_VAE = disc_repr_loss + image_loss + kl_div * 1.0

            VAE_optimizer.zero_grad()
            loss_VAE.backward()
            _ = nn.utils.clip_grad_norm_(VAE.parameters(), 1_000)
            VAE_optimizer.step()

            loss_discriminator = disc_loss

            DISCmodel_optimizer.zero_grad()    
                
            loss_discriminator.backward()
            #_ = nn.utils.clip_grad_norm_(DISCmodel.parameters(), 1_000)

            DISCmodel_optimizer.step()

            for k, v in losses.items():
                training_epoch_data[k] += list(v.cpu().data.numpy())
            disc_data['disc_false_negatives'] = (1 - outputs['disc_real_pred'])
            disc_data['disc_true_positives'] = outputs['disc_real_pred']
            disc_data['disc_true_negatives'] = (1 - outputs['disc_fake_pred'])
            disc_data['disc_false_positives'] = outputs['disc_fake_pred']
            for k, v in disc_data.items():
                disc_training_epoch_data[k] += list(v.cpu().data.numpy())

        for k, v in training_epoch_data.items():
            training_data[k] += [np.mean(training_epoch_data[k])]

        with torch.no_grad():
            validation_epoch_data = defaultdict(list)
            disc_validation_epoch_data = defaultdict(list)
            _ = VAE.eval()
            _ = DISCmodel.eval()        
            for x, _ in validation_loader:
                # batchwise normalization. Only to be used if imagewise normalization has been ocmmented out.
                # x = batch_normalize_images(x)
                x = x.to(device)
                losses_mean, losses, outputs = vi(VAE, DISCmodel, x)

                # unfolding losses:
                image_loss = losses_mean['image_loss']
                kl_div = losses_mean['kl_div']
                disc_loss = losses_mean['disc_loss']
                disc_repr_loss = losses_mean['disc_repr_loss']
                loss_VAE = disc_repr_loss + image_loss + kl_div * 1.0
                loss_discriminator = disc_repr_loss

                for k, v in losses.items():
                    validation_epoch_data[k] += list(v.cpu().data.numpy())
                disc_data['disc_false_negatives'] = (1 - outputs['disc_real_pred'])
                disc_data['disc_true_positives'] = outputs['disc_real_pred']
                disc_data['disc_true_negatives'] = (1 - outputs['disc_fake_pred'])
                disc_data['disc_false_positives'] = outputs['disc_fake_pred']
                for k, v in disc_data.items():
                    disc_validation_epoch_data[k] += list(v.cpu().data.numpy())

            for k, v in validation_epoch_data.items():
                validation_data[k] += [np.mean(validation_epoch_data[k])]
                    
            if epoch % print_every == 0:
                cprint("\n", logfile)
                cprint(f"epoch: {epoch}/{num_epochs}", logfile)
                train_string = StatusString("training", training_epoch_data)
                evalString = StatusString("evaluation", validation_epoch_data)
                cprint(train_string, logfile)
                cprint(evalString, logfile)

                train_string = DiscStatusString("training Discriminator accurracy", disc_training_epoch_data)
                evalString = DiscStatusString("evaluation Discriminator accurracy", disc_validation_epoch_data)
                cprint(train_string, logfile)
                cprint(evalString, logfile)


                #cprint("vi.beta: {}".format(vi.beta), logfile)
                #cprint("vi.alpha: {}".format(vi.alpha), logfile)        

        VAE.update_()
        DISCmodel.update_()
        vi.update_vi()

    return validation_data, training_data, params, models