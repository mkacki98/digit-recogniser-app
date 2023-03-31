import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import logging
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from src.utils.utils import get_model_name, nmf_activation_fn

torch.multiprocessing.set_sharing_strategy("file_system")
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def train_validate(model, config, train_loader, val_loader, optimizer, criterion, device, save = True):
    """ Train the model given config parameters, optimizer, loss function and dataloaders
    and validate its performance on unseen data."""

    logging.info(f"\n Started training a {config.model.upper()} for {config.epoch_n} epochs with batch size of {config.batch_size}, learning rate of {config.lr}.")

    writer = SummaryWriter(f"logs/fit/{config.model} | BatchSize{config.batch_size} | LearningRate {config.lr} | Epochs {config.epoch_n}")
    
    is_neumorphic = model.__class__.__name__ == "NeuromorphicClassifier"

    with trange(config.epoch_n) as t:
        for epoch in t:

            epoch_train_loss, epoch_train_acc = epoch_train(
                model, train_loader, optimizer, criterion, writer, epoch, device, is_neumorphic)

            epoch_val_loss, epoch_val_acc = epoch_validate(model, val_loader, criterion, writer, epoch, device, is_neumorphic)

            t.set_postfix(train_acc=epoch_train_acc, val_acc = epoch_val_acc)
            
            writer.add_scalar(
                "epoch train loss", epoch_train_loss, epoch
            )
            writer.add_scalar(
                "epoch val loss", epoch_val_loss, epoch
            )
            writer.add_hparams(
                {"bs": config.batch_size, "lr": config.lr}, {"acc": epoch_val_acc, "loss": epoch_val_loss}
            )
            writer.flush()

    if save:
        torch.save(
            model,
            f"models/{get_model_name(config)}",
        )

def epoch_train(model, train_loader, optimizer, criterion, writer, epoch, device, is_neumorphic):
    """ Train the model through a single data pass (epoch). Report the result after every `reports_n` examples."""

    model.train(True)

    total = 0
    running_hits = 0
    running_loss = 0.0
    n_steps = len(train_loader)
    accuracies = []
    losses = []

    model.eval()

    for i, data in enumerate(train_loader):
        images, labels = data

        if is_neumorphic:
            images = images.flatten(start_dim=1, end_dim = 3)

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        prediction = model(images)

        loss = criterion(prediction, labels)     

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if is_neumorphic:
            predicted = torch.argmax(prediction, 1)
        else:
            _, predicted = torch.max(prediction.data, 1)
        
        total += labels.size(0)
        running_hits += (predicted == labels).sum().item()

        report = 50

        if (i + 1) % report == 0:

            acc = (running_hits / total)
            loss = (running_loss / total)
            
            accuracies.append(acc)
            losses.append(loss)
            
            writer.add_scalar(
                "train loss", loss, epoch * n_steps + i
            )
            writer.add_scalar(
                "train accuracy", acc, epoch * n_steps + i
            )

            running_loss = 0.0
            running_hits = 0
            total = 0

    epoch_loss = sum(losses)/len(losses)
    epoch_acc = sum(accuracies)/len(accuracies)
    
    model.train(False)

    return epoch_loss, epoch_acc

def epoch_validate(model, val_loader, criterion, writer, epoch, device, is_neumorphic):
    """Collect the validation loss for that epoch."""

    total = 0

    running_hits = 0
    running_loss = 0.0

    n_steps = len(val_loader)
    model.eval()

    accuracies = []
    losses = []

    for i, data in enumerate(val_loader):
        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        if is_neumorphic:
            images = images.flatten(start_dim=1, end_dim = 3)

        prediction = model(images)

        loss = criterion(prediction, labels)
        running_loss += loss.item()

        if is_neumorphic:
            predicted = torch.argmax(prediction, 1)
        else:
            _, predicted = torch.max(prediction.data, 1)
        
        total += labels.size(0)
        running_hits += (predicted == labels).sum().item()

        report = 50

        if (i + 1) % report == 0:
            
            acc = (running_hits / total)
            loss = (running_loss / total)
            
            accuracies.append(acc)
            losses.append(loss)
            
            writer.add_scalar(
                "val loss", loss, epoch * n_steps + i
            )
            writer.add_scalar(
                "val accuracy", acc, epoch * n_steps + i
            )

            running_loss = 0.0
            running_hits = 0
            total = 0

    epoch_loss = sum(losses)/len(losses)
    epoch_acc = sum(accuracies)/len(accuracies)
    
    return epoch_loss, epoch_acc

def get_pretrained_synapses(train_loader, 
                            config,
                            from_pretrained = False, 
                            save = True):
    """ 
    Pre-train synapses according to a differential equation introduced in (Krotov, 2019).
    
    ---
    Parameters:
        train_loader; train dataloader (MNIST)
        config; parser with all the arguments
        from_pretrained (bool); whether to return pre-trained synapses or train them from scratch
        save (bool); whether to save the synapses

        (from config)
            lr (float); initial learning rate (will be changed as the process progresses)
            epochs (int); number of full passes through the data

            delta (float); strength of the anti-hebbian learning
            p (int); Lebesgue norm, for p = 2 activation function becomes ReLU
            rank (int); ranking parameter (how many hidden units do we consider), has to be >= 2
            tau_l (float); constant to define the time scale of the process
    Returns:
        synapses; tensor of trained synapses, (n_hidden x 784)
    """

    if from_pretrained:
        return torch.load("models/synapses_100.pkl")

    synapses = torch.rand((config.hid, 784), dtype = torch.float)
    lr = config.lr 

    with trange(config.hid_epoch_n) as t:
        for epoch in t:

            lr *= (1 - epoch / config.hid_epoch_n)

            for i, data in enumerate(train_loader):

                # v; input batch
                images, _ = data
                images = images.flatten(start_dim=1, end_dim=3)
                current_batch_size = images.size(0)

                # I; sum(r(Wv); != k); r = max(x, 0) if p == 2.0
                input_current = torch.mm(nmf_activation_fn(config.p, synapses), torch.transpose(images, 0, 1))

                # g; applying activation/inhibition depending on current received by a neuron
                y = torch.argsort(input_current, dim=0) 

                w = torch.zeros((config.hid, current_batch_size), dtype = torch.float)
                w[y[config.hid-1, :], np.arange(current_batch_size)] = 1.0
                w[y[config.hid-config.rank], np.arange(current_batch_size)] = -1 * config.delta
                
                # dW_i/dt = g * v - g * sum(r(Wv), != v) * W_i 
                dW_dt = torch.mm(w, images) - torch.sum(w * input_current, 1).unsqueeze(1) * synapses

                # Update the tau, tau << tau_l
                tau = torch.max(np.absolute(dW_dt))
                tau = config.tau_l if tau < config.tau_l else tau

                delta_w = np.true_divide(dW_dt, tau)

                # Update the synapses
                synapses += config.lr * delta_w
    
    if save:
        torch.save(synapses, f'models/synapses_hid_{config.hid}_batch_size_{config.batch_size}')

    return synapses