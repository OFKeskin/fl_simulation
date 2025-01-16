import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from model import MLP
from utils import get_dataset, average_weights, plot_and_save

if __name__ == '__main__':
    os.makedirs("figures", exist_ok=True)
    start_time = time.time()

    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    device = 'cuda'

    train_dataset, test_dataset, user_groups = get_dataset(args)

    img_size = train_dataset[0][0].shape
    len_in = 1
    for x in img_size:
        len_in *= x
    

    K_values = [1,5,40,800]

    train_loss, train_accuracy = [], []
    test_losses, test_accuracy = [], []

    for K in K_values:
        global_model = MLP(dim_in=len_in, dim_out=10)
        global_model.to(device)
        global_model.train()

        global_weights = global_model.state_dict()

        for epoch in tqdm(range(args.epochs)):
            local_grads, local_weights, local_losses = [], [], []
            
            global_model.train()
            
            for m in range(args.num_users):
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[m], logger=logger)
                g, w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                local_grads.append(copy.deepcopy(g))
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            global_weights = average_weights(local_grads, global_model, K, args.z_std, epoch)
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            list_acc, list_loss = [], []
            global_model.eval()

            for m in range(args.num_users):
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[m], logger=logger)
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc)
                list_loss.append(loss)
            train_accuracy.append(sum(list_acc)/len(list_acc))
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            test_accuracy.append(test_acc)
            test_losses.append(test_loss)
            
    # Plot and save the training accuracies
    plot_and_save(
        y_data=[train_accuracy[i*args.epochs:(i+1)*args.epochs] for i in range(len(K_values))],
        title='Training Accuracy vs Epoch',
        ylabel='Training Accuracy',
        filename='training_accuracy.png',
        labels=K_values
    )

    # Plot and save the test accuracies
    plot_and_save(
        y_data=[test_accuracy[i*args.epochs:(i+1)*args.epochs] for i in range(len(K_values))],
        title='Test Accuracy vs Epoch',
        ylabel='Test Accuracy',
        filename='test_accuracy.png',
        labels=K_values
    )

    # Plot and save the training losses
    plot_and_save(
        y_data=[train_loss[i*args.epochs:(i+1)*args.epochs] for i in range(len(K_values))],
        title='Training Loss vs Epoch',
        ylabel='Training Loss',
        filename='training_loss.png',
        labels=K_values
    )

    # Plot and save the test losses
    plot_and_save(
        y_data=[test_losses[i*args.epochs:(i+1)*args.epochs] for i in range(len(K_values))],
        title='Test Loss vs Epoch',
        ylabel='Test Loss',
        filename='test_loss.png',
        labels=K_values
    )

    print("Figures have been saved in the 'figures' directory.")
    