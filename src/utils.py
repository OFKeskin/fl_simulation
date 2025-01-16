import copy
import torch
import numpy as np
from torchvision import datasets, transforms
from data_dist import mnist_iid
import matplotlib.pyplot as plt

def get_dataset(args):
    #data_dir = '../data/mnist'
    data_dir = '~/.cache/torch'

    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

    user_groups = mnist_iid(train_dataset, args.num_users, args.sample_size)

    return train_dataset, test_dataset, user_groups


"""
def average_weights(g, global_weights, K, z_std, epoch):
    M = len(g)
    alpha = 1 + epoch/1000

    y = {}
    for key in g[0].keys():
        size = g[0][key].size()
        y[key] = torch.zeros(size[0]/2,size[1])

    for k in range(K):
        h_list = {}
        y_k = {}
        for key in g[0].keys():
            size = g[0][key].size()
            y_k[key] = torch.zeros(size[0]/2,size[1])
            h_list[key] = torch.zeros(size[0]/2,size[1])
        for m in range(M):
            for key in g[m].keys():
                size = g[m][key].size()
                h_m_k = (torch.randn(size[0]/2, size[1], dtype=torch.float64, device='cuda') + 1j * torch.randn(size[0]/2, size[1], dtype=torch.float64, device='cuda'))
                h_list[key] += h_m_k
                g_m = g[m][key][:size[0]/2,:] + 1j*g[m][key][size[0]/2:,:]
                z = torch.randn(size[0]/2,size[1],dtype=torch.float64, device='cuda') * (z_std/2) + 1j * torch.randn(size[0]/2,size[1],dtype=torch.float64, device='cuda') * (z_std/2)
                y_k[key] += alpha * h_m_k * g_m + z
        for key in y_k.keys():
            h_conj = h_list[key].conj()
            y[key] += h_conj * y_k[key] / K

    #y_recover = {}
    #for key in y.keys():
        #y_recover[key] = torch.cat((torch.real(y[key]),torch.imag(y[key])),0) / (alpha * M)
    
    for key in global_weights.items():
        global_weights[key] -= torch.cat((torch.real(y[key]),torch.imag(y[key])),0) / (alpha * M)


    return updated_global_weights
""" 
def average_weights(local_gradients, global_model, K, z_std, epoch):
    device = 'cuda'

    M = len(local_gradients)
    optimizer = torch.optim.Adam(global_model.parameters(), lr=0.01, weight_decay=1e-4)

    alpha = 1 + (epoch+1)/1000

    total_elements = sum(g.numel() for g in local_gradients[0].values())

    y = torch.zeros(int(total_elements/2), dtype=torch.complex128, device='cuda')

    for k in range(K):
        h_matrix = (torch.randn(int(total_elements/2), M, dtype=torch.float64, device='cuda') + 
            1j * torch.randn(int(total_elements/2), M, dtype=torch.float64, device='cuda'))

        y_k = torch.zeros(int(total_elements/2), dtype=torch.complex128, device='cuda')

        for m,g in enumerate(local_gradients):
            gradients = list(g.values())
            flattened_gradients = torch.cat([gr.flatten() for gr in gradients])

            real_part = flattened_gradients[:len(flattened_gradients)//2]
            imag_part = flattened_gradients[len(flattened_gradients)//2:]
            g_signal = real_part + 1j*imag_part
            g_signal = g_signal.to(device)
            z = torch.randn(int(total_elements/2),dtype=torch.float64, device='cuda') * (z_std/2) + 1j * torch.randn(int(total_elements/2),dtype=torch.float64, device='cuda') * (z_std/2)
            y_k += alpha * h_matrix[:,m] * g_signal + z
        
        y += y_k * torch.sum(h_matrix, dim=1).conj() / K

    real_part_recovered = torch.real(y) / (alpha * M)
    imag_part_recovered = torch.imag(y) / (alpha * M)
    recovered_gradients = torch.concatenate((real_part_recovered, imag_part_recovered))

    optimizer.zero_grad()

    idx = 0
    for name, param in global_model.named_parameters():
        num_elements = param.numel()
        gradient_slice = recovered_gradients[idx:idx + num_elements]
        param.grad = gradient_slice.reshape(param.shape).to(param.dtype).to(device)
        idx += num_elements

    optimizer.step()  

    return global_model.state_dict()
    """
    idx = 0
    updated_global_weights = {}
    for key, param in global_weights.items():
        num_elements = param.numel() 
        gradient_slice = recovered_gradients[idx:idx + num_elements]
        updated_param = param - 0.01 * torch.tensor(gradient_slice, dtype=param.dtype).reshape(param.shape)  
        updated_global_weights[key] = updated_param 
        idx += num_elements 
    
    return updated_global_weights
    """

def plot_and_save(y_data, title, ylabel, filename, labels):
    plt.figure()
    for i, y in enumerate(y_data):
        plt.plot(range(1, len(y) + 1), y, label=f'K = {labels[i]}')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'figures/{filename}')
    plt.close()

