import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users, sample_size):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs,sample_size,replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users