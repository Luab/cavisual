import torch
import numpy as np
import os
import h5py

use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_activations(model, output_dir, data_loader, concept_name, layer_names,
                    max_samples, save_activations):
    '''
    The function to generate the activations of all layers for ONE concept only
    :param model:
    :param output_dir:
    :param data_loader: the dataloader for the input of ONE concept
    :param layer_names:
    :param max_samples:
    :return:
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.eval()
    activations = {}
    with torch.no_grad():
        for l in layer_names:
            activations[l] = []

        for i, data in enumerate(data_loader):
            if i == max_samples:
                break
            data = data['img'].to(device)
            _ = model(data)
            for l in layer_names:
                z = model.intermediate_activations[l].cpu().detach().numpy()
                activations[l].append(z)

    for l in layer_names:
        activations[l] = np.concatenate(activations[l], axis=0)

    if save_activations:
        with h5py.File(
                os.path.join(output_dir, 'activations_%s.h5' % concept_name),
                'w') as f:
            for l in layer_names:
                f.create_dataset(l, data=activations[l])
    return activations


def load_activations(path):
    activations = {}
    with h5py.File(path, 'r') as f:
        for k, v in f.items():
            activations[k] = np.array(v)
    return activations
