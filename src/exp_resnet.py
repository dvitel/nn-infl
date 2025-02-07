from collections import defaultdict
import fcntl
from functools import partial
import json
import os
import re

from tqdm import tqdm

from cifar import DatasetSplits, OneDataset
from resnet import ResNet34
os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache')
import pickle
import random
from torch.autograd import Variable
import torch.nn.functional as F

import argh
import numpy as np
import torch

seed = int(os.environ.get("INFL_SEED", 0))
cwd = os.environ.get("INFL_CWD", "./data/dev")
deterministic = bool(os.environ.get("INFL_DETERMINISTIC", False))

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


torch.serialization.add_safe_globals({'DatasetSplits': DatasetSplits})

# noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
# args.noise_type = noise_type_map[args.noise_type]
# # load dataset
# if args.noise_path is None:
#     if args.dataset == 'cifar10':
#         args.noise_path = './data/CIFAR-10_human.pt'
#     elif args.dataset == 'cifar100':
#         args.noise_path = './data/CIFAR-100_human.pt'
#     else: 
#         raise NameError(f'Undefined dataset {args.dataset}')

def preprocess(dataset = 'cifar10', gen_new_noise = True,
                noise_path = './data/cifar-raw/CIFAR-10_human.pt', cache_dir='./data/cifar-raw'):
    ''' Preprocoess CIFAR dataset and saves it as separate tensor file on disk'''

    from cifar import load_cifar_dataset, add_noise
    
    # if noise_type not in supported_noise_types:
    #     raise ValueError(f'Noise type {noise_type} not supported: {supported_noise_types.keys()}')
    
    ds = load_cifar_dataset(dataset, cache_dir)

    if noise_path != '':
        ds = add_noise(ds, noise_path, gen_new_noise, random_seed=seed)

    # config = dict(seed = seed, dataset = dataset)

    # config_path = os.path.join(cwd, f'c_{dataset}_{seed}.json')
    # with open(config_path, 'w') as file:
    #     json.dump(config, file)

    dataset_path = os.path.join(cwd, f'd_{dataset}_{seed}')
    torch.save(ds, dataset_path)
    print(f'Dataset saved at {dataset_path}')

supported_models = {
    'cifar10': {
        "resnet34": ResNet34,
    }
}

def adjust_learning_rate(optimizer, epoch, alpha_plan = [0.1] * 60 + [0.01] * 40):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res    

def evaluate_model(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader):
    model.eval()    # Change model to 'eval' mode.
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader):
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)

    return acc    

def train_model(model: torch.nn.Module, train_dataset: torch.utils.data.Dataset, device = 'cuda', num_epochs = 100, 
                batch_size = 128, num_workers = 4, lr = 0.1, weight_decay = 0.0005, momentum = 0.9,
                validation_dataset = None):

    if validation_dataset is None:
        validation_dataset = train_dataset

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                    batch_size = batch_size,
                                    num_workers=num_workers,
                                    shuffle=False, #True originally
                                    drop_last = False)    
    
    #used as validation set
    validation_loader = torch.utils.data.DataLoader(dataset = validation_dataset,
                                    batch_size = batch_size,
                                    num_workers=num_workers,
                                    shuffle=False,
                                    drop_last = False)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    validation_accuracy = []
    train_loss = []

    for epoch in range(num_epochs):

        # print(f'epoch {epoch}')
        adjust_learning_rate(optimizer, epoch)  
        model.train()      

        train_loss_history = []

        for images, labels in tqdm(train_loader):
            inputs = images.to(device)
            outputs = labels.to(device)
        
            # Forward + Backward + Optimize
            logits = model(inputs)

            loss: torch.Tensor = F.cross_entropy(logits, outputs, reduction = 'mean')
            train_loss_history.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (i+1) % print_freq == 0:
            #     print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
            #         %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data))

        validation_acc = evaluate_model(model, validation_loader)
        validation_accuracy.append(validation_acc)
        train_loss.append(loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {validation_acc:.4f}, Training Loss: {loss.data:.4f}')

    # train_acc=float(train_correct)/float(train_total)
    return {"accuracy": validation_accuracy, "train_loss": train_loss}

def finetune(dataset_path: str = "./cifar10", model_name = "resnet34", noise_type = "clean", lr = 0.1, num_epochs = 100, batch_size = 128,
             num_workers = 4, device = 'cuda'):
    ''' Finetune a model on a dataset with noise'''

    config = dict(seed = seed, dataset_path = dataset_path, 
                  model_name=model_name, batch_size=batch_size, lr=lr, 
                  num_epochs=num_epochs, noise_type=noise_type, device = device)
    
    dataset_splits: DatasetSplits = torch.load(dataset_path, weights_only = False)
    config_path = os.path.join(cwd, f'c_{dataset_splits.name}_{model_name}_{seed}.json')

    train_dataset = OneDataset(dataset_splits, noise_type=noise_type, for_training=True)
    test_dataset = OneDataset(dataset_splits, for_training=False)

    builder_params = dict(num_classes = dataset_splits.num_classes)
    model_builder_fn = supported_models[dataset_splits.name][model_name]
    model = model_builder_fn(**builder_params)
    model.to(device)

    train_metrics = \
        train_model(model, train_dataset, device=device, num_epochs=num_epochs, 
                    batch_size=batch_size, num_workers=num_workers, lr=lr,
                    validation_dataset = test_dataset)

    config['finetune'] = train_metrics

    with open(config_path, 'w') as file:
        json.dump(config, file)       

    model_path = os.path.join(cwd, f'm_{dataset_splits.name}_{model_name}_{seed}')

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "builder": {
            "dataset_name": dataset_splits.name,
            "model_name": model_name,
            "params": builder_params
        },
        "training": { #meta info of checkpoint            
            "dataset_path": dataset_path,
            "noise_type": noise_type,
            "num_epochs": num_epochs,
            "config_path": config_path
        }
    }

    torch.save(checkpoint, model_path)

    pass

def checkpoint(cp: str = ""):
    checkpoint = torch.load(cp, weights_only=True)
    dataset_name = checkpoint["builder"]["dataset_name"]
    model_name = checkpoint["builder"]["model_name"]
    builder_params = checkpoint["builder"]["params"]
    model_builder_fn = supported_models[dataset_name][model_name]
    model = model_builder_fn(**builder_params) #empty model build
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)
    model.to('cuda')
    dataset_file = checkpoint["training"]["dataset_path"]
    dataset_splits: DatasetSplits = torch.load(dataset_file)
    test_dataset = OneDataset(dataset_splits, for_training=False)
    test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset,
                                    batch_size = 128,
                                    num_workers=4,
                                    shuffle=False,
                                    drop_last = False)
    test_acc = evaluate_model(model, test_dataloader)
    print("Loaded model accuracy: ", test_acc)
    pass


parser = argh.ArghParser()
parser.add_commands([preprocess, finetune, checkpoint])

if __name__ == '__main__':
    try:
        parser.dispatch()
    except argh.CommandError as e:
        print(f"Error: {e}")
        parser.print_help()
        exit(2)