from collections import defaultdict
import fcntl
from functools import partial
import json
import os
import re

from tqdm import tqdm

from cifar import DatasetSplits, OneDataset
from influence import  compute_infl_from_model, datainf_fn, hessian_free_fn, lissa_fn
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
             num_workers = 2, device = 'cuda'):
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

def load_checkpoint(cp: str = ""):
    checkpoint = torch.load(cp, weights_only=True)
    dataset_name = checkpoint["builder"]["dataset_name"]
    model_name = checkpoint["builder"]["model_name"]
    builder_params = checkpoint["builder"]["params"]
    model_builder_fn = supported_models[dataset_name][model_name]
    model = model_builder_fn(**builder_params) #empty model build
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)
    model.to('cuda')
    return model, checkpoint

def _compute_grads(model, dataloader, device="cuda", bring_to_cpu=False):
    ''' Builds tensor of grads, collected accross the model '''
    model.eval()

    module_grads = {}
    num_samples = len(dataloader)
    model.to(device)
    for k, v in model.named_parameters():
        grad = torch.empty((num_samples, v.numel()), device=device)
        module_grads[k] = grad
    for step, (images, labels) in enumerate(tqdm(dataloader)):
        inputs = images.to(device)
        outputs = labels.to(device)
        model.zero_grad() # zeroing out gradient
        logits = model(inputs)
        loss: torch.Tensor = F.cross_entropy(logits, outputs, reduction = 'mean')

        loss.backward()
        
        for k, v in model.named_parameters():
            if k in module_grads:
                module_grads[k][step] = v.grad.view(-1)
            else:
                pass
    if bring_to_cpu:
        for k, v in module_grads.items():
            module_grads[k] = v.cpu()
            del v
    return module_grads    

def grads(dataset_name = 'cifar10', model_name: str  = 'resnet34', no_val = False, return_grads = False, config = None):
    ''' Computes gradients for modules of the model'''
    if config is None:
        config_path = os.path.join(cwd, f'c_{dataset_name}_{model_name}_{seed}.json')
        with open(config_path, 'r') as file:
            config = json.load(file)

    device = config['device']
    # model_path = os.path.join(cwd, f'm_{task}_{seed}')
    model_path = os.path.join(cwd, f'm_{dataset_name}_{model_name}_{seed}')
    model, checkpoint = load_checkpoint(model_path)
    dataset_file = checkpoint["training"]["dataset_path"]
    noise_type = checkpoint["training"]["noise_type"]
    dataset_splits: DatasetSplits = torch.load(dataset_file, weights_only = False)
    train_dataset = OneDataset(dataset_splits, noise_type=noise_type, for_training=True)
    test_dataset = OneDataset(dataset_splits, for_training=False)
    train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,
                                    batch_size = 1,
                                    num_workers = 2,
                                    shuffle=False,
                                    drop_last = False)
    eval_dataloader = torch.utils.data.DataLoader(dataset = test_dataset,
                                    batch_size = 1,
                                    num_workers = 2,
                                    shuffle=False,
                                    drop_last = False)

    train_grads = _compute_grads(model, train_dataloader, device=device, bring_to_cpu=not return_grads)
    if no_val:
        val_grads = {}
    else:
        val_grads = _compute_grads(model, eval_dataloader, device=device, bring_to_cpu=not return_grads)

    if return_grads:
        return {'train': train_grads, 'validation': val_grads}
    else:
        grad_path = os.path.join(cwd, f'g_{dataset_name}_{model_name}_{seed}.pt')
        torch.save({'train': train_grads, 'validation': val_grads}, grad_path)

influence_fns = \
    {
        "hf": hessian_free_fn,
        "datainf": datainf_fn,
        "lissa": lissa_fn,
        # "exact": compute_accurate_influences
    }

def infl(dataset_name = 'cifar10', model_name: str = "resnet34", method = "datainf", self_influence = False):
    config_path = os.path.join(cwd, f'c_{dataset_name}_{model_name}_{seed}.json')
    with open(config_path, 'r') as file:
        config = json.load(file)

    device = config['device']

    model_path = os.path.join(cwd, f'm_{dataset_name}_{model_name}_{seed}')
    model, checkpoint = load_checkpoint(model_path)
    dataset_file = checkpoint["training"]["dataset_path"]
    noise_type = checkpoint["training"]["noise_type"]
    dataset_splits: DatasetSplits = torch.load(dataset_file, weights_only = False)
    train_dataset = OneDataset(dataset_splits, noise_type=noise_type, for_training=True) # max_size = 1000)
    if self_influence:
        test_dataset = train_dataset
    else:
        test_dataset = OneDataset(dataset_splits, for_training=False) #, max_size=1000)

    method_fn = influence_fns[method]
    runtine, inf_tensors = compute_infl_from_model(model, train_dataset, test_dataset, device = device, infl_fn=method_fn,
                                                   module_patterns=['layer1\\..*','layer2\\..*','layer3\\..*','layer4\\..*'])

    config.setdefault("infl_runtimes", {})[method] = runtine

    infl_path = os.path.join(cwd, f'i_{method}_{dataset_name}_{model_name}_{seed}.pt')
    torch.save(inf_tensors, infl_path)

    with open(config_path, 'w') as file:
        json.dump(config, file)


def finetune2(dataset_name = 'cifar10', model_name: str = 'resnet34',
         num_epochs = 100, filter_method='infl', infl_method='datainf', module_pattern='', filter_perc = 0.7):
    config_path = os.path.join(cwd, f'c_{dataset_name}_{model_name}_{seed}.json')
    with open(config_path, 'r') as file:
        config = json.load(file)    
    config.update(finetune2=dict(num_epochs=num_epochs, filter_method=filter_method,
                                 infl_method=infl_method, module_pattern=module_pattern, filter_perc=filter_perc))
    print(f"Finetune with modules: {module_pattern}")
    device = config['device']
    lr = config['lr']
    model = config['model']
    batch_size = config['batch_size']
    model_path = os.path.join(cwd, f'm_{dataset_name}_{model_name}_{seed}')
    checkpoint = torch.load(model_path, weights_only=True)
    dataset_name = checkpoint["builder"]["dataset_name"]
    model_name = checkpoint["builder"]["model_name"]
    builder_params = checkpoint["builder"]["params"]
    model_builder_fn = supported_models[dataset_name][model_name]
    model = model_builder_fn(**builder_params) #empty model build
    noise_type = checkpoint["training"]["noise_type"]
    dataset_path = os.path.join(cwd, f'd_{dataset_name}_{seed}')
    dataset_splits: DatasetSplits = torch.load(dataset_path, weights_only = False)
    if filter_method == "infl":
        influences = torch.load(os.path.join(cwd, f'i_{infl_method}_{dataset_name}_{model_name}_{seed}.pt'))

        module_names = [] # all modules
        if module_pattern == '':
            influence = influences[''].to(device)
        else:
            pattern = re.compile(module_pattern)
            influence = None
            for module in influences:
                if pattern.match(module): 
                    if influence is None:
                        influence = influences[module].to(device)
                    else:
                        influence += influences[module].to(device)
                    module_names.append(module)
        config['finetune2']['module_names'] = module_names
        
        # fine-tuning models
        high_to_low_quality = torch.argsort(influence)
        filter_len = int(filter_perc*len(high_to_low_quality))
        filtered_indexes = high_to_low_quality[:filter_len].cpu()

        def infl_filter(inputs: torch.Tensor):
            return inputs[filtered_indexes]
        filter_fn = infl_filter
    elif filter_method == 'rand':
        filtered_indexes = None
        def rand_filter(inputs: torch.Tensor):
            nonlocal filtered_indexes
            if filtered_indexes is None:
                filter_len = int(filter_perc*inputs.shape[0])
                filtered_indexes = torch.randperm(inputs.shape[0])[:filter_len]
            return inputs[filtered_indexes]
        filter_fn = rand_filter        
    elif filter_method == 'denoise':
        noise_type = 'clean'
        filter_fn = None
    else:
        filter_fn = None

    train_dataset = OneDataset(dataset_splits, noise_type=noise_type, for_training=True, filter_fn=filter_fn)
    test_dataset = OneDataset(dataset_splits, for_training=False)

    train_metrics = \
        train_model(model, train_dataset, device=device, num_epochs=num_epochs, 
                    batch_size=batch_size, num_workers=2, lr=lr,
                    validation_dataset = test_dataset)

    config['finetune2'] = train_metrics

    with open(config_path, 'w') as file:
        json.dump(config, file)       

    metric_file = os.path.join(cwd, f'metrics.jsonlist')
    with open(metric_file, "a") as f:                
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({**config, "metrics": train_metrics}) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)    

    with torch.no_grad():
        torch.cuda.empty_cache()

parser = argh.ArghParser()
parser.add_commands([preprocess, finetune, checkpoint, grads, infl, finetune2])

if __name__ == '__main__':
    try:
        parser.dispatch()
    except argh.CommandError as e:
        print(f"Error: {e}")
        parser.print_help()
        exit(2)