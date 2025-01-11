from collections import defaultdict
import fcntl
import json
import os
import re
os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache')
import pickle
import random

import argh
import numpy as np
from lora_model import build_LORA_model, train_LORA_model, load_pretrained_LORA_model, compute_grads
from influence import IFEngine, compute_hessian_free_influences, compute_datainf_influences, compute_lissa_influences, compute_accurate_influences
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader

seed = int(os.environ.get("INFL_SEED", 0))
cwd = os.environ.get("INFL_CWD", "./data/dev")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

task_to_keys = {
    "cola": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "wnli": ("sentence1", "sentence2"),
}

def flip_label(example, ind, noise_index):
    if ind in noise_index:
        example["label"] = 1 - example["label"]
        example["noise"] = True
    else:
        example["noise"] = False
    return example

def select_shuffled_balanced(dataset, total_size, num_groups = 1):
    possible_labels = len(dataset.features['label'].names)
    num_samples_per_class_per_group = total_size // possible_labels
    num_samples_per_class = num_groups * num_samples_per_class_per_group
    labels = np.array(dataset['label'])
    all_indices = [[] for _ in range(num_groups)]
    for i in range(possible_labels):
        class_0_indices = np.where(labels == i)[0]
        sampled_class_0_indices = np.random.choice(class_0_indices, num_samples_per_class, replace=False)
        for i, g in enumerate(all_indices):
            g.append(sampled_class_0_indices[i * num_samples_per_class_per_group:(i + 1) * num_samples_per_class_per_group])
    datasets = []
    for group in all_indices:
        sampled_indices = np.concatenate(group)
        np.random.shuffle(sampled_indices)
        sampled_dataset = dataset.select(sampled_indices)
        datasets.append(sampled_dataset)
    return datasets

def load_noisy_dataset_by_task(task, train_size, val_size, noise_ratio=0.2):
    glue_datasets = load_dataset("glue", task) 
    trainsets = select_shuffled_balanced(glue_datasets['train'], train_size, num_groups=1)
    valsets = select_shuffled_balanced(glue_datasets['validation'], val_size, num_groups=1)
    for i, valset in enumerate(valsets):
        glue_datasets[f'validation{i}'] = valset

    if noise_ratio > 0.0:
        noise_index = set(np.random.choice(train_size, size=int(noise_ratio*train_size), replace=False))
    else:
        noise_index = []

    for i, trainset in enumerate(trainsets):
        glue_datasets[f'train{i}'] = trainset.map(flip_label, with_indices=True, fn_kwargs={'noise_index':noise_index})
    
    glue_datasets.pop('test')
    glue_datasets.pop('train')
    glue_datasets.pop('validation')
    
    return glue_datasets

def load_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def preprocess(task = 'mrpc', noise_ratio = 0.2, train_size = 4500, val_size = 500,
                tokenizer_name='roberta-large'):
    ''' Preprocoess GLUE dataset of specific task '''
    config = dict(seed = seed, task=task, noise_ratio=noise_ratio, 
                  train_size = train_size, val_size = val_size,
                  tokenizer_name=tokenizer_name)

    sentence1_key, sentence2_key = task_to_keys[task]
    tokenizer = load_tokenizer(tokenizer_name)
    def tokenize_function(examples, max_length=128):
        # max_length=None => use the model max length (it's actually the default)
        if sentence2_key is None:
            outputs = tokenizer(examples[sentence1_key], truncation=True, max_length=max_length)
        else:
            outputs = tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, max_length=max_length)
        return outputs

    noisy_datasets = load_noisy_dataset_by_task(task, train_size, val_size, noise_ratio=noise_ratio)
    if sentence2_key is None:
        tokenized_datasets = noisy_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", sentence1_key],
        )
    else:
        tokenized_datasets = noisy_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", sentence1_key, sentence2_key],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
    with open(config_path, 'w') as file:
        json.dump(config, file)

    dataset_path = os.path.join(cwd, f'd_{task}_{seed}')
    tokenized_datasets.save_to_disk(dataset_path)

def build_loaders(dataset_path, tokenizer_name, batch_size = 32, shuffle_train = True, 
                    filter_fn = None, train_set_id = 0, val_set_id = 0):
    datasets = load_from_disk(dataset_path)
    trainset = datasets[f'train{train_set_id}']        
    valset = datasets[f'validation{val_set_id}']
    if filter_fn is not None:
        trainset = filter_fn(trainset)
    trainset = trainset.remove_columns(['noise'])
    tokenizer = load_tokenizer(tokenizer_name)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")  
        
    train_dataloader = DataLoader(trainset,
                                  shuffle=shuffle_train, 
                                  collate_fn=collator,
                                  batch_size=batch_size)
    eval_dataloader = DataLoader(valset,
                                 shuffle=False, 
                                 collate_fn=collator, 
                                 batch_size=batch_size)
    return train_dataloader, eval_dataloader, tokenizer

def convert_metrics(eval_metrics):
    metrics = {}
    for epoch in eval_metrics:
        for metric, value in epoch.items():
            metrics.setdefault(metric, []).append(value)
    return metrics
    
def finetune(task = 'mrpc', low_rank = 4,
         device = 'cuda', lr = 3e-4, model = 'roberta-large', batch_size = 32, train_set_id = 0, val_set_id = 0,
         num_epochs = 10, target_modules = ['value']):
    ''' Fine tune specific model on specific task and save it to disk for later postprocessing'''
    config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
    with open(config_path, 'r') as file:
        config = json.load(file)

    config.update(low_rank=low_rank, train_set_id=train_set_id, val_set_id=val_set_id,
                  device=device, lr=lr, model=model, batch_size=batch_size,
                  num_epochs=num_epochs, target_modules=target_modules)

    dataset_path = os.path.join(cwd, f'd_{task}_{seed}')
    train_dataloader, eval_dataloader, tokenizer = \
        build_loaders(dataset_path, config['tokenizer_name'], batch_size, 
                      train_set_id=train_set_id, val_set_id=val_set_id)

    lora_model = build_LORA_model(model_name_or_path=model,
                                target_modules=target_modules, 
                                low_rank=low_rank)
    eval_metrics = train_LORA_model(lora_model, train_dataloader, eval_dataloader, device, num_epochs, lr, task)

    config['finetune'] = convert_metrics(eval_metrics)
    
    with open(config_path, 'w') as file:
        json.dump(config, file)       

    model_path = os.path.join(cwd, f'm_{task}_{seed}')

    ## next code is for testing weights preservation 
    # lora_model.to('cpu')
    # another_lora_model = load_pretrained_LORA_model(model_name_or_path=model_path)
    # for (name1, param1), (name2, param2) in zip(lora_model.named_parameters(), another_lora_model.named_parameters()):
    #     if "original_module" not in name1:
    #         assert torch.allclose(param1, param2, rtol=1e-05, atol=1e-08), f'Parameters are not equal: {name1} {name2}'

    lora_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    del lora_model, train_dataloader, eval_dataloader

    with torch.no_grad():
        torch.cuda.empty_cache()

def grads(task = 'mrpc'):
    ''' Computes gradients for modules of the model'''
    config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
    with open(config_path, 'r') as file:
        config = json.load(file)
    device = config['device']
    train_set_id = config['train_set_id']
    val_set_id = config['val_set_id']
    model_path = os.path.join(cwd, f'm_{task}_{seed}')
    lora_model = load_pretrained_LORA_model(model_name_or_path=model_path)
    dataset_path = os.path.join(cwd, f'd_{task}_{seed}')
    train_dataloader, eval_dataloader, _ = \
        build_loaders(dataset_path, config['tokenizer_name'], batch_size=1, 
                        train_set_id=train_set_id, val_set_id=val_set_id, shuffle_train=False)
    train_grads = compute_grads(lora_model, train_dataloader, device=device, bring_to_cpu=True)
    val_grads = compute_grads(lora_model, eval_dataloader, device=device, bring_to_cpu=True)

    grad_path = os.path.join(cwd, f'g_{task}_{seed}.pt')
    torch.save({'train': train_grads, 'validation': val_grads}, grad_path)

def infl(task = 'mrpc', compute_accurate = False, self_influence = False):
    config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
    with open(config_path, 'r') as file:
        config = json.load(file)

    device = config['device']

    grads = torch.load(os.path.join(cwd, f'g_{task}_{seed}.pt'))

    train_grads = {k:v.to(device) for k, v in grads['train'].items()}
    val_grads = {k:v.to(device) for k, v in grads['validation'].items()}

    # only for some infl methods
    def compute_all(train_grads, val_grads):
        avg_val_grads = {module_name: torch.mean(module_grads, dim=0) for module_name, module_grads in val_grads.items()}
        # influence functions
        runtimes = {}
        influences = {}
        hessian_free_runtime, hessian_free_infl = compute_hessian_free_influences(train_grads, val_grads, avg_val_grads, bring_to_cpu=True)
        influences['HessianFree'] = hessian_free_infl
        runtimes['HessianFree'] = hessian_free_runtime
        datainf_runtime, datainf_infl = compute_datainf_influences(train_grads, val_grads, avg_val_grads, bring_to_cpu=True)
        influences['DataInf'] = datainf_infl
        runtimes['DataInf'] = datainf_runtime
        lissa_runtime, lissa_infl = compute_lissa_influences(train_grads, val_grads, avg_val_grads, bring_to_cpu=True)
        influences['Lissa'] = lissa_infl
        runtimes['Lissa'] = lissa_runtime
        if compute_accurate:
            accurate_runtime, accurate_infl = compute_accurate_influences(train_grads, val_grads, avg_val_grads, bring_to_cpu=True)
            influences['Exact'] = accurate_infl
            runtimes['Exact'] = accurate_runtime
        return influences, runtimes

    prefix = ""
    if self_influence:
        selfinfluences, selfruntimes = compute_all(train_grads, train_grads)
        all_data = {'influences': selfinfluences, 'runtimes': selfruntimes}
        prefix = "s"
    else:
        influences, runtimes = compute_all(train_grads, val_grads)
        all_data = {'influences': influences, 'runtimes': runtimes}

    # selfinfluence_engine = IFEngine(tr_grad_dict, tr_grad_dict)
    # selfinfluence_engine.preprocess_gradients(tr_grad_dict, tr_grad_dict, noise_index)
    # selfinfluence_engine.compute_hvps(compute_accurate=compute_accurate)
    # selfinfluences = selfinfluence_engine.compute_all_influences()


    infl_path = os.path.join(cwd, f'{prefix}i_{task}_{seed}.pt')
    torch.save(all_data, infl_path)

def finetune2(task = 'mrpc',
         num_epochs = 10, infl_key='influences', infl_method='DataInf', infl_module='',
         val_set_id = 0, filter_perc = 0.7):
    config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
    with open(config_path, 'r') as file:
        config = json.load(file)    
    config.update(finetune2=dict(num_epochs=num_epochs, infl_key=infl_key, 
                                 infl_method=infl_method, infl_module=infl_module, filter_perc=filter_perc,
                                 val_set_id=val_set_id))
    train_set_id = config['train_set_id']
    all_influences = torch.load(os.path.join(cwd, f'i_{task}_{seed}.pt'))
    if infl_key in all_influences:
        module_influences = all_influences[infl_key][infl_method]

        device = config['device']
        lr = config['lr']
        model = config['model']
        batch_size = config['batch_size']
        target_modules = config['target_modules']
        if infl_module == '':
            num_modules = -1
            influence = module_influences[''].to(device)
        else:
            num_modules = 0
            pattern = re.compile(re.escape(infl_module))
            influence = None
            for module in module_influences:
                if pattern.match(module): 
                    if influence is None:
                        influence = module_influences[module].to(device)
                    else:
                        influence += module_influences[module].to(device)
                    num_modules += 1
        if num_modules == 0:
            print(f"No modules match patter {infl_module}")
            return
        config['finetune2']['num_modules'] = num_modules
        
        # fine-tuning models

        def infl_filter(train_dataset):
            high_to_low_quality = torch.argsort(influence)
            filter_len = int(filter_perc*len(high_to_low_quality))
            filtered_indexes = high_to_low_quality[:filter_len].cpu().numpy()
            return train_dataset.select(filtered_indexes)
        filter_fn = infl_filter
    elif infl_key == 'rand':
        def rand_filter(train_dataset):
            filter_len = int(filter_perc*len(train_dataset))
            filtered_indexes = np.random.choice(len(train_dataset), filter_len, replace=False)
            return train_dataset.select(filtered_indexes)
        filter_fn = rand_filter        
    elif infl_key == 'denoise':
        def denoise_map(example):
            if example['noise']:
                example['labels'] = 1 - example['labels'] 
            return example               
        def denoise_filter(train_dataset):
            train_dataset.map(denoise_map, remove_columns=['noise'])
        filter_fn = denoise_filter
    else:
        filter_fn = None

    dataset_path = os.path.join(cwd, f'd_{task}_{seed}')
    train_dataloader, eval_dataloader, _ = \
        build_loaders(dataset_path, config['tokenizer_name'], batch_size,
            train_set_id = train_set_id, val_set_id = val_set_id, filter_fn=filter_fn)

    lora_model = build_LORA_model(model_name_or_path=model,
                                target_modules=target_modules, 
                                low_rank=config['low_rank'])
    eval_metrics = train_LORA_model(lora_model, train_dataloader, eval_dataloader, device, num_epochs, lr, task)
    metrics = convert_metrics(eval_metrics)

    del lora_model, train_dataloader, eval_dataloader

    metric_file = os.path.join(cwd, f'metrics.jsonlist')
    with open(metric_file, "a") as f:                
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({**config, "metrics": metrics}) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)    

    with torch.no_grad():
        torch.cuda.empty_cache()

parser = argh.ArghParser()
parser.add_commands([preprocess, finetune, grads, infl, finetune2])

if __name__ == '__main__':
    parser.dispatch()