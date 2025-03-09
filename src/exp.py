from collections import defaultdict
import fcntl
from functools import partial
from itertools import product
import json
import os
import re
from time import time

from tqdm import tqdm
os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache')
import pickle
import random

import argh
import numpy as np
from lora_model import build_LORA_model, train_LORA_model, load_pretrained_LORA_model, compute_grads
from influence import IFEngine, compute_hessian_free_influences, compute_datainf_influences, compute_infl_from_model, compute_lissa_influences, compute_accurate_influences, datainf_fn, lissa_fn
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset, load_from_disk, Dataset
from torch.utils.data import DataLoader

seed = int(os.environ.get("INFL_SEED", 0))
cwd = os.environ.get("INFL_CWD", "./data/dev")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

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

# def select_shuffled_balanced(dataset, total_size, num_groups = 1):
#     possible_labels = len(dataset.features['label'].names)
#     num_samples_per_class_per_group = total_size // possible_labels
#     num_samples_per_class = num_groups * num_samples_per_class_per_group
#     labels = np.array(dataset['label'])
#     all_indices = [[] for _ in range(num_groups)]
#     for i in range(possible_labels):
#         class_0_indices = np.where(labels == i)[0]
#         sampled_class_0_indices = np.random.choice(class_0_indices, num_samples_per_class, replace=False)
#         for i, g in enumerate(all_indices):
#             g.append(sampled_class_0_indices[i * num_samples_per_class_per_group:(i + 1) * num_samples_per_class_per_group])
#     datasets = []
#     for group in all_indices:
#         sampled_indices = np.concatenate(group)
#         np.random.shuffle(sampled_indices)
#         sampled_dataset = dataset.select(sampled_indices)
#         datasets.append(sampled_dataset)
#     return datasets

def load_noisy_dataset_by_task(task, infl_ratio = 0.5, max_val_size = None, max_train_size = None, noise_ratio=0.2):
    glue_datasets = load_dataset("glue", task) 
    if max_train_size is not None and max_train_size < len(glue_datasets['train']):
        tmpsets = glue_datasets['train'].train_test_split(train_size = max_train_size, shuffle=True, seed=seed, stratify_by_column='label')
        glue_datasets['train'] = tmpsets['train']
    if max_val_size is not None and max_val_size < len(glue_datasets['validation']):
        tmpsets = glue_datasets['validation'].train_test_split(train_size = max_val_size, shuffle=True, seed=seed, stratify_by_column='label')  
        glue_datasets['validation'] = tmpsets['train']

    infl_size = int(infl_ratio * len(glue_datasets['validation']))
    tmpsets = glue_datasets['validation'].train_test_split(train_size = infl_size, shuffle=True, seed=seed, stratify_by_column='label') 
    glue_datasets['infl'] = tmpsets['train']
    glue_datasets['validation'] = tmpsets['test']

    train_size = len(glue_datasets['train'])

    if noise_ratio > 0.0:
        noise_index = set(np.random.choice(train_size, size=int(noise_ratio*train_size), replace=False))
    else:
        noise_index = []

    glue_datasets['train'] = glue_datasets['train'].map(flip_label, with_indices=True, fn_kwargs={'noise_index':noise_index})
    
    glue_datasets.pop('test')
    
    return glue_datasets

def load_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def preprocess(task = 'qnli', noise_ratio = 0.2, tokenizer_name='roberta-large'):
    ''' Preprocoess GLUE dataset of specific task '''
    config = dict(seed = seed, task=task, noise_ratio=noise_ratio, 
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

    noisy_datasets = load_noisy_dataset_by_task(task, max_train_size = 4500, max_val_size = 1000, infl_ratio=0.5, noise_ratio=noise_ratio)
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
                    filter_fn = None):
    datasets = load_from_disk(dataset_path)
    trainset = datasets['train']  
    valset = datasets['validation']
    if filter_fn is not None:
        trainset = filter_fn(trainset)
    trainset = trainset.remove_columns(['noise'])
    tokenizer = load_tokenizer(tokenizer_name)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")  
        
    train_dataloader = DataLoader(trainset, #.select(range(100)),
                                  shuffle=shuffle_train, 
                                  collate_fn=collator,
                                  batch_size=batch_size)
    eval_dataloader = DataLoader(valset, #.select(range(100)),
                                 shuffle=False, 
                                 collate_fn=collator, 
                                 batch_size=batch_size)
    return train_dataloader, eval_dataloader, tokenizer
    
def finetune(task = 'mrpc', low_rank = 4,
         device = 'cuda', lr = 3e-4, model = 'roberta-large', batch_size = 32,
         num_epochs = 10, target_modules = ['value'], unfreeze_regex = None):
    ''' Fine tune specific model on specific task and save it to disk for later postprocessing'''
    config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
    with open(config_path, 'r') as file:
        config = json.load(file)

    config.update(low_rank=low_rank, device=device, lr=lr, model=model, batch_size=batch_size,
                  num_epochs=num_epochs, target_modules=target_modules, unfreeze_regex = unfreeze_regex)

    dataset_path = os.path.join(cwd, f'd_{task}_{seed}')
    train_dataloader, eval_dataloader, tokenizer = \
        build_loaders(dataset_path, config['tokenizer_name'], batch_size)

    lora_model = build_LORA_model(model_name_or_path=model,
                                target_modules=target_modules, 
                                low_rank=low_rank, unfreeze_modules_regex=unfreeze_regex)
    eval_metrics = train_LORA_model(lora_model, train_dataloader, eval_dataloader, device, num_epochs, lr, task,
                                    compute_cancellation=False, compute_gold_val_predictions=False)

    config['finetune'] = eval_metrics
    
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

def grads(task = 'mrpc', no_val = False, return_grads = False, config = None):
    ''' Computes gradients for modules of the model'''
    if config is None:
        config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
        with open(config_path, 'r') as file:
            config = json.load(file)
    device = config['device']
    model_path = os.path.join(cwd, f'm_{task}_{seed}')
    lora_model = load_pretrained_LORA_model(model_name_or_path=model_path)
    dataset_path = os.path.join(cwd, f'd_{task}_{seed}')
    train_dataloader, eval_dataloader, _ = \
        build_loaders(dataset_path, config['tokenizer_name'], batch_size=1, 
                        shuffle_train=False)
    train_grads = compute_grads(lora_model, train_dataloader, device=device, bring_to_cpu=not return_grads)
    if no_val:
        val_grads = {}
    else:
        val_grads = compute_grads(lora_model, eval_dataloader, device=device, bring_to_cpu=not return_grads)

    if return_grads:
        return {'train': train_grads, 'validation': val_grads}
    else:
        grad_path = os.path.join(cwd, f'g_{task}_{seed}.pt')
        torch.save({'train': train_grads, 'validation': val_grads}, grad_path)

# add new methods here
influence_methods = \
    {
        "hf": compute_hessian_free_influences,
        "datainf": compute_datainf_influences,
        "lissa": compute_lissa_influences,
        "exact": compute_accurate_influences
    }

def infl(task = 'mrpc', methods = "datainf,lissa", self_influence = False, with_grads = False):
    config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
    with open(config_path, 'r') as file:
        config = json.load(file)

    device = config['device']

    if with_grads:
        gradients = grads(task = task, return_grads=True, config=config, no_val=self_influence)
    else:
        gradients = torch.load(os.path.join(cwd, f'g_{task}_{seed}.pt'))
    
    train_grads = {k:v.to(device) for k, v in gradients['train'].items()}

    if self_influence:
        val_grads = train_grads
    else:        
        val_grads = {k:v.to(device) for k, v in gradients['validation'].items()}

    runtimes = {}
    influences = {}
    for infl_method in methods.split(','):
        if infl_method not in influence_methods:
            continue
        # NOTE: not all methods require mean gradients
        avg_val_grads = {module_name: torch.mean(module_grads, dim=0) for module_name, module_grads in val_grads.items()}
        method_fn = influence_methods[infl_method]
        runtine, inf_tensors = method_fn(train_grads, val_grads, avg_val_grads, bring_to_cpu=True)
                
        # model_path = os.path.join(cwd, f'm_{task}_{seed}')
        # lora_model = load_pretrained_LORA_model(model_name_or_path=model_path)
        # dataset_path = os.path.join(cwd, f'd_{task}_{seed}')
        # train_dataloader, eval_dataloader, _ = \
        #     build_loaders(dataset_path, config['tokenizer_name'], batch_size=1, 
        #                     shuffle_train=False, val_size=500)

        # inf_tensors2 = compute_infl_from_model(lora_model, train_dataloader, eval_dataloader, device=device, infl_fn=lissa_fn)
        # pass
        
        
        influences[infl_method] = inf_tensors
        runtimes[infl_method] = runtine

    config["infl_runtimes"] = runtimes

    for infl_method, infls in influences.items():
        infl_path = os.path.join(cwd, f'i_{infl_method}_{task}_{seed}.pt')
        torch.save(infls, infl_path)

    with open(config_path, 'w') as file:
        json.dump(config, file)


def get_dataset_splits(dataset: Dataset, numel: int): 
    ''' Generator that returns split flow, datasets of max of numel size from dataset '''    
    cur_dataset = dataset
    # cur_dataset.cache_files = []
    total_shift = 0
    while len(cur_dataset) > numel:        
        new_dict = cur_dataset.train_test_split(train_size = numel, shuffle=False)  #keep_in_memory = True, load_from_cache_file=False, test_indices_cache_file_name = None, train_indices_cache_file_name= None)
        cur_dataset = new_dict['test']
        yield (total_shift, new_dict['train'])
        total_shift += len(new_dict['train'])
    if len(cur_dataset) > 0:
        yield (total_shift, cur_dataset)

class DatasetSplits:
    def __init__(self, trainset: Dataset, train_size: int, valset: Dataset, val_size: int):
        self.trainset = trainset
        self.train_size = train_size
        self.valset = valset
        self.val_size = val_size
        self.tran_batches = len(trainset) // train_size + (1 if len(trainset) % train_size > 0 else 0)
        self.val_batches = len(valset) // val_size + (1 if len(valset) % val_size > 0 else 0)
        self.total_batches = self.tran_batches * self.val_batches

    def __iter__(self):
        for val_shift, val_ds in get_dataset_splits(self.valset, self.val_size):
            for train_shift, train_ds in get_dataset_splits(self.trainset, self.train_size):
                yield (val_shift, val_ds, train_shift, train_ds)
    
    def __len__(self):
        return self.total_batches
    
# numel_levels = [*[ 100*(i + 1) for i in range(10)], *[ 1000*(i + 1) for i in range(10)], *[ 10000*(i + 1) for i in range(10)]]        
# numel_levels_with_ids = list(enumerate(numel_levels))

def pick_modules_and_split_size(active_modules_sorted: list[tuple[str, int]], train_num_samples: int, val_num_samples: int, 
                            method_memory_koef: float = 1.0,
                            memory_delta: float = 0.5,
                            device = "cuda"):
    ''' Computes necessary memory for the grads and outputs suggestion for the adjusted dataset size
        Param active_modules - list of module_name and its size in bytes
        Param method_memory_koef allows to scale memory requirement, for instance hessian-free influence would not require more memory than grads,
        but exact influence computation would require. 
        Param memory_delta - in GB, memory to leave free 
    '''

    # the atomic unit of grad computation is one parameter tensor of the model - therefore we pick the biggest module
    # max_size_module_name, max_module_byte_size = max([(param_name, param.numel() * (torch.finfo(param.dtype).bits // 8)) for param_name, param in model.named_parameters()], key=lambda x: x[1])
    # estimated_memory_size = max_module_byte_size * (train_size + val_size) * method_memory_koef
    allocated_memory = torch.cuda.memory_allocated(device)
    total_memory_bytes = torch.cuda.get_device_properties(device).total_memory - allocated_memory - (memory_delta * 1024 ** 3)
    total_memory_GB = total_memory_bytes / 1024 ** 3
    total_num_samples = train_num_samples + val_num_samples

    selected_byte_size = 0 
    selected_index = 0
    estimated_num_samples = total_num_samples

    while selected_index < len(active_modules_sorted):
        cur_mod_byte_size = active_modules_sorted[selected_index][1]
        selected_byte_size += cur_mod_byte_size
        estimated_num_samples = round(total_memory_bytes / (selected_byte_size * method_memory_koef))
        selected_index += 1
        if estimated_num_samples <= total_num_samples:
            if estimated_num_samples == 0:
                if selected_byte_size == 0:
                    estimated_num_samples = total_num_samples
                else:
                    estimated_num_samples = round(total_memory_bytes / (selected_byte_size * method_memory_koef))
                selected_byte_size -= cur_mod_byte_size
                selected_index -= 1        
            break
        
    # too_big_WARN = False
    if selected_index == 0: # module at all do noto fit this GPU - just allow to crash 
        selected_byte_size = active_modules_sorted[0][1]
        selected_index = 1     
        # too_big_WARN = True 

    if estimated_num_samples >= total_num_samples:
        estimated_num_samples = total_num_samples   
        estimated_train_size = train_num_samples
        estimated_val_size = val_num_samples
    else:
        estimated_train_size = round(train_num_samples * estimated_num_samples / total_num_samples)
        estimated_val_size = estimated_num_samples - estimated_train_size

    if estimated_val_size == 0:
        estimated_train_size -= 1
        estimated_val_size = 1

    if estimated_train_size == 0:
        # not enough memory to process
        estimated_train_size = 1

    selected_module_names = [module[0] for module in active_modules_sorted[:selected_index]]
    selected_module_names_str = ','.join(selected_module_names)

    selected_GB_size = round(selected_byte_size * (estimated_train_size + estimated_val_size) / 1024.0 ** 3)
    # if too_big_WARN:
    #     print(f"WARNING: Module {selected_module_names_str} is too big {selected_GB_size}GB for device {device} with memory {total_memory_GB}GB (delta {memory_delta}GB)")

    train_num_batches = train_num_samples // estimated_train_size + (1 if train_num_samples % estimated_train_size > 0 else 0)
    val_num_batches = val_num_samples // estimated_val_size + (1 if val_num_samples % estimated_val_size > 0 else 0)
    print(f"Modules: {selected_module_names_str}\nSize: {selected_GB_size:.2f}GB/{total_memory_GB:.2f}GB\nTrain-val split: {estimated_train_size}/{train_num_samples} ({train_num_batches} splits), {estimated_val_size}/{val_num_samples} ({val_num_batches} splits). Total splits {train_num_batches * val_num_batches}")
    return (selected_index, estimated_train_size, estimated_val_size)

class CurrentActiveModules:
    def __init__(self, cur_active_modules: list[tuple[str, int, torch.nn.Parameter]], active_modules: list[tuple[str, int, torch.nn.Parameter]]):
        self.cur_active_modules = cur_active_modules
        self.all_active_modules = active_modules
        self.cur_params = [param for _, _, param in self.cur_active_modules]
        self.enumerated_names = [name for name, _, _ in self.cur_active_modules]
        pass

    def __enter__(self):
        for _, _, param in self.all_active_modules:
            param.requires_grad = False
        for _, _, param in self.cur_active_modules:
            param.requires_grad = True

    def __exit__(self, exc_type, exc_value, traceback):
        for _, _, param in self.all_active_modules:
            param.requires_grad = True

    def numel(self):
        return sum([param.numel() for _, _, param in self.cur_active_modules])
    
    def alloc_grads(self, num_samples):
        ''' Preserves the shape of params but adds dimension for samples '''
        new_grads = [torch.zeros((num_samples, *param.shape), device=param.device) for _, _, param in self.cur_active_modules ]
        return new_grads
    
    def get_cur_params(self):
        return self.cur_params
    
    def enumerate_names(self):
        return enumerate(self.enumerated_names)
    
def get_infl_loader(dataset: Dataset, collator):
    dataloader = DataLoader(dataset,
                    shuffle=False, 
                    collate_fn=collator,
                    batch_size=1)
    return dataloader

def set_grad_values(all_grads: list[torch.Tensor], model: torch.nn.Module, active_params: list, dataloader: DataLoader, device = "cuda"):     
    for sampleId, batch in enumerate(dataloader):
        model.zero_grad()
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        for g, p in zip(all_grads, active_params):
            g[sampleId] = p.grad
    model.zero_grad()

# here we assume that positive 'infl' is good and negative is 'bad'

def matrix_hf_fn(int_view: torch.Tensor, val_grad: torch.Tensor, train_grad: torch.Tensor, **_) -> None:
    val_grad_flat = val_grad.view(val_grad.shape[0], -1)
    train_grad_flat = train_grad.view(train_grad.shape[0], -1)
    tmp_prods = torch.einsum('ik,jk->ij', val_grad_flat, train_grad_flat)
    # https://github.com/pytorch/pytorch/issues/75701
    # NOTE: out param is not supported yet
    int_view[:] = tmp_prods
    del tmp_prods, val_grad_flat, train_grad_flat
    pass 

def matrix_cos_fn(int_view: torch.Tensor, val_grad: torch.Tensor, train_grad: torch.Tensor, **_) -> None:
    val_grad_flat = val_grad.view(val_grad.shape[0], -1)
    train_grad_flat = train_grad.view(train_grad.shape[0], -1)
    tmp_prods = torch.einsum('ik,jk->ij', val_grad_flat, train_grad_flat)
    int_view[:] = tmp_prods / torch.norm(train_grad_flat, dim=-1) / torch.norm(val_grad_flat, dim=-1).view(-1, 1)
    del tmp_prods
    pass

def matrix_cov_fn(int_view: torch.Tensor, val_grad: torch.Tensor, train_grad: torch.Tensor, **_) -> None:
    val_grad_flat = val_grad.view(val_grad.shape[0], -1)
    train_grad_flat = train_grad.view(train_grad.shape[0], -1)
    train_means = torch.mean(train_grad_flat, dim=-1)
    val_means = torch.mean(val_grad_flat, dim=-1)
    train_centered = train_grad_flat - train_means.view(-1, 1)
    val_centered = val_grad_flat - val_means.view(-1, 1)
    del train_means, val_means
    tmp_prods = torch.einsum('ik,jk->ij', val_centered, train_centered) 
    int_view[:] = tmp_prods / (train_centered.shape[-1] - 1)
    del train_centered, val_centered, tmp_prods
    pass 

def matrix_datainf_one_sample_fn(int_view: torch.Tensor, val_grad: torch.Tensor, train_grad: torch.Tensor, *, lambda_const_param = 10, **_) -> None:
    val_grad_flat = val_grad.view(val_grad.shape[0], -1)
    train_grad_flat = train_grad.view(train_grad.shape[0], -1)
    module_numel = train_grad_flat.shape[-1]
    denom = torch.einsum('ki,ki->k', train_grad_flat, train_grad_flat)
    tmp_prod = torch.einsum('ik,jk->ij', val_grad_flat, train_grad_flat)
    tmp_prod /= denom
    tmp_prod *= (module_numel / ((1 / lambda_const_param) + module_numel)) 
    del denom 
    int_view[:] = tmp_prod
    del tmp_prod
    pass

def matrix_datainf_continuation(module_int_matrices: dict[str, torch.Tensor], module_val_train_products: dict[str, torch.Tensor], 
                                    module_train_train_products: dict[str, torch.Tensor], module_num_params: dict[str, int], 
                                    lambda_const_param, **_) -> None:
    for module_name, int_matrix in module_int_matrices.items():
        num_train = int_matrix.shape[1]
        num_params = module_num_params[module_name]
        val_train_prods = module_val_train_products[module_name]
        train_train_prods = module_train_train_products[module_name]
        train_train_diag = torch.diag(train_train_prods)
        lambda_const = torch.sum(train_train_diag) / (num_train * lambda_const_param * num_params)
        lambda_const_n = lambda_const * num_train
        train_train_diag += lambda_const
        train_train_prods /= train_train_diag
        tmp_prods = torch.einsum("ji,ki -> jk", val_train_prods, train_train_prods) 
        tmp_prods /= lambda_const_n
        val_train_prods /= lambda_const #NOTE: here we have difference from original impl which divides by n 
        int_matrix[:] = val_train_prods - tmp_prods # we ignore negation 
        del tmp_prods, train_train_diag
        del val_train_prods, train_train_prods
        pass

# NOTE: the following method does not work because we need to iterate through training samples twice if we want to implement datainf
# it only works when train_grad contains all training samples
def matrix_datainf_fn(__: torch.Tensor, val_grad: torch.Tensor, train_grad: torch.Tensor, lambda_const_param = 10,
                        *, module_name: str, infl_context: dict, train_shift: int, val_shift: int, full_train_size: int, full_val_size: int,
                        **_) -> None:
    ''' Here we just prepare necessary vector products, which will be used in matrix_datainf_continuation '''
    if "continuation" not in infl_context:
        infl_context["continuation"] = matrix_datainf_continuation
        infl_context["module_val_train_products"] = {}
        infl_context["module_train_train_products"] = {}
        infl_context["module_num_params"] = {}
        infl_context["lambda_const_param"] = lambda_const_param

    val_grad_flat = val_grad.view(val_grad.shape[0], -1)
    train_grad_flat = train_grad.view(train_grad.shape[0], -1)

    if module_name not in infl_context["module_val_train_products"]:
        infl_context["module_val_train_products"][module_name] = torch.zeros((full_val_size, full_train_size), device=train_grad.device, dtype = train_grad.dtype)
        infl_context["module_train_train_products"][module_name] = torch.zeros((full_train_size, full_train_size), device=train_grad.device, dtype = train_grad.dtype)
    infl_context["module_num_params"].setdefault(module_name, train_grad_flat.shape[1])
    val_batch_size = val_grad_flat.shape[0]
    train_batch_size = train_grad_flat.shape[0]
    tmp_prod1 = torch.einsum('ik,jk->ij', val_grad_flat, train_grad_flat)
    infl_context["module_val_train_products"][module_name][val_shift:val_shift + val_batch_size, train_shift:train_shift + train_batch_size] = tmp_prod1
    del tmp_prod1
    # NOTE: to compute the following we need additional loop over training samples --> TODO: finish if necessary, but for now
    tmp_prod2 = torch.einsum('ik,jk->ij', train_grad_flat, train_grad_flat)
    infl_context["module_train_train_products"][module_name][train_shift:train_shift + train_batch_size, train_shift:train_shift + train_batch_size] = tmp_prod2
    del tmp_prod2
    pass

# def matrix_lissa_fn(int_view: torch.Tensor, val_grad: torch.Tensor, train_grad: torch.Tensor, lambda_const_param = 10, n_iteration=10, alpha_const=1.) -> None:
#     n_train = train_grad.shape[0]
#     module_train_grad_squares = train_grad ** 2
#     lambda_const = torch.mean(module_train_grad_squares) / lambda_const_param

#     # hvp computation
#     running_hvp = val_grad
#     for _ in range(n_iteration):
#         new_running_hvp = val_grad + running_hvp - alpha_const * torch.sum((torch.sum(train_grad * running_hvp, dim=-1).view(-1, 1) * train_grad - lambda_const * running_hvp) / n_train, dim=0)
#         del running_hvp
#         running_hvp = new_running_hvp

#     module_infl_values = running_hvp * train_grad
#     int_view[:] = module_infl_values.sum(dim=-1)
#     del module_train_grad_squares, running_hvp, module_infl_values

def common_we(int_view: torch.Tensor, val_grad: torch.Tensor, train_grad: torch.Tensor, *, base_method_fn, train_shift, val_shift, common_tokens: dict[tuple[int, int], dict[int, tuple[int, int]]] = {}, **kwargs) -> None:
    ''' 
        Works only on embedding layer grads: i.e. 50K * 768 dense representation for BERT models 
        Modifies base_method_fn by zeroing out gradients for tokens that are not common between train and val samples
        Param common_tokens is dict of (train_id, val_id) to dict of token id to counts in (train_occur, val_occur), common to train and val sample
    '''
    # NOTE: it is infeasible to allocate mask tensor = 50K * 1K * 5K * 8 ~ 2K GB of memory 
    # grad_mask = torch.zeros((train_grad.shape[0], val_grad.shape[0], train_grad.shape[1]), dtype=train_grad.device, device=train_grad.device)
    uncommon_mask = torch.ones(train_grad.shape[1], device=train_grad.device, dtype=torch.bool)
    train_grad_clone = train_grad.clone()
    val_grad_clone = val_grad.clone()
    for train_id in range(train_grad.shape[0]):
        for val_id in range(val_grad.shape[0]):
            common_token_ids_list = list(common_tokens.get((train_shift + train_id, val_shift + val_id), {}).keys())
            common_token_ids = torch.tensor(common_token_ids_list, device=train_grad.device, dtype=torch.int)
            uncommon_mask[:] = True
            uncommon_mask[common_token_ids] = False 
            train_grad_clone[train_id, uncommon_mask] = 0
            val_grad_clone[val_id, uncommon_mask] = 0
            del common_token_ids
    base_method_fn(int_view, val_grad_clone, train_grad_clone, train_shift=train_shift, val_shift=val_shift, common_tokens = common_tokens, **kwargs)
    del uncommon_mask, train_grad_clone, val_grad_clone
    pass

def common_we_topk(int_view: torch.Tensor, val_grad: torch.Tensor, train_grad: torch.Tensor, *, base_method_fn, train_shift, val_shift, common_tokens: dict[tuple[int, int], dict[int, tuple[int, int]]] = {}, topk = 10, **kwargs) -> None:
    ''' 
        Similar to common_we, but selects topk grads by norm.
        https://proceedings.neurips.cc/paper_files/paper/2022/file/d07022783ff6f7bf7a288c207b7dcbd1-Paper-Conference.pdf
        topk = {10,20,...,100}
    '''
    # NOTE: it is infeasible to allocate mask tensor = 50K * 1K * 5K * 8 ~ 2K GB of memory 
    # grad_mask = torch.zeros((train_grad.shape[0], val_grad.shape[0], train_grad.shape[1]), dtype=train_grad.device, device=train_grad.device)
    uncommon_mask = torch.ones(train_grad.shape[1], device=train_grad.device, dtype=torch.bool)
    train_grad_clone = train_grad.clone()
    val_grad_clone = val_grad.clone()
    for train_id in range(train_grad.shape[0]):
        for val_id in range(val_grad.shape[0]):
            common_token_ids_dict = common_tokens.get((train_shift + train_id, val_shift + val_id), {})
            common_token_ids = torch.tensor(list(common_token_ids_dict.keys()), device=train_grad.device, dtype=torch.int)
            train_denom = torch.tensor([common_token_ids_dict[token_id][0] for token_id in common_token_ids_dict.keys()], device=train_grad.device, dtype=train_grad.dtype)
            train_token_scores = torch.norm(train_grad_clone[train_id, common_token_ids], dim = -1) / train_denom            
            train_token_ids_ids = torch.argsort(train_token_scores, descending=True)[:topk]
            train_token_ids = common_token_ids[train_token_ids_ids]
            val_denom = torch.tensor([common_token_ids_dict[token_id][1] for token_id in common_token_ids_dict.keys()], device=train_grad.device, dtype=val_grad.dtype)
            val_token_scores = torch.norm(val_grad_clone[val_id, common_token_ids], dim = -1) / val_denom
            val_token_ids_ids = torch.argsort(val_token_scores, descending=True)[:topk]
            val_token_ids = common_token_ids[val_token_ids_ids]
            uncommon_mask[:] = True
            uncommon_mask[train_token_ids] = False 
            train_grad_clone[train_id, uncommon_mask] = 0
            uncommon_mask[:] = True
            uncommon_mask[val_token_ids] = False 
            val_grad_clone[val_id, uncommon_mask] = 0
            del common_token_ids, train_denom, train_token_scores, train_token_ids_ids, val_denom, val_token_scores, val_token_ids_ids
    base_method_fn(int_view, val_grad_clone, train_grad_clone, train_shift=train_shift, val_shift=val_shift, common_tokens = common_tokens, **kwargs)
    del uncommon_mask, train_grad_clone, val_grad_clone
    pass

# def matrix_tf_idf_head():
#     ''' 
#         Executes on head, tf_idf of train sample on val sample tokens 
#     '''
#     pass 

matrix_infl_methods = {
    "hf": matrix_hf_fn,
    "hf_we_": partial(common_we, base_method_fn=matrix_hf_fn),
    "hw_we_topk_10": partial(common_we_topk, base_method_fn=matrix_hf_fn, topk=10),
    # "hw_we_topk_20": partial(common_we_topk, base_method_fn=matrix_hf_fn, topk=20),
    "cos": matrix_cos_fn,
    # "cos_we": partial(common_we, base_method_fn=matrix_cos_fn),
    "cov": matrix_cov_fn,
    # "cov_we": partial(common_we, base_method_fn=matrix_cov_fn),
    "datainf_one": matrix_datainf_one_sample_fn,
    'datainf': matrix_datainf_fn,
    # "datainf_we": partial(common_we, base_method_fn=matrix_datainf_fn),
    # "lissa": matrix_lissa_fn,
    # "lissa_we": partial(common_we, base_method_fn=matrix_lissa_fn),
}

# Mem koef NOTE: hf (1.1, 0.3), hf_we_ (2, 0.3), hf_we_topk (2, 0.3), cos (1.1, 0.3), cov (2, 0.3), datainf_one (1.1, 0.3), datainf (2, 0.3), 
def infl_matrix(task = 'mrpc', methods = "hf,hf_we_,hw_we_topk_10,cos,cov,datainf_one,datainf", mem_koef: float = 2.0, mem_delta: float = 0.3):
    config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
    with open(config_path, 'r') as file:
        config = json.load(file)

    # method_list = methods.split(',')
    device = config['device']
    unfreeze_regex = config.get('unfreeze_regex', None)

    # if with_grads:
    #     gradients = grads(task = task, return_grads=True, config=config, no_val=self_influence)
    # else:
    #     gradients = torch.load(os.path.join(cwd, f'g_{task}_{seed}.pt'))

    model_path = os.path.join(cwd, f'm_{task}_{seed}')

    # todo check if reequires_grad is attached to embedding
    lora_model = load_pretrained_LORA_model(model_name_or_path=model_path, unfreeze_modules_regex=unfreeze_regex)
    lora_model.to(device)
    lora_model.eval()  

    active_modules = [(name, param.numel() * (torch.finfo(param.dtype).bits // 8), param) for name, param in lora_model.named_parameters() if param.requires_grad]
    active_modules.sort(key=lambda x: x[1], reverse=True)

    dataset_path = os.path.join(cwd, f'd_{task}_{seed}')

    datasets = load_from_disk(dataset_path) # add validation dataset 

    trainset = datasets['train']
    trainset = trainset.remove_columns(['noise'])
    tokenizer = load_tokenizer(config['tokenizer_name'])
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")  

    # method_fn = matrix_infl_methods[method]
    method_names = [name for name in methods.split(',') if name in matrix_infl_methods]

    valset = datasets['infl']

    common_tokens = {}

    if any('_we_' in method_name for method_name in method_names):
        common_tokens_ds = os.path.join(cwd, f'common_tokens_{task}_{seed}.pkl')
        if os.path.exists(common_tokens_ds):
            with open(common_tokens_ds, 'rb') as file:
                common_tokens = pickle.load(file)
        else:
            # need to compute common tokens         
            train_token_counts = {}
            for train_id in range(len(trainset)):
                cur_train_token_counts = train_token_counts.setdefault(train_id, {})
                for token_id in trainset[train_id]['input_ids']:
                    cur_train_token_counts[token_id] = cur_train_token_counts.get(token_id, 0) + 1
            val_token_counts = {}
            for val_id in range(len(valset)):
                cur_val_token_counts = val_token_counts.setdefault(val_id, {})
                for token_id in valset[val_id]['input_ids']:
                    cur_val_token_counts[token_id] = cur_val_token_counts.get(token_id, 0) + 1
            for train_id, train_token_count in train_token_counts.items():
                for val_id, val_token_count in val_token_counts.items():
                    common_token_ids = set(train_token_count.keys()).intersection(val_token_count.keys())
                    common_tokens[(train_id, val_id)] = {token_id: (train_token_count[token_id], val_token_count[token_id]) for token_id in common_token_ids}
            with open(common_tokens_ds, 'wb') as file:
                pickle.dump(common_tokens, file)

    interaction_matrices = {method_name: {module_name: torch.zeros((len(valset), len(trainset)), device=device)                                             
                                            for module_name, _, module_params in active_modules
                                            if (method_name == 'datainf' and (module_params.numel() < 100000)) or
                                                ((method_name != 'datainf') and ('_we_' not in method_name)) or 
                                                ((method_name != 'datainf') and ('_we_' in method_name) and ('.word_embeddings.' in module_name))} 
                                for method_name in method_names}
    
    interaction_modules = set([module_name for int_matrices in interaction_matrices.values() for module_name in int_matrices.keys()])

    all_active_modules = [(name, size, params) for name, size, params in active_modules if name in interaction_modules]    
    
    while len(all_active_modules) > 0:
        selected_module_count, train_size, test_size = pick_modules_and_split_size(all_active_modules, len(trainset), len(valset), method_memory_koef=mem_koef, memory_delta=mem_delta, device=device)
        cur_active_modules = all_active_modules[:selected_module_count]
        all_active_modules = all_active_modules[selected_module_count:]
        cur_params = CurrentActiveModules(cur_active_modules, active_modules)
        infl_contexts = {method_name: {} for method_name in method_names} # methods could store arbitrary data here between calls to method_fn
        with cur_params:
            train_val_splits = DatasetSplits(trainset, train_size, valset, test_size)

            val_grads = None

            for val_shift, cur_val_dataset, train_shift, cur_train_dataset in tqdm(train_val_splits): 
                if train_shift == 0:
                    if val_grads is not None:
                        del val_grads
                    val_grads = cur_params.alloc_grads(len(cur_val_dataset))
                    val_dataloader = get_infl_loader(cur_val_dataset, collator)
                    set_grad_values(val_grads, lora_model, cur_params.get_cur_params(), val_dataloader, device=device)
                train_grads = cur_params.alloc_grads(len(cur_train_dataset))
                train_dataloader = get_infl_loader(cur_train_dataset, collator)
                set_grad_values(train_grads, lora_model, cur_params.get_cur_params(), train_dataloader, device=device)
                for method_name, module_int_matrices in interaction_matrices.items():
                    for cur_module_id, module_name in cur_params.enumerate_names():
                        if module_name in module_int_matrices:
                            cur_int_matrix_view = module_int_matrices[module_name][val_shift:val_shift + len(cur_val_dataset), train_shift:train_shift + len(cur_train_dataset)]
                            method_fn = matrix_infl_methods[method_name]
                            method_fn(cur_int_matrix_view, val_grads[cur_module_id], train_grads[cur_module_id],
                                    module_name = module_name, infl_context = infl_contexts[method_name],
                                    train_shift=train_shift, val_shift=val_shift, 
                                    full_train_size = len(trainset), full_val_size = len(valset),
                                    common_tokens=common_tokens)
                del train_grads
                torch.cuda.empty_cache() 
            if val_grads is not None:
                del val_grads
            for method_name, method_infl_context in infl_contexts.items():
                if 'continuation' in method_infl_context:
                    cur_int_matrices = interaction_matrices[method_name]
                    method_infl_context['continuation'](cur_int_matrices, **method_infl_context)
        pass 


    for method_name, infl_matrices in interaction_matrices.items():
        infl_path = os.path.join(cwd, f'i_{method_name}_{task}_{seed}.pt')
        torch.save(infl_matrices, infl_path)

    # with open(config_path, 'w') as file:
    #     fcntl.flock(file, fcntl.LOCK_EX)
    #     json.dump(config, file)
    #     fcntl.flock(file, fcntl.LOCK_UN)    

def finetune2(task = 'mrpc',
         num_epochs = 10, filter_method='infl', infl_method='datainf', module_pattern='', filter_perc = 0.7):
    config_path = os.path.join(cwd, f'c_{task}_{seed}.json')
    with open(config_path, 'r') as file:
        config = json.load(file)    
    config.update(finetune2=dict(num_epochs=num_epochs, filter_method=filter_method,
                                 infl_method=infl_method, module_pattern=module_pattern, filter_perc=filter_perc))
    print(f"Finetune with modules: {module_pattern}")
    device = config['device']
    lr = config['lr']
    model = config['model']
    batch_size = config['batch_size']
    target_modules = config['target_modules']
    if filter_method == "infl":
        influences = torch.load(os.path.join(cwd, f'i_{infl_method}_{task}_{seed}.pt'))

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

        def infl_filter(train_dataset):
            high_to_low_quality = torch.argsort(influence)
            filter_len = int(filter_perc*len(high_to_low_quality))
            filtered_indexes = high_to_low_quality[:filter_len].cpu().numpy()
            return train_dataset.select(filtered_indexes)
        filter_fn = infl_filter
    elif filter_method == 'rand':
        def rand_filter(train_dataset):
            filter_len = int(filter_perc*len(train_dataset))
            filtered_indexes = np.random.choice(len(train_dataset), filter_len, replace=False)
            return train_dataset.select(filtered_indexes)
        filter_fn = rand_filter        
    elif filter_method == 'denoise':
        def denoise_map(example):
            if example['noise']:
                example['labels'] = 1 - example['labels'] 
            return example               
        def denoise_filter(train_dataset):
            return train_dataset.map(denoise_map)
        filter_fn = denoise_filter
    else:
        filter_fn = None

    dataset_path = os.path.join(cwd, f'd_{task}_{seed}')
    train_dataloader, eval_dataloader, _ = \
        build_loaders(dataset_path, config['tokenizer_name'], batch_size, filter_fn=filter_fn)

    lora_model = build_LORA_model(model_name_or_path=model,
                                target_modules=target_modules, 
                                low_rank=config['low_rank'])
    eval_metrics = train_LORA_model(lora_model, train_dataloader, eval_dataloader, device, num_epochs, lr, task)

    del lora_model, train_dataloader, eval_dataloader

    metric_file = os.path.join(cwd, f'metrics.jsonlist')
    with open(metric_file, "a") as f:                
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({**config, "metrics": eval_metrics}) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)    

    with torch.no_grad():
        torch.cuda.empty_cache()

parser = argh.ArghParser()
parser.add_commands([preprocess, finetune, grads, infl, infl_matrix, finetune2])

if __name__ == '__main__':
    parser.dispatch()


# import torch

# a = torch.zeros(10, dtype=torch.int)

# i = torch.tensor([0, 1, 1, 1, 2, 2, 5])
# i_u, i_c = torch.unique(i, return_counts=True)

# a[i_u] += i_c
# n = 100
# a = torch.rand(n, dtype = torch.float)
# b = torch.rand(n, dtype = torch.float)
# ab = torch.outer(a, b)

# d = torch.rand(n, dtype = torch.float)
# l = torch.mm(d.view(1, -1), ab)
# e = torch.rand(n, dtype = torch.float)
# f1 = torch.dot(l.view(-1), e)

# l2 = torch.dot(d, a)
# l3 = torch.dot(b, e)
# l2 * l3

# torch.mm(d.view(1, -1), torch.eye(n, dtype = torch.float))