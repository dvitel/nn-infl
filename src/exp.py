import fcntl
import json
import os
import re
os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache')
import pickle
import random

import argh
import numpy as np
from dataloader import create_dataloaders, create_filtered_dataloaders
from lora_model import LORAEngine
from influence import IFEngine
import torch

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def infl(run_id=0, task = 'mrpc', low_rank = 4, noise_ratio = 0.2,
         device = 'cuda', lr = 3e-4, model = 'distilbert/distilroberta-base', batch_size = 32,
         num_epochs = 10, target_modules = ['value'], compute_accurate = False,
         cwd = "."):
    config = dict(run_id=run_id, task=task, low_rank=low_rank, noise_ratio=noise_ratio,
                  device=device, lr=lr, model=model, batch_size=batch_size,
                  num_epochs=num_epochs, target_modules=target_modules, compute_accurate=compute_accurate)
    print(config)
    _set_seed(run_id)
    if low_rank > 4:
        compute_accurate=False
    
    # fine-tuning models
    train_dataloader, eval_dataloader, noise_index, tokenized_datasets, collate_fn = \
        create_dataloaders(run_id, model_name_or_path=model,
            task=task,
            noise_ratio=noise_ratio,
            batch_size=batch_size)

    lora_engine = LORAEngine(model_name_or_path=model,
                                target_modules=target_modules,
                                train_dataloader=train_dataloader,
                                eval_dataloader=eval_dataloader,
                                device=device,
                                num_epochs=num_epochs,
                                lr=lr,
                                task=task,
                                low_rank=low_rank)
    lora_engine.build_LORA_model()
    eval_metrics = lora_engine.train_LORA_model()
    tr_grad_dict, val_grad_dict = lora_engine.compute_gradient(tokenized_datasets, collate_fn)

    del lora_engine, train_dataloader, eval_dataloader, tokenized_datasets, collate_fn

    # influence functions
    influence_engine = IFEngine()
    influence_engine.preprocess_gradients(tr_grad_dict, val_grad_dict, noise_index)
    influence_engine.compute_hvps(compute_accurate=compute_accurate)
    influences = influence_engine.compute_all_influences()
    influences['config'] = config
    influences['eval_metrics'] = eval_metrics
    influences['noise_index'] = noise_index

    work_path = os.path.join(cwd, f'infl_{task}_{run_id}.pkl')
    with open(work_path, 'wb') as file:
        pickle.dump(influences, file)

    del tr_grad_dict, val_grad_dict, noise_index, influence_engine
    with torch.no_grad():
        torch.cuda.empty_cache()


def filter(run_id=0, task = 'mrpc',
         num_epochs = 10, infl_method='DataInf', infl_module='',
         filter_perc = 0.7, cwd = "."):
    work_path = os.path.join(cwd, f'infl_{task}_{run_id}.pkl')
    with open(work_path, 'rb') as file:
        all_influences = pickle.load(file)
    config = all_influences['config']
    config['num_epochs'] = num_epochs
    config['infl_method'] = infl_method
    config['infl_module'] = infl_module
    config['filter_perc'] = filter_perc
    low_rank = config['low_rank']
    noise_ratio = config['noise_ratio']
    device = config['device']
    lr = config['lr']
    model = config['model']
    batch_size = config['batch_size']
    target_modules = config['target_modules']
    influences = all_influences['influences'][infl_method]
    if infl_module == '':
        num_modules = -1
        influence = influences['']
    else:
        num_modules = 0
        pattern = re.compile(re.escape(infl_module))
        influence = None
        for module in influences:
            if pattern.match(module): 
                if influence is None:
                    influence = np.array(influences[module])
                else:
                    influence += np.array(influences[module])
                num_modules += 1
    if num_modules == 0:
        return
    config['num_modules'] = num_modules
    _set_seed(run_id)
    
    # fine-tuning models
    train_dataloader, eval_dataloader, noise_index, tokenized_datasets, collate_fn = \
        create_filtered_dataloaders(run_id, model_name_or_path=model,
            task=task, noise_ratio=noise_ratio,
            batch_size=batch_size, influence = influence, filter_perc=filter_perc)

    lora_engine = LORAEngine(model_name_or_path=model,
                                target_modules=target_modules,
                                train_dataloader=train_dataloader,
                                eval_dataloader=eval_dataloader,
                                device=device,
                                num_epochs=num_epochs,
                                lr=lr,
                                task=task,
                                low_rank=low_rank)
    lora_engine.build_LORA_model()
    eval_metrics = lora_engine.train_LORA_model()
    metrics = {}
    for epoch in eval_metrics:
        for metric, value in epoch.items():
            metrics.setdefault(metric, []).append(value)

    del lora_engine, train_dataloader, eval_dataloader, tokenized_datasets, collate_fn

    metric_file = os.path.join(cwd, f'metrics.jsonlist')
    with open(metric_file, "a") as f:                
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({**config, "metrics": metrics}) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)    

    with torch.no_grad():
        torch.cuda.empty_cache()

parser = argh.ArghParser()
parser.add_commands([infl, filter])

if __name__ == '__main__':
    parser.dispatch()