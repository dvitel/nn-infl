import fcntl
import json
import os
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

def infl(run_id=0, task = 'mrpc', low_rank = 4, noise_ratio = 0.2,
         device = 'cuda', lr = 3e-4, model = 'distilbert/distilroberta-base', batch_size = 32,
         num_epochs = 10, target_modules = ['value'], compute_accurate = False,
         cwd = "."):
    config = locals()
    print(config)
    del config['cwd']    
    _set_seed(run_id)
    if low_rank > 4:
        compute_accurate=False
    
    # fine-tuning models
    train_dataloader, eval_dataloader, noise_index, tokenized_datasets, collate_fn = \
        create_dataloaders(model_name_or_path=model,
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
    influence_engine.compute_IF()
    influences = influence_engine.get_metrics()
    influences['config'] = config
    influences['eval_metrics'] = eval_metrics

    work_path = os.path.join(cwd, f'infl_{run_id}.pkl')
    with open(work_path, 'wb') as file:
        pickle.dump(influences, file)

    del tr_grad_dict, val_grad_dict, noise_index, influence_engine
    with torch.no_grad():
        torch.cuda.empty_cache()


def filter(run_id=0, task = 'mrpc', low_rank = 4, noise_ratio = 0.2,
         device = 'cuda', lr = 3e-4, model = 'distilbert/distilroberta-base', batch_size = 32,
         num_epochs = 10, target_modules = ['value'], infl_method='DataInf', infl_module='',
         filter_perc = 0.7, cwd = "."):
    config = locals()
    print(config)
    del config['cwd']
    work_path = os.path.join(cwd, f'infl_{run_id}.pkl')
    with open(work_path, 'rb') as file:
        all_influences = pickle.load(file)
    influences = all_influences[infl_method]
    if infl_module == '':
        influence = influences['']
    else:
        for module in influences:
            if infl_module in module: 
                influence = influences[module]
                break
    _set_seed(run_id)
    
    # fine-tuning models
    train_dataloader, eval_dataloader, noise_index, tokenized_datasets, collate_fn = \
        create_filtered_dataloaders(model_name_or_path=model,
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

    del lora_engine, train_dataloader, eval_dataloader, tokenized_datasets, collate_fn

    metric_file = os.path.join(cwd, f'metrics.jsonlist')
    with open(metric_file, "a") as f:                
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({**config, **eval_metrics}) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)    

    with torch.no_grad():
        torch.cuda.empty_cache()

parser = argh.ArghParser()
parser.add_commands([infl, filter])

if __name__ == '__main__':
    parser.dispatch()