import re
from time import time
from typing import Optional
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import pickle, os
import torch

from cifar import OneDataset

def hessian_free_fn(module_train_grad, module_avg_val_grads):
    module_infl_values = module_avg_val_grads * module_train_grad
    module_infs = module_infl_values.sum(dim=-1)
    del module_infl_values
    return module_infs

def hessian_free_vec_fn(module_train_grad, module_val_grads):
    ''' Return infl matrix '''
    # module_infl_values = module_val_grads * module_train_grad
    infl_matrix = torch.einsum('ik,jk->ij', module_val_grads, module_train_grad)
    return infl_matrix

def cosine_vec_fn(module_train_grad, module_val_grads):
    infl_matrix = torch.einsum('ik,jk->ij', module_val_grads, module_train_grad)
    module_train_grad_norm = torch.norm(module_train_grad, dim=-1)
    infl_matrix /= module_train_grad_norm
    del module_train_grad_norm
    module_val_grads_norm = torch.norm(module_val_grads, dim=-1)
    infl_matrix /= module_val_grads_norm.view(-1, 1)
    del module_val_grads_norm
    return infl_matrix

# a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
# b = torch.tensor([[5, 6], [7, 8], [9, 10]], dtype=torch.float)
# res = cosine_vec_fn(a, b)
# print(res)
# pass

def datainf_fn(module_train_grad, module_avg_val_grads, lambda_const_param=10):    
    module_train_grad_squares = module_train_grad ** 2
    lambda_const = torch.mean(module_train_grad_squares) / lambda_const_param
    denom_values = lambda_const + torch.sum(module_train_grad_squares, dim=-1)
    del module_train_grad_squares
    const_val = lambda_const * module_train_grad.shape[0]

    C_tmp_values = module_avg_val_grads * module_train_grad
    nom_values = torch.sum(C_tmp_values, dim=-1)
    del C_tmp_values
    C_tmp = nom_values / denom_values
    del nom_values

    C_tmp_grad_values = C_tmp.view(-1, 1) * module_train_grad
    del C_tmp
    module_hvp_values = (module_avg_val_grads - C_tmp_grad_values) / const_val
    del C_tmp_grad_values

    module_hvp = torch.sum(module_hvp_values, dim=0)
    del module_hvp_values

    module_infl_values = module_hvp * module_train_grad
    module_infls = module_infl_values.sum(dim=-1)
    del module_hvp, denom_values, module_infl_values
    return module_infls

def lissa_fn(module_train_grad, module_avg_val_grads, lambda_const_param=10, n_iteration=10, alpha_const=1.):
    n_train = module_train_grad.shape[0]
    module_train_grad_squares = module_train_grad ** 2
    lambda_const = torch.mean(module_train_grad_squares) / lambda_const_param

    # hvp computation
    running_hvp = module_avg_val_grads
    for _ in range(n_iteration):
        hvp_tmp_values = module_train_grad * running_hvp
        hvp_tmp_sum = torch.sum(hvp_tmp_values, dim=-1)
        del hvp_tmp_values
        hvp_tmp_0 = (hvp_tmp_sum.view(-1, 1) * module_train_grad - lambda_const * running_hvp) / n_train
        del hvp_tmp_sum
        hvp_tmp = torch.sum(hvp_tmp_0, dim=0)
        del hvp_tmp_0
        new_running_hvp = module_avg_val_grads + running_hvp - alpha_const * hvp_tmp
        del hvp_tmp, running_hvp
        running_hvp = new_running_hvp

    module_infl_values = running_hvp * module_train_grad
    module_infls = module_infl_values.sum(dim=-1)
    del module_train_grad_squares, running_hvp, module_infl_values
    return module_infls

def print_tensors_in_use(device="cuda"):
    summary = torch.cuda.memory_summary(device=device, abbreviated=False)
    print(summary)

def compute_infl_from_model(model: torch.nn.Module, train_dataset: OneDataset, 
                                    val_dataset: OneDataset, device = "cuda",
                                    module_patterns: list[str] = [], filter_list = None,
                                    infl_fn = hessian_free_fn, max_num_el = 10000, size_koef = 0.5):
    ''' Use this for large models - does not require to store all gradients in memory, but
        computes influence on request '''

    start_time = time()
    
    patterns = [ (re.compile(p), p) for p in module_patterns ]
    model.to(device)
    model.eval()

    module_infls = {}
    num_val_samples = len(val_dataset)
    total_num_train_samples = train_dataset.total_len()
    module_infls[''] = torch.zeros(total_num_train_samples, device = device, dtype=torch.float)
    module_groups = {}
    active_modules = {}
    # filter_list = ['lora_A', 'lora_B', 'modules_to_save.default.out_proj.weight']
    for module_name, module_params in model.named_parameters():
        if not module_params.requires_grad:
            continue
        if filter_list is not None and not any(f in module_name for f in filter_list):
            continue
        active_modules[module_name] = module_params
        for p, p_str in patterns:
            if p.match(module_name):
                module_groups.setdefault(module_name, []).append(p_str)
                if p_str not in module_infls:
                    module_infls[p_str] = torch.zeros(total_num_train_samples, device = device, dtype=module_params.dtype)

    
    val_dataloader = torch.utils.data.DataLoader(dataset = val_dataset,
                                        batch_size = 1,
                                        num_workers = 2,
                                        shuffle=False,
                                        drop_last = False)        
    model.zero_grad()
    for batch in tqdm(val_dataloader):
        if type(batch) == list:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            # model.zero_grad() # we aggregate all gradients in .grad
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
        else:
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

    avg_val_grad = {}
    
    for module_name, module_params in active_modules.items():
        avg_val_grad[module_name] = module_params.grad.reshape(-1) / num_val_samples

    active_module_list = list(active_modules.keys())
    active_module_list.sort(key=lambda x: active_modules[x].numel(), reverse=True)

    total_memory = (torch.cuda.get_device_properties(device).total_memory / (1024.0 ** 3) - 0.5)

    adjusted_sizes = [*[ 100*(i + 1) for i in range(10)], *[ 1000*(i + 1) for i in range(10)], *[ 10000*(i + 1) for i in range(10)]]        
    adjusted_sizes_pairs = list(enumerate(adjusted_sizes))

    # testing
    # active_module_list = ['linear.weight', 'conv1.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.shortcut.1.weight', 'layer4.0.shortcut.1.bias']    

    while True:
        total_numels = 0        
        current_active_modules = {}
        while len(active_module_list) > 0:
            module_name = active_module_list[0]
            module_params = active_modules[module_name]
            cur_numel = module_params.numel()
            if (total_numels + cur_numel > max_num_el) and (total_numels > 0):
                break 
            current_active_modules[module_name] = module_params
            total_numels += cur_numel
            active_module_list.pop(0)
        if len(current_active_modules) == 0:
            break
        allocated_memory = round(torch.cuda.memory_allocated() / (1024. ** 3), 2)
        reserved_memory = round(torch.cuda.memory_reserved() / (1024. ** 3), 2)
        el_mem = round(((len(train_dataset) * total_numels) * 8. / (1024.0 ** 3)), 2)
        print(f">> Infl of {list(current_active_modules.keys())} [{total_numels} els, {el_mem}GB].\nCUDA MEM: {allocated_memory}GB, reserved {reserved_memory}GB")
        # print_tensors_in_use(device=device)
        for module_params_i in active_modules.values():
            module_params_i.requires_grad = False
        for module_params_i in current_active_modules.values():
            module_params_i.requires_grad = True

        prec_size = round(total_memory * (1024. ** 3) / (total_numels * 8) * size_koef)

        adjusted_size_i = next((i - 1 for i, size in adjusted_sizes_pairs if prec_size < size), None)
        adjusted_size = train_dataset.total_len() if adjusted_size_i is None or adjusted_size_i == -1 else adjusted_sizes[adjusted_size_i]

        if adjusted_size is None:
            adjusted_size = adjusted_sizes[-1]

        train_dataset.load_next_dataset(start_index=0, size = adjusted_size)  
        
        print(f">> tot mem {total_memory}, prec size {prec_size}, adjusted size {adjusted_size}, db size {len(train_dataset)}")    

        train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = 1,
                                            num_workers = 2,
                                            shuffle=False,
                                            drop_last = False)       
        has_dataset = True 
        while has_dataset:  
            active_train_grads = {}
            for module_name, module_params in current_active_modules.items():
                module_train_grad = torch.zeros((len(train_dataset), module_params.numel()), device = device, dtype=module_params.dtype)
                active_train_grads[module_name] = (module_params, module_train_grad)
            # module_train_grad = torch.zeros((len(train_dataset), total_numels), device = device, dtype=torch.float)
            for step, batch in enumerate(tqdm(train_dataloader)):
                model.zero_grad() # we aggregate all gradients in .grad
                if type(batch) == list:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    logits = model(inputs)
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    loss.backward()
                else:
                    batch.to(device)
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()  

                for module_name, (module_params, module_train_grad) in active_train_grads.items():
                    module_train_grad[step] = module_params.grad.view(-1)

            model.zero_grad()

            for module_name, (module_params, module_train_grad) in active_train_grads.items():
                module_infls_one = infl_fn(module_train_grad, avg_val_grad[module_name])
                module_infls[''][train_dataset.start_index:train_dataset.end_index] += module_infls_one
                for p_str in module_groups.get(module_name, []):
                    module_infls[p_str][train_dataset.start_index:train_dataset.end_index] += module_infls_one
                del module_infls_one, module_train_grad
            has_dataset = train_dataset.load_next_dataset()       
        torch.cuda.empty_cache() 
    for module_params_i in active_modules.values():
        module_params_i.requires_grad = True
    for infl in module_infls.values():
        infl.neg_()         
    timespan = time() - start_time
    return timespan, module_infls


def compute_infl_matrix_from_model(model: torch.nn.Module, train_dataset: OneDataset, 
                                    val_dataset: OneDataset, device = "cuda",
                                    module_patterns: list[str] = [], filter_list = None,
                                    infl_vec_fn = hessian_free_vec_fn, max_num_el = 10000, size_koef = 0.5,
                                    val_set_batch_size = 1000):
    ''' Instead of aggregating of influence accross all validation samples, builds matrix Infl(v, x) '''

    start_time = time()

    
    patterns = [ (re.compile(p), p) for p in module_patterns ]
    model.half()
    model.to(device)
    model.eval()
    first_params = next(model.parameters())

    infl_matrices = {}
    num_val_samples = val_dataset.total_len()
    total_num_train_samples = train_dataset.total_len()
    infl_matrices[''] = torch.zeros(num_val_samples, total_num_train_samples, device = first_params.dtype, dtype=first_params.dtype)
    module_groups = {}
    active_modules = {}
    # filter_list = ['lora_A', 'lora_B', 'modules_to_save.default.out_proj.weight']
    for module_name, module_params in model.named_parameters():
        if not module_params.requires_grad:
            continue
        if filter_list is not None and not any(f in module_name for f in filter_list):
            continue
        active_modules[module_name] = module_params
        for p, p_str in patterns:
            if p.match(module_name):
                module_groups.setdefault(module_name, []).append(p_str)
                if p_str not in infl_matrices:
                    infl_matrices[p_str] = torch.zeros(num_val_samples, total_num_train_samples, device = device, dtype=module_params.dtype)

    # avg_val_grad = {}
    
    # for module_name, module_params in active_modules.items():
    #     avg_val_grad[module_name] = module_params.grad.reshape(-1) / num_val_samples

    active_module_list = list(active_modules.keys())
    active_module_list.sort(key=lambda x: active_modules[x].numel(), reverse=True)

    total_memory = (torch.cuda.get_device_properties(device).total_memory / (1024.0 ** 3) - 0.5)

    adjusted_sizes = [*[ 100*(i + 1) for i in range(10)], *[ 1000*(i + 1) for i in range(10)], *[ 10000*(i + 1) for i in range(10)]]        
    adjusted_sizes_pairs = list(enumerate(adjusted_sizes))

    # testing
    # active_module_list = ['linear.weight', 'conv1.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.shortcut.1.weight', 'layer4.0.shortcut.1.bias']    

    while True:
        total_numels = 0        
        current_active_modules = {}
        while len(active_module_list) > 0:
            module_name = active_module_list[0]
            module_params = active_modules[module_name]
            cur_numel = module_params.numel()
            if (total_numels + cur_numel > max_num_el) and (total_numels > 0):
                break 
            current_active_modules[module_name] = module_params
            total_numels += cur_numel
            active_module_list.pop(0)
        if len(current_active_modules) == 0:
            break
        allocated_memory = round(torch.cuda.memory_allocated() / (1024. ** 3), 2)
        reserved_memory = round(torch.cuda.memory_reserved() / (1024. ** 3), 2)
        el_mem = round(((len(train_dataset) * total_numels) * 8. / (1024.0 ** 3)), 2)
        print(f">> Infl of {list(current_active_modules.keys())} [{total_numels} els, {el_mem}GB].\nCUDA MEM: {allocated_memory}GB, reserved {reserved_memory}GB")
        # print_tensors_in_use(device=device)
        for module_params_i in active_modules.values():
            module_params_i.requires_grad = False
        for module_params_i in current_active_modules.values():
            module_params_i.requires_grad = True

        prec_size = round(total_memory * (1024. ** 3) / (total_numels * 8) * size_koef)

        adjusted_size_i = next((i - 1 for i, size in adjusted_sizes_pairs if prec_size < size), None)
        adjusted_size = train_dataset.total_len() if adjusted_size_i is None or adjusted_size_i == -1 else adjusted_sizes[adjusted_size_i]

        if adjusted_size is None:
            adjusted_size = adjusted_sizes[-1]

        train_dataset.load_next_dataset(start_index=0, size = adjusted_size)  
        val_dataset.load_next_dataset(start_index=0, size = val_set_batch_size)
        
        print(f">> tot mem {total_memory}, prec size {prec_size}, adjusted size {adjusted_size}, db size {len(train_dataset)}")    

        train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = 1,
                                            num_workers = 2,
                                            shuffle=False,
                                            drop_last = False)       
        
        val_dataloader = torch.utils.data.DataLoader(dataset = val_dataset,
                                            batch_size = 1,
                                            num_workers = 2,
                                            shuffle=False,
                                            drop_last = False)            
        has_dataset = True 
        while has_dataset:  
            active_train_grads = {}
            for module_name, module_params in current_active_modules.items():
                module_train_grad = torch.zeros((len(train_dataset), module_params.numel()), device = device, dtype=module_params.dtype)
                active_train_grads[module_name] = (module_params, module_train_grad)
            # module_train_grad = torch.zeros((len(train_dataset), total_numels), device = device, dtype=torch.float)
            for step, batch in enumerate(tqdm(train_dataloader)):
                model.zero_grad() # we aggregate all gradients in .grad
                if type(batch) == list:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    logits = model(inputs)
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    loss.backward()
                else:
                    batch.to(device)
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()  

                for module_name, (module_params, module_train_grad) in active_train_grads.items():
                    module_train_grad[step] = module_params.grad.view(-1)
    
            has_val_dataset = True 

            while has_val_dataset:
                active_grads = {}
                for module_name, (module_params, module_train_grad) in active_train_grads.items():
                    module_val_grad = torch.zeros((len(val_dataset), module_params.numel()), device = device, dtype=module_params.dtype)
                    active_grads[module_name] = (module_params, module_val_grad, module_train_grad)

                for step, batch in enumerate(tqdm(val_dataloader)):
                    model.zero_grad()
                    if type(batch) == list:
                        inputs, labels = batch
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        # model.zero_grad() # we aggregate all gradients in .grad
                        logits = model(inputs)
                        loss = torch.nn.functional.cross_entropy(logits, labels)
                        loss.backward()
                    else:
                        batch.to(device)
                        outputs = model(**batch)
                        loss = outputs.loss
                        loss.backward()

                    for module_name, (module_params, module_val_grad, module_train_grad) in active_grads.items():
                        module_val_grad[step] = module_params.grad.view(-1)                                    

            for module_name, (module_params, module_val_grad, module_train_grad) in active_grads.items():
                module_infls_submatrix = infl_vec_fn(module_train_grad, module_val_grad)
                del module_train_grad, module_val_grad
                infl_matrices[''][val_dataset.start_index:val_dataset.end_index,train_dataset.start_index:train_dataset.end_index] += module_infls_submatrix
                for p_str in module_groups.get(module_name, []):
                    infl_matrices[p_str][val_dataset.start_index:val_dataset.end_index, train_dataset.start_index:train_dataset.end_index] += module_infls_submatrix
                del module_infls_submatrix
            has_dataset = train_dataset.load_next_dataset()       
        torch.cuda.empty_cache() 
    for module_params_i in active_modules.values():
        module_params_i.requires_grad = True
    for infl in infl_matrices.values():
        infl.neg_()         
    timespan = time() - start_time
    return timespan, infl_matrices

def compute_influence_from_hvp(modules_hvp, modules_grad, bring_to_cpu=False):
    # if_tmp_dict = {}
    module_influences = defaultdict(dict)
    total_infl = None
    for module_name, module_grad in modules_grad.items():
        module_hvp = modules_hvp[module_name]
        module_infl_values = module_hvp * module_grad
        module_infl = torch.sum(module_infl_values, dim=-1)
        if total_infl is None:
            total_infl = module_infl.clone()
        else:
            total_infl += module_infl
        module_influences[module_name] = module_infl
        del module_infl_values
    module_influences[''] = total_infl
    for infl in module_influences.values():
        infl.neg_() 
    if bring_to_cpu:
        module_influences_cpu = {}
        for module_name, module_infls in module_influences.items():
            module_influences_cpu[module_name] = module_infls.cpu() 
            del module_infls
        module_influences = module_influences_cpu
    return module_influences
        
    # self.influences[method_name] = {layer_name:pd.Series(influences, dtype=float).to_numpy() for layer_name, influences in module_influences.items() }
    # self.influences[method_name][''] = pd.Series(if_tmp_dict, dtype=float).to_numpy()
    # # return {"runtime": self.time_dict, "influences": self.influences}

def avg_grad(modules_grad):
    avg_grads = {module_name: torch.mean(module_grads, dim=0) for module_name, module_grads in modules_grad.items()}        
    return avg_grads

def compute_hessian_free_influences(modules_train_grad, modules_val_grad, modules_avg_val_grads = None, 
                                        bring_to_cpu = False):
    should_free_avg_grad = False
    start_time = time()
    if modules_avg_val_grads is None:
        modules_avg_val_grads = avg_grad(modules_val_grad)
        should_free_avg_grad = True
    modules_hvp = modules_avg_val_grads.copy()
    module_infls = compute_influence_from_hvp(modules_hvp, modules_train_grad, bring_to_cpu=bring_to_cpu)
    timespan = time() - start_time
    del modules_hvp
    if should_free_avg_grad:
        del modules_avg_val_grads
    return timespan, module_infls

def compute_datainf_influences(modules_train_grad, modules_val_grad, modules_avg_val_grads = None, 
                                    bring_to_cpu = False, lambda_const_param=10):
    should_free_avg_grad = False
    start_time = time()
    if modules_avg_val_grads is None:
        modules_avg_val_grads = avg_grad(modules_val_grad)
        should_free_avg_grad = True
    modules_hvp = {}
    for module_name, module_avg_val_grads in modules_avg_val_grads.items():
        module_train_grad = modules_train_grad[module_name]
        module_train_grad_squares = module_train_grad ** 2
        lambda_const = torch.mean(module_train_grad_squares) / lambda_const_param

        C_tmp_values = module_avg_val_grads * module_train_grad
        nom_values = torch.sum(C_tmp_values, dim=-1)
        denom_values = lambda_const + torch.sum(module_train_grad_squares, dim=-1)
        C_tmp = nom_values / denom_values

        C_tmp_grad_values = C_tmp.view(-1, 1) * module_train_grad

        const_val = lambda_const * module_train_grad.shape[0]

        module_hvp_values = (module_avg_val_grads - C_tmp_grad_values) / const_val

        module_hvp = torch.sum(module_hvp_values, dim=0)

        modules_hvp[module_name] = module_hvp

        del module_train_grad_squares, C_tmp_values, nom_values, denom_values, C_tmp, C_tmp_grad_values, module_hvp_values
        # lambda_const computation
        # S = torch.zeros(len(self.tr_grad_dict.keys()))
        # for tr_id in self.tr_grad_dict:
        #     tmp_grad = self.tr_grad_dict[tr_id][weight_name]
        #     S[tr_id]=torch.mean(tmp_grad**2)
        # lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda
        
        # hvp computation
        # hvp=torch.zeros(self.val_grad_avg_dict[weight_name].shape)
        # for tr_id in self.tr_grad_dict:
        #     tmp_grad = self.tr_grad_dict[tr_id][weight_name]
        #     C_tmp = torch.sum(self.val_grad_avg_dict[weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
        #     hvp += (self.val_grad_avg_dict[weight_name] - C_tmp*tmp_grad) / (self.n_train*lambda_const)
        # hvp_proposed_dict[weight_name] = hvp 
    module_infls = compute_influence_from_hvp(modules_hvp, modules_train_grad, bring_to_cpu=bring_to_cpu)
    timespan = time() - start_time
    if should_free_avg_grad:
        del modules_avg_val_grads
    return timespan, module_infls

def compute_accurate_influences(modules_train_grad, modules_val_grad, modules_avg_val_grads = None, 
                                    bring_to_cpu = False, lambda_const_param=10):
    should_free_avg_grad = False
    start_time = time()
    if modules_avg_val_grads is None:
        modules_avg_val_grads = avg_grad(modules_val_grad)
        should_free_avg_grad = True
    modules_hvp = {}
    for module_name, module_avg_val_grads in modules_avg_val_grads.items():
        module_train_grad = modules_train_grad[module_name]
        module_train_grad_squares = module_train_grad ** 2
        lambda_const = torch.mean(module_train_grad_squares) / lambda_const_param

        # module_train_grad_flat = module_train_grad.reshape(-1)
        AAt_matrix = torch.einsum("ki,kj->ij", module_train_grad, module_train_grad)
        # AAt_matrix = torch.sum(AAt_matrix_values, dim=0)
        L, V = torch.linalg.eig(AAt_matrix)
        L, V = L.float(), V.float()
        module_hvp_0 = module_avg_val_grads.reshape(-1) @ V
        denom_values = lambda_const + L / module_train_grad.shape[0]
        module_hvp_1 = module_hvp_0 / denom_values
        module_hvp = module_hvp_1 @ V.T
        modules_hvp[module_name] = module_hvp

        del module_train_grad_squares, AAt_matrix, L, V, module_hvp_0, denom_values, module_hvp_1, module_hvp  # to save memory

        # lambda_const computation
        # S=torch.zeros(len(self.tr_grad_dict.keys()))
        # for tr_id in self.tr_grad_dict:
        #     tmp_grad = self.tr_grad_dict[tr_id][weight_name]
        #     S[tr_id]=torch.mean(tmp_grad**2)
        # lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

        # hvp computation (eigenvalue decomposition)
        # AAt_matrix = torch.zeros(torch.outer(self.tr_grad_dict[0][weight_name].reshape(-1), 
        #                                         self.tr_grad_dict[0][weight_name].reshape(-1)).shape)
        # for tr_id in self.tr_grad_dict: 
        #     tmp_mat = torch.outer(self.tr_grad_dict[tr_id][weight_name].reshape(-1), 
        #                             self.tr_grad_dict[tr_id][weight_name].reshape(-1))
        #     AAt_matrix += tmp_mat
            
        # L, V = torch.linalg.eig(AAt_matrix)
        # L, V = L.float(), V.float()
        # hvp = self.val_grad_avg_dict[weight_name].reshape(-1) @ V
        # hvp = (hvp / (lambda_const + L/ self.n_train)) @ V.T

        # hvp_accurate_dict[weight_name] = hvp.reshape(len(self.tr_grad_dict[0][weight_name]), -1)
        # del tmp_mat, AAt_matrix, V # to save memory
    module_infls = compute_influence_from_hvp(modules_hvp, modules_train_grad, bring_to_cpu=bring_to_cpu)
    timespan = time() - start_time
    if should_free_avg_grad:
        del modules_avg_val_grads    
    return timespan, module_infls

def compute_lissa_influences(modules_train_grad, modules_val_grad, modules_avg_val_grads = None, 
                                bring_to_cpu = False, lambda_const_param=10, n_iteration=10, alpha_const=1.):
    should_free_avg_grad = False
    start_time = time()
    if modules_avg_val_grads is None:
        modules_avg_val_grads = avg_grad(modules_val_grad)
        should_free_avg_grad = True
    modules_hvp = {}
    for module_name, module_avg_val_grads in modules_avg_val_grads.items():
        # lambda_const computation
        module_train_grad = modules_train_grad[module_name]
        n_train = module_train_grad.shape[0]
        module_train_grad_squares = module_train_grad ** 2
        lambda_const = torch.mean(module_train_grad_squares) / lambda_const_param

        # hvp computation
        running_hvp = module_avg_val_grads
        for _ in range(n_iteration):
            hvp_tmp_values = module_train_grad * running_hvp
            hvp_tmp_sum = torch.sum(hvp_tmp_values, dim=-1)
            hvp_tmp_0 = (hvp_tmp_sum.view(-1, 1) * module_train_grad - lambda_const * running_hvp) / n_train
            hvp_tmp = torch.sum(hvp_tmp_0, dim=0)
            running_hvp = module_avg_val_grads + running_hvp - alpha_const * hvp_tmp
        modules_hvp[module_name] = running_hvp 
    module_infls = compute_influence_from_hvp(modules_hvp, modules_train_grad, bring_to_cpu=bring_to_cpu)
    timespan = time() - start_time
    if should_free_avg_grad:
        del modules_avg_val_grads    
    return timespan, module_infls

class IFEngine(object):
    def __init__(self, tr_grad_dict, val_grad_dict):
        self.time_dict=defaultdict(list)
        self.hvp_dict=defaultdict(list)
        self.influences=defaultdict(dict)
        self.tr_grad_dict = tr_grad_dict
        self.val_grad_dict = val_grad_dict

        self.n_train = len(self.tr_grad_dict.keys())
        self.n_val = len(self.val_grad_dict.keys())
        self.compute_val_grad_avg()

    def compute_val_grad_avg(self):
        # Compute the avg gradient on the validation dataset
        self.val_grad_avg_dict={}
        for weight_name in self.val_grad_dict[0]:
            self.val_grad_avg_dict[weight_name]=torch.zeros(self.val_grad_dict[0][weight_name].shape)
            for val_id in self.val_grad_dict:
                self.val_grad_avg_dict[weight_name] += self.val_grad_dict[val_id][weight_name] / self.n_val

    def compute_hvps(self, lambda_const_param=10, compute_accurate=True):
        self.compute_hvp_identity()
        self.compute_hvp_proposed(lambda_const_param=lambda_const_param)
        self.compute_hvp_LiSSA(lambda_const_param=lambda_const_param)
        if compute_accurate:
            self.compute_hvp_accurate(lambda_const_param=lambda_const_param)

    def compute_hvp_identity(self):
        start_time = time()
        self.hvp_dict['identity'] = self.val_grad_avg_dict.copy()
        self.time_dict['identity'] = time()-start_time

    def compute_hvp_proposed(self, lambda_const_param=10):
        start_time = time()
        hvp_proposed_dict={}
        for weight_name in self.val_grad_avg_dict:
            # lambda_const computation
            S=torch.zeros(len(self.tr_grad_dict.keys()))
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                S[tr_id]=torch.mean(tmp_grad**2)
            lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda
            
            # hvp computation
            hvp=torch.zeros(self.val_grad_avg_dict[weight_name].shape)
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                C_tmp = torch.sum(self.val_grad_avg_dict[weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
                hvp += (self.val_grad_avg_dict[weight_name] - C_tmp*tmp_grad) / (self.n_train*lambda_const)
            hvp_proposed_dict[weight_name] = hvp 
        self.hvp_dict['DataInf'] = hvp_proposed_dict
        self.time_dict['DataInf'] = time()-start_time

    def compute_hvp_accurate(self, lambda_const_param=10):
        start_time = time()
        hvp_accurate_dict={}
        for weight_name in self.val_grad_avg_dict:
            # lambda_const computation
            S=torch.zeros(len(self.tr_grad_dict.keys()))
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                S[tr_id]=torch.mean(tmp_grad**2)
            lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

            # hvp computation (eigenvalue decomposition)
            AAt_matrix = torch.zeros(torch.outer(self.tr_grad_dict[0][weight_name].reshape(-1), 
                                                 self.tr_grad_dict[0][weight_name].reshape(-1)).shape)
            for tr_id in self.tr_grad_dict: 
                tmp_mat = torch.outer(self.tr_grad_dict[tr_id][weight_name].reshape(-1), 
                                      self.tr_grad_dict[tr_id][weight_name].reshape(-1))
                AAt_matrix += tmp_mat
                
            L, V = torch.linalg.eig(AAt_matrix)
            L, V = L.float(), V.float()
            hvp = self.val_grad_avg_dict[weight_name].reshape(-1) @ V
            hvp = (hvp / (lambda_const + L/ self.n_train)) @ V.T

            hvp_accurate_dict[weight_name] = hvp.reshape(len(self.tr_grad_dict[0][weight_name]), -1)
            del tmp_mat, AAt_matrix, V # to save memory
        self.hvp_dict['accurate'] = hvp_accurate_dict
        self.time_dict['accurate'] = time()-start_time 

    def compute_hvp_LiSSA(self, lambda_const_param=10, n_iteration=10, alpha_const=1.):
        start_time = time()
        hvp_LiSSA_dict={}
        for weight_name in self.val_grad_avg_dict:
            # lambda_const computation
            S=torch.zeros(len(self.tr_grad_dict.keys()))
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                S[tr_id]=torch.mean(tmp_grad**2)
            lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

            # hvp computation
            running_hvp=self.val_grad_avg_dict[weight_name]
            for _ in range(n_iteration):
                hvp_tmp=torch.zeros(self.val_grad_avg_dict[weight_name].shape)
                for tr_id in self.tr_grad_dict:
                    tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                    hvp_tmp += (torch.sum(tmp_grad*running_hvp)*tmp_grad - lambda_const*running_hvp) / self.n_train
                running_hvp = self.val_grad_avg_dict[weight_name] + running_hvp - alpha_const*hvp_tmp
            hvp_LiSSA_dict[weight_name] = running_hvp 
        self.hvp_dict['LiSSA'] = hvp_LiSSA_dict
        self.time_dict['LiSSA'] = time()-start_time 

    def compute_all_influences(self):
        for method_name in self.hvp_dict:
            if_tmp_dict = {}
            module_influences = defaultdict(dict)
            for tr_id in self.tr_grad_dict:
                if_tmp_value = 0
                for weight_name in self.val_grad_avg_dict:
                    layer_influence = torch.sum(self.hvp_dict[method_name][weight_name]*self.tr_grad_dict[tr_id][weight_name])
                    module_influences[weight_name][tr_id] = -layer_influence
                    if_tmp_value += layer_influence
                if_tmp_dict[tr_id]= -if_tmp_value 
                
            self.influences[method_name] = {layer_name:pd.Series(influences, dtype=float).to_numpy() for layer_name, influences in module_influences.items() }
            self.influences[method_name][''] = pd.Series(if_tmp_dict, dtype=float).to_numpy()
        return {"runtime": self.time_dict, "influences": self.influences}

class IFEngineGeneration(object):
    '''
    This class computes the influence function for every validation data point
    '''
    def __init__(self):
        self.time_dict = defaultdict(list)
        self.hvp_dict = defaultdict(list)
        self.IF_dict = defaultdict(list)

    def preprocess_gradients(self, tr_grad_dict, val_grad_dict):
        self.tr_grad_dict = tr_grad_dict
        self.val_grad_dict = val_grad_dict

        self.n_train = len(self.tr_grad_dict.keys())
        self.n_val = len(self.val_grad_dict.keys())

    def compute_hvps(self, lambda_const_param=10):
        self.compute_hvp_identity()
        self.compute_hvp_proposed(lambda_const_param=lambda_const_param)

    def compute_hvp_identity(self):
        start_time = time()
        self.hvp_dict["identity"] = self.val_grad_dict.copy()
        self.time_dict["identity"] = time() - start_time

    def compute_hvp_proposed(self, lambda_const_param=10):
        start_time = time()
        hvp_proposed_dict=defaultdict(dict)
        for val_id in tqdm(self.val_grad_dict.keys()):
            for weight_name in self.val_grad_dict[val_id]:
                # lambda_const computation
                S=torch.zeros(len(self.tr_grad_dict.keys()))
                for tr_id in self.tr_grad_dict:
                    tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                    S[tr_id]=torch.mean(tmp_grad**2)
                lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

                # hvp computation
                hvp=torch.zeros(self.val_grad_dict[val_id][weight_name].shape)
                for tr_id in self.tr_grad_dict:
                    tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                    C_tmp = torch.sum(self.val_grad_dict[val_id][weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
                    hvp += (self.val_grad_dict[val_id][weight_name] - C_tmp*tmp_grad) / (self.n_train*lambda_const)
                hvp_proposed_dict[val_id][weight_name] = hvp
        self.hvp_dict['DataInf'] = hvp_proposed_dict
        self.time_dict['DataInf'] = time()-start_time

    def compute_IF(self):
        for method_name in self.hvp_dict:
            print("Computing IF for method: ", method_name)
            if_tmp_dict = defaultdict(dict)
            for tr_id in self.tr_grad_dict:
                for val_id in self.val_grad_dict:
                    if_tmp_value = 0
                    for weight_name in self.val_grad_dict[0]:
                        if_tmp_value += torch.sum(self.hvp_dict[method_name][val_id][weight_name]*self.tr_grad_dict[tr_id][weight_name])
                    if_tmp_dict[tr_id][val_id]=-if_tmp_value

            self.IF_dict[method_name] = pd.DataFrame(if_tmp_dict, dtype=float)   

    def save_result(self, run_id=0):
        results={}
        results['runtime']=self.time_dict
        results['influence']=self.IF_dict

        with open(f"./results_{run_id}.pkl",'wb') as file:
            pickle.dump(results, file)


def test_infl_are_same():
    tran_grads = {"layer1": torch.randn(30, 100, 5), "layer2": torch.randn(30, 5, 300) }
    val_grads = {"layer1": torch.randn(20, 100, 5), "layer2": torch.randn(20, 5, 300) }
    tran_grad_dict = defaultdict(dict)
    for layer, grad in tran_grads.items():
        for i in range(grad.shape[0]):
            tran_grad_dict[i][layer] = grad[i]
    val_grad_dict = defaultdict(dict)
    for layer, grad in val_grads.items():
        for i in range(grad.shape[0]):
            val_grad_dict[i][layer] = grad[i]  
    tran_grads = {layer: grad.reshape(grad.shape[0], -1) for layer, grad in tran_grads.items()}
    val_grads = {layer: grad.reshape(grad.shape[0], -1) for layer, grad in val_grads.items()}
    _, hf = compute_hessian_free_influences(tran_grads, val_grads)
    _, di = compute_datainf_influences(tran_grads, val_grads)
    _, li = compute_lissa_influences(tran_grads, val_grads)
    _, ai = compute_accurate_influences(tran_grads, val_grads)
    eng = IFEngine(tran_grad_dict, val_grad_dict)
    eng.compute_hvps(compute_accurate=True)
    res2 = eng.compute_all_influences()
    infls = res2['influences']
    hf2 = infls['identity']    
    di2 = infls['DataInf']
    li2 = infls['LiSSA']
    ai2 = infls['accurate']
    for l in hf.keys():
        x, y = hf[l], torch.tensor(hf2[l], dtype=torch.float)
        assert torch.allclose(x, y, atol=1e-4), f'Hessian Free for layer {l} in not same: {x} and {y}'
    for l in di.keys():
        x, y = di[l], torch.tensor(di2[l], dtype=torch.float)
        assert torch.allclose(x, y, atol=1e-4), f'DataInf for layer {l} in not same: {x} and {y}'
    for l in li.keys():
        x, y = li[l], torch.tensor(li2[l], dtype=torch.float)
        assert torch.allclose(x, y, rtol=1e-2), f'LISSA for layer {l} in not same: {x} and {y}'
    for l in ai.keys():
        x, y = ai[l], torch.tensor(ai2[l], dtype=torch.float)
        assert torch.allclose(x, y, atol=1e-3), f'Exact for layer {l} in not same: {x} and {y}'
    pass

if __name__ == "__main__":
    test_infl_are_same()
    pass