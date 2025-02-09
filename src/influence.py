import re
from time import time
from typing import Optional
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import pickle, os
import torch

def hessian_free_fn(module_train_grad, module_avg_val_grads):
    module_train_grad = module_train_grad.reshape(module_train_grad.shape[0], -1)
    module_avg_val_grads = module_avg_val_grads.reshape(-1)    
    module_infl_values = module_avg_val_grads * module_train_grad
    module_infs = module_infl_values.reshape(module_train_grad.shape[0], -1).sum(dim=-1)
    return module_infs

def datainf_fn(module_train_grad, module_avg_val_grads, lambda_const_param=10):
    module_train_grad = module_train_grad.reshape(module_train_grad.shape[0], -1)
    module_avg_val_grads = module_avg_val_grads.reshape(-1)
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

    module_infl_values = module_hvp * module_train_grad
    module_infls = module_infl_values.sum(dim=-1)
    return module_infls

def lissa_fn(module_train_grad, module_avg_val_grads, lambda_const_param=10, n_iteration=10, alpha_const=1.):
    module_train_grad = module_train_grad.reshape(module_train_grad.shape[0], -1)
    module_avg_val_grads = module_avg_val_grads.reshape(-1)    
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

    module_infl_values = running_hvp * module_train_grad
    module_infls = module_infl_values.sum(dim=-1)
    return module_infls

def compute_infl_from_model(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, 
                                    val_dataloader: Optional[torch.utils.data.DataLoader] = None, device = "cuda",
                                    module_patterns: list[str] = [], return_all = False, filter_list = None,
                                    infl_fn = hessian_free_fn):
    ''' Use this for large models - does not require to store all gradients in memory, but
        computes influence on request '''
    
    if val_dataloader is None:
        val_dataloader = train_dataloader

    start_time = time()
    
    patterns = [ (re.compile(p), p) for p in module_patterns ]
    model.to(device)
    model.eval()

    module_infls = {}
    num_val_samples = len(val_dataloader)
    num_train_samples = len(train_dataloader)
    module_infls[''] = torch.zeros(num_train_samples, device = device, dtype=torch.float)
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
                    module_infls[p_str] = torch.zeros(num_train_samples, device = device, dtype=module_params.dtype)

    # model.zero_grad()

    # def compute_loss_func(params, batch):
    #     if type(batch) == list:
    #         output_is_logits = True
    #         inputs, labels = batch 
    #         args = (inputs,)
    #         kwargs = {}
    #     else:
    #         output_is_logits = False
    #         labels = batch.pop("labels")
    #         args = ()
    #         kwargs = batch
    #     output = torch.func.functional_call(model, params, args, kwargs=kwargs)
    #     if output_is_logits:
    #         logits = output
    #     else:
    #         logits = output.logits
    #     loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    #     return loss

    # loss2_jac_fn = torch.func.jacrev(compute_loss_func, has_aux=False)
    # # trainable_params = {nm:pval for nm, pval in model.named_parameters() if pval.requires_grad}

    # new_dataloader = torch.utils.data.DataLoader(val_dataloader.dataset, batch_size=16, shuffle=False)

    # for step, batch in enumerate(tqdm(new_dataloader)):
    #     if type(batch) == list:
    #         inputs, labels = batch
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #         batch = [inputs, labels]
    #     else:
    #         batch.to(device)
    #     # labels = batch.pop("labels")
        
    #     with torch.no_grad():
    #         loss2_jacobian = loss2_jac_fn(active_modules, batch)
        
    #     for nm, pval in active_modules.items():
    #         if pval.grad is not None:
    #             pval.grad += loss2_jacobian[nm].sum(dim=0)
    #         else:
    #             pval.grad = loss2_jacobian[nm].sum(dim=0)

    # avg_val_grad2 = { module_name: module_params.grad / num_val_samples for module_name, module_params in active_modules.items() }          
    
    model.zero_grad()
    for step, batch in enumerate(tqdm(val_dataloader)):
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

    avg_val_grad = { module_name: module_params.grad / num_val_samples for module_name, module_params in active_modules.items() }
        
    for module_name, module_params in active_modules.items():
        for module_params_i in active_modules.values():
            module_params_i.requires_grad = False
        module_params.requires_grad = True                
        module_avg_val_grad = avg_val_grad[module_name]

        module_infl_values = torch.zeros((num_train_samples, *module_params.shape), device = device, dtype=module_params.dtype)
        module_train_grad = torch.zeros((num_train_samples, *module_params.shape), device = device, dtype=module_params.dtype)
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

            module_train_grad[step] = module_params.grad

        module_infls_one = infl_fn(module_train_grad, module_avg_val_grad)
        module_infls[''] += module_infls_one
        for p_str in module_groups.get(module_name, []):
            module_infls[p_str] += module_infls_one
        if return_all:
            module_infls[module_name] = module_infls_one
    for module_params_i in active_modules.values():
        module_params_i.requires_grad = True
    for infl in module_infls.values():
        infl.neg_()         
    timespan = time() - start_time
    return timespan, module_infls


def compute_influence_from_hvp(modules_hvp, modules_grad, bring_to_cpu=False):
    if_tmp_dict = {}
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
        
    self.influences[method_name] = {layer_name:pd.Series(influences, dtype=float).to_numpy() for layer_name, influences in module_influences.items() }
    self.influences[method_name][''] = pd.Series(if_tmp_dict, dtype=float).to_numpy()
    # return {"runtime": self.time_dict, "influences": self.influences}

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
        assert torch.allclose(x, y, atol=1e-5), f'Hessian Free for layer {l} in not same: {x} and {y}'
    for l in di.keys():
        x, y = di[l], torch.tensor(di2[l], dtype=torch.float)
        assert torch.allclose(x, y, atol=1e-5), f'DataInf for layer {l} in not same: {x} and {y}'
    for l in li.keys():
        x, y = li[l], torch.tensor(li2[l], dtype=torch.float)
        assert torch.allclose(x, y, rtol=1e-3), f'LISSA for layer {l} in not same: {x} and {y}'
    for l in ai.keys():
        x, y = ai[l], torch.tensor(ai2[l], dtype=torch.float)
        assert torch.allclose(x, y, atol=1e-4), f'Exact for layer {l} in not same: {x} and {y}'
    pass

if __name__ == "__main__":
    test_infl_are_same()
    pass