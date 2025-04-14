from collections import defaultdict
from functools import partial
import json
import os
import pickle
import pandas as pd
import re
from typing import Optional
from tabulate import tabulate

import datasets
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
from scipy import stats
import torch
from datasets import load_from_disk
from scipy import stats as sci_stats
import scikit_posthocs as sci_posthocs

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{times}"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = ['Times']    

from cifar import DatasetSplits

def draw_mislabel_detection_rate(task="mrpc", res_folder = "./data/", datasets_folder = "./data/",  
                                    module_pattern='', out = "./data/mdr/qnli.png", title = "All modules"):
    # res_pattern = re.compile(res_filer)
    res_path = os.path.join(res_folder, task)
    all_influences = defaultdict(list)
    if module_pattern == '':
        ptrn = None 
    else:
        ptrn = re.compile(module_pattern)
    seeds = defaultdict(list)
    for file_name in os.listdir(res_path):
        name_parts = file_name.split('.')[0].split('_')
        method_name = name_parts[1]
        seed = int(name_parts[-1])
        seeds[method_name].append(seed)
        tensor_dict = torch.load(os.path.join(res_path, file_name))
        if ptrn is None:
            all_influences[method_name].append(tensor_dict[''])
        else:
            module_influences = None
            for module_name, influences in tensor_dict.items():
                if ptrn.match(module_name):
                    print(f"{module_name}")
                    if module_influences is None: 
                        module_influences = influences
                    else:
                        module_influences += influences
            all_influences[method_name].append(module_influences)
    data_detection_per_method = {}
    datasets_path = os.path.join(datasets_folder, task)
    for method_name, tensors in all_influences.items():
        method_seeds = seeds[method_name]
        for infls_tensor, seed in zip(tensors, method_seeds):
            infls = infls_tensor.numpy()
            dataset = load_from_disk(os.path.join(datasets_path, f"d_{task}_{seed}"))
            noises = np.array(dataset['train']['noise'])
            noise_index = np.where(noises)[0]
            n_train = len(infls)
            detection_rate_list=[]
            low_quality_to_high_quality=np.argsort(infls)[::-1]
            for ind in range(1, len(low_quality_to_high_quality)+1):
                detected_samples = set(low_quality_to_high_quality[:ind]).intersection(noise_index)
                detection_rate = 100*len(detected_samples)/len(noise_index)
                detection_rate_list.append(detection_rate)
            data_detection_per_method.setdefault(method_name, []).append(detection_rate_list)
    method_names = sorted(data_detection_per_method.keys())
    plt.ioff()
    for method in method_names:
        detection_rate_lists = data_detection_per_method[method]
        drls = np.array(detection_rate_lists)
        drl = np.mean(drls, axis=0)
        confidence_level = 0.95
        degrees_freedom = drls.shape[0] - 1
        sample_standard_error = stats.sem(drls, axis=0)
        confidence_interval = stats.t.interval(confidence_level, degrees_freedom, drl, sample_standard_error)
        min_v = confidence_interval[0]
        max_v = confidence_interval[1]
        xs = 100*np.arange(n_train)/n_train
        plt.plot(xs, drl, label=method)
        plt.fill_between(xs, min_v, max_v, alpha=.1, linewidth=0)
    plt.axvline(x=30, color='r', linestyle='--', linewidth=1)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('Data inspected (%)')
    plt.ylabel('Detection Rate (%)')
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.legend(fontsize='small')
    plt.title(f'{(task.upper())}, {title}', fontsize=15)
    plt.tight_layout()
    plt.savefig(out)  
    plt.clf()  


def draw_mislabel_detection_rate2(infl_folder = "./data/cifar-resnet/infl", dataset_file = "./data/cifar-resnet/ds/d_cifar10_0",  
                                    out = "./data/mdr/cifar.png", module_name = '', step = 500, title = "All modules", task = "cifar"):
    ''' For cifar and resnet '''
    # res_pattern = re.compile(res_filer)
    all_influences = defaultdict(list)
    # seeds = defaultdict(list)
    for file_name in os.listdir(infl_folder):
        name_parts = file_name.split('.')[0].split('_')
        method_name = name_parts[1]
        seed = int(name_parts[-1])
        # seeds[method_name].append(seed)
        tensor_dict = torch.load(os.path.join(infl_folder, file_name))
        t = tensor_dict[module_name]
        if not(torch.any(torch.isnan(t))):
            all_influences[method_name].append(t)
        else:
            print(f"Found nan in {file_name}")
    data_detection_per_method = {}
    dataset_splits: DatasetSplits = torch.load(dataset_file, weights_only = False)
    for method_name, tensors in all_influences.items():
        # method_seeds = seeds[method_name]
        for infls_tensor in tensors:
            infls = infls_tensor.to('cpu').numpy()
            noises = (dataset_splits.noisy_labels["worst"] != dataset_splits.train_clean_labels).to('cpu').numpy()
            noise_index = np.where(noises)[0]
            n_train = len(infls)
            detection_rate_list=[]
            low_quality_to_high_quality=np.argsort(infls)[::-1]
            for ind in range(1, len(low_quality_to_high_quality)+1, step):
                detected_samples = set(low_quality_to_high_quality[:ind]).intersection(noise_index)
                detection_rate = 100*len(detected_samples)/len(noise_index)
                detection_rate_list.append(detection_rate)
            data_detection_per_method.setdefault(method_name, []).append(detection_rate_list)
    method_names = sorted(data_detection_per_method.keys())
    plt.ioff()
    for method in method_names:
        detection_rate_lists = data_detection_per_method[method]
        drls = np.array(detection_rate_lists)
        drl = np.mean(drls, axis=0)
        confidence_level = 0.95
        degrees_freedom = drls.shape[0] - 1
        sample_standard_error = stats.sem(drls, axis=0)
        confidence_interval = stats.t.interval(confidence_level, degrees_freedom, drl, sample_standard_error)
        min_v = confidence_interval[0]
        max_v = confidence_interval[1]
        xs = 100*np.arange(0, n_train, step)/ n_train
        plt.plot(xs, drl, label=method)
        plt.fill_between(xs, min_v, max_v, alpha=.1, linewidth=0)
    plt.axvline(x=30, color='r', linestyle='--', linewidth=1)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('Data inspected (%)')
    plt.ylabel('Detection Rate (%)')
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.legend(fontsize='small')
    plt.title(f'{(task.upper())}, {title}', fontsize=15)
    plt.tight_layout()
    plt.savefig(out)  
    plt.clf()  


def draw_ft2_metric(task: str, infile: str, outfile: str, metric = 'accuracy', influence_method = "", module_pattern_to_name = {}, allowed_filter_methods = ['rand', 'denoise', 'none']):
    with open(infile, 'r') as f:
        json_lines = f.readlines()
    all_metrics = [json.loads(l) for l in json_lines]
    method_metrics = defaultdict(list)
    for metrics in all_metrics:
        metric_values = metrics['metrics'][metric]
        infl_method = metrics['finetune2']['infl_method']
        filter_method = metrics['finetune2']['filter_method']                
        if (influence_method != "" and influence_method != infl_method) and (filter_method not in allowed_filter_methods):
            continue
        module_pattern = metrics['finetune2']['module_pattern']
        if module_pattern not in module_pattern_to_name:
            continue
        module_pattern = module_pattern_to_name[module_pattern]
        is_infl = False
        if filter_method  == 'infl':
            filter_method = infl_method
            is_infl = True
        if module_pattern != "" and is_infl:
            filter_method = f'{filter_method}, {module_pattern}'
        method_metrics[filter_method].append(metric_values)

    method_names = sorted(method_metrics.keys())
    plt.ioff()
    for method in method_names:
        metrics = method_metrics[method]
        metric_values = np.array(metrics) * 100
        mean = np.mean(metric_values, axis=0)
        confidence_level = 0.95
        degrees_freedom = metric_values.shape[0] - 1
        sample_standard_error = stats.sem(metric_values, axis=0)
        confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
        min_v = confidence_interval[0]
        max_v = confidence_interval[1]
        default_args = dict(marker='o', markersize=5, linewidth=1)
        if method == 'denoise':
            default_args = dict(linewidth=1, color='darkgray', linestyle='--')
        if method == 'rand':
            default_args = dict(linewidth=1, color='gray', linestyle='-.')
        line = plt.plot(np.arange(1, 11), mean, label=method, **default_args)
        plt.fill_between(np.arange(1, 11), min_v, max_v, alpha=.05, color = line[0].get_color(), linewidth=0)
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy, %')
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.legend(fontsize='small')
    plt.title(f'{task.upper()} 70% filtered finetuning', fontsize=15)
    plt.tight_layout()
    plt.savefig(outfile)  
    plt.clf()  

def list_modules(infl_file: str, module_pattern = ''):
    tensor_dict = torch.load(infl_file)
    selected = []
    if module_pattern == '':
        for module_name, infls in tensor_dict.items():
            selected.append((module_name, infls))
    else:
        pattern = re.compile(module_pattern)
        for module_name, infls in tensor_dict.items():
            if pattern.match(module_name): 
                selected.append((module_name, infls))
    for module_name, infls in selected:
        print(f"{module_name}")

# NOTE: Obserted effect: noisy group for some layers has less % (on average) of codirected infl samples than clean group 
def dir_majority_indicator(inf_matrix: torch.Tensor, *, additional_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    ''' Returns 1 for each train sample that has majority of codirected influences (positive influences)'''
    same_dir_mask = inf_matrix > 0
    if additional_mask is not None:
        same_dir_mask &= additional_mask
        half_infl = additional_mask.sum(dim=0) // 2
    else:
        half_infl = same_dir_mask.shape[0] // 2
    maj = torch.sum(same_dir_mask, dim=0) > half_infl
    return maj

# scores - bigger is better (more influential example).
def mean_score(inf_matrix: torch.Tensor, *, additional_mask: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
    if additional_mask is not None:
        inf_matrix_clone = inf_matrix.clone()
        inf_matrix_clone[~additional_mask] = 0
        inf_matrix = inf_matrix_clone
        denoms = additional_mask.sum(dim=0)
    else:
        denoms = inf_matrix.shape[0]
    res = inf_matrix.sum(dim=0) / denoms
    return res

def mean_matrix_score(int_matrix: torch.Tensor, *, trim_ratio = None, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample '''
    if trim_ratio is not None:
        start_id = round(int_matrix.shape[-1] * trim_ratio / 2)
        end_id = int_matrix.shape[-1] - start_id
        int_matrix_sorted, int_matrix_sorted_ids = torch.sort(int_matrix, dim = -1)
        scores = torch.mean(int_matrix_sorted[:, :, start_id:end_id], dim=(-2,-1))
        del int_matrix_sorted, int_matrix_sorted_ids
    else:  
        scores = int_matrix.mean(dim=(-2,-1))
    return scores

def median_matrix_score(int_matrix: torch.Tensor, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample ''' 
    scores = int_matrix.nanmedian(dim=(-2,-1))
    return scores

def rank_matrix_score(int_matrix: torch.Tensor, *, rank_score_fn = mean_matrix_score, **_):
    ''' 3-D tensor: train sample * module * infl sample '''
    # int_matrix_view = int_matrix.view(int_matrix.shape[0], -1)
    # int_matrix_t = int_matrix.transpose(0, 2)
    ranks = torch.zeros_like(int_matrix)
    for int_matrix_view in torch.split(int_matrix, 4, dim = 1):
        sort_indexes = torch.argsort(int_matrix_view, dim = 0)
        rank_range = torch.arange(int_matrix.shape[0], device = int_matrix.device, dtype=int_matrix.dtype)
        # rank_range += 1.0
        # int_matrix_view = int_matrix.view(int_matrix.shape[0], -1)
        rank_ranges = rank_range.view(-1, 1, 1).repeat(1, int_matrix.shape[1], int_matrix.shape[2])
        # ranks[sort_indexes] = rank_range
        ranks.scatter_(0, sort_indexes, rank_ranges)
        # for module_id in range(int_matrix.shape[1]):
        #     for val_id in range(int_matrix.shape[2]):
        #         ranks[sort_indexes[val_id, module_id], module_id, val_id] = rank_range
        del rank_ranges, rank_range, sort_indexes
        # del sort_indexes, rank_range
    scores = rank_score_fn(ranks)
    del ranks
    return scores

def dir_matrix_score(int_matrix: torch.Tensor, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample '''
    same_dir_mask = int_matrix > 0
    one_scores = same_dir_mask.float()
    scores = torch.mean(one_scores, dim=(-2, -1))
    del same_dir_mask, one_scores
    return scores

# test_inf_matrix = torch.tensor([[10, 20, 10, 5], [5, 15, 15, 7]], dtype=torch.float)
# mean_rank_score(test_inf_matrix)

def median_score(inf_matrix: torch.Tensor, *, additional_mask: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
    if additional_mask is not None:
        inf_matrix_clone = inf_matrix.clone()
        inf_matrix_clone[~additional_mask] = torch.nan
        inf_matrix = inf_matrix_clone
    res = torch.nanmedian(inf_matrix, dim=0)[0]
    return res

def mean_dir_score(inf_matrix: torch.Tensor, *, additional_mask: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
    same_dir_mask = inf_matrix > 0
    if additional_mask is not None:
        same_dir_mask &= additional_mask
        denoms = additional_mask.sum(dim=0)
    else:
        denoms = inf_matrix.shape[0]
    res = same_dir_mask.sum(dim=0).float() / denoms
    return res

def mean_on_preds(inf_matrix: torch.Tensor, *, mask_cache, base_dir, task, seed, **_) -> torch.Tensor:
    key = (base_dir, task, seed)
    if key not in mask_cache:
        logits_path = os.path.join(base_dir, f"m_b_{task}_{seed}", "infl_logits.pt")
        ds_path = os.path.join(base_dir, f"d_{task}_{seed}")
        ds = datasets.load_from_disk(ds_path)
        inflset_labels = torch.tensor(ds["infl"]['labels'], device = "cuda")
        inflset_logits = torch.load(logits_path)
        inflset_preds = torch.argmax(inflset_logits, dim=1)
        inflset_correct = inflset_preds == inflset_labels
        mask_cache[key] = inflset_correct
    inflset_correct = mask_cache[key]
    filtered_infl_matrix = inf_matrix[inflset_correct]
    return mean_score(filtered_infl_matrix)



def compute_noise_detection_metrics(base_dir_path: str, task: str, 
                                    m_prefix="m_b", i_prefix="i_b", 
                                    plot_title = "", cache_file = "",
                                    out_chart_file = "",
                                    module_groups_regex: dict[str, str] = {}):

    # loading dataset data for noise:
    mean_on_preds_fn = partial(mean_on_preds, mask_cache = {}, base_dir = base_dir_path, task = task)
    score_fns = [mean_score, mean_dir_score, mean_on_preds_fn] #[mean_dir_score, mean_score, median_score] #[mean_score] #, median_score, mean_dir_score]

    score_names = {mean_score: "mean", mean_dir_score: "meandir", median_score: "median", mean_on_preds_fn: "meanpred"}

    infl_method_names = {"cos": "cos", "hf": "hf", "hf_we_": "hf_we", "hf_we_topk_10": "hf_we top 10", "datainf": "datainf", "datainf0": "datainf0", "datainf_one": "datainf1"}

    replace_name = {'layer': 'L', 'classifier': 'C', 'word_embeddings': 'WE' }

    # glob_mask = torch.zeros((len(infl), len(trainset)), dtype = torch.bool, device = device)

    # here we deal with binary confusion matrix
    # i0 = torch.where(infl_labels == 0)[0]
    # i1 = torch.where(infl_labels == 1)[0]
    # t0_n = torch.where((trainset_labels == 0) & noise_mask)[0]
    # t0_c = torch.where((trainset_labels == 0) & ~noise_mask)[0]
    # t1_n = torch.where((trainset_labels == 1) & noise_mask)[0]
    # t1_c = torch.where((trainset_labels == 1) & ~noise_mask)[0]

    selected_infl_methods = ['cos', 'hf', 'hf_we_', 'datainf'] #'hf_we_topk_10',
    # selected_infl_methods = ['datainf', 'datainf0', 'datainf_one']    
    # selected_infl_methods = ['hf_we_', 'hf_we_topk_10']    

    module_groups_patterns = {name: re.compile(pattern) for name, pattern in module_groups_regex.items()}

    all_dict_loaded = False
    if cache_file != "": # loading all post computed metrics from cache file for visualization
        try:
            all_dict = torch.load(cache_file)
            auc_rocs_by_methods_and_layer = all_dict['auc_rocs_by_methods_and_layer']
            first_30_score_by_methods_and_layer = all_dict['first_30_score_by_methods_and_layer']
            curves_by_methods_and_layer = all_dict['curves_by_methods_and_layer']
            auc_rocs_mean_std = all_dict['auc_rocs_mean_std']
            first_30_mean_std = all_dict['first_30_mean_std']
            curves_mean_confidence_interval = all_dict['curves_mean_confidence_interval']
            sorted_method_keys = all_dict['sorted_method_keys']
            n_train = all_dict['n_train']
            num_noise = all_dict['num_noise']
            all_dict_loaded = True
        except:
            all_dict_loaded = False 
    if not all_dict_loaded:

        auc_rocs_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is list of auc roc scores
        first_30_score_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is filtered noise in first 30 percent
        curves_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is list of lists of agg_method filtering scores

        noise_list_dict = {} # by run_id
        for file_name in os.listdir(base_dir_path):
            if not file_name.startswith("d_"):
                continue
            ds_file_parts = file_name.split('_')
            *_, run_id = ds_file_parts
            ds_path = os.path.join(base_dir_path, file_name)
            ds = datasets.load_from_disk(ds_path)
            trainset = ds['train']
            # noise_mask = torch.tensor(trainset['noise'], device = device)
            noise_list_dict[run_id] = trainset['noise']

        n_train = len(noise_list_dict['0'])
        for file_name_with_ext in os.listdir(base_dir_path):
            if not file_name_with_ext.startswith(i_prefix):
                continue
            file_name = file_name_with_ext.split('.')[0]
            fine_name_parts = file_name[len(i_prefix):].split('_')[1:]
            *method_parts, _, run_id_str = fine_name_parts
            run_id = int(run_id_str)
            infl_method = '_'.join(method_parts)
            if infl_method not in selected_infl_methods:
                continue
            file_path = os.path.join(base_dir_path, file_name_with_ext)
            matrix_dict = torch.load(file_path)
            if len(matrix_dict) > 1:
                total_layer_matrix = None 
                for module_name, inf_matrix in matrix_dict.items():
                    if total_layer_matrix is None:
                        total_layer_matrix = torch.clone(inf_matrix)
                    else:
                        total_layer_matrix += inf_matrix
                matrix_dict['total'] = total_layer_matrix
            if len(module_groups_patterns) > 0:
                for pattern_name, pattern in module_groups_patterns.items():
                    total_layer_matrix = None 
                    for module_name, inf_matrix in matrix_dict.items():
                        if pattern.match(module_name):
                            if total_layer_matrix is None:
                                total_layer_matrix = torch.clone(inf_matrix)
                            else:
                                total_layer_matrix += inf_matrix
                    matrix_dict[pattern_name] = total_layer_matrix
            noise_list = noise_list_dict[run_id_str]
            num_noise = sum(noise_list)
            num_clean = len(noise_list) - num_noise
            first_30_idx = round(0.3 * len(noise_list))
            ideal_area = num_noise / 2 + num_clean
            for module_name, inf_matrix in matrix_dict.items():
                if inf_matrix is None:
                    continue
                for agg_method in score_fns:
                    scores = agg_method(inf_matrix, seed = run_id)
                    train_ids = torch.argsort(scores).tolist()
                    noise_perc_curve = []
                    noise_count = 0
                    for train_id in train_ids:
                        if noise_list[train_id]:
                            noise_count += 1
                        noise_perc_curve.append(noise_count / num_noise)
                    first_30 = noise_perc_curve[first_30_idx]
                    auc_roc = sum((noise_perc_curve[i] + noise_perc_curve[i + 1]) / 2 for i in range(len(noise_perc_curve) - 1)) / ideal_area
                    auc_rocs_by_methods_and_layer[(infl_method, agg_method, module_name)].append(auc_roc)
                    curves_by_methods_and_layer[(infl_method, agg_method, module_name)].append(noise_perc_curve)
                    first_30_score_by_methods_and_layer[(infl_method, agg_method, module_name)].append(first_30)

        first_30_mean_std = {}

        for key, first_30_values in first_30_score_by_methods_and_layer.items():
            mean = np.mean(first_30_values)
            std = np.std(first_30_values)
            first_30_mean_std[key] = (mean, std)

        auc_rocs_mean_std = {}

        for key, auc_rocs in auc_rocs_by_methods_and_layer.items():
            mean = np.mean(auc_rocs)
            std = np.std(auc_rocs)
            auc_rocs_mean_std[key] = (mean, std)

        curves_mean_confidence_interval = {}

        for key, curves in curves_by_methods_and_layer.items():
            curves = np.array(curves)
            mean = np.mean(curves, axis=0)
            confidence_level = 0.95
            degrees_freedom = curves.shape[0] - 1
            sample_standard_error = stats.sem(curves, axis=0)
            confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
            min_v = confidence_interval[0]
            max_v = confidence_interval[1]
            curves_mean_confidence_interval[key] = (mean * 100, min_v * 100, max_v * 100)

        sorted_method_keys = sorted(first_30_mean_std.keys(), key = lambda x: first_30_mean_std[x][0], reverse=True)

        if cache_file != "":
            all_dict = {'auc_rocs_by_methods_and_layer': auc_rocs_by_methods_and_layer,
                        'first_30_score_by_methods_and_layer': first_30_score_by_methods_and_layer,
                        'curves_by_methods_and_layer': curves_by_methods_and_layer,
                        "auc_rocs_mean_std": auc_rocs_mean_std, "first_30_mean_std": first_30_mean_std,
                        "curves_mean_confidence_interval": curves_mean_confidence_interval,
                        "sorted_method_keys": sorted_method_keys,
                        "n_train": n_train, "num_noise": num_noise}
            torch.save(all_dict, cache_file)


    # first_5 = sorted_method_keys[:5]

    # pick top of each excluding total
    tops = []
    for infl_method in selected_infl_methods:
        # infl_key = next((key for key in sorted_method_keys if key[0] == infl_method and key[1] == mean_score and key[2] != 'total'), None)
        # infl_key = next((key for key in sorted_method_keys if key[0] == infl_method and key[1] == mean_score and key[2] in module_groups_patterns), None)
        infl_key = next((key for key in sorted_method_keys if key[0] == infl_method and key[2] != 'total'), None)
        if infl_key:
            tops.append(infl_key)

    selected_keys = tops #first_5

    xs = 100*np.arange(n_train)/n_train
    simple_module_names = {}
    plt.ioff()
    for key in selected_keys:
        module_name = key[2]
        if module_name not in simple_module_names:
            simple_name_parts = [replace_name.get(n, n) for n in module_name.split('.') if n not in ['base_model', 'model', 'roberta', 'encoder', 'embeddings', 'default', 'value', 'weight', 'attention', 'self', 'modules_to_save']]
            simple_module_name = ' '.join(simple_name_parts)
            simple_module_names[module_name] = simple_module_name
        simple_module_name = simple_module_names[module_name]
        label = f"{infl_method_names[key[0]]}, {score_names[key[1]]}, {simple_module_name}"
        curve_means, curve_mins, curve_maxs = curves_mean_confidence_interval[key]
        p = plt.plot(xs, curve_means, label=label, linewidth=1)
        plt.fill_between(xs, curve_mins, curve_maxs, alpha=.1, linewidth=0, color = p[0].get_color())
     
    best_xs =100*np.arange(num_noise)/ n_train
    best_ys = [cnt * 100 / num_noise for cnt in range(num_noise)]
    plt.plot(best_xs, best_ys, color='gray', linestyle='--', linewidth=1)
    default_ys = [cnt * 100 / n_train for cnt in range(n_train) ]
    plt.plot(xs, default_ys, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=num_noise * 100 / n_train, color='gray', linestyle='--', linewidth=1)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('Data inspected (\\%)')
    plt.ylabel('Detection Rate (\\%)')
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.legend(fontsize='small')
    plt.title(plot_title, fontsize=15)
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_chart_file)  
    plt.clf()  

    # perc = DataFrame(rows, columns = ['module', *column_names])
    # print(perc)

    # perc.to_csv(f'tmp.csv', index = False)

    pass



def compute_noise_detection_metrics_per_sample(base_dir_path: str, 
                                    m_prefix="m_b", i_prefix="i_b", 
                                    plot_title = "", cache_file = "",
                                    out_chart_file = ""):   
    ''' In contrast to compute_noise_detection_metrics, uses each infl sample for scoring and filtering and then avg '''

    infl_method_names = {"cos": "cos", "cov": "cov", "hf": "hf", "hf_we_": "hf_we", "hf_we_topk_10": "hf_we top 10", "datainf": "datainf"}

    replace_name = {'layer': 'L', 'classifier': 'C', 'word_embeddings': 'WE' }


    all_dict_loaded = False
    if cache_file != "": # loading all post computed metrics from cache file for visualization
        try:
            all_dict = torch.load(cache_file)
            auc_ndr_by_methods_and_layer = all_dict['auc_rocs_by_methods_and_layer']
            # first_30_score_by_methods_and_layer = all_dict['first_30_score_by_methods_and_layer']
            curves_by_methods_and_layer = all_dict['curves_by_methods_and_layer']
            auc_ndr_mean_std = all_dict['auc_rocs_mean_std']
            curves_mean_confidence_interval = all_dict['curves_mean_confidence_interval']
            sorted_method_keys = all_dict['sorted_method_keys']
            n_train = all_dict['n_train']
            num_noise = all_dict['num_noise']
            all_dict_loaded = True
        except:
            all_dict_loaded = False 
    if not all_dict_loaded:

        auc_ndr_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is list of auc roc scores
        # first_30_score_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is filtered noise in first 30 percent
        curves_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is list of lists of agg_method filtering scores

        noise_list_dict = {} # by run_id
        for file_name in os.listdir(base_dir_path):
            if not file_name.startswith("d_"):
                continue
            ds_file_parts = file_name.split('_')
            *_, run_id = ds_file_parts
            ds_path = os.path.join(base_dir_path, file_name)
            ds = datasets.load_from_disk(ds_path)
            trainset = ds['train']
            # noise_mask = torch.tensor(trainset['noise'], device = device)
            noise_list_dict[run_id] = trainset['noise']

        n_train = len(noise_list_dict['0'])
        for file_name_with_ext in os.listdir(base_dir_path):
            if not file_name_with_ext.startswith(i_prefix):
                continue
            file_name = file_name_with_ext.split('.')[0]
            fine_name_parts = file_name[len(i_prefix):].split('_')[1:]
            *method_parts, _, run_id_str = fine_name_parts
            infl_method = '_'.join(method_parts)
            file_path = os.path.join(base_dir_path, file_name_with_ext)
            matrix_dict = torch.load(file_path)
            noise_list = noise_list_dict[run_id_str]
            noise_one_tensor = torch.tensor(noise_list, dtype = torch.float, device = "cuda")
            num_noise = sum(noise_list)
            num_clean = len(noise_list) - num_noise
            # first_30_idx = round(0.3 * len(noise_list))
            ideal_area = num_noise / 2 + num_clean
            for module_name, inf_matrix in matrix_dict.items():
                train_ids = torch.argsort(inf_matrix, dim = 1)
                noise_tensor = noise_one_tensor[train_ids]

                noise_perc_curve = torch.cumsum(noise_tensor, dim = 1)
                noise_perc_curve /= num_noise

                auc_ndr = torch.sum((noise_perc_curve[:, :-1] + noise_perc_curve[:, 1:]) / 2, dim = 1) / ideal_area

                auc_ndr_mean = auc_ndr.mean().item()
                auc_ndr_by_methods_and_layer[(infl_method, module_name)].append(auc_ndr_mean)
                noise_perc_curve_mean = noise_perc_curve.mean(dim=0).cpu().numpy()
                curves_by_methods_and_layer[(infl_method, module_name)].append(noise_perc_curve_mean)

                del train_ids, noise_tensor, noise_perc_curve, auc_ndr

        auc_ndr_mean_std = {}

        for key, auc_ndr in auc_ndr_by_methods_and_layer.items():
            mean = np.mean(auc_ndr)
            std = np.std(auc_ndr)
            auc_ndr_mean_std[key] = (mean, std)

        curves_mean_confidence_interval = {}

        for key, curves in curves_by_methods_and_layer.items():
            curves = np.array(curves)
            mean = np.mean(curves, axis=0)
            confidence_level = 0.95
            degrees_freedom = curves.shape[0] - 1
            sample_standard_error = stats.sem(curves, axis=0)
            confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
            min_v = confidence_interval[0]
            max_v = confidence_interval[1]
            curves_mean_confidence_interval[key] = (mean * 100, min_v * 100, max_v * 100)

        sorted_method_keys = sorted(auc_ndr_mean_std.keys(), key = lambda x: auc_ndr_mean_std[x][0], reverse=True)

        if cache_file != "":
            all_dict = {'auc_rocs_by_methods_and_layer': auc_ndr_by_methods_and_layer,
                        # 'first_30_score_by_methods_and_layer': first_30_score_by_methods_and_layer,
                        'curves_by_methods_and_layer': curves_by_methods_and_layer,
                        "auc_rocs_mean_std": auc_ndr_mean_std,
                        "curves_mean_confidence_interval": curves_mean_confidence_interval,
                        "sorted_method_keys": sorted_method_keys,
                        "n_train": n_train, "num_noise": num_noise}
            torch.save(all_dict, cache_file)


    # first_5 = sorted_method_keys[:5]

    # pick top of each excluding total
    selected_infl_methods = ['cos', 'cov', 'hf', 'hf_we_', 'hf_we_topk_10', 'datainf']
    tops = []
    for infl_method in selected_infl_methods:
        infl_key = next((key for key in sorted_method_keys if key[0] == infl_method), None)
        if infl_key:
            tops.append(infl_key)

    selected_keys = tops #first_5

    xs = 100*np.arange(n_train)/n_train
    simple_module_names = {}
    plt.ioff()
    for key in selected_keys:
        module_name = key[1]
        if module_name not in simple_module_names:
            simple_name_parts = [replace_name.get(n, n) for n in module_name.split('.') if n not in ['base_model', 'model', 'roberta', 'encoder', 'embeddings', 'default', 'value', 'weight', 'attention', 'self', 'modules_to_save']]
            simple_module_name = ' '.join(simple_name_parts)
            simple_module_names[module_name] = simple_module_name
        simple_module_name = simple_module_names[module_name]
        label = f"{infl_method_names[key[0]]}, {simple_module_name}"
        curve_means, curve_mins, curve_maxs = curves_mean_confidence_interval[key]
        p = plt.plot(xs, curve_means, label=label, linewidth=1)
        plt.fill_between(xs, curve_mins, curve_maxs, alpha=.1, linewidth=0, color = p[0].get_color())
     
    best_xs =100*np.arange(num_noise)/ n_train
    best_ys = [cnt * 100 / num_noise for cnt in range(num_noise)]
    plt.plot(best_xs, best_ys, color='gray', linestyle='--', linewidth=1)
    default_ys = [cnt * 100 / n_train for cnt in range(n_train) ]
    plt.plot(xs, default_ys, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=num_noise * 100 / n_train, color='gray', linestyle='--', linewidth=1)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('Data inspected (\\%)')
    plt.ylabel('Detection Rate (\\%)')
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.legend(fontsize='small')
    plt.title(plot_title, fontsize=15)
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_chart_file)  
    plt.clf()  

    # perc = DataFrame(rows, columns = ['module', *column_names])
    # print(perc)

    # perc.to_csv(f'tmp.csv', index = False)

    pass

def get_simple_module_name(module_name, *, replace_name = {'layer': 'L', 'classifier': 'CL', 'word_embeddings': 'WE', 'lora_A': 'A', 'lora_B': 'B' }, simple_module_names = {}):
    if module_name not in simple_module_names:
        simple_name_parts = [replace_name.get(n, n) for n in module_name.split('.') if n not in ['base_model', 'model', 'roberta', 'encoder', 'embeddings', 'default', 'value', 'weight', 'attention', 'self', 'modules_to_save']]
        simple_module_name = ' '.join(simple_name_parts)
        simple_module_names[module_name] = simple_module_name   
    return simple_module_names[module_name] 

layer_regex = re.compile(r'layer\.(.+?)\.')
def get_simple_module_and_layer_name(module_name: str, *, replace_name = {'layer': 'L', 'classifier': 'CL', 'word_embeddings': 'WE', 'lora_A': 'A', 'lora_B': 'B' }, simple_module_names = {}):
    if module_name not in simple_module_names:
        module_layer_match = layer_regex.search(module_name)
        if module_layer_match:
            layer_name = module_layer_match.group(1)
        elif (".classifier." in module_name) or (module_name == 'classifier'):
            layer_name = "CL"
        elif (".word_embeddings." in module_name) or (module_name == 'word_embeddings'):
            layer_name = "WE"
        else:
            layer_name = ""
        simple_name_parts = [replace_name.get(n, n) for n in module_name.split('.') if n not in ['base_model', 'model', 'roberta', 'encoder', 'embeddings', 'default', 'value', 'weight', 'attention', 'self', 'modules_to_save']]
        num_to_skip = 0 
        if layer_name not in [ 'CL', 'WE', '']:
            num_to_skip = 2
        elif layer_name == 'CL':
            num_to_skip = 1
        simple_module_name = ' '.join(simple_name_parts[num_to_skip:])
        simple_module_names[module_name] = (layer_name, simple_module_name)
    return simple_module_names[module_name] 


def get_simple_agg_name(agg_method):
    score_names = {mean_score: "mean", mean_dir_score: "dir", median_score: "median", mean_rank_score: "rank"} #, mean_on_preds_fn: "meanpred"}
    return score_names.get(agg_method, agg_method.__name__)

def get_simple_infl_name(infl_method):
    infl_method_names = {"cos": "cos", "hf": "hf", "hf_we_": "hf_we", "hf_we_topk_10": "hf_we top 10", "datainf": "datainf", "datainf0": "datainf0", "datainf_one": "datainf1"}    
    return infl_method_names.get(infl_method, infl_method)


def compute_noise_detection_metrics_per_group2(base_dir_path: str, task: str, 
                                    m_prefix="m_b", i_prefix="i_b", 
                                    plot_title = "", cache_file = "",
                                    out_chart_file = "",
                                    module_groups_regex: dict[str, str] = {},
                                    selected_infl_methods = ['hf']):

    # loading dataset data for noise:
    # mean_on_preds_fn = partial(mean_on_preds, mask_cache = {}, base_dir = base_dir_path, task = task)
    score_fns = [mean_score] #, mean_dir_score, mean_on_preds_fn] #[mean_dir_score, mean_score, median_score] #[mean_score] #, median_score, mean_dir_score]        

    # glob_mask = torch.zeros((len(infl), len(trainset)), dtype = torch.bool, device = device)

    # here we deal with binary confusion matrix
    # i0 = torch.where(infl_labels == 0)[0]
    # i1 = torch.where(infl_labels == 1)[0]
    # t0_n = torch.where((trainset_labels == 0) & noise_mask)[0]
    # t0_c = torch.where((trainset_labels == 0) & ~noise_mask)[0]
    # t1_n = torch.where((trainset_labels == 1) & noise_mask)[0]
    # t1_c = torch.where((trainset_labels == 1) & ~noise_mask)[0]

    # selected_infl_methods = ['hf'] #'hf_we_topk_10',
    # selected_infl_methods = ['datainf', 'datainf0', 'datainf_one']    
    # selected_infl_methods = ['hf_we_', 'hf_we_topk_10']    

    module_groups_patterns = {name: re.compile(pattern) for name, pattern in module_groups_regex.items()}

    all_dict_loaded = False
    if cache_file != "": # loading all post computed metrics from cache file for visualization
        try:
            all_dict = torch.load(cache_file)
            auc_rocs_by_methods_and_layer = all_dict['auc_rocs_by_methods_and_layer']
            first_30_score_by_methods_and_layer = all_dict['first_30_score_by_methods_and_layer']
            curves_by_methods_and_layer = all_dict['curves_by_methods_and_layer']
            auc_rocs_mean_std = all_dict['auc_rocs_mean_std']
            first_30_mean_std = all_dict['first_30_mean_std']
            curves_mean_confidence_interval = all_dict['curves_mean_confidence_interval']
            sorted_method_keys = all_dict['sorted_method_keys']
            n_train = all_dict['n_train']
            num_noise = all_dict['num_noise']
            all_dict_loaded = True
        except:
            all_dict_loaded = False 
    if not all_dict_loaded:

        cancel_abs_per_module = defaultdict(list)
        cancel_norm_per_module = defaultdict(list)

        for file_name_with_ext in os.listdir(base_dir_path):
            if not file_name_with_ext.startswith("c_"):
                continue
            config_path = os.path.join(base_dir_path, file_name_with_ext)
            with open(config_path, 'r') as f:
                config = json.load(f)
            cur_cancel_abs = config['finetune']['cancel_abs']
            cur_cancel_norm = config['finetune']['cancel_norm']
            for module_name, score in cur_cancel_abs.items():
                cancel_abs_per_module[module_name].append(score)
            for module_name, score in cur_cancel_norm.items():
                cancel_norm_per_module[module_name].append(score)

        avg_cancel_abs_per_module = {module_name: (np.mean(scores), np.std(scores)) for module_name, scores in cancel_abs_per_module.items()}
        avg_cancel_norm_per_module = {module_name: (np.mean(scores), np.std(scores)) for module_name, scores in cancel_norm_per_module.items()}

        auc_rocs_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is list of auc roc scores
        first_30_score_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is filtered noise in first 30 percent
        curves_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is list of lists of agg_method filtering scores

        noise_list_dict = {} # by run_id
        for file_name in os.listdir(base_dir_path):
            if not file_name.startswith("d_"):
                continue
            ds_file_parts = file_name.split('_')
            *_, run_id = ds_file_parts
            ds_path = os.path.join(base_dir_path, file_name)
            ds = datasets.load_from_disk(ds_path)
            trainset = ds['train']
            # noise_mask = torch.tensor(trainset['noise'], device = device)
            noise_list_dict[run_id] = trainset['noise']

        n_train = len(noise_list_dict['0'])
        for file_name_with_ext in os.listdir(base_dir_path):
            if not file_name_with_ext.startswith(i_prefix):
                continue
            file_name = file_name_with_ext.split('.')[0]
            fine_name_parts = file_name[len(i_prefix):].split('_')[1:]
            *method_parts, _, run_id_str = fine_name_parts
            run_id = int(run_id_str)
            infl_method = '_'.join(method_parts)
            if infl_method not in selected_infl_methods:
                continue
            file_path = os.path.join(base_dir_path, file_name_with_ext)
            matrix_dict = torch.load(file_path)
            if len(matrix_dict) > 1:
                total_layer_matrix = None 
                group_cancel_abs = []
                group_cancel_norm = []
                for module_name, inf_matrix in matrix_dict.items():
                    group_cancel_abs.append(cancel_abs_per_module[module_name])
                    group_cancel_norm.append(cancel_norm_per_module[module_name])
                    if total_layer_matrix is None:
                        total_layer_matrix = torch.clone(inf_matrix)
                    else:
                        total_layer_matrix += inf_matrix
                matrix_dict['total'] = total_layer_matrix
                avg_cancel_abs_per_module['total'] = (np.mean(group_cancel_abs), np.std(group_cancel_abs))
                avg_cancel_norm_per_module['total'] = (np.mean(group_cancel_norm), np.std(group_cancel_norm))
            if len(module_groups_patterns) > 0:
                for pattern_name, pattern in module_groups_patterns.items():
                    total_layer_matrix = None 
                    group_cancel_abs = []
                    group_cancel_norm = []
                    for module_name, inf_matrix in matrix_dict.items():
                        if pattern.match(module_name):
                            group_cancel_abs.append(cancel_abs_per_module[module_name])
                            group_cancel_norm.append(cancel_norm_per_module[module_name])
                            if total_layer_matrix is None:
                                total_layer_matrix = torch.clone(inf_matrix)
                            else:
                                total_layer_matrix += inf_matrix
                    matrix_dict[pattern_name] = total_layer_matrix
                    avg_cancel_abs_per_module[pattern_name] = (np.mean(group_cancel_abs), np.std(group_cancel_abs))
                    avg_cancel_norm_per_module[pattern_name] = (np.mean(group_cancel_norm), np.std(group_cancel_norm))
            noise_list = noise_list_dict[run_id_str]
            num_noise = sum(noise_list)
            num_clean = len(noise_list) - num_noise
            first_30_idx = round(0.3 * len(noise_list))
            ideal_area = num_noise / 2 + num_clean
            for module_name, inf_matrix in matrix_dict.items():
                if inf_matrix is None:
                    continue
                for agg_method in score_fns:
                    scores = agg_method(inf_matrix, seed = run_id)
                    train_ids = torch.argsort(scores).tolist()
                    noise_perc_curve = []
                    noise_count = 0
                    for train_id in train_ids:
                        if noise_list[train_id]:
                            noise_count += 1
                        noise_perc_curve.append(noise_count / num_noise)
                    first_30 = noise_perc_curve[first_30_idx]
                    auc_roc = sum((noise_perc_curve[i] + noise_perc_curve[i + 1]) / 2 for i in range(len(noise_perc_curve) - 1)) / ideal_area
                    auc_rocs_by_methods_and_layer[(infl_method, agg_method, module_name)].append(auc_roc)
                    curves_by_methods_and_layer[(infl_method, agg_method, module_name)].append(noise_perc_curve)
                    first_30_score_by_methods_and_layer[(infl_method, agg_method, module_name)].append(first_30)

        first_30_mean_std = {}

        for key, first_30_values in first_30_score_by_methods_and_layer.items():
            mean = np.mean(first_30_values)
            std = np.std(first_30_values)
            first_30_mean_std[key] = (mean, std)

        auc_rocs_mean_std = {}

        for key, auc_rocs in auc_rocs_by_methods_and_layer.items():
            mean = np.mean(auc_rocs)
            std = np.std(auc_rocs)
            auc_rocs_mean_std[key] = (mean, std)

        curves_mean_confidence_interval = {}

        for key, curves in curves_by_methods_and_layer.items():
            curves = np.array(curves)
            mean = np.mean(curves, axis=0)
            confidence_level = 0.95
            degrees_freedom = curves.shape[0] - 1
            sample_standard_error = stats.sem(curves, axis=0)
            confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
            min_v = confidence_interval[0]
            max_v = confidence_interval[1]
            curves_mean_confidence_interval[key] = (mean * 100, min_v * 100, max_v * 100)

        sorted_method_keys = sorted(first_30_mean_std.keys(), key = lambda x: first_30_mean_std[x][0], reverse=True)

        if cache_file != "":
            all_dict = {'auc_rocs_by_methods_and_layer': auc_rocs_by_methods_and_layer,
                        'first_30_score_by_methods_and_layer': first_30_score_by_methods_and_layer,
                        'curves_by_methods_and_layer': curves_by_methods_and_layer,
                        "auc_rocs_mean_std": auc_rocs_mean_std, "first_30_mean_std": first_30_mean_std,
                        "curves_mean_confidence_interval": curves_mean_confidence_interval,
                        "sorted_method_keys": sorted_method_keys,
                        "n_train": n_train, "num_noise": num_noise}
            torch.save(all_dict, cache_file)

    rows = []
    for sk in sorted_method_keys:
        infl_method, agg_method, module_name = sk
        module_simple_name = get_simple_module_name(module_name)
        agg_simple_name = get_simple_agg_name(agg_method)
        infl_simple_name = get_simple_infl_name(infl_method)        
        f30_mean, f30_std = first_30_mean_std[sk]
        auc_ndr_mean, auc_ndr_std = auc_rocs_mean_std[sk]
        cancel_abs_mean, cancel_abs_std = avg_cancel_abs_per_module[module_name]
        cancel_norm_mean, cancel_norm_std = avg_cancel_norm_per_module[module_name]
        rows.append([infl_simple_name, agg_simple_name, module_simple_name, f30_mean, f30_std, auc_ndr_mean, auc_ndr_std, cancel_abs_mean, cancel_abs_std, cancel_norm_mean, cancel_norm_std])

    selected_infl_methods_str = "-".join(selected_infl_methods)
    stats_file_path = os.path.join(base_dir_path, "..", "postprocess", task, f"stats_{task}_{selected_infl_methods_str}.txt")
    with open(stats_file_path, "w") as stats_file:
        print(tabulate(rows, headers = ['infl', 'agg', 'module', 'f30_mean', 'f30_std', 'auc_ndr_mean', 'auc_ndr_std', 'cancel_abs_mean', 'cancel_abs_std', 'cancel_norm_mean', 'cancel_norm_std'], tablefmt="github", floatfmt=".3f"), file = stats_file)
    pass
    # first_5 = sorted_method_keys[:5]

    # pick top of each excluding total
    selected_keys = [k for k in sorted_method_keys if k[2] in module_groups_patterns]

    # selected_keys = tops #first_5

    xs = 100*np.arange(n_train)/n_train
    simple_module_names = {}
    plt.ioff()
    for key in selected_keys:
        module_name = key[2]

        module_simple_name = get_simple_module_name(module_name)
        agg_simple_name = get_simple_agg_name(key[1])
        infl_simple_name = get_simple_infl_name(key[0])
        label = f"{infl_simple_name}, {agg_simple_name}, {module_simple_name}"
        curve_means, curve_mins, curve_maxs = curves_mean_confidence_interval[key]
        p = plt.plot(xs, curve_means, label=label, linewidth=1)
        plt.fill_between(xs, curve_mins, curve_maxs, alpha=.1, linewidth=0, color = p[0].get_color())
     
    best_xs =100*np.arange(num_noise)/ n_train
    best_ys = [cnt * 100 / num_noise for cnt in range(num_noise)]
    plt.plot(best_xs, best_ys, color='gray', linestyle='--', linewidth=1)
    default_ys = [cnt * 100 / n_train for cnt in range(n_train) ]
    plt.plot(xs, default_ys, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=num_noise * 100 / n_train, color='gray', linestyle='--', linewidth=1)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('Data inspected (\\%)')
    plt.ylabel('Detection Rate (\\%)')
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.legend(fontsize='small')
    plt.title(plot_title, fontsize=15)
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_chart_file)  
    plt.clf()  

    # perc = DataFrame(rows, columns = ['module', *column_names])
    # print(perc)

    # perc.to_csv(f'tmp.csv', index = False)

    pass

import pandas as pd

def split_dict(d: dict, filter_fn:callable):
    d1 = {}
    d2 = {}
    for k, v in d.items():
        if filter_fn(k):
            d1[k] = v
        else:
            d2[k] = v
    return d1, d2

def get_avg_ranks(setup_score_values: dict[str, list[float]], *, ascending = True) -> dict[str, float]:
    ''' ranks setups, assumes that same trials are given as list, index of trial defines same trial accross setups  '''
    setup_names = list(setup_score_values.keys())
    total_measures = np.array([setup_score_values[setup_name] for setup_name in setup_names])
    df = pd.DataFrame(total_measures.T)
    # sort_indeces = np.argsort(total_measures_per_trial, axis = 1)
    ranks = df.rank(axis = 1, method="min", ascending=ascending)
    mean_ranks = ranks.mean(axis = 0)
    rank_dict = {setup_name: mean_rank for setup_name, mean_rank in zip(setup_names, mean_ranks)}
    return rank_dict

# data = {'setup1': [10, 20, 10], 'setup2': [20, 10, 5], 'setup3': [5, 15, 15], 'setup4': [5, 5, 5]}
# get_ranks(data)


    # n_trials = len(setup_score_values[setup_names[0]])
    # setup_ranks = {setup_name: [] for setup_name in setup_names}
    # for trial_id in range(n_trials):
    #     trial_scores = [(setup_id, setup_score_values[setup_name][trial_id]) for setup_id, setup_name in enumerate(setup_names)]
    #     trial_scores_sorted = sorted(trial_scores, key=lambda x:x[1], reverse = reverse)
    #     for rank_id, (setup_id, _) in enumerate(trial_scores_sorted):
    #         setup_ranks[setup_names[setup_id]].append(rank_id + 1)
    #     trial_rank = {score: rank for rank, score in enumerate(trial_scores_sorted, 1)}
    #     for setup_name in setup_names:
    #         setup_score_values[setup_name][trial_id] = trial_rank[setup_score_values[setup_name][trial_id]]


def load_ds_info(task_in_dir: str):
    noise_list_dict = {}
    trainset_labels_dict = {}
    inflset_labels_dict = {}
    for file_name in os.listdir(task_in_dir):
        if not file_name.startswith("d_"):
            continue
        ds_file_parts = file_name.split('_')
        *_, run_id = ds_file_parts
        ds_path = os.path.join(task_in_dir, file_name)
        ds = datasets.load_from_disk(ds_path)
        trainset = ds['train']
        noise_list_dict[run_id] = trainset['noise']
        trainset_labels_dict[run_id] = trainset['labels']
        inflset = ds['infl']
        inflset_labels_dict[run_id] = inflset['labels']
    return noise_list_dict, trainset_labels_dict, inflset_labels_dict

def load_m_info(task_in_dir: str, m_prefix: str):
    ''' Loads infl samples logits on the specified chehckpoint '''
    inflset_logits = {}
    for file_name in os.listdir(task_in_dir):
        if not file_name.startswith(m_prefix):
            continue
        m_file_parts = file_name.split('_')
        *_, run_id = m_file_parts
        logits_m_path = os.path.join(task_in_dir, file_name, "infl_logits.pt")
        logits = torch.load(logits_m_path)
        inflset_logits[run_id] = logits
    return inflset_logits

def get_cancellation_metrics_table(base_dir_path: str, task='qnli',
                                    module_groups_regex: dict[str, str] = {}):
    cancel_abs_per_module = defaultdict(list)
    cancel_norm_per_module = defaultdict(list)    
    module_groups_patterns = {name: re.compile(pattern) for name, pattern in module_groups_regex.items()}
    task_in_dir = os.path.join(base_dir_path, task)
    for file_name_with_ext in os.listdir(task_in_dir):
        if not file_name_with_ext.startswith("c_"):
            continue
        config_path = os.path.join(task_in_dir, file_name_with_ext)
        with open(config_path, 'r') as f:
            config = json.load(f)
        cur_cancel_abs = config['finetune']['cancel_abs']
        cur_cancel_norm = config['finetune']['cancel_norm']
        for module_name, score in cur_cancel_abs.items():
            cancel_abs_per_module[module_name].append(score)
        for module_name, score in cur_cancel_norm.items():
            cancel_norm_per_module[module_name].append(score)

    module_names = list(cancel_abs_per_module.keys())
    for pattern_name, pattern in module_groups_patterns.items():
        group_cancel_abs = []
        group_cancel_norm = []
        for module_name in module_names:
            if pattern.match(module_name):
                group_cancel_abs.append(cancel_abs_per_module[module_name])
                group_cancel_norm.append(cancel_norm_per_module[module_name])
        cancel_abs_per_module[pattern_name] = np.array(group_cancel_abs).mean(axis = 0)
        cancel_norm_per_module[pattern_name] = np.array(group_cancel_norm).mean(axis = 0)

    cancel_abs_per_group, cancel_abs_per_orig_module = split_dict(cancel_abs_per_module, lambda k: (k in module_groups_patterns) or (k == 'total'))
    cancel_norm_per_group, cancel_norm_per_orig_module = split_dict(cancel_norm_per_module, lambda k: (k in module_groups_patterns) or (k == 'total'))
    cancel_abs_module_ranks = get_avg_ranks(cancel_abs_per_orig_module)
    cancel_abs_group_ranks = get_avg_ranks(cancel_abs_per_group)
    cancel_norm_module_ranks = get_avg_ranks(cancel_norm_per_orig_module)
    cancel_norm_group_ranks = get_avg_ranks(cancel_norm_per_group)
    avg_cancel_abs_per_module = {module_name: (np.mean(scores), np.std(scores)) for module_name, scores in cancel_abs_per_module.items()}
    avg_cancel_norm_per_module = {module_name: (np.mean(scores), np.std(scores)) for module_name, scores in cancel_norm_per_module.items()}

    rows = []
    for module_name in module_names:
        cancel_abs_mean, cancel_abs_std = avg_cancel_abs_per_module[module_name]
        cancel_norm_mean, cancel_norm_std = avg_cancel_norm_per_module[module_name]
        cancel_abs_module_rank = cancel_abs_module_ranks.get(module_name, pd.NA)
        cancel_abs_group_rank = cancel_abs_group_ranks.get(module_name, pd.NA)
        cancel_norm_module_rank = cancel_norm_module_ranks.get(module_name, pd.NA)
        cancel_norm_group_rank = cancel_norm_group_ranks.get(module_name, pd.NA)
        cancel_row_data = {
            "cancel_norm_module_rank": cancel_norm_module_rank,
            "cancel_norm_group_rank": cancel_norm_group_rank,
            "cancel_norm_mean": cancel_norm_mean,
            "cancel_norm_std": cancel_norm_std,
            "cancel_abs_module_rank": cancel_abs_module_rank,
            "cancel_abs_group_rank": cancel_abs_group_rank,
            "cancel_abs_mean": cancel_abs_mean,
            "cancel_abs_std": cancel_abs_std                
        }
        rows.append(cancel_row_data)
    df = pd.DataFrame(rows)
    return df

def compute_ndr_metrics_table(base_dir_path: str, task='qnli', 
                                    module_groups_regex: dict[str, str] = {},
                                    infl_methods = ['hf'],
                                    agg_methods: dict[str, callable] = {"mean": mean_matrix_score},
                                    include_total = True,
                                    m_prefix = "m_b", i_prefix="i_b"):

    agg_method_names = list(agg_methods.keys())
    module_groups_patterns = {name: re.compile(pattern) for name, pattern in module_groups_regex.items()}

    task_in_dir = os.path.join(base_dir_path, task)

    auc_ndr_by_infl = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is list of auc roc scores
    f30_score_by_infl = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is filtered noise in first 30 percent
    # curves_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is list of lists of agg_method filtering scores

    noise_list_dict, trainset_labels_dict, inflset_labels_dict = load_ds_info(task_in_dir)

    inflset_logits_dict = load_m_info(task_in_dir, m_prefix)

    for file_name_with_ext in os.listdir(task_in_dir):
        if not file_name_with_ext.startswith(i_prefix):
            continue
        file_name = file_name_with_ext.split('.')[0]
        fine_name_parts = file_name[len(i_prefix):].split('_')[1:]
        *method_parts, _, run_id_str = fine_name_parts
        # run_id = int(run_id_str)
        infl_method = '_'.join(method_parts)
        if infl_method not in infl_methods:
            continue
        file_path = os.path.join(task_in_dir, file_name_with_ext)
        matrix_dict = torch.load(file_path)        
        module_names = list(matrix_dict.keys())
        module_and_group_names = list(module_names)
        # transpose all matrices 
        for module_name in module_names:
            matrix_dict[module_name] = matrix_dict[module_name].t() # first dim is train_sample now and second is infl val sample
        if len(module_names) == 0:
            continue
        all_interactions = torch.stack([matrix_dict[module_name] for module_name in module_names], dim = 1)
        for int_matrix in matrix_dict.values():
            del int_matrix
        # now the dimensions is train_sample * module * infl_val_sample

        # create module views
        module_interactions = [all_interactions[:, module_id:module_id+1] for module_id in range(len(module_names))]
        group_modules = defaultdict(list)
        for group_name, pattern in module_groups_patterns.items():
            for module_id, module_name in enumerate(module_names):
                if pattern.match(module_name):
                    group_modules[group_name].append(module_id)

        group_names = list(group_modules.keys())
        module_and_group_names.extend(group_names)
        for group_name in group_names:
            group_module_ids = torch.tensor(group_modules[group_name], device = all_interactions.device)
            module_interactions.append(all_interactions[:, group_module_ids])
            del group_module_ids

        if include_total:
            module_interactions.append(all_interactions)
            module_and_group_names.append('total')
        
        noise_list = noise_list_dict[run_id_str]
        num_noise = sum(noise_list)
        num_clean = len(noise_list) - num_noise
        first_30_idx = round(0.3 * len(noise_list))
        ideal_area = num_noise / 2 + num_clean

        noise_mask = torch.tensor(noise_list, device = all_interactions.device)
        trainset_labels = torch.tensor(trainset_labels_dict[run_id_str], device = all_interactions.device)
        inflset_labels = torch.tensor(inflset_labels_dict[run_id_str], device = all_interactions.device)
        inflset_logits = inflset_logits_dict[run_id_str]
        
        scores = torch.zeros((len(agg_methods), len(module_interactions), all_interactions.shape[0]), device = all_interactions.device)

        for agg_method_id, agg_method_name in enumerate(agg_method_names):
            for matrix_id, inf_matrix in enumerate(module_interactions):
                agg_method_fn = agg_methods[agg_method_name] 
                new_scores = agg_method_fn(inf_matrix, noise_mask = noise_mask, 
                                       trainset_labels = trainset_labels, inflset_labels = inflset_labels, 
                                       inflset_logits = inflset_logits, task = task, run_id = run_id_str)
                scores[agg_method_id, matrix_id] = new_scores
                del new_scores
                torch.cuda.empty_cache() 
                
        del trainset_labels, inflset_labels
                
        train_ids = torch.argsort(scores, dim = -1)

        noise_tensor = noise_mask[train_ids]

        del noise_mask

        noise_detection_curves = torch.cumsum(noise_tensor, dim = -1, dtype = torch.float)
        noise_detection_curves /= num_noise

        auc_ndrs = noise_detection_curves.sum(dim = -1) 
        auc_ndrs /= ideal_area

        first_30 = noise_detection_curves[:, :, first_30_idx]

        auc_ndrs_cpu = auc_ndrs.cpu()
        first_30_cpu = first_30.cpu()
        del auc_ndrs, first_30, noise_detection_curves, scores, train_ids

        for agg_method_id, agg_method_name in enumerate(agg_method_names):
            for module_id, module_name in enumerate(module_and_group_names):
                auc_ndr_by_infl[(infl_method, agg_method_name, module_name)].append(auc_ndrs_cpu[agg_method_id, module_id].item())
                f30_score_by_infl[(infl_method, agg_method_name, module_name)].append(first_30_cpu[agg_method_id, module_id].item())
                # curves_by_methods_and_layer[(infl_method, agg_method, module_name)].append(noise_perc_curve)
        torch.cuda.empty_cache() 

    first_30_mean_std = {key: (np.mean(first_30_values), np.std(first_30_values)) for key, first_30_values in f30_score_by_infl.items() }
    first_30_ranks = get_avg_ranks(f30_score_by_infl, ascending=False)

    auc_rocs_mean_std = {key: (np.mean(auc_ndr), np.std(auc_ndr)) for key, auc_ndr in auc_ndr_by_infl.items() }
    auc_ndr_ranks = get_avg_ranks(auc_ndr_by_infl, ascending=False)

    sorted_method_keys = sorted(first_30_ranks.keys(), key = lambda x: (first_30_ranks[x], x))

    rows = []
    for sk in sorted_method_keys:
        infl_method, agg_method_name, module_name = sk
        if module_name in module_groups_patterns or module_name == 'total':
            module_layer, module_simple_name = module_name, "*"
        else:
            module_layer, module_simple_name = get_simple_module_and_layer_name(module_name)
        f30_mean, f30_std = first_30_mean_std[sk]
        auc_ndr_mean, auc_ndr_std = auc_rocs_mean_std[sk]
        f30_rank = first_30_ranks[sk]
        auc_ndr_rank = auc_ndr_ranks[sk]
        row_data = {
            "infl": infl_method,
            "agg": agg_method_name,
            "layer": module_layer,
            "module": module_simple_name,
            "f30_rank": f30_rank,
            "f30_mean": f30_mean,
            "f30_std": f30_std,
            "auc_ndr_rank": auc_ndr_rank,
            "auc_ndr_mean": auc_ndr_mean,
            "auc_ndr_std": auc_ndr_std,
        }
        rows.append(row_data)

    df = pd.DataFrame(rows)

    return df

def output_table(df: pd.DataFrame, base_path: str, task: str):
    ndr_stats_file_path = os.path.join(base_path, f"{task}_ndr_stats.csv")
    
    df.to_csv(ndr_stats_file_path, index = False)

    stats_file_path = os.path.join(base_path, f"{task}_ndr_stats_simple.txt")
    with open(stats_file_path, "w") as stats_file:
        print(tabulate(df, headers = 'keys', tablefmt="github", floatfmt=".3f", showindex=True), file = stats_file)
    pass 

def draw_ft2_metric2(task: str, infile: str, outfile: str, metric = 'accuracy', infl_methods = [], module_pattern_to_name = {},
                        colors = {}, legend_order = {}, legend_names = {}, infl_vs_module_filter = [],
                        with_rand_denoise = True):
    with open(infile, 'r') as f:
        json_lines = f.readlines()
    all_metrics = [json.loads(l) for l in json_lines]
    method_metrics = defaultdict(list)
    for metrics in all_metrics:
        metric_values = metrics[metric]
        infl_method = metrics['config']['infl_method']
        filter_method = metrics['config']['filter_method']                
        module_pattern = metrics['config']['module_pattern']
        # if module_pattern == '':
        #     module_pattern = filter_method
        module_name = module_pattern_to_name.get(module_pattern, module_pattern)
        selected = (((infl_method in infl_methods) or (filter_method in infl_methods)) and \
                           module_pattern in module_pattern_to_name) or \
                          (infl_method, module_name) in infl_vs_module_filter or \
                          (filter_method, module_name) in infl_vs_module_filter
        if not selected:
            continue
        is_infl = False
        if filter_method  == 'infl':
            filter_method = infl_method
            is_infl = True
        if module_pattern != "" and is_infl:
            filter_method = f'{filter_method}, {module_name}'
        method_metrics[filter_method].append(metric_values)

    baseline_metrics = defaultdict(list)
    baseline_metric_names = []
    if with_rand_denoise:
        baseline_metric_names = ['rand', 'denoise']
        for metrics in all_metrics:
            metric_values = metrics[metric]
            filter_method = metrics['config']['filter_method']
            module_name = module_pattern_to_name.get(module_pattern, module_pattern)
            selected = filter_method in ["rand", "denoise"]
            if not selected:
                continue
            baseline_metrics[filter_method].append(metric_values)
    
    method_metrics_flat = {k: [v3 for v2 in v for v3 in v2] for k, v in method_metrics.items()}
    method_metric_ranks = get_avg_ranks(method_metrics_flat)

    method_names = sorted(method_metrics.keys(), key = method_metric_ranks.get)
    plt.ioff()

    handles_dict = {}
    labels_dict = {}

    for method in baseline_metric_names:
        metrics = baseline_metrics[method]
        metric_values = np.array(metrics) * 100
        mean = np.mean(metric_values, axis=0)
        confidence_level = 0.95
        degrees_freedom = metric_values.shape[0] - 1
        sample_standard_error = stats.sem(metric_values, axis=0)
        confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
        min_v = confidence_interval[0]
        max_v = confidence_interval[1]
        default_args = dict(marker='o', markersize=4, linewidth=1)
        if method == 'denoise':
            default_args = dict(linewidth=1, color="gray", linestyle='--')
        if method == 'rand':
            default_args = dict(linewidth=1, color="gray", linestyle='-.')
        # xs = np.arange(len(mean)) + 1
        xs = np.arange(len(mean)) + 1 # Shift x-coordinates slightly
        line = plt.plot(xs, mean, **default_args)
        plt.fill_between(xs, min_v, max_v, alpha=.05, color = line[0].get_color(), linewidth=0)
        # plt.errorbar(xs, mean, yerr=[mean - min_v, max_v - mean], alpha=.5, fmt='none', ecolor=line[0].get_color(), capsize=2, linewidth=1, zorder=0)
        handles_dict[method] = line[0]
        labels_dict[method] = legend_names.get(method, method)  

    for i, method in enumerate(method_names):
        metrics = method_metrics[method]
        metric_values = np.array(metrics) * 100
        mean = np.mean(metric_values, axis=0)
        confidence_level = 0.95
        degrees_freedom = metric_values.shape[0] - 1
        sample_standard_error = stats.sem(metric_values, axis=0)
        confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
        min_v = confidence_interval[0]
        max_v = confidence_interval[1]
        default_args = dict(marker='o', markersize=4, linewidth=1, color = colors[method])
        if method == 'denoise':
            default_args = dict(linewidth=1, linestyle='--')
        if method == 'rand':
            default_args = dict(linewidth=1, linestyle='-.')
        # xs = np.arange(len(mean)) + 1
        xs = np.arange(len(mean)) + 1 + ((i - len(method_metrics) // 2) * 0.075)  # Shift x-coordinates slightly
        line = plt.plot(xs, mean, linestyle='none', zorder=1, **default_args)
        # plt.fill_between(xs, min_v, max_v, alpha=.05, color = line[0].get_color(), linewidth=0)
        plt.errorbar(xs, mean, yerr=[mean - min_v, max_v - mean], alpha=.5, fmt='none', ecolor=line[0].get_color(), capsize=2, linewidth=1, zorder=0)
        handles_dict[method] = line[0]
        labels_dict[method] = legend_names.get(method, method)      
    
    ordered_legend_names = sorted([k for k in handles_dict.keys() if k not in baseline_metric_names], key = legend_order.get)
    ordered_legend_names = ordered_legend_names + baseline_metric_names
    ordered_handles = [handles_dict[k] for k in ordered_legend_names]
    ordered_labels = [labels_dict[k] for k in ordered_legend_names]
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy, \\%')
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.legend(ordered_handles, ordered_labels, fontsize='small')
    plt.title(f'{task.upper()} 70\\% filtered finetuning', fontsize=15)
    plt.tight_layout()
    plt.savefig(outfile)  
    plt.clf()  

def draw_ft2_metric3(infile: str, outfile: str, metric = 'accuracy', method_dict: dict = {}, draw_diff = False):
    with open(infile, 'r') as f:
        json_lines = f.readlines()
    all_metrics = [json.loads(l) for l in json_lines]
    method_metrics = defaultdict(list)
    for metrics in all_metrics:
        if draw_diff:
            before_metric_values = metrics['first_finetune'][metric] 
            after_metric_values = metrics[metric]
            metric_values = [a - b for a, b in zip(after_metric_values, before_metric_values)]
        else:
            metric_values = metrics[metric]
        task = metrics['config']['task']
        infl_method = metrics['config']['infl_method']
        agg_method = metrics['config']['agg_method']                
        module_name = metrics['config']['module_name']

        key = (infl_method, agg_method, module_name)
        if key not in method_dict:
            continue
        method_metrics[key].append(metric_values)

    method_metrics_flat = {k: [v3 for v2 in v for v3 in v2] for k, v in method_metrics.items()}
    method_metric_ranks = get_avg_ranks(method_metrics_flat)

    method_names = sorted(method_metrics.keys(), key = method_metric_ranks.get)
    plt.ioff()

    handles_ = []

    for i, (infl_method, agg_method, module_name) in enumerate(method_names):
        metrics = method_metrics[(infl_method, agg_method, module_name)]
        metric_values = np.array(metrics) * 100
        mean = np.mean(metric_values, axis=0)
        confidence_level = 0.95
        degrees_freedom = metric_values.shape[0] - 1
        sample_standard_error = stats.sem(metric_values, axis=0)
        confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
        min_v = confidence_interval[0]
        max_v = confidence_interval[1]
        method_settings = method_dict[(infl_method, agg_method, module_name)]
        default_args = dict(marker='o', markersize=4, linestyle='none', linewidth=1, color = method_settings['color'])
        draw_full = False
        if infl_method == 'denoise':
            default_args['linestyle']='--'
            default_args['markersize'] = 0
            draw_full = True 
        if infl_method == 'rand':
            default_args['linestyle']='-.'
            default_args['markersize'] = 0
            draw_full = True 
        # xs = np.arange(len(mean)) + 1
        xs = np.arange(len(mean)) + 1 + ((i - len(method_metrics) // 2) * 0.075)  # Shift x-coordinates slightly
        line = plt.plot(xs, mean, zorder=1, **default_args)
        if draw_full:
            plt.fill_between(xs, min_v, max_v, alpha=.05, color = line[0].get_color(), linewidth=0)
        else:
            plt.errorbar(xs, mean, yerr=[mean - min_v, max_v - mean], alpha=.5, fmt='none', ecolor=line[0].get_color(), capsize=2, linewidth=1, zorder=0)
        handles_.append((line[0], method_settings['legend_name'], method_settings['legend_order']))
    
    handles_.sort(key = lambda x: x[2])
    ordered_handles = [h[0] for h in handles_]
    ordered_labels = [h[1] for h in handles_]
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy, \\%')
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.legend(ordered_handles, ordered_labels, fontsize='small')
    plt.title(f'{task.upper()} 70\\% filtered finetuning', fontsize=15)
    plt.tight_layout()
    plt.savefig(outfile)  
    plt.clf()  


# RQ1: what checkpoint it is better to use with infl methods to detect noise 
def get_df_from_file(metric_file: str, suffix):

    # file_name = os.path.basename(metric_file).split(".")[0]
    with open(metric_file, 'r') as f:
        runs = [json.loads(l) for l in f.readlines()]

    rows = []
    rand_accuracies = {}
    for r in runs: 

        accuracy0 = r["first_finetune"]["accuracy"]
        accuracy = r["accuracy"]
        best_accuracy0 = np.max(accuracy0)
        best_accuracy = np.max(accuracy)
        accuracy_delta = best_accuracy - best_accuracy0

        infl_accuracy0 = r["first_finetune"]["infl_accuracy"]
        infl_accuracy = r["infl_accuracy"]
        best_infl_accuracy0 = np.max(infl_accuracy0)
        best_infl_accuracy = np.max(infl_accuracy)
        infl_accuracy_delta = best_infl_accuracy - best_infl_accuracy0

        infl_loss0 = r["first_finetune"]["infl_loss"]
        infl_loss = r["infl_loss"]
        min_infl_loss0 = np.min(infl_loss0)
        min_infl_loss = np.min(infl_loss)        
        infl_loss_delta = min_infl_loss - min_infl_loss0

        logits_change = r["logits_change"]

        auc_ndr = r["auc_ndr"]
        filtered = r["filtered"]

        noise_curve = r["ndr_curve"]
        index_30 = round(0.3 * len(noise_curve))
        noise_30  = noise_curve[index_30]
        num_noise = round(0.2 * len(noise_curve))
        ideal_area = num_noise / 2 + (len(noise_curve) - num_noise)        
        auc_ndr2 = sum((noise_curve[i] + noise_curve[i + 1]) / 2 for i in range(len(noise_curve) - 1)) / ideal_area

        assert np.allclose(auc_ndr, auc_ndr2, atol=1e-3)
        assert np.allclose(noise_30, filtered, atol=1e-3)

        task = r["config"]["task"]
        infl_method = r["config"]["infl_method"]
        seed0 = r["config"]["seed"]
        seed1 = r["config"]["seed2"]

        if infl_method == "rand":
            rand_accuracies[(suffix, task, seed0, seed1)] = (best_accuracy, best_infl_accuracy)

        row = {"checkpoint": suffix,
               "task": task,
               "infl_method": infl_method,
               "agg_method": r["config"]["agg_method"],
               "module": r["config"]["module_name"],
               
               "best_accuracy_delta": accuracy_delta,
               "best_infl_accuracy_delta": infl_accuracy_delta,
                "infl_loss_delta": infl_loss_delta,
               "logits_change": logits_change,
               "noise_30": filtered,
               "auc_ndr": auc_ndr,

               "best_accuracy_0": best_accuracy0,
               "best_accuracy_1": best_accuracy,

               "best_infl_accuracy_0": best_infl_accuracy0,
               "best_infl_accuracy_1": best_infl_accuracy,

                "infl_loss_0": min_infl_loss0,
                'infl_loss_1': min_infl_loss,

               "seed0": seed0,
               "seed1": seed1,

               }
        rows.append(row)

    for r in rows:
        key = (r["checkpoint"], r["task"], r["seed0"], r["seed1"])
        rand_accuracy, rand_infl_accuracy = rand_accuracies[key]
        accuracy_rand_delta = r["best_accuracy_1"] - rand_accuracy
        infl_accuracy_rand_delta = r["best_infl_accuracy_1"] - rand_infl_accuracy
        r["accuracy_rand_delta"] = accuracy_rand_delta
        r["infl_accuracy_rand_delta"] = infl_accuracy_rand_delta

    df = pd.DataFrame(rows)

    return df

def get_agg_df(df: DataFrame, key_columns = ['checkpoint', 'task', 'infl_method', 'agg_method', 'module'],
                drop_columns=['seed0', 'seed1'], selected_columns=None, agg_types = ['mean', 'std']):
    if selected_columns is None:
        grouped = df.drop(columns=drop_columns).groupby(key_columns)
    else:
        grouped = df[key_columns + selected_columns].drop(columns=drop_columns).groupby(key_columns)
    aggregated = grouped.agg(agg_types).reset_index()
    aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]
    aggregated_sorted = aggregated.sort_values(by='best_accuracy_delta_mean', ascending=False)
    std_columns = [col for col in aggregated_sorted.columns if col.endswith('std')]
    other_columns = [col for col in aggregated_sorted.columns if not col.endswith('std')]

    # Reorder the columns
    aggregated_sorted = aggregated_sorted[other_columns + std_columns]    
    return aggregated_sorted

def get_all_df(base_path = "data/roberta-2", datasets = ["mrpc", "qnli", "sst2", "qqp"],
                checkpoints = ["b", "l", "bl"],
                tmp_file = "tmp.pkl"):
    if os.path.exists(tmp_file):
        df1 = pd.read_pickle(tmp_file)
    else:
        plain_dataframes = []
        for dataset in datasets:
            for checkpoint in checkpoints:
                file = os.path.join(base_path, f"{dataset}-{checkpoint}.jsonlist")
                df = get_df_from_file(file, checkpoint)
                plain_dataframes.append(df)
        
        df1 = pd.concat(plain_dataframes, ignore_index=True)
        df1.to_pickle(tmp_file)
    return df1

def rank_checkpoints(base_path = "data/roberta-2", metric = "noise_30",
                        datasets = ["mrpc", "qnli", "sst2", "qqp"],
                        checkpoints = ["b", "l", "bl"]):
    df1 = get_all_df(base_path = base_path, datasets = datasets,
                        checkpoints = checkpoints)
    key_columns = ["checkpoint", "task", "infl_method", "agg_method", "module", "seed0", "seed1"]

    df2 = df1[(df1["agg_method"] != "")]
    df3 = df2[key_columns + [metric]]
    df4 = df3.pivot(index=["task", "infl_method", "agg_method", "module", "seed0", "seed1"], columns="checkpoint", values=metric)
    df4 = df4[~df4.isna().any(axis=1)]
    df5: DataFrame = df4[df4[checkpoints].gt(0).any(axis=1)]

    stats_data = df5.T.to_numpy()

    friedman_res = sci_stats.friedmanchisquare(*stats_data)

    nemenyi_res = sci_posthocs.posthoc_nemenyi_friedman(stats_data.T) 
    print(f"\n----------------------------------")
    print(f"Friedman: {friedman_res}")
    from tabulate import tabulate
    names = ["b", "bl", "l"]
    rows = []
    for i in range(len(names)):
        row = []
        row.append(names[i])
        for j in range(len(names)):
            row.append(nemenyi_res[i][j])
        rows.append(row)
    print(tabulate(rows, headers=["", *names], tablefmt="grid", numalign="center", stralign="center"))


    # df5.groupby(["task", "infl_method", "agg_method", "module"]).agg(["mean"]).groupby(["task", "infl_method", "agg_method"]).agg(["max"]).reset_index()    
    mean_ranks = df5.rank(axis=1, ascending=False).mean(axis=0)
    mean_metric = df5.mean(axis=0)
    print(mean_ranks)
    print(mean_metric)
    # score = df6.groupby(["task"]).agg(["mean"]).reset_index()
    return mean_ranks, mean_metric


def estimate_ndr(folder:str, prefix: str):
    noise_30 = {}
    noise_list_dict, trainset_labels_dict, inflset_labels_dict = load_ds_info(folder)
    for file_name in os.listdir(folder):
        if not file_name.startswith(prefix):
            continue 
        run_id = file_name.split(".")[0].split("_")[-1]
        noise_list = noise_list_dict[run_id]
        noise_mask = torch.tensor(noise_list, device = "cpu")
        total_noise = noise_mask.sum()
        path = os.path.join(folder, file_name)
        score_dict = torch.load(path)
        for key, scores in score_dict.items():
            train_idxs = torch.argsort(scores)
            filter_len = int(0.3*len(train_idxs))
            train_idxs_left = train_idxs[:filter_len].cpu().numpy()            
            noise_tensor = noise_mask[train_idxs_left]            
            num_noise = noise_tensor.sum()
            ndr = num_noise / total_noise
            noise_30.setdefault(key, []).append(ndr)
        pass
    rows = []
    for (infl_method, agg_method, module), ndrs in noise_30.items():
        rows.append({
            "infl_method": infl_method, "agg_method": agg_method,
            "module": module, "ndr_30_mean": np.mean(ndrs), "ndr_30_std": np.std(ndrs)
        })
    df = DataFrame(rows)
    df = df.sort_values(by='ndr_30_mean', ascending=False)
    pass 

if __name__ == "__main__":
    # estimate_ndr("./data/roberta-2/sst2", "s_bl_")

    rank_checkpoints(metric = "infl_accuracy_rand_delta")
    pass

    # draw_ft2_metric3('./data/mistral/sst2.jsonlist',
    #                  './data/mistral/sst2-acc-hf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-07'): {'color': '#2ca02c', 'legend_name': 'hf, 00-07', 'legend_order': 3},
    #                      ('hf', 'mean', '08-15'): {'color': '#d62728', 'legend_name': 'hf, 08-15', 'legend_order': 4},
    #                      ('hf', 'mean', '16-23'): {'color': '#8c564b', 'legend_name': 'hf, 16-23', 'legend_order': 5},
    #                      ('hf', 'mean', '24-31'): {'color': '#e377c2', 'legend_name': 'hf, 24-31', 'legend_order': 6},
    #                     #  ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/mistral/sst2.jsonlist',
    #                  './data/mistral/sst2-acc-cos.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-07'): {'color': '#2ca02c', 'legend_name': 'cos, 00-07', 'legend_order': 3},
    #                      ('cos', 'mean', '08-15'): {'color': '#d62728', 'legend_name': 'cos, 08-15', 'legend_order': 4},
    #                      ('cos', 'mean', '16-23'): {'color': '#8c564b', 'legend_name': 'cos, 16-23', 'legend_order': 5},
    #                      ('cos', 'mean', '24-31'): {'color': '#e377c2', 'legend_name': 'cos, 24-31', 'legend_order': 6},
    #                     #  ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/mistral/sst2.jsonlist',
    #                  './data/mistral/sst2-acc-datainf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-07'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-07', 'legend_order': 3},
    #                      ('datainf', 'mean', '08-15'): {'color': '#d62728', 'legend_name': 'datainf, 08-15', 'legend_order': 4},
    #                      ('datainf', 'mean', '16-23'): {'color': '#8c564b', 'legend_name': 'datainf, 16-23', 'legend_order': 5},
    #                      ('datainf', 'mean', '24-31'): {'color': '#e377c2', 'legend_name': 'datainf, 24-31', 'legend_order': 6},
    #                     #  ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass     

    # draw_ft2_metric3('./data/mistral/qqp.jsonlist',
    #                  './data/mistral/qqp-acc-hf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-07'): {'color': '#2ca02c', 'legend_name': 'hf, 00-07', 'legend_order': 3},
    #                      ('hf', 'mean', '08-15'): {'color': '#d62728', 'legend_name': 'hf, 08-15', 'legend_order': 4},
    #                      ('hf', 'mean', '16-23'): {'color': '#8c564b', 'legend_name': 'hf, 16-23', 'legend_order': 5},
    #                      ('hf', 'mean', '24-31'): {'color': '#e377c2', 'legend_name': 'hf, 24-31', 'legend_order': 6},
    #                     #  ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/mistral/qqp.jsonlist',
    #                  './data/mistral/qqp-acc-cos.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-07'): {'color': '#2ca02c', 'legend_name': 'cos, 00-07', 'legend_order': 3},
    #                      ('cos', 'mean', '08-15'): {'color': '#d62728', 'legend_name': 'cos, 08-15', 'legend_order': 4},
    #                      ('cos', 'mean', '16-23'): {'color': '#8c564b', 'legend_name': 'cos, 16-23', 'legend_order': 5},
    #                      ('cos', 'mean', '24-31'): {'color': '#e377c2', 'legend_name': 'cos, 24-31', 'legend_order': 6},
    #                     #  ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                     #  ('cos', 'mean', ''): {'color': '#7f7f7f', 'legend_name': 'cos, total', 'legend_order': 7.5},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    

    # pass
    
    # draw_ft2_metric3('./data/mistral/qqp.jsonlist',
    #                  './data/mistral/qqp-acc-datainf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-07'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-07', 'legend_order': 3},
    #                      ('datainf', 'mean', '08-15'): {'color': '#d62728', 'legend_name': 'datainf, 08-15', 'legend_order': 4},
    #                      ('datainf', 'mean', '16-23'): {'color': '#8c564b', 'legend_name': 'datainf, 16-23', 'legend_order': 5},
    #                      ('datainf', 'mean', '24-31'): {'color': '#e377c2', 'legend_name': 'datainf, 24-31', 'legend_order': 6},
    #                     #  ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass     

    # draw_ft2_metric3('./data/mistral/qnli.jsonlist',
    #                  './data/mistral/qnli-acc-hf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-07'): {'color': '#2ca02c', 'legend_name': 'hf, 00-07', 'legend_order': 3},
    #                      ('hf', 'mean', '08-15'): {'color': '#d62728', 'legend_name': 'hf, 08-15', 'legend_order': 4},
    #                      ('hf', 'mean', '16-23'): {'color': '#8c564b', 'legend_name': 'hf, 16-23', 'legend_order': 5},
    #                      ('hf', 'mean', '24-31'): {'color': '#e377c2', 'legend_name': 'hf, 24-31', 'legend_order': 6},
    #                     #  ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/mistral/qnli.jsonlist',
    #                  './data/mistral/qnli-acc-cos.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-07'): {'color': '#2ca02c', 'legend_name': 'cos, 00-07', 'legend_order': 3},
    #                      ('cos', 'mean', '08-15'): {'color': '#d62728', 'legend_name': 'cos, 08-15', 'legend_order': 4},
    #                      ('cos', 'mean', '16-23'): {'color': '#8c564b', 'legend_name': 'cos, 16-23', 'legend_order': 5},
    #                      ('cos', 'mean', '24-31'): {'color': '#e377c2', 'legend_name': 'cos, 24-31', 'legend_order': 6},
    #                     #  ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/mistral/qnli.jsonlist',
    #                  './data/mistral/qnli-acc-datainf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-07'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-07', 'legend_order': 3},
    #                      ('datainf', 'mean', '08-15'): {'color': '#d62728', 'legend_name': 'datainf, 08-15', 'legend_order': 4},
    #                      ('datainf', 'mean', '16-23'): {'color': '#8c564b', 'legend_name': 'datainf, 16-23', 'legend_order': 5},
    #                      ('datainf', 'mean', '24-31'): {'color': '#e377c2', 'legend_name': 'datainf, 24-31', 'legend_order': 6},
    #                     #  ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass 

    # draw_ft2_metric3('./data/mistral/mrpc.jsonlist',
    #                  './data/mistral/mrpc-acc-hf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-07'): {'color': '#2ca02c', 'legend_name': 'hf, 00-07', 'legend_order': 3},
    #                      ('hf', 'mean', '08-15'): {'color': '#d62728', 'legend_name': 'hf, 08-15', 'legend_order': 4},
    #                      ('hf', 'mean', '16-23'): {'color': '#8c564b', 'legend_name': 'hf, 16-23', 'legend_order': 5},
    #                      ('hf', 'mean', '24-31'): {'color': '#e377c2', 'legend_name': 'hf, 24-31', 'legend_order': 6},
    #                     #  ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/mistral/mrpc.jsonlist',
    #                  './data/mistral/mrpc-acc-cos.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-07'): {'color': '#2ca02c', 'legend_name': 'cos, 00-07', 'legend_order': 3},
    #                      ('cos', 'mean', '08-15'): {'color': '#d62728', 'legend_name': 'cos, 08-15', 'legend_order': 4},
    #                      ('cos', 'mean', '16-23'): {'color': '#8c564b', 'legend_name': 'cos, 16-23', 'legend_order': 5},
    #                      ('cos', 'mean', '24-31'): {'color': '#e377c2', 'legend_name': 'cos, 24-31', 'legend_order': 6},
    #                     #  ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/mistral/mrpc.jsonlist',
    #                  './data/mistral/mrpc-acc-datainf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-07'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-07', 'legend_order': 3},
    #                      ('datainf', 'mean', '08-15'): {'color': '#d62728', 'legend_name': 'datainf, 08-15', 'legend_order': 4},
    #                      ('datainf', 'mean', '16-23'): {'color': '#8c564b', 'legend_name': 'datainf, 16-23', 'legend_order': 5},
    #                      ('datainf', 'mean', '24-31'): {'color': '#e377c2', 'legend_name': 'datainf, 24-31', 'legend_order': 6},
    #                     #  ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        
    
    # pass    
    

    # draw_ft2_metric3('./data/roberta-1/sst2.jsonlist',
    #                  './data/roberta-1/sst2-iacc-hf.pdf',  metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
    #                      ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
    #                      ('hf', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'hf, 12-17', 'legend_order': 5},
    #                      ('hf', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'hf, 18-23', 'legend_order': 6},
    #                     #  ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/roberta-1/sst2.jsonlist',
    #                  './data/roberta-1/sst2-iacc-cos.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'cos, 00-05', 'legend_order': 3},
    #                      ('cos', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'cos, 06-11', 'legend_order': 4},
    #                      ('cos', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'cos, 12-17', 'legend_order': 5},
    #                      ('cos', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'cos, 18-23', 'legend_order': 6},
    #                     #  ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/roberta-1/sst2.jsonlist',
    #                  './data/roberta-1/sst2-iacc-datainf.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-05', 'legend_order': 3},
    #                      ('datainf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'datainf, 06-11', 'legend_order': 4},
    #                      ('datainf', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'datainf, 12-17', 'legend_order': 5},
    #                      ('datainf', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'datainf, 18-23', 'legend_order': 6},
    #                     #  ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass     

    # draw_ft2_metric3('./data/roberta-1/qqp.jsonlist',
    #                  './data/roberta-1/qqp-iacc-hf.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
    #                      ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
    #                      ('hf', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'hf, 12-17', 'legend_order': 5},
    #                      ('hf', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'hf, 18-23', 'legend_order': 6},
    #                     #  ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/roberta-1/qqp.jsonlist',
    #                  './data/roberta-1/qqp-iacc-cos.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'cos, 00-05', 'legend_order': 3},
    #                      ('cos', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'cos, 06-11', 'legend_order': 4},
    #                      ('cos', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'cos, 12-17', 'legend_order': 5},
    #                      ('cos', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'cos, 18-23', 'legend_order': 6},
    #                     #  ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                     #  ('cos', 'mean', ''): {'color': '#7f7f7f', 'legend_name': 'cos, total', 'legend_order': 7.5},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    

    # pass
    
    # draw_ft2_metric3('./data/roberta-1/qqp.jsonlist',
    #                  './data/roberta-1/qqp-iacc-datainf.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-05', 'legend_order': 3},
    #                      ('datainf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'datainf, 06-11', 'legend_order': 4},
    #                      ('datainf', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'datainf, 12-17', 'legend_order': 5},
    #                      ('datainf', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'datainf, 18-23', 'legend_order': 6},
    #                     #  ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass     

    # draw_ft2_metric3('./data/roberta-1/qnli.jsonlist',
    #                  './data/roberta-1/qnli-iacc-hf.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
    #                      ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
    #                      ('hf', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'hf, 12-17', 'legend_order': 5},
    #                      ('hf', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'hf, 18-23', 'legend_order': 6},
    #                     #  ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/roberta-1/qnli.jsonlist',
    #                  './data/roberta-1/qnli-iacc-cos.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'cos, 00-05', 'legend_order': 3},
    #                      ('cos', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'cos, 06-11', 'legend_order': 4},
    #                      ('cos', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'cos, 12-17', 'legend_order': 5},
    #                      ('cos', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'cos, 18-23', 'legend_order': 6},
    #                     #  ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/roberta-1/qnli.jsonlist',
    #                  './data/roberta-1/qnli-iacc-datainf.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-05', 'legend_order': 3},
    #                      ('datainf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'datainf, 06-11', 'legend_order': 4},
    #                      ('datainf', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'datainf, 12-17', 'legend_order': 5},
    #                      ('datainf', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'datainf, 18-23', 'legend_order': 6},
    #                     #  ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass 

    # draw_ft2_metric3('./data/roberta-1/mrpc.jsonlist',
    #                  './data/roberta-1/mrpc-iacc-hf.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
    #                      ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
    #                      ('hf', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'hf, 12-17', 'legend_order': 5},
    #                      ('hf', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'hf, 18-23', 'legend_order': 6},
    #                     #  ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/roberta-1/mrpc.jsonlist',
    #                  './data/roberta-1/mrpc-iacc-cos.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'cos, 00-05', 'legend_order': 3},
    #                      ('cos', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'cos, 06-11', 'legend_order': 4},
    #                      ('cos', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'cos, 12-17', 'legend_order': 5},
    #                      ('cos', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'cos, 18-23', 'legend_order': 6},
    #                     #  ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/roberta-1/mrpc.jsonlist',
    #                  './data/roberta-1/mrpc-iacc-datainf.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-05', 'legend_order': 3},
    #                      ('datainf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'datainf, 06-11', 'legend_order': 4},
    #                      ('datainf', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'datainf, 12-17', 'legend_order': 5},
    #                      ('datainf', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'datainf, 18-23', 'legend_order': 6},
    #                     #  ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        
    
    # pass    

    # draw_ft2_metric3('./data/roberta-1/sst2.jsonlist',
    #                  './data/roberta-1/sst2-acc-hf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
    #                      ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
    #                      ('hf', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'hf, 12-17', 'legend_order': 5},
    #                      ('hf', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'hf, 18-23', 'legend_order': 6},
    #                     #  ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/roberta-1/sst2.jsonlist',
    #                  './data/roberta-1/sst2-acc-cos.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'cos, 00-05', 'legend_order': 3},
    #                      ('cos', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'cos, 06-11', 'legend_order': 4},
    #                      ('cos', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'cos, 12-17', 'legend_order': 5},
    #                      ('cos', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'cos, 18-23', 'legend_order': 6},
    #                     #  ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/roberta-1/sst2.jsonlist',
    #                  './data/roberta-1/sst2-acc-datainf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-05', 'legend_order': 3},
    #                      ('datainf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'datainf, 06-11', 'legend_order': 4},
    #                      ('datainf', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'datainf, 12-17', 'legend_order': 5},
    #                      ('datainf', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'datainf, 18-23', 'legend_order': 6},
    #                     #  ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass     

    # draw_ft2_metric3('./data/roberta-1/qqp.jsonlist',
    #                  './data/roberta-1/qqp-acc-hf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
    #                      ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
    #                      ('hf', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'hf, 12-17', 'legend_order': 5},
    #                      ('hf', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'hf, 18-23', 'legend_order': 6},
    #                     #  ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/roberta-1/qqp.jsonlist',
    #                  './data/roberta-1/qqp-acc-cos.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'cos, 00-05', 'legend_order': 3},
    #                      ('cos', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'cos, 06-11', 'legend_order': 4},
    #                      ('cos', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'cos, 12-17', 'legend_order': 5},
    #                      ('cos', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'cos, 18-23', 'legend_order': 6},
    #                     #  ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                     #  ('cos', 'mean', ''): {'color': '#7f7f7f', 'legend_name': 'cos, total', 'legend_order': 7.5},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    

    # pass
    
    # draw_ft2_metric3('./data/roberta-1/qqp.jsonlist',
    #                  './data/roberta-1/qqp-acc-datainf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-05', 'legend_order': 3},
    #                      ('datainf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'datainf, 06-11', 'legend_order': 4},
    #                      ('datainf', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'datainf, 12-17', 'legend_order': 5},
    #                      ('datainf', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'datainf, 18-23', 'legend_order': 6},
    #                     #  ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass     

    # draw_ft2_metric3('./data/roberta-1/qnli.jsonlist',
    #                  './data/roberta-1/qnli-acc-hf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
    #                      ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
    #                      ('hf', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'hf, 12-17', 'legend_order': 5},
    #                      ('hf', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'hf, 18-23', 'legend_order': 6},
    #                     #  ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/roberta-1/qnli.jsonlist',
    #                  './data/roberta-1/qnli-acc-cos.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'cos, 00-05', 'legend_order': 3},
    #                      ('cos', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'cos, 06-11', 'legend_order': 4},
    #                      ('cos', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'cos, 12-17', 'legend_order': 5},
    #                      ('cos', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'cos, 18-23', 'legend_order': 6},
    #                     #  ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/roberta-1/qnli.jsonlist',
    #                  './data/roberta-1/qnli-acc-datainf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-05', 'legend_order': 3},
    #                      ('datainf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'datainf, 06-11', 'legend_order': 4},
    #                      ('datainf', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'datainf, 12-17', 'legend_order': 5},
    #                      ('datainf', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'datainf, 18-23', 'legend_order': 6},
    #                     #  ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass 

    # draw_ft2_metric3('./data/roberta-1/mrpc.jsonlist',
    #                  './data/roberta-1/mrpc-acc-hf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
    #                      ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
    #                      ('hf', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'hf, 12-17', 'legend_order': 5},
    #                      ('hf', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'hf, 18-23', 'legend_order': 6},
    #                     #  ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/roberta-1/mrpc.jsonlist',
    #                  './data/roberta-1/mrpc-acc-cos.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'cos, 00-05', 'legend_order': 3},
    #                      ('cos', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'cos, 06-11', 'legend_order': 4},
    #                      ('cos', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'cos, 12-17', 'legend_order': 5},
    #                      ('cos', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'cos, 18-23', 'legend_order': 6},
    #                     #  ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/roberta-1/mrpc.jsonlist',
    #                  './data/roberta-1/mrpc-acc-datainf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-05', 'legend_order': 3},
    #                      ('datainf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'datainf, 06-11', 'legend_order': 4},
    #                      ('datainf', 'mean', '12-17'): {'color': '#8c564b', 'legend_name': 'datainf, 12-17', 'legend_order': 5},
    #                      ('datainf', 'mean', '18-23'): {'color': '#e377c2', 'legend_name': 'datainf, 18-23', 'legend_order': 6},
    #                     #  ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        
    
    # pass


    # draw_ft2_metric3('./data/llama/sst2.jsonlist',
    #                  './data/llama/sst2-accd-hf.pdf', metric="infl_accuracy", draw_diff=True,
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'hf, 00-03', 'legend_order': 3},
    #                      ('hf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'hf, 04-07', 'legend_order': 4},
    #                      ('hf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'hf, 08-11', 'legend_order': 5},
    #                      ('hf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'hf, 12-15', 'legend_order': 6},
    #                      ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/llama/sst2.jsonlist',
    #                  './data/llama/sst2-accd-cos.pdf', metric="infl_accuracy", draw_diff=True,
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'cos, 00-03', 'legend_order': 3},
    #                      ('cos', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'cos, 04-07', 'legend_order': 4},
    #                      ('cos', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'cos, 08-11', 'legend_order': 5},
    #                      ('cos', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'cos, 12-15', 'legend_order': 6},
    #                      ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/llama/sst2.jsonlist',
    #                  './data/llama/sst2-accd-datainf.pdf', metric="infl_accuracy", draw_diff=True,
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-03', 'legend_order': 3},
    #                      ('datainf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'datainf, 04-07', 'legend_order': 4},
    #                      ('datainf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'datainf, 08-11', 'legend_order': 5},
    #                      ('datainf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'datainf, 12-15', 'legend_order': 6},
    #                      ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass     

    # draw_ft2_metric3('./data/llama/qqp.jsonlist',
    #                  './data/llama/qqp-accd-hf.pdf', metric="infl_accuracy", draw_diff=True,
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'hf, 00-03', 'legend_order': 3},
    #                      ('hf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'hf, 04-07', 'legend_order': 4},
    #                      ('hf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'hf, 08-11', 'legend_order': 5},
    #                      ('hf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'hf, 12-15', 'legend_order': 6},
    #                      ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/llama/qqp.jsonlist',
    #                  './data/llama/qqp-accd-cos.pdf', metric="infl_accuracy", draw_diff=True,
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'cos, 00-03', 'legend_order': 3},
    #                      ('cos', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'cos, 04-07', 'legend_order': 4},
    #                      ('cos', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'cos, 08-11', 'legend_order': 5},
    #                      ('cos', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'cos, 12-15', 'legend_order': 6},
    #                      ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('cos', 'mean', ''): {'color': '#7f7f7f', 'legend_name': 'cos, total', 'legend_order': 7.5},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    

    # pass
    
    # draw_ft2_metric3('./data/llama/qqp.jsonlist',
    #                  './data/llama/qqp-accd-datainf.pdf', metric="infl_accuracy", draw_diff=True,
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-03', 'legend_order': 3},
    #                      ('datainf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'datainf, 04-07', 'legend_order': 4},
    #                      ('datainf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'datainf, 08-11', 'legend_order': 5},
    #                      ('datainf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'datainf, 12-15', 'legend_order': 6},
    #                      ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass     

    # draw_ft2_metric3('./data/llama/qnli.jsonlist',
    #                  './data/llama/qnli-accd-hf.pdf', metric="infl_accuracy", draw_diff=True,
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'hf, 00-03', 'legend_order': 3},
    #                      ('hf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'hf, 04-07', 'legend_order': 4},
    #                      ('hf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'hf, 08-11', 'legend_order': 5},
    #                      ('hf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'hf, 12-15', 'legend_order': 6},
    #                      ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/llama/qnli.jsonlist',
    #                  './data/llama/qnli-accd-cos.pdf', metric="infl_accuracy", draw_diff=True,
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'cos, 00-03', 'legend_order': 3},
    #                      ('cos', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'cos, 04-07', 'legend_order': 4},
    #                      ('cos', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'cos, 08-11', 'legend_order': 5},
    #                      ('cos', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'cos, 12-15', 'legend_order': 6},
    #                      ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/llama/qnli.jsonlist',
    #                  './data/llama/qnli-accd-datainf.pdf', metric="infl_accuracy", draw_diff=True,
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-03', 'legend_order': 3},
    #                      ('datainf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'datainf, 04-07', 'legend_order': 4},
    #                      ('datainf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'datainf, 08-11', 'legend_order': 5},
    #                      ('datainf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'datainf, 12-15', 'legend_order': 6},
    #                      ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass 

    # draw_ft2_metric3('./data/llama/mrpc.jsonlist', 
    #                  './data/llama/mrpc-accd-hf.pdf', metric="infl_accuracy", draw_diff=True,
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'hf, 00-03', 'legend_order': 3},
    #                      ('hf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'hf, 04-07', 'legend_order': 4},
    #                      ('hf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'hf, 08-11', 'legend_order': 5},
    #                      ('hf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'hf, 12-15', 'legend_order': 6},
    #                      ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/llama/mrpc.jsonlist',
    #                  './data/llama/mrpc-accd-cos.pdf', metric="infl_accuracy", draw_diff=True,
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'cos, 00-03', 'legend_order': 3},
    #                      ('cos', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'cos, 04-07', 'legend_order': 4},
    #                      ('cos', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'cos, 08-11', 'legend_order': 5},
    #                      ('cos', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'cos, 12-15', 'legend_order': 6},
    #                      ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/llama/mrpc.jsonlist',
    #                  './data/llama/mrpc-accd-datainf.pdf', metric="infl_accuracy", draw_diff=True,
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-03', 'legend_order': 3},
    #                      ('datainf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'datainf, 04-07', 'legend_order': 4},
    #                      ('datainf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'datainf, 08-11', 'legend_order': 5},
    #                      ('datainf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'datainf, 12-15', 'legend_order': 6},
    #                      ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        
    
    # pass


    # draw_ft2_metric3('./data/llama/sst2.jsonlist',
    #                  './data/llama/sst2-iacc-hf.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'hf, 00-03', 'legend_order': 3},
    #                      ('hf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'hf, 04-07', 'legend_order': 4},
    #                      ('hf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'hf, 08-11', 'legend_order': 5},
    #                      ('hf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'hf, 12-15', 'legend_order': 6},
    #                      ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/llama/sst2.jsonlist',
    #                  './data/llama/sst2-iacc-cos.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'cos, 00-03', 'legend_order': 3},
    #                      ('cos', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'cos, 04-07', 'legend_order': 4},
    #                      ('cos', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'cos, 08-11', 'legend_order': 5},
    #                      ('cos', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'cos, 12-15', 'legend_order': 6},
    #                      ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/llama/sst2.jsonlist',
    #                  './data/llama/sst2-iacc-datainf.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-03', 'legend_order': 3},
    #                      ('datainf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'datainf, 04-07', 'legend_order': 4},
    #                      ('datainf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'datainf, 08-11', 'legend_order': 5},
    #                      ('datainf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'datainf, 12-15', 'legend_order': 6},
    #                      ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass     

    # draw_ft2_metric3('./data/llama/qqp.jsonlist',
    #                  './data/llama/qqp-iacc-hf.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'hf, 00-03', 'legend_order': 3},
    #                      ('hf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'hf, 04-07', 'legend_order': 4},
    #                      ('hf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'hf, 08-11', 'legend_order': 5},
    #                      ('hf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'hf, 12-15', 'legend_order': 6},
    #                      ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/llama/qqp.jsonlist',
    #                  './data/llama/qqp-iacc-cos.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'cos, 00-03', 'legend_order': 3},
    #                      ('cos', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'cos, 04-07', 'legend_order': 4},
    #                      ('cos', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'cos, 08-11', 'legend_order': 5},
    #                      ('cos', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'cos, 12-15', 'legend_order': 6},
    #                      ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('cos', 'mean', ''): {'color': '#7f7f7f', 'legend_name': 'cos, total', 'legend_order': 7.5},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    

    # pass
    
    # draw_ft2_metric3('./data/llama/qqp.jsonlist',
    #                  './data/llama/qqp-iacc-datainf.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-03', 'legend_order': 3},
    #                      ('datainf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'datainf, 04-07', 'legend_order': 4},
    #                      ('datainf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'datainf, 08-11', 'legend_order': 5},
    #                      ('datainf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'datainf, 12-15', 'legend_order': 6},
    #                      ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass     

    # draw_ft2_metric3('./data/llama/qnli.jsonlist',
    #                  './data/llama/qnli-iacc-hf.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'hf, 00-03', 'legend_order': 3},
    #                      ('hf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'hf, 04-07', 'legend_order': 4},
    #                      ('hf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'hf, 08-11', 'legend_order': 5},
    #                      ('hf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'hf, 12-15', 'legend_order': 6},
    #                      ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/llama/qnli.jsonlist',
    #                  './data/llama/qnli-iacc-cos.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'cos, 00-03', 'legend_order': 3},
    #                      ('cos', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'cos, 04-07', 'legend_order': 4},
    #                      ('cos', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'cos, 08-11', 'legend_order': 5},
    #                      ('cos', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'cos, 12-15', 'legend_order': 6},
    #                      ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/llama/qnli.jsonlist',
    #                  './data/llama/qnli-iacc-datainf.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-03', 'legend_order': 3},
    #                      ('datainf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'datainf, 04-07', 'legend_order': 4},
    #                      ('datainf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'datainf, 08-11', 'legend_order': 5},
    #                      ('datainf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'datainf, 12-15', 'legend_order': 6},
    #                      ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass 

    # draw_ft2_metric3('./data/llama/mrpc.jsonlist',
    #                  './data/llama/mrpc-iacc-hf.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'hf, 00-03', 'legend_order': 3},
    #                      ('hf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'hf, 04-07', 'legend_order': 4},
    #                      ('hf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'hf, 08-11', 'legend_order': 5},
    #                      ('hf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'hf, 12-15', 'legend_order': 6},
    #                      ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/llama/mrpc.jsonlist',
    #                  './data/llama/mrpc-iacc-cos.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'cos, 00-03', 'legend_order': 3},
    #                      ('cos', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'cos, 04-07', 'legend_order': 4},
    #                      ('cos', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'cos, 08-11', 'legend_order': 5},
    #                      ('cos', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'cos, 12-15', 'legend_order': 6},
    #                      ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/llama/mrpc.jsonlist',
    #                  './data/llama/mrpc-iacc-datainf.pdf', metric="infl_accuracy",
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-03', 'legend_order': 3},
    #                      ('datainf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'datainf, 04-07', 'legend_order': 4},
    #                      ('datainf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'datainf, 08-11', 'legend_order': 5},
    #                      ('datainf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'datainf, 12-15', 'legend_order': 6},
    #                      ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        
    
    # pass


    # base_path = './data/llama'
    # tasks = ['qnli', 'mrpc', 'sst2', 'qqp']

    # module_groups_regex = { "WE": ".*\\.embed_tokens\\..*",     
    #                         "00-03": ".*\\.layers\\.([0-3])\\..*\\.lora_(A|B)\\..*",
    #                         "04-07": ".*\\.layers\\.([4-7])\\..*\\.lora_(A|B)\\..*",
    #                         "08-11": ".*\\.layers\\.([8-9]|1[0-1])\\..*\\.lora_(A|B)\\..*",
    #                         "12-15": ".*\\.layers\\.(1[2-5])\\..*\\.lora_(A|B)\\..*",

    #                         "00-03 A": ".*\\.layers\\.([0-3])\\..*\\.lora_A\\..*",
    #                         "04-07 A": ".*\\.layers\\.([4-7])\\..*\\.lora_A\\..*",
    #                         "08-11 A": ".*\\.layers\\.([8-9]|1[0-1])\\..*\\.lora_A\\..*",
    #                         "12-15 A": ".*\\.layers\\.(1[2-5])\\..*\\.lora_A\\..*",
                            
    #                         "00-03 B": ".*\\.layers\\.([0-3])\\..*\\.lora_B\\..*",
    #                         "04-07 B": ".*\\.layers\\.([4-7])\\..*\\.lora_B\\..*",
    #                         "08-11 B": ".*\\.layers\\.([8-9]|1[0-1])\\..*\\.lora_B\\..*",
    #                         "12-15 B": ".*\\.layers\\.(1[2-5])\\..*\\.lora_B\\..*",
    #                         "CL": ".*\\.score\\..*"
    #                      }

    # infl_methods = [
    #     'hf',
    #     'cos',
    #     'datainf',
    #     'hf_we_', 
    #     'hf_we_topk_10',
    # ]
    # agg_methods = {
    #     "rank": rank_matrix_score, 
    #     "mean": mean_matrix_score, 
    #     "mean_10": partial(mean_matrix_score, trim_ratio=0.1),
    #     "mean_50": partial(mean_matrix_score, trim_ratio=0.5),
    #     "dir": dir_matrix_score,
    # }
    # for task in tasks:
    #     df = compute_ndr_metrics_table(base_path, task=task,
    #                                     module_groups_regex = module_groups_regex,
    #                                     agg_methods=agg_methods,
    #                                     infl_methods = infl_methods)
    #     print(df)
    #     pass 
    #     output_table(df, base_path, task)
    #     pass

    # pass
        

    # draw_ft2_metric3('./data/llama/sst2.jsonlist',
    #                  './data/llama/sst2-acc-hf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'hf, 00-03', 'legend_order': 3},
    #                      ('hf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'hf, 04-07', 'legend_order': 4},
    #                      ('hf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'hf, 08-11', 'legend_order': 5},
    #                      ('hf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'hf, 12-15', 'legend_order': 6},
    #                      ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/llama/sst2.jsonlist',
    #                  './data/llama/sst2-acc-cos.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'cos, 00-03', 'legend_order': 3},
    #                      ('cos', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'cos, 04-07', 'legend_order': 4},
    #                      ('cos', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'cos, 08-11', 'legend_order': 5},
    #                      ('cos', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'cos, 12-15', 'legend_order': 6},
    #                      ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/llama/sst2.jsonlist',
    #                  './data/llama/sst2-acc-datainf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-03', 'legend_order': 3},
    #                      ('datainf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'datainf, 04-07', 'legend_order': 4},
    #                      ('datainf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'datainf, 08-11', 'legend_order': 5},
    #                      ('datainf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'datainf, 12-15', 'legend_order': 6},
    #                      ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass     

    # draw_ft2_metric3('./data/llama/qqp.jsonlist',
    #                  './data/llama/qqp-acc-hf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'hf, 00-03', 'legend_order': 3},
    #                      ('hf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'hf, 04-07', 'legend_order': 4},
    #                      ('hf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'hf, 08-11', 'legend_order': 5},
    #                      ('hf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'hf, 12-15', 'legend_order': 6},
    #                      ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/llama/qqp.jsonlist',
    #                  './data/llama/qqp-acc-cos.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'cos, 00-03', 'legend_order': 3},
    #                      ('cos', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'cos, 04-07', 'legend_order': 4},
    #                      ('cos', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'cos, 08-11', 'legend_order': 5},
    #                      ('cos', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'cos, 12-15', 'legend_order': 6},
    #                      ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('cos', 'mean', ''): {'color': '#7f7f7f', 'legend_name': 'cos, total', 'legend_order': 7.5},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    

    # pass
    
    # draw_ft2_metric3('./data/llama/qqp.jsonlist',
    #                  './data/llama/qqp-acc-datainf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-03', 'legend_order': 3},
    #                      ('datainf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'datainf, 04-07', 'legend_order': 4},
    #                      ('datainf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'datainf, 08-11', 'legend_order': 5},
    #                      ('datainf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'datainf, 12-15', 'legend_order': 6},
    #                      ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass     

    # draw_ft2_metric3('./data/llama/qnli.jsonlist',
    #                  './data/llama/qnli-acc-hf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'hf, 00-03', 'legend_order': 3},
    #                      ('hf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'hf, 04-07', 'legend_order': 4},
    #                      ('hf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'hf, 08-11', 'legend_order': 5},
    #                      ('hf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'hf, 12-15', 'legend_order': 6},
    #                      ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/llama/qnli.jsonlist',
    #                  './data/llama/qnli-acc-cos.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'cos, 00-03', 'legend_order': 3},
    #                      ('cos', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'cos, 04-07', 'legend_order': 4},
    #                      ('cos', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'cos, 08-11', 'legend_order': 5},
    #                      ('cos', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'cos, 12-15', 'legend_order': 6},
    #                      ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/llama/qnli.jsonlist',
    #                  './data/llama/qnli-acc-datainf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-03', 'legend_order': 3},
    #                      ('datainf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'datainf, 04-07', 'legend_order': 4},
    #                      ('datainf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'datainf, 08-11', 'legend_order': 5},
    #                      ('datainf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'datainf, 12-15', 'legend_order': 6},
    #                      ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        

    # pass 

    # draw_ft2_metric3('./data/llama/mrpc.jsonlist',
    #                  './data/llama/mrpc-acc-hf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                      ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                      ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('hf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'hf, WE', 'legend_order': 2},
    #                      ('hf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'hf, 00-03', 'legend_order': 3},
    #                      ('hf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'hf, 04-07', 'legend_order': 4},
    #                      ('hf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'hf, 08-11', 'legend_order': 5},
    #                      ('hf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'hf, 12-15', 'legend_order': 6},
    #                      ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })

    # draw_ft2_metric3('./data/llama/mrpc.jsonlist',
    #                  './data/llama/mrpc-acc-cos.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('cos', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'cos, WE', 'legend_order': 2},
    #                      ('cos', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'cos, 00-03', 'legend_order': 3},
    #                      ('cos', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'cos, 04-07', 'legend_order': 4},
    #                      ('cos', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'cos, 08-11', 'legend_order': 5},
    #                      ('cos', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'cos, 12-15', 'legend_order': 6},
    #                      ('cos', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'cos, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })    
    
    # draw_ft2_metric3('./data/llama/mrpc.jsonlist',
    #                  './data/llama/mrpc-acc-datainf.pdf',
    #                  method_dict={
    #                      ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': -1},
    #                     #  ('hf_we_', 'mean', 'WE'): {'color': '#1f77b4', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #                     #  ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$_{we}^{10}$', 'legend_order': 1},
    #                      ('datainf', 'mean', 'WE'): {'color': '#ff7f0e', 'legend_name': 'datainf, WE', 'legend_order': 2},
    #                      ('datainf', 'mean', '00-03'): {'color': '#2ca02c', 'legend_name': 'datainf, 00-03', 'legend_order': 3},
    #                      ('datainf', 'mean', '04-07'): {'color': '#d62728', 'legend_name': 'datainf, 04-07', 'legend_order': 4},
    #                      ('datainf', 'mean', '08-11'): {'color': '#8c564b', 'legend_name': 'datainf, 08-11', 'legend_order': 5},
    #                      ('datainf', 'mean', '12-15'): {'color': '#e377c2', 'legend_name': 'datainf, 12-15', 'legend_order': 6},
    #                      ('datainf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'datainf, CL', 'legend_order': 7},
    #                      ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': 8},
    #                  })        
    
    # pass

    # compute_noise_detection_metrics_per_sample('./data/roberta/qnli', 
    #                                 plot_title = "QNLI on Roberta-large with WE",
    #                                 cache_file="./data/roberta/qnli/postprocess-per-sample.pt",
    #                                 out_chart_file="./data/roberta/postprocess/qnli/best_modules_per_sample.pdf")

    # compute_noise_detection_metrics_per_sample('./data/roberta/mrpc', 
    #                                 plot_title = "MRPC on Roberta-large with WE",
    #                                 cache_file="./data/roberta/mrpc/postprocess-per-sample.pt",
    #                                 out_chart_file="./data/roberta/postprocess/mrpc/best_modules_per_sample.pdf")

    # compute_noise_detection_metrics_per_sample('./data/roberta/sst2', 
    #                                 plot_title = "SST2 on Roberta-large with WE",
    #                                 cache_file="./data/roberta/sst2/postprocess-per-sample.pt",
    #                                 out_chart_file="./data/roberta/postprocess/sst2/best_modules_per_sample.pdf")

    # compute_noise_detection_metrics_per_sample('./data/roberta/qqp', 
    #                                 plot_title = "QQP on Roberta-large with WE",
    #                                 cache_file="./data/roberta/qqp/postprocess-per-sample.pt",
    #                                 out_chart_file="./data/roberta/postprocess/qqp/best_modules_per_sample.pdf")


    # base_path = './data/roberta'
    # tasks = ['qnli', 'mrpc', 'sst2', 'qqp']

    # module_groups_regex = { "WE": ".*\\.word_embeddings\\..*",
                            
    #                         "00-05": ".*\\.layer\\.([0-5])\\..*\\.lora_(A|B)\\..*",
    #                         "06-11": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_(A|B)\\..*",
    #                         "12-17": ".*\\.layer\\.(1[2-7])\\..*\\.lora_(A|B)\\..*",
    #                         "18-23": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_(A|B)\\..*",

    #                         "00-05 A": ".*\\.layer\\.([0-5])\\..*\\.lora_A\\..*",
    #                         "06-11 A": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_A\\..*",
    #                         "12-17 A": ".*\\.layer\\.(1[2-7])\\..*\\.lora_A\\..*",
    #                         "18-23 A": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_A\\..*",

    #                         "00-05 B": ".*\\.layer\\.([0-5])\\..*\\.lora_B\\..*",
    #                         "06-11 B": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_B\\..*",
    #                         "12-17 B": ".*\\.layer\\.(1[2-7])\\..*\\.lora_B\\..*",
    #                         "18-23 B": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_B\\..*",                            

    #                         "CL": ".*\\.classifier\\..*",
    #                      }

    # infl_methods = [
    #     'hf',
    #     'cos',
    #     'datainf',
    #     'hf_we_', 
    #     'hf_we_topk_10',
    # ]
    # agg_methods = {
    #     "rank": rank_matrix_score, 
    #     "mean": mean_matrix_score, 
    #     "mean_10": partial(mean_matrix_score, trim_ratio=0.1),
    #     "mean_50": partial(mean_matrix_score, trim_ratio=0.5),
    #     "dir": dir_matrix_score,
    # }
    # for task in tasks:
    #     df = compute_ndr_metrics_table(base_path, task=task,
    #                                     module_groups_regex = module_groups_regex,
    #                                     agg_methods=agg_methods,
    #                                     infl_methods = infl_methods)
    #     print(df)
    #     pass 
    #     output_table(df, base_path, task)
    #     pass


    # infl_methods = ["hf", "hf_we_", "hf_we_topk_10"]
    # infl_methods = ["cos"]
    # module_groups_regex = { "WE": ".*\\.word_embeddings\\..*",
                            
    #                         "00-05": ".*\\.layer\\.([0-5])\\..*\\.lora_(A|B)\\..*",
    #                         "06-11": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_(A|B)\\..*",
    #                         "12-17": ".*\\.layer\\.(1[2-7])\\..*\\.lora_(A|B)\\..*",
    #                         "18-23": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_(A|B)\\..*",

    #                         # "00-05 A": ".*\\.layer\\.([0-5])\\..*\\.lora_A\\..*",
    #                         # "06-11 A": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_A\\..*",
    #                         # "12-17 A": ".*\\.layer\\.(1[2-7])\\..*\\.lora_A\\..*",
    #                         # "18-23 A": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_A\\..*",

    #                         # "00-05 B": ".*\\.layer\\.([0-5])\\..*\\.lora_B\\..*",
    #                         # "06-11 B": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_B\\..*",
    #                         # "12-17 B": ".*\\.layer\\.(1[2-7])\\..*\\.lora_B\\..*",
    #                         # "18-23 B": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_B\\..*",                            

    #                         "CL": ".*\\.classifier\\..*",
    #                      }
    
    # colors = {
    #     # "hf_we_, WE": '#1f77b4',
    #     "cos, WE": "#ff7f0e",
    #     "cos, 00-05": "#2ca02c",
    #     "cos, 18-23": "#d62728",
    #     "cos, CL": "#9467bd",
    #     "cos, 06-11": "#8c564b",
    #     "cos, 12-17": "#e377c2",
    #     # "hf_we_topk_10, WE": '#7f7f7f',
    #     # '#bcbd22', '#17becf'
    # }

    # legend_order = {
    #     # "hf_we_, WE": 0,
    #     # "hf_we_topk_10, WE": 1,
    #     "cos, WE": 2,
    #     "cos, 00-05": 3,
    #     "cos, 06-11": 4,
    #     "cos, 12-17": 5,
    #     "cos, 18-23": 6,
    #     "cos, CL": 7
    # }
    # legend_names = {
    #     "hf_we_, WE": "hf$_{we}$",
    #     "hf_we_topk_10, WE": "hf$_{we}^{10}$"
    # }
    # module_groups_regex_rev = {v:k for k,v in module_groups_regex.items()}
    # tasks = ['mrpc', 'qnli', 'sst2', 'qqp']
    # for d in tasks:
    #     draw_ft2_metric2(d, infile = f'./data/roberta/{d}/metrics.jsonlist', outfile = f'./data/roberta/{d}/postprocess/T-acc-cos-layers.pdf',
    #                     infl_methods = infl_methods, metric = 'accuracy', module_pattern_to_name = module_groups_regex_rev,
    #                     colors=colors, legend_order = legend_order, legend_names = legend_names)

    # pass

    # infl_methods = ["hf", "hf_we_", "hf_we_topk_10"]
    # module_groups_regex = { "WE": ".*\\.word_embeddings\\..*",
                            
    #                         # "00-05": ".*\\.layer\\.([0-5])\\..*\\.lora_(A|B)\\..*",
    #                         # "06-11": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_(A|B)\\..*",
    #                         # "12-17": ".*\\.layer\\.(1[2-7])\\..*\\.lora_(A|B)\\..*",
    #                         # "18-23": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_(A|B)\\..*",

    #                         # "00-05 A": ".*\\.layer\\.([0-5])\\..*\\.lora_A\\..*",
    #                         # "06-11 A": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_A\\..*",
    #                         "12-17 A": ".*\\.layer\\.(1[2-7])\\..*\\.lora_A\\..*",
    #                         "18-23 A": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_A\\..*",

    #                         # "00-05 B": ".*\\.layer\\.([0-5])\\..*\\.lora_B\\..*",
    #                         # "06-11 B": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_B\\..*",
    #                         "12-17 B": ".*\\.layer\\.(1[2-7])\\..*\\.lora_B\\..*",
    #                         "18-23 B": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_B\\..*",                            

    #                         "CL": ".*\\.classifier\\..*",
    #                      }
    
    # colors = {
    #     "hf_we_, WE": '#1f77b4',
    #     "hf_we_topk_10, WE": '#7f7f7f',
    #     "hf, WE": "#ff7f0e",
    #     "hf, 12-17 A": "#2ca02c",
    #     "hf, 12-17 B": "#d62728",
    #     "hf, 18-23 A": "#8c564b",
    #     "hf, 18-23 B": "#e377c2",
    #     "hf, CL": "#9467bd",
    #     # '#bcbd22', '#17becf'
    # }

    # legend_order = {
    #     "hf_we_, WE": 0,
    #     "hf_we_topk_10, WE": 1,
    #     "hf, WE": 2,
    #     "hf, 12-17 A": 3,
    #     "hf, 12-17 B": 4,
    #     "hf, 18-23 A": 5,
    #     "hf, 18-23 B": 6,
    #     "hf, CL": 7,
    # }
    # legend_names = {
    #     "hf_we_, WE": "hf$_{we}$",
    #     "hf_we_topk_10, WE": "hf$_{we}^{10}$"
    # }
    # module_groups_regex_rev = {v:k for k,v in module_groups_regex.items()}
    # for d in tasks:
    #     draw_ft2_metric2(d, infile = f'./data/roberta/{d}/metrics.jsonlist', outfile = f'./data/roberta/{d}/postprocess/T-acc-hf-AB.pdf',
    #                     infl_methods = infl_methods, metric = 'accuracy', module_pattern_to_name = module_groups_regex_rev,
    #                     colors=colors, legend_order = legend_order, legend_names = legend_names)


    # infl_vs_module_filter = [("datainf", "12-17 A"), ("hf", "12-17"), ("cos", "18-23 B"), ("hf_we_", "WE"), ("hf_we_topk_10", "WE")]
    # module_groups_regex = { "WE": ".*\\.word_embeddings\\..*",
                            
    #                         "00-05": ".*\\.layer\\.([0-5])\\..*\\.lora_(A|B)\\..*",
    #                         "06-11": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_(A|B)\\..*",
    #                         "12-17": ".*\\.layer\\.(1[2-7])\\..*\\.lora_(A|B)\\..*",
    #                         "18-23": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_(A|B)\\..*",

    #                         "00-05 A": ".*\\.layer\\.([0-5])\\..*\\.lora_A\\..*",
    #                         "06-11 A": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_A\\..*",
    #                         "12-17 A": ".*\\.layer\\.(1[2-7])\\..*\\.lora_A\\..*",
    #                         "18-23 A": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_A\\..*",

    #                         "00-05 B": ".*\\.layer\\.([0-5])\\..*\\.lora_B\\..*",
    #                         "06-11 B": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_B\\..*",
    #                         "12-17 B": ".*\\.layer\\.(1[2-7])\\..*\\.lora_B\\..*",
    #                         "18-23 B": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_B\\..*",                            

    #                         "CL": ".*\\.classifier\\..*",
    #                      }
    
    # colors = {
    #     "hf_we_, WE": '#1f77b4',
    #     "hf_we_topk_10, WE": '#7f7f7f',
    #     # "hf, WE": "#ff7f0e",
    #     "hf, 12-17": "#2ca02c",
    #     "cos, 18-23 B": "#ff7f0e",
    #     # "hf, 12-17 B": "#d62728",
    #     # "hf, 18-23 A": "#8c564b",
    #     # "hf, 18-23 B": "#e377c2",
    #     # "hf, CL": "#9467bd",
    #     "datainf, 12-17 A":"#9467bd",
    #     # '#bcbd22', '#17becf'
    # }

    # legend_order = {
    #     "hf_we_, WE": 0,
    #     "hf_we_topk_10, WE": 1,
    #     "datainf, 12-17 A": 2,
    #     "hf, 12-17": 3,
    #     "cos, 18-23 B": 4,
    # }
    # legend_names = {
    #     "hf_we_, WE": "hf$_{we}$",
    #     "hf_we_topk_10, WE": "hf$_{we}^{10}$"
    # }
    # tasks = ['mrpc']
    # module_groups_regex_rev = {v:k for k,v in module_groups_regex.items()}
    # for d in tasks:
    #     draw_ft2_metric2(d, infile = f'./data/roberta/{d}/metrics.jsonlist', outfile = f'./data/roberta/{d}/postprocess/T-acc-hf-top.pdf',
    #                     metric = 'accuracy', module_pattern_to_name = module_groups_regex_rev,
    #                     colors=colors, legend_order = legend_order, legend_names = legend_names,
    #                     infl_vs_module_filter = infl_vs_module_filter)
        
    # pass

    # infl_vs_module_filter = [("cos", ""), ("datainf", "18-23 B"), ("hf", "18-23 B"), ("hf_we_", "WE"), ("hf_we_topk_10", "WE")]
    # module_groups_regex = { "WE": ".*\\.word_embeddings\\..*",
                            
    #                         "00-05": ".*\\.layer\\.([0-5])\\..*\\.lora_(A|B)\\..*",
    #                         "06-11": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_(A|B)\\..*",
    #                         "12-17": ".*\\.layer\\.(1[2-7])\\..*\\.lora_(A|B)\\..*",
    #                         "18-23": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_(A|B)\\..*",

    #                         "00-05 A": ".*\\.layer\\.([0-5])\\..*\\.lora_A\\..*",
    #                         "06-11 A": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_A\\..*",
    #                         "12-17 A": ".*\\.layer\\.(1[2-7])\\..*\\.lora_A\\..*",
    #                         "18-23 A": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_A\\..*",

    #                         "00-05 B": ".*\\.layer\\.([0-5])\\..*\\.lora_B\\..*",
    #                         "06-11 B": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_B\\..*",
    #                         "12-17 B": ".*\\.layer\\.(1[2-7])\\..*\\.lora_B\\..*",
    #                         "18-23 B": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_B\\..*",                            

    #                         "CL": ".*\\.classifier\\..*",
    #                      }
    
    # colors = {
    #     "hf_we_, WE": '#1f77b4',
    #     "hf_we_topk_10, WE": '#7f7f7f',
    #     "hf, 18-23 B": "#2ca02c",
    #     "datainf, 18-23 B":"#9467bd",
    #     "cos": "#ff7f0e",
    #     # "hf, 12-17 B": "#d62728",
    #     # "hf, 18-23 A": "#8c564b",
    #     # "hf, 18-23 B": "#e377c2",
    #     # "hf, CL": "#9467bd",
    #     # '#bcbd22', '#17becf'
    # }

    # legend_order = {
    #     "hf_we_, WE": 0,
    #     "hf_we_topk_10, WE": 1,
    #     "hf, 18-23 B": 2,
    #     "datainf, 18-23 B": 3,
    #     "cos": 4,
    # }
    # legend_names = {
    #     "hf_we_, WE": "hf$_{we}$",
    #     "hf_we_topk_10, WE": "hf$_{we}^{10}$"
    # }
    # tasks = ['qnli']
    # module_groups_regex_rev = {v:k for k,v in module_groups_regex.items()}
    # for d in tasks:
    #     draw_ft2_metric2(d, infile = f'./data/roberta/{d}/metrics.jsonlist', outfile = f'./data/roberta/{d}/postprocess/T-acc-hf-top.pdf',
    #                     metric = 'accuracy', module_pattern_to_name = module_groups_regex_rev,
    #                     colors=colors, legend_order = legend_order, legend_names = legend_names,
    #                     infl_vs_module_filter = infl_vs_module_filter)
        
    # pass    

    # infl_vs_module_filter = [("datainf", "18-23 A"), ("cos", "18-23 B"), ("hf", "CL"), ("hf_we_", "WE"), ("hf_we_topk_10", "WE")]
    # module_groups_regex = { "WE": ".*\\.word_embeddings\\..*",
                            
    #                         "00-05": ".*\\.layer\\.([0-5])\\..*\\.lora_(A|B)\\..*",
    #                         "06-11": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_(A|B)\\..*",
    #                         "12-17": ".*\\.layer\\.(1[2-7])\\..*\\.lora_(A|B)\\..*",
    #                         "18-23": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_(A|B)\\..*",

    #                         "00-05 A": ".*\\.layer\\.([0-5])\\..*\\.lora_A\\..*",
    #                         "06-11 A": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_A\\..*",
    #                         "12-17 A": ".*\\.layer\\.(1[2-7])\\..*\\.lora_A\\..*",
    #                         "18-23 A": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_A\\..*",

    #                         "00-05 B": ".*\\.layer\\.([0-5])\\..*\\.lora_B\\..*",
    #                         "06-11 B": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_B\\..*",
    #                         "12-17 B": ".*\\.layer\\.(1[2-7])\\..*\\.lora_B\\..*",
    #                         "18-23 B": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_B\\..*",                            

    #                         "CL": ".*\\.classifier\\..*",
    #                      }
    
    # colors = {
    #     "hf_we_, WE": '#1f77b4',
    #     "hf_we_topk_10, WE": '#7f7f7f',
    #     # "hf, WE": "#ff7f0e",
    #     "hf, CL": "#2ca02c",
    #     "cos, 18-23 B": "#ff7f0e",
    #     # "hf, 12-17 B": "#d62728",
    #     # "hf, 18-23 A": "#8c564b",
    #     # "hf, 18-23 B": "#e377c2",
    #     # "hf, CL": "#9467bd",
    #     "datainf, 18-23 A":"#9467bd",
    #     # '#bcbd22', '#17becf'
    # }

    # legend_order = {
    #     "hf_we_, WE": 0,
    #     "hf_we_topk_10, WE": 1,
    #     "datainf, 18-23 A": 2,
    #     "cos, 18-23 B": 3,
    #     "hf, CL": 4,
    # }
    # legend_names = {
    #     "hf_we_, WE": "hf$_{we}$",
    #     "hf_we_topk_10, WE": "hf$_{we}^{10}$"
    # }
    # tasks = ['qqp']
    # module_groups_regex_rev = {v:k for k,v in module_groups_regex.items()}
    # for d in tasks:
    #     draw_ft2_metric2(d, infile = f'./data/roberta/{d}/metrics.jsonlist', outfile = f'./data/roberta/{d}/postprocess/T-acc-hf-top.pdf',
    #                     metric = 'accuracy', module_pattern_to_name = module_groups_regex_rev,
    #                     colors=colors, legend_order = legend_order, legend_names = legend_names,
    #                     infl_vs_module_filter = infl_vs_module_filter)
        
    # pass    

    # infl_vs_module_filter = [("cos", "18-23 B"), ("datainf", "18-23 B"), ("hf", "18-23 B"), ("hf_we_", "WE"), ("hf_we_topk_10", "WE")]
    # module_groups_regex = { "WE": ".*\\.word_embeddings\\..*",
                            
    #                         "00-05": ".*\\.layer\\.([0-5])\\..*\\.lora_(A|B)\\..*",
    #                         "06-11": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_(A|B)\\..*",
    #                         "12-17": ".*\\.layer\\.(1[2-7])\\..*\\.lora_(A|B)\\..*",
    #                         "18-23": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_(A|B)\\..*",

    #                         "00-05 A": ".*\\.layer\\.([0-5])\\..*\\.lora_A\\..*",
    #                         "06-11 A": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_A\\..*",
    #                         "12-17 A": ".*\\.layer\\.(1[2-7])\\..*\\.lora_A\\..*",
    #                         "18-23 A": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_A\\..*",

    #                         "00-05 B": ".*\\.layer\\.([0-5])\\..*\\.lora_B\\..*",
    #                         "06-11 B": ".*\\.layer\\.([6-9]|1[0-1])\\..*\\.lora_B\\..*",
    #                         "12-17 B": ".*\\.layer\\.(1[2-7])\\..*\\.lora_B\\..*",
    #                         "18-23 B": ".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_B\\..*",                            

    #                         "CL": ".*\\.classifier\\..*",
    #                      }
    
    # colors = {
    #     "hf_we_, WE": '#1f77b4',
    #     "hf_we_topk_10, WE": '#7f7f7f',
    #     # "hf, WE": "#ff7f0e",
    #     "hf, 18-23 B": "#2ca02c",
    #     "cos, 18-23 B": "#ff7f0e",
    #     # "hf, 12-17 B": "#d62728",
    #     # "hf, 18-23 A": "#8c564b",
    #     # "hf, 18-23 B": "#e377c2",
    #     # "hf, CL": "#9467bd",
    #     "datainf, 18-23 B":"#9467bd",
    #     # '#bcbd22', '#17becf'
    # }

    # legend_order = {
    #     "hf_we_, WE": 0,
    #     "hf_we_topk_10, WE": 1,
    #     "hf, 18-23 B": 3,
    #     "datainf, 18-23 B": 4,
    #     "cos, 18-23 B": 5,
    # }
    # legend_names = {
    #     "hf_we_, WE": "hf$_{we}$",
    #     "hf_we_topk_10, WE": "hf$_{we}^{10}$"
    # }
    # tasks = ['sst2']
    # module_groups_regex_rev = {v:k for k,v in module_groups_regex.items()}
    # for d in tasks:
    #     draw_ft2_metric2(d, infile = f'./data/roberta/{d}/metrics.jsonlist', outfile = f'./data/roberta/{d}/postprocess/T-acc-hf-top.pdf',
    #                     metric = 'accuracy', module_pattern_to_name = module_groups_regex_rev,
    #                     colors=colors, legend_order = legend_order, legend_names = legend_names,
    #                     infl_vs_module_filter = infl_vs_module_filter)
        
    # pass        

    #------------------------------------------------------------------------
    # OLD code from here

    # compute_noise_detection_metrics('./data/roberta/qnli', 
    #                                 plot_title = "QNLI on Roberta-large with WE",
    #                                 # cache_file="./data/roberta/qnli/postprocess.pt",
    #                                 out_chart_file="./data/roberta/postprocess/qnli/best_modules_on_mean_groups2.pdf",
    #                                 module_groups_regex = module_groups_regex)

    # compute_noise_detection_metrics('./data/roberta/mrpc', 
    #                                 plot_title = "MRPC on Roberta-large with WE",
    #                                 # cache_file="./data/roberta/mrpc/postprocess.pt",
    #                                 out_chart_file="./data/roberta/postprocess/mrpc/best_modules_on_mean_groups2.pdf",
    #                                 module_groups_regex = module_groups_regex)

    # compute_noise_detection_metrics('./data/roberta/sst2', 
    #                                 plot_title = "SST2 on Roberta-large with WE",
    #                                 # cache_file="./data/roberta/sst2/postprocess.pt",
    #                                 out_chart_file="./data/roberta/postprocess/sst2/best_modules_on_mean_groups2.pdf",
    #                                 module_groups_regex = module_groups_regex)

    # compute_noise_detection_metrics('./data/roberta/qqp', 
    #                                 plot_title = "QQP on Roberta-large with WE",
    #                                 # cache_file="./data/roberta/qqp/postprocess.pt",
    #                                 out_chart_file="./data/roberta/postprocess/qqp/best_modules_on_mean_groups2.pdf",
    #                                 module_groups_regex = module_groups_regex)






    # draw_mislabel_detection_rate2(infl_folder = "./data/cifar-resnet/infl", dataset_file = "./data/cifar-resnet/ds/d_cifar10_0",
    #                                 out = "./data/mdr/cifar.png", module_name = '')

    # draw_mislabel_detection_rate2(infl_folder = "./data/cifar-resnet/infl", dataset_file = "./data/cifar-resnet/ds/d_cifar10_0",
    #                                 out = "./data/mdr/cifar-layer4.png", module_name = 'layer4\\..*', title='Layer 4, worst noise', task="cifar10")
        
    # for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
    #     draw_mislabel_detection_rate(task=ds, res_folder = "./data/infl", module_pattern = '', 
    #                                 datasets_folder = './data/datasets', out = f"./data/mdr/{ds}.png")
        
    # for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
    #     draw_mislabel_detection_rate(task=ds, res_folder = "./data/infl", module_pattern = '.*\.layer\.[0-7]\..*\.lora_(A|B)\..*', 
    #                                 datasets_folder = './data/datasets', out = f"./data/mdr-0/{ds}.png", title="Layers 0-7")

    # for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
    #     draw_mislabel_detection_rate(task=ds, res_folder = "./data/infl", module_pattern = '.*\.layer\.([8-9]|1[0-5])\..*\.lora_(A|B)\..*', 
    #                                 datasets_folder = './data/datasets', out = f"./data/mdr-1/{ds}.png", title="Layers 8-15")

    # for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
    #     draw_mislabel_detection_rate(task=ds, res_folder = "./data/infl", module_pattern = '.*\.layer\.(1[6-9]|2[0-3])\..*\.lora_(A|B)\..*', 
    #                                 datasets_folder = './data/datasets', out = f"./data/mdr-2/{ds}.png", title="Layers 16-23")
        
    # for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
    #     draw_mislabel_detection_rate(task=ds, res_folder = "./data/infl", module_pattern = '.*\.classifier\..*', 
    #                                 datasets_folder = './data/datasets', out = f"./data/mdr-3/{ds}.png", title="Classifier")        

    # for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
    #     draw_ft2_metric(task = ds, infile = f'./data/ft2-infl/{ds}.jsonlist', outfile = f'./data/accuracy/{ds}.png', metric = 'accuracy')
    # list_modules('./data/infl/mrpc/i_datainf_mrpc_0.pt', '') #'.*\.layer\.(1[6-9]|2[0-3])\..*\.lora_(A|B)\..*')
    # draw_curve(res_filer='infl_qnli_', out = "./data/auc/qnli.png")    
    # draw_curve(task="qnli", res_folder = "./data/self-infl", module_pattern = '', datasets_folder = './data/datasets', out = "./data/auc/qnli.png")
    
    # module_pattern_to_name = {".*\\.layer\\.(1[6-9]|2[0-3])\\..*\\.lora_(A|B)\\..*": "last 8", 
    #                           "": "all", 
    #                           ".*\\.classifier\\..*": "classifier", 
    #                           ".*\\.layer\\.([0-9]|1[0-5])\\..*\\.lora_(A|B)\\..*": "first 16", 
    #                           "rand": "rand", 
    #                           ".*\\.layer\\.[0-8]\\..*\\.lora_(A|B)\\..*": "first 9"}
    # for d in ['mrpc', 'qnli', 'qqp', 'sst2']:
    #     for m in ['datainf', 'hf', 'lissa']:
    #         draw_ft2_metric(d, infile = f'./data/ft2-infl/{d}.jsonlist', outfile = f'./data/accuracy/{m}/{d}.png', 
    #                         influence_method = m, metric = 'accuracy', module_pattern_to_name = module_pattern_to_name)
    
    # module_pattern_to_name = {".*\\.layer\\.(1[6-9]|2[0-3])\\..*\\.lora_(A|B)\\..*": "last 8", "": "all"}
    # module_pattern_to_name = {
    #                           ".*\\.word_embeddings\\..*": "WE",
    #                           ".*\\.classifier\\..*": "classifier", 
    #                           ".*\\.layer\\.([0-9]|1[0-5])\\..*\\.lora_(A|B)\\..*": "first 16", 
    #                           ".*\\.layer\\.(1[6-9]|2[0-3])\\..*\\.lora_(A|B)\\..*": "last 8",   
    #                           "rand": "rand",
    #                           "denoise": "denoise"}
    # for d in ['mrpc', 'qnli', 'qqp', 'sst2']:
    #     draw_ft2_metric2(d, infile = f'./data/roberta/{d}/metrics.jsonlist', outfile = f'./data/roberta/postprocess/{d}/tun2-acc-hf-we.pdf',
    #                     influence_method = "hf_we_", metric = 'accuracy', module_pattern_to_name = module_pattern_to_name)
