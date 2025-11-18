from collections import defaultdict
from functools import partial
import json
import os
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pandas as pd
import re
from typing import Literal, Optional
from tabulate import tabulate

import datasets
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
from scipy import stats
import torch
from datasets import load_from_disk
from scipy import stats as sci_stats
from sklearn.preprocessing import StandardScaler

# from pyclustering.cluster.xmeans import xmeans
from sklearn.metrics import silhouette_samples
# import warnings
# np.warnings = warnings

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{times}"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = ['Times']    

benchmark = ["qnli", "mrpc", "sst2", "cola", "qqp", "mnli", "rte", "stsb"]

# from cifar import DatasetSplits

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

def compute_ndr_histogram(scores: torch.Tensor, noise_mask_cpu: torch.Tensor, bins = 10):
    ''' scores is 3-D tensor: agg_method * modules_id * train_sample_scores 
        we compute histograms for each agg_method * modules_id
    '''
    histograms = torch.zeros((scores.shape[0], scores.shape[1], bins), dtype=torch.float, device = "cpu")     

    quantiles = torch.linspace(0, 1, bins + 1, device = scores.device)

    bin_edges = torch.quantile(scores, quantiles, dim=-1)
    bin_ranges = bin_edges[-1] - bin_edges[0]
    bin_sizes = (bin_edges[1:] - bin_edges[:-1]) / bin_ranges

    for method_id in range(scores.shape[0]):
        for module_id in range(scores.shape[1]):
            one_scores = scores[method_id, module_id].cpu()
            one_bin_edges = bin_edges[:, method_id, module_id].cpu()
            # NOTE: histogram does not work on CUDA
            hist, _ = torch.histogram(one_scores, bins = one_bin_edges, weight=noise_mask_cpu, density=False)
            histograms[method_id, module_id] = hist

    histograms = histograms / torch.sum(noise_mask_cpu)

    bin_sizes_p = bin_sizes.permute(1, 2, 0).cpu()

    return histograms, bin_sizes_p

def compute_histograms(infl_matrix: torch.Tensor, noise_mask: torch.Tensor, bins = 10):
    ''' infl_matrix is 2-D tensor train_sample * infl_scores,
         infl_matrix.t() is infl_sample * infl_scores 
    '''
    infl_matrix_t = infl_matrix.t()
    histograms = torch.zeros((infl_matrix_t.shape[0], bins), dtype=torch.float, device = infl_matrix.device)     
    bin_sizes_all = torch.zeros((infl_matrix_t.shape[0], bins), dtype=torch.float, device = infl_matrix.device)     

    quantiles = torch.linspace(0, 1, bins + 1, device = infl_matrix.device)
    for infl_id in range(infl_matrix_t.shape[0]):
        bin_edges = torch.quantile(infl_matrix_t[infl_id], quantiles)
        # count_in_bin = np.sum((module_interactions[infl_id] >= bin_edges[0]) & (module_interactions[infl_id] <= bin_edges[1]))
        hist, _ = torch.histogram(infl_matrix_t[infl_id], bins = bin_edges, weight=noise_mask, density=False)
        histograms[infl_id] = hist
        bin_sizes_all[infl_id] = bin_edges[1:] - bin_edges[:-1]

    mean_histogram = histograms.mean(dim=0)
    return histograms, mean_histogram, bin_sizes_all


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

def mean_matrix_score(int_matrix: torch.Tensor, *, use_correct = None, correct_infl_preds, trim_ratio = None, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample '''
    if use_correct is not None:
        if use_correct:
            int_matrix = int_matrix[:, :, correct_infl_preds]
        else:
            int_matrix = int_matrix[:, :, ~correct_infl_preds]
    if trim_ratio is not None:
        start_id = round(int_matrix.shape[-1] * trim_ratio / 2)
        end_id = int_matrix.shape[-1] - start_id
        int_matrix_sorted, int_matrix_sorted_ids = torch.sort(int_matrix, dim = -1)
        scores = torch.mean(int_matrix_sorted[:, :, start_id:end_id], dim=(-2,-1))
        del int_matrix_sorted, int_matrix_sorted_ids
    else:  
        scores = int_matrix.mean(dim=(-2,-1))

    # mean_histogram, histograms, _ = compute_histograms(int_matrix, noise_mask)
    # mean_histogram2, _, _ = compute_histograms(int_matrix, noise_mask, bins = 100)
    return scores

def median_matrix_score(int_matrix: torch.Tensor, *, noise_mask, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample '''
    scores = int_matrix.median(dim=(-2,-1))

    # mean_histogram, histograms, _ = compute_histograms(int_matrix, noise_mask)
    # mean_histogram2, _, _ = compute_histograms(int_matrix, noise_mask, bins = 100)
    return scores    

# common set on missed influence samples
# def csmi_matrix_score(int_matrix: torch.Tensor, *, noise_mask: torch.Tensor, 
#                               trainset_labels: torch.Tensor, inflset_labels: torch.Tensor,
#                               inflset_logits: torch.Tensor,
#                               same_class = False, descending = False, vote_ratio = 0.3, noise_ratio = 0.3, **_) -> torch.Tensor:
#     ''' 3-D tensor: train sample * module * infl sample '''
    
#     global_total_int_matrix = int_matrix.mean(dim=-2)

#     infl_predictions = torch.argmax(inflset_logits, dim = -1)

#     unique_classes = torch.unique(trainset_labels)

#     infl_groups = {}

#     for label in unique_classes:
#         infl1 = torch.where((inflset_labels == label) & (inflset_labels == infl_predictions))[0]
#         infl2 = torch.where((inflset_labels == label) & (inflset_labels != infl_predictions))[0]
#         infl3 = torch.where((inflset_labels != label) & (inflset_labels == infl_predictions))[0]
#         infl4 = torch.where((inflset_labels != label) & (inflset_labels != infl_predictions))[0]

#         infl_groups[label] = (infl1, infl2, infl3, infl4)        


#     train_ids = []
#     infl_ids = []
#     for label in unique_classes:
#         infl1 = torch.where((inflset_labels == label) & (inflset_labels == infl_predictions))[0]
#         infl2 = torch.where((inflset_labels == label) & (inflset_labels != infl_predictions))[0]
#         infl3 = torch.where((inflset_labels != label) & (inflset_labels == infl_predictions))[0]
#         infl4 = torch.where((inflset_labels != label) & (inflset_labels != infl_predictions))[0]

#         label_train_ids = torch.where(trainset_labels == label)[0]
#         train_ids.append(label_train_ids)
#         infl_ids.append(infl1)
#         train_ids.append(label_train_ids)
#         infl_ids.append(infl2)
#         train_ids.append(label_train_ids)
#         infl_ids.append(infl3)
#         train_ids.append(label_train_ids)
#         infl_ids.append(infl4)                        
#         # if same_class:
#         #     infl_ids.append(torch.where((inflset_labels == label) & (inflset_labels == infl_predictions))[0])
#         # else:
#         #     infl_ids.append(torch.where((inflset_labels != label) & (inflset_labels == infl_predictions))[0])
#     scores = torch.zeros(global_total_int_matrix.shape[0], dtype = torch.float, device = global_total_int_matrix.device)
#     for one_train_ids, one_infl_ids in zip(train_ids, infl_ids):
#         total_int_matrix = global_total_int_matrix[one_train_ids][:, one_infl_ids]
#         test_ids_ordered = total_int_matrix.argsort(dim=0, descending = descending)
#         # infl_sets = [set() for _ in range(total_int_matrix.shape[-1])]
#         noise_size = int(noise_ratio * total_int_matrix.shape[0])
#         mask = torch.zeros_like(test_ids_ordered, dtype=torch.bool, device = test_ids_ordered.device)
#         col_range = torch.arange(test_ids_ordered.shape[-1], device = test_ids_ordered.device)
#         for t in range(total_int_matrix.shape[0]):
#             mask[test_ids_ordered[t], col_range] = True 
#             votes = torch.sum(mask, dim=-1, dtype=torch.float) / test_ids_ordered.shape[-1]
#             commonset = torch.where(votes >= vote_ratio)[0]
#             if len(commonset) >= noise_size:
#                 break
#         one_scores = 1.0 - votes

#         selected_ids = torch.argsort(one_scores)[:noise_size]
#         noise_ndr1 = noise_mask[selected_ids].sum().item() / 900
#         print(noise_ndr1)
#         # selected_ids = torch.asgsort(one_scores)[noise_size:]
#         # noise_ndr2 = noise_mask[selected_ids].sum().item()
#         # print(noise_ndr2)
#         scores[one_train_ids] = one_scores
#     # scores = torch.zeros(int_matrix.shape[0], dtype = torch.float, device = total_int_matrix.device)
#     # scores[commonset] = -1.0     
#     return scores    

def commonsubset_matrix_score(int_matrix: torch.Tensor, *, 
                              trainset_labels: torch.Tensor, inflset_labels: torch.Tensor,
                              same_class = False, descending = False, vote_ratio = 0.4, noise_ratio = 0.3, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample '''
    
    global_total_int_matrix = int_matrix.mean(dim=-2)

    unique_classes = torch.unique(trainset_labels)
    train_ids = []
    infl_ids = []
    for label in unique_classes:
        train_ids.append(torch.where(trainset_labels == label)[0])
        if same_class:
            infl_ids.append(torch.where(inflset_labels == label)[0])
        else:
            infl_ids.append(torch.where(inflset_labels != label)[0])
    scores = torch.zeros(global_total_int_matrix.shape[0], dtype = torch.float, device = global_total_int_matrix.device)
    for one_train_ids, one_infl_ids in zip(train_ids, infl_ids):
        total_int_matrix = global_total_int_matrix[one_train_ids][:, one_infl_ids]
        test_ids_ordered = total_int_matrix.argsort(dim=0, descending = descending)
        # infl_sets = [set() for _ in range(total_int_matrix.shape[-1])]
        noise_size = int(noise_ratio * total_int_matrix.shape[0])
        mask = torch.zeros_like(test_ids_ordered, dtype=torch.bool, device = test_ids_ordered.device)
        col_range = torch.arange(test_ids_ordered.shape[-1], device = test_ids_ordered.device)
        for t in range(total_int_matrix.shape[0]):
            mask[test_ids_ordered[t], col_range] = True 
            votes = torch.sum(mask, dim=-1, dtype=torch.float) / test_ids_ordered.shape[-1]
            commonset = torch.where(votes >= vote_ratio)[0]
            if len(commonset) >= noise_size:
                break
        one_scores = 1.0 - votes
        scores[one_train_ids] = one_scores
    # scores = torch.zeros(int_matrix.shape[0], dtype = torch.float, device = total_int_matrix.device)
    # scores[commonset] = -1.0     
    return scores    

def commonset_matrix_score(int_matrix: torch.Tensor, *, vote_ratio = 0.2, noise_ratio = 0.3, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample '''
    total_int_matrix = int_matrix.mean(dim=-2)
    test_ids_ordered = total_int_matrix.argsort(dim=0)
    # infl_sets = [set() for _ in range(total_int_matrix.shape[-1])]
    noise_size = int(noise_ratio * total_int_matrix.shape[0])
    mask = torch.zeros_like(test_ids_ordered, dtype=torch.bool, device = test_ids_ordered.device)
    col_range = torch.arange(test_ids_ordered.shape[-1], device = test_ids_ordered.device)
    for t in range(total_int_matrix.shape[0]):
        mask[test_ids_ordered[t], col_range] = True 
        votes = torch.sum(mask, dim=-1, dtype=torch.float) / test_ids_ordered.shape[-1]
        commonset = torch.where(votes >= vote_ratio)[0]
        if len(commonset) >= noise_size:
            break
    scores = 1.0 - votes
    # scores = torch.zeros(int_matrix.shape[0], dtype = torch.float, device = total_int_matrix.device)
    # scores[commonset] = -1.0     
    return scores

def mean_min_matrix_score(int_matrix: torch.Tensor, *, min_ratio = None, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample '''
    total_int_matrix = int_matrix.view(int_matrix.shape[0], -1)
    if min_ratio is not None:
        infl_threshold = int(min_ratio * total_int_matrix.shape[-1])
        infl_set_ids = torch.argsort(total_int_matrix, dim = -1)[:, :infl_threshold]
        selected_infl_scores = total_int_matrix.gather(1, infl_set_ids)
        scores = torch.mean(selected_infl_scores, dim = -1)
    else:
        scores = torch.min(total_int_matrix, dim = -1).values
    return scores

def cset_matrix_score(int_matrix: torch.Tensor, *, correct_infl_preds, vote_ratio = 0.3, noise_ratio = 0.3, use_correct = None,
                                                    both_sides = False, **_) -> torch.Tensor:    
    ''' 3-D tensor: train sample * module * infl sample '''
    if use_correct is not None:
        if use_correct:
            int_matrix = int_matrix[:, :, correct_infl_preds]
        else:
            int_matrix = int_matrix[:, :, ~correct_infl_preds]        
    total_int_matrix = int_matrix.view(int_matrix.shape[0], -1)
    votes = torch.zeros(int_matrix.shape[0], dtype = torch.float, device = total_int_matrix.device)
    vote_threshold = int(total_int_matrix.shape[-1] * vote_ratio)
    noise_size = int(noise_ratio * total_int_matrix.shape[0])
    test_ids_ordered = total_int_matrix.argsort(dim=0) 
    for t in range(total_int_matrix.shape[0]):
        votes[test_ids_ordered[t]] += 1 
        if both_sides:
            votes[test_ids_ordered[-t-1]] += 1
        enough_votes = torch.sum(votes >= vote_threshold)
        if enough_votes >= noise_size:
            break
    scores = 1.0 - votes / total_int_matrix.shape[-1]
    return scores

# def maj_matrix_score(int_matrix: torch.Tensor, *, vote_ratio = 0.5, noise_ratio = 0.3, **_) -> torch.Tensor:    
#     ''' 3-D tensor: train sample * module * infl sample '''
#     total_int_matrix = int_matrix.view(int_matrix.shape[0], -1)
#     votes = torch.zeros(int_matrix.shape[0], dtype = torch.float, device = total_int_matrix.device)
#     vote_threshold = int(total_int_matrix.shape[-1] * vote_ratio)
#     noise_size = int(noise_ratio * total_int_matrix.shape[0])
#     test_ids_ordered = total_int_matrix.argsort(dim=0)
#     voters = torch.zeros(total_int_matrix.shape[-1], dtype = torch.int64, device = total_int_matrix.device)
#     voter_ids = torch.arange(total_int_matrix.shape[-1], device = total_int_matrix.device)
#     while True:
#         cur_test_ids = torch.gather(test_ids_ordered, 0, voters.unsqueeze(0)).view(-1)
#         min_val_voter_id = torch.argmin(total_int_matrix[cur_test_ids, voter_ids])
#         cur_test_id = test_ids_ordered[voters[min_val_voter_id]]
#         votes[cur_test_id] += 1
#         voters[min_val_voter_id] += 1
#         enough_votes = torch.sum(votes >= vote_threshold)
#         if enough_votes >= noise_size:
#             break
#     scores = 1.0 - votes / total_int_matrix.shape[-1]
#     del test_ids_ordered, voters, voter_ids
#     return scores

# def get_pareto_front_indexes_neg(int_matrix: torch.Tensor) -> np.ndarray:
#     ''' Get the pareto front from a population. 
#         NOTE: greater is better here. Invert your fitness if it is the opposite.
#     '''
#     # mask = np.ones(int_matrix.shape[0], dtype=bool)
#     # mask[exclude_indexes] = False
#     # index_remap = np.where(mask)[0]
#     domination_matrix = torch.all(int_matrix[:, None] >= int_matrix, dim=2) & torch.any(int_matrix[:, None] > int_matrix, axis=2)
#     indexes = torch.where(~torch.any(domination_matrix, axis=1))[0]
#     return indexes

def get_pareto_front_indexes(fitnesses: np.ndarray, exclude_indexes: np.array = []) -> np.ndarray:
    ''' Get the pareto front from a population. 
        NOTE: greater is better here. Invert your fitness if it is the opposite.
    '''
    mask = np.ones(fitnesses.shape[0], dtype=bool)
    mask[exclude_indexes] = False
    index_remap = np.where(mask)[0]
    domination_matrix = np.all(fitnesses[mask][:, None] <= fitnesses[mask], axis=2) & np.any(fitnesses[mask][:, None] < fitnesses[mask], axis=2)
    indexes = np.where(~np.any(domination_matrix, axis=1))[0]
    return index_remap[indexes]

def pareto_matrix_score(int_matrix: torch.Tensor, *, noise_mask: torch.Tensor, noise_ratio = 0.3, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample '''
    total_int_matrix = int_matrix.mean(dim=-2)
    total_mean = total_int_matrix.mean(dim=-1)
    binary_mask = total_int_matrix < total_mean[:, None]
    # total_int_matrix = total_int_matrix[:, :10]
    scores = torch.zeros(total_int_matrix.shape[0], dtype=total_int_matrix.dtype, device = total_int_matrix.device)
    dom_ids = get_pareto_front_indexes_neg(total_int_matrix)
    scores[dom_ids] -= 1
    # infl_sets = [set() for _ in range(total_int_matrix.shape[-1])]
    # noise_size = int(noise_ratio * total_int_matrix.shape[0])
    # mask = torch.zeros_like(test_ids_ordered, dtype=torch.bool, device = test_ids_ordered.device)
    # col_range = torch.arange(test_ids_ordered.shape[-1], device = test_ids_ordered.device)
    # for t in range(total_int_matrix.shape[0]):
    #     mask[test_ids_ordered[t], col_range] = True 
    #     votes = torch.sum(mask, dim=-1, dtype=torch.float) / test_ids_ordered.shape[-1]
    #     commonset = torch.where(votes >= vote_ratio)[0]
    #     if len(commonset) >= noise_size:
    #         break
    # scores = 1.0 - votes
    # scores = torch.zeros(int_matrix.shape[0], dtype = torch.float, device = total_int_matrix.device)
    # scores[commonset] = -1.0     
    return scores    

def confident_matrix_score(int_matrix: torch.Tensor, *, base_method_fn = mean_matrix_score, 
                            trainset_labels: torch.Tensor, inflset_labels: torch.Tensor, correct_infl_preds,
                            inflset_logits: torch.Tensor, n_confident = 50, **_) -> torch.Tensor:    
    ''' 3-D tensor: train sample * module * infl sample '''
    # TODO: for multiclass we should use maximal non-golden conterpart for confidence
    logit_dist = inflset_logits[torch.arange(inflset_logits.shape[0]), inflset_labels] - inflset_logits[torch.arange(inflset_logits.shape[0]), 1 - inflset_labels]
    num_bigger = torch.sum(logit_dist > 0)
    n_confident_local = min(num_bigger.item(), n_confident)
    infl_ids = torch.argsort(logit_dist, descending=True)[:n_confident_local]
    selected_int_matrix = int_matrix[:, :, infl_ids]
    del logit_dist
    scores = base_method_fn(selected_int_matrix, trainset_labels = trainset_labels, inflset_labels = inflset_labels[infl_ids], 
                            inflset_logits = inflset_logits[infl_ids],
                            correct_infl_preds = correct_infl_preds)
    del infl_ids
    return scores


def median_matrix_score(int_matrix: torch.Tensor, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample ''' 
    total_int_matrix = int_matrix.view(int_matrix.shape[0], -1)
    scores = torch.median(total_int_matrix, dim = -1).values
    return scores

def condorcet_matrix_score(int_matrix: torch.Tensor, *, noise_mask: torch.Tensor, **_) -> torch.Tensor:
    pass 

def borda_matrix_score(int_matrix: torch.Tensor, *, noise_mask: torch.Tensor, **_) -> torch.Tensor:
    pass

def vote_matrix_score(rank_matrix: torch.Tensor, *, chunk_size = 10000, filter_perc = 0.3, **_) -> torch.Tensor:
    # total_rank_matrix = rank_matrix.view(rank_matrix.shape[0], -1)
    votes = torch.zeros(rank_matrix.shape[0], dtype = torch.float, device = rank_matrix.device)
    filter_threshold = round(filter_perc * rank_matrix.shape[0])
    for (ranks_view, votes_view) in zip(torch.split(rank_matrix, chunk_size, dim = 0),
                                                torch.split(votes, chunk_size, dim = 0)):    
        votes_view[:] = torch.sum(ranks_view >= filter_threshold, dim=(-2, -1), dtype=torch.float)
    return votes

def vote2_matrix_score(rank_matrix: torch.Tensor, *, chunk_size = 10000, filter_perc = 0.2, **_) -> torch.Tensor:
    # total_rank_matrix = rank_matrix.view(rank_matrix.shape[0], -1)
    votes = torch.zeros(rank_matrix.shape[0], dtype = torch.float, device = rank_matrix.device)
    filter_threshold = round(filter_perc * rank_matrix.shape[0])
    # max_rank_value = rank_matrix.shape[0] - 1
    for (ranks_view, votes_view) in zip(torch.split(rank_matrix, chunk_size, dim = 0),
                                                torch.split(votes, chunk_size, dim = 0)):    
        tmp = filter_threshold - ranks_view 
        tmp[tmp < 0] = 0
        votes_view[:] = torch.sum(tmp, dim=(-2, -1), dtype=torch.float)
        del tmp
    votes.neg_()
    return votes

def min_matrix_score(int_matrix: torch.Tensor, **_) -> torch.Tensor:
    scores = torch.min(int_matrix.view(int_matrix.shape[0], -1), dim = -1).values
    return scores

def max_matrix_score(int_matrix: torch.Tensor, **_) -> torch.Tensor:
    scores = torch.max(int_matrix.view(int_matrix.shape[0], -1), dim = -1).values
    return scores

def rank_matrix_score(int_matrix: torch.Tensor, *, 
                        use_correct = None, correct_infl_preds, 
                        chunk_size = 10000, rank_score_fn = mean_matrix_score, **_):
    ''' 3-D tensor: train sample * module * infl sample '''
    if use_correct is not None:
        if use_correct:
            int_matrix = int_matrix[:, :, correct_infl_preds]
        else:
            int_matrix = int_matrix[:, :, ~correct_infl_preds]    
    
    total_int_matrix = int_matrix.view(int_matrix.shape[0], -1)
    ranks = torch.zeros_like(total_int_matrix, dtype=torch.float, device = total_int_matrix.device)
    rank_ranges = torch.arange(total_int_matrix.shape[0], device = total_int_matrix.device, dtype=torch.float).view(-1, 1).repeat(1, chunk_size)
    for (int_matrix_view, ranks_view) in zip(torch.split(total_int_matrix, chunk_size, dim = 1),
                                                torch.split(ranks, chunk_size, dim = 1)):
        sort_indexes = torch.argsort(int_matrix_view, dim = 0)
        # rank_range += 1.0
        # int_matrix_view = int_matrix.view(int_matrix.shape[0], -1)        
        # ranks[sort_indexes] = rank_range
        ranks_view.scatter_(0, sort_indexes, rank_ranges)
        # for module_id in range(int_matrix.shape[1]):
        #     for val_id in range(int_matrix.shape[2]):
        #         ranks[sort_indexes[val_id, module_id], module_id, val_id] = rank_range
        del sort_indexes
        # del sort_indexes, rank_range
    del rank_ranges

    # noisy_ranks = ranks[noise_mask] / (int_matrix.shape[0] - 1)
    # mean_ranker_noise_scores = torch.mean(noisy_ranks, dim=0) 
    # good_rankers_mask = mean_ranker_noise_scores < 0.3
    # good_rankers_mask_sz = good_rankers_mask.sum()
    # # good_rankers_mask = (mean_ranker_noise_scores.view(-1, int_matrix.shape[-1]).mean(dim=0, dtype=torch.float)) < 0.5
    # # match_labels = trainset_labels == inflset_labels
    # # missmatch_labels = trainset_labels != inflset_labels

    # pred_labels = torch.argmax(inflset_logits, dim = -1)
    # correct_mask = inflset_labels == pred_labels

    # good_and_correct = good_rankers_mask.view(-1, correct_mask.shape[0]) & correct_mask.unsqueeze(0)
    # good_and_correct_counts = torch.sum(good_and_correct, dim=1) / good_rankers_mask_sz  #per layer
    # discovered_count = torch.mean(good_and_correct_counts, dtype=torch.float)

    # good_and_incorrect = good_rankers_mask.view(-1, correct_mask.shape[0]) & (~correct_mask).unsqueeze(0)
    # good_and_incorrect_counts = torch.sum(good_and_incorrect, dim=1) / good_rankers_mask_sz  #per layer
    # discovered_count2 = torch.mean(good_and_incorrect_counts, dtype=torch.float)


    scores = rank_score_fn(ranks.view(int_matrix.shape), correct_infl_preds = correct_infl_preds)
    del ranks
    return scores

def draw_clusters(X, y, G, module_id = 0):
    import umap

    umap_model = umap.UMAP(n_neighbors=25, min_dist = 0.01, random_state=42)
    X_umap = umap_model.fit_transform(X)

    distinct_y = np.unique(y)

    # colors = ['#03fcc2', '#ed91e4', '#03fcc2', '#ed91e4', '#f04354', '#696667']

    legends = []
    plt.ioff()
    plt.clf()

    for label_id, label in enumerate(distinct_y):

        X0 = X_umap[(G == 0) & (y == label)]
        X1 = X_umap[(G == 1) & (y == label)]
        # color = colors[label_id]

        h = plt.scatter(X0[:, 0], X0[:, 1], marker="o", s=10, alpha=0.5)
        legends.append(f"Cluster {label}, clear")
        plt.scatter(X1[:, 0], X1[:, 1], marker="x", s=20, alpha = 1, color=h.get_facecolor()[0])
        legends.append(f"Cluster {label}, noise")
    
    plt.legend(legends)
    plt.savefig("tmp.pdf")
    plt.savefig(f"data/clusters/l-{module_id}.pdf")
    pass

def draw_clusters2(X1, y1, G1, X2, y2, G2, module_id = 0):

    plt.ioff()
    plt.clf()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    import umap

    umap_model = umap.UMAP(n_neighbors=25, min_dist = 0.01, random_state=42)
    X1_umap = umap_model.fit_transform(X1)

    distinct_y = np.unique(y1)

    legends = []
    for label_id, label in enumerate(distinct_y):

        xx0 = X1_umap[(G1 == 0) & (y1 == label)]
        xx1 = X1_umap[(G1 == 1) & (y1 == label)]
        # color = colors[label_id]

        ax[0].scatter(xx0[:, 0], xx0[:, 1], marker="o", s=10, alpha=0.5)
        legends.append(f"Cluster {label}, clear")
        ax[0].scatter(xx1[:, 0], xx1[:, 1], marker="x", s=20, alpha = 1)
        legends.append(f"Cluster {label}, noise")
    
    ax[0].legend(legends)

    umap_model = umap.UMAP(n_neighbors=25, min_dist = 0.01, random_state=42)
    X2_umap = umap_model.fit_transform(X2)

    distinct_y = np.unique(y2)

    legends = []
    for label_id, label in enumerate(distinct_y):

        xx0 = X2_umap[(G2 == 0) & (y2 == label)]
        xx1 = X2_umap[(G2 == 1) & (y2 == label)]
        # color = colors[label_id]

        ax[1].scatter(xx0[:, 0], xx0[:, 1], marker="o", s=10, alpha=0.5)
        legends.append(f"Cluster {label}, clear")
        ax[1].scatter(xx1[:, 0], xx1[:, 1], marker="x", s=20, alpha = 1) # color=h.get_facecolor()[0])
        legends.append(f"Cluster {label}, noise")
    
    ax[1].legend(legends)


    plt.savefig("tmp.pdf")
    plt.savefig(f"data/clusters/l-{module_id}.pdf")
    plt.close(fig)
    pass

def distance_to_centroid(X, _):
    centroid = np.mean(X, axis = 0)
    d = np.linalg.norm(X - centroid, axis=1)
    max_d = np.max(d)
    return 1 - d / max_d


def cluster_matrix_score(int_matrix: torch.Tensor, *, noise_mask: torch.Tensor, 
                         trainset_labels: torch.Tensor, inflset_labels: torch.Tensor, 
                         measure_clusters = silhouette_samples, kmax = 4, **_):
    ''' 3-D tensor: train sample * module * infl sample '''

    int_matrix_X = int_matrix.float().cpu().numpy()
    # same_class = trainset_labels[:, None] == inflset_labels[None, :]
    labels = trainset_labels.cpu().numpy()
    # labels = same_class.to(torch.int)
    # labels[:, trainset_labels == 1] += 2
    # labels = labels.cpu().numpy()
    noise_mask_np = noise_mask.cpu().numpy()
    scores = []
    thr = 1350
    noise_size = 900
    ndrs = []
    for module_id in range(int_matrix.shape[1]):
        # random_state = np.random.randint(0, 100000)
        module_int_matrix_X = int_matrix_X[:, module_id]
        X_normalized = StandardScaler().fit_transform(module_int_matrix_X)
        # X_normalized = module_int_matrix_X
        # xmeans_instance = KMeans(n_clusters = kmax, random_state = random_state).fit(X_normalized)
        # # xmeans_instance.process()
        # # clusters = xmeans_instance.get_clusters()
        # # labels = np.zeros(int_matrix.shape[0], dtype=int)
        # # for cluster_id, smaple_ids in enumerate(clusters):
        # #     labels[smaple_ids] = cluster_id
        # labels = xmeans_instance.labels_


        draw_clusters(X_normalized, labels, noise_mask_np, module_id = module_id)

        scores_by_clusters = measure_clusters(X_normalized, labels)
        idxs = np.argsort(scores_by_clusters)
        start_noise_30 = noise_mask_np[idxs[:thr]].sum() / noise_size
        end_noise_30 = noise_mask_np[idxs[-thr:]].sum() / noise_size
        print(f"Module {module_id} start noise {round(start_noise_30 * 100)}, end noise {round(end_noise_30 * 100)}")
        ndrs.append({"module_id": module_id, "start_noise_30": start_noise_30, "end_noise_30": end_noise_30})
        scores.append(scores_by_clusters)

    df = pd.DataFrame(ndrs)
    df.to_csv("data/clusters/ndrs.csv")
    mean_scores = np.mean(scores, axis = 0)
    idxs = np.argsort(mean_scores)
    start_noise_30 = noise_mask_np[idxs[:thr]].sum() / noise_size
    end_noise_30 = noise_mask_np[idxs[-thr:]].sum() / noise_size
    print(f"Total: start noise {round(start_noise_30 * 100)}, end noise {round(end_noise_30 * 100)}")

    return mean_scores

def cluster_matrix_score2(int_matrix: torch.Tensor, *, noise_mask: torch.Tensor, 
                         trainset_labels: torch.Tensor, inflset_labels: torch.Tensor, 
                         measure_clusters = distance_to_centroid, kmax = 4, **_):
    ''' 3-D tensor: train sample * module * infl sample '''

    int_matrix_X = int_matrix.float().cpu().numpy()
    train_labels = trainset_labels.cpu().numpy()
    infl_labels = inflset_labels.cpu().numpy()
    all_labels = np.unique(train_labels)
    train_ids = []
    infl_ids = []
    for label in all_labels:
        train_ids.append(np.where(train_labels == label)[0])
        infl_ids.append(np.where(infl_labels == label)[0])
    # same_class = trainset_labels[:, None] == inflset_labels[None, :]
    # labels = same_class.to(torch.int)
    # labels[:, trainset_labels == 1] += 2
    # labels = labels.cpu().numpy()
    noise_mask_np = noise_mask.cpu().numpy()
    scores = []
    thr = 1350
    noise_size = 900
    ndrs = []
    for module_id in range(int_matrix.shape[1]):
        # random_state = np.random.randint(0, 100000)
        module_int_matrix_X = int_matrix_X[:, module_id]
        X_normalized = StandardScaler().fit_transform(module_int_matrix_X)
        # X_normalized = module_int_matrix_X
        # xmeans_instance = KMeans(n_clusters = kmax, random_state = random_state).fit(X_normalized)
        # # xmeans_instance.process()
        # # clusters = xmeans_instance.get_clusters()
        # # labels = np.zeros(int_matrix.shape[0], dtype=int)
        # # for cluster_id, smaple_ids in enumerate(clusters):
        # #     labels[smaple_ids] = cluster_id
        # labels = xmeans_instance.labels_

        X1 = X_normalized[train_ids[0]][:, infl_ids[0]]
        y1 = train_labels[train_ids[0]]
        G1 = noise_mask_np[train_ids[0]]

        X2 = X_normalized[train_ids[1]][:, infl_ids[1]]
        y2 = train_labels[train_ids[1]]
        G2 = noise_mask_np[train_ids[1]]
        all_scores_by_clusters = np.zeros(int_matrix_X.shape[0], dtype = float)

        draw_clusters2(X1, y1, G1, X2, y2, G2, module_id = module_id)

        scores_by_clusters = measure_clusters(X1, y1)
        all_scores_by_clusters[train_ids[0]] = scores_by_clusters

        scores_by_clusters = measure_clusters(X2, y2)
        all_scores_by_clusters[train_ids[1]] = scores_by_clusters 

        idxs = np.argsort(all_scores_by_clusters)
        start_noise_30 = noise_mask_np[idxs[:thr]].sum() / noise_size
        end_noise_30 = noise_mask_np[idxs[-thr:]].sum() / noise_size
        print(f"Module {module_id} start noise {round(start_noise_30 * 100)}, end noise {round(end_noise_30 * 100)}")
        ndrs.append({"module_id": module_id, "start_noise_30": start_noise_30, "end_noise_30": end_noise_30})
        scores.append(all_scores_by_clusters)

    df = pd.DataFrame(ndrs)
    df.to_csv("data/clusters/ndrs.csv")
    mean_scores = np.mean(scores, axis = 0)
    idxs = np.argsort(mean_scores)
    start_noise_30 = noise_mask_np[idxs[:thr]].sum() / noise_size
    end_noise_30 = noise_mask_np[idxs[-thr:]].sum() / noise_size
    print(f"Total: start noise {round(start_noise_30 * 100)}, end noise {round(end_noise_30 * 100)}")

    return mean_scores


def dir_matrix_score(int_matrix: torch.Tensor, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample '''
    same_dir_mask = int_matrix > 0
    one_scores = same_dir_mask.float()
    scores = torch.mean(one_scores, dim=(-2, -1))
    del same_dir_mask, one_scores
    return scores


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
        simple_name_parts = [replace_name.get(n, n) for n in module_name.split('.') if n not in ['base_model', 'model', 'roberta', 'encoder', 'embeddings', 'default', 'weight', 'attention', 'self', 'modules_to_save']]
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

def output_table(df: pd.DataFrame, base_path: str, task: str):
    ndr_stats_file_path = os.path.join(base_path, f"{task}_ndr_stats.csv")
    
    df.to_csv(ndr_stats_file_path, index = False)

    stats_file_path = os.path.join(base_path, f"{task}_ndr_stats_simple.txt")
    with open(stats_file_path, "w") as stats_file:
        print(tabulate(df, headers = 'keys', tablefmt="github", floatfmt=".3f", showindex=True), file = stats_file)
    pass 

# RQ1: what checkpoint it is better to use with infl methods to detect noise 
def get_df_from_file(metric_file: str):    

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

        # logits_change = r["logits_change"]

        auc_ndr = r["auc_ndr"]
        filtered = r["filtered"]

        noise_curve = r["ndr_curve"]
        num_noise = round(0.2 * len(noise_curve))
        ideal_area = num_noise / 2 + (len(noise_curve) - num_noise)        
        auc_ndr2 = sum((noise_curve[i] + noise_curve[i + 1]) / 2 for i in range(len(noise_curve) - 1)) / ideal_area

        task = r["config"]["task"]
        infl_method = r["config"]["infl_method"]
        seed = r["config"]["seed"]
        filter_perc = r["config"]["filter_perc"]

        index_30 = round(filter_perc * len(noise_curve))
        noise_30  = noise_curve[index_30]

        assert np.allclose(auc_ndr, auc_ndr2, atol=1e-2)
        assert np.allclose(noise_30, filtered, atol=1e-2)

        if infl_method == "rand":
            rand_accuracies[(task, filter_perc, seed)] = (best_accuracy, best_infl_accuracy)

        module_name = r["config"]["module_name"]

        row = {
               "task": task,
               "filter_perc": filter_perc,
               "infl_method": infl_method,
               "agg_method": r["config"]["agg_method"],
               "module": module_name,
               
               "best_accuracy_delta": accuracy_delta,
               "best_infl_accuracy_delta": infl_accuracy_delta,
                "infl_loss_delta": infl_loss_delta,
            #    "logits_change": logits_change,
               "noise_30": filtered,
               "auc_ndr": auc_ndr,

               "best_accuracy_0": best_accuracy0,
               "best_accuracy_1": best_accuracy,

               "best_infl_accuracy_0": best_infl_accuracy0,
               "best_infl_accuracy_1": best_infl_accuracy,

                "infl_loss_0": min_infl_loss0,
                'infl_loss_1': min_infl_loss,

               "seed": seed

               }
        rows.append(row)

    if len(rand_accuracies) > 0:
        for r in rows:
            key = (r["task"], r["filter_perc"], r["seed"])
            rand_accuracy, rand_infl_accuracy = rand_accuracies[key]
            accuracy_rand_delta = r["best_accuracy_1"] - rand_accuracy
            infl_accuracy_rand_delta = r["best_infl_accuracy_1"] - rand_infl_accuracy
            r["accuracy_rand_delta"] = accuracy_rand_delta
            r["infl_accuracy_rand_delta"] = infl_accuracy_rand_delta

    df = pd.DataFrame(rows)

    return df

def get_agg_df(df: DataFrame, key_columns = ['task', 'filter_perc', 'infl_method', 'agg_method', 'module'],
                drop_columns=['seed'], selected_columns=None, agg_types = ['mean', 'std']):
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

def get_all_df(base_path = "data/roberta/filter-30", datasets = ["mrpc", "qnli", "sst2", "qqp", "cola", "mnli", "rte", "stsb"],
                res_suffix = "bl", keep_only = None):
    # if os.path.exists(tmp_file):
    #     df1 = pd.read_pickle(tmp_file)
    # else:
    keep_only = ["task", "infl_method", "agg_method", "module", "seed", *keep_only]
    plain_dataframes = []
    for dataset in datasets:
        file = os.path.join(base_path, "metrics",  f"{dataset}-{res_suffix}.jsonlist")
        df = get_df_from_file(file)
        if keep_only is not None:
            df.drop(columns=[c for c in df.columns if c not in keep_only], inplace=True)
        plain_dataframes.append(df)
    
    df1 = pd.concat(plain_dataframes, ignore_index=True)
    # df1.to_pickle(tmp_file)
    return df1

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

def run_friedman_tests(metric_name: str = "best_accuracy_1",
                       datasets = benchmark,
                        out_folder = "data/roberta/filter-30",
                        res_suffix = "bl"):

    all_df = get_all_df(base_path = out_folder, datasets = datasets,
                        res_suffix = res_suffix, keep_only = [metric_name])

    metrics_per_ds = all_df.pivot(index=["infl_method", "agg_method", "module"], columns=["task", "seed"], values=metric_name)

    idx = pd.MultiIndex.from_tuples(metrics_per_ds.columns)

    metrics_per_ds.columns = idx

    ranks = metrics_per_ds.rank(axis = 0, ascending=False, method="average") # method="dense")

    metrics_per_ds['rank'] = ranks.mean(axis=1).round(1)
    metrics_per_ds['rank_std'] = ranks.std(axis=1).round(1)
    
    metrics_per_ds = metrics_per_ds.sort_values(by=['rank', 'rank_std'], ascending=[True, True]).drop(columns=["rank", 'rank_std'])    

    methods = [ (m if m in ['denoise', 'rand'] else f'{m}, all') if l == "" else f"{m}, {l}" for m, _, l in metrics_per_ds.index.to_list()]

    stats_data = metrics_per_ds.dropna(axis=1, how='any').to_numpy()

    friedman_res = sci_stats.friedmanchisquare(*stats_data)

    import scikit_posthocs as sci_posthocs

    nemenyi_res = sci_posthocs.posthoc_nemenyi_friedman(stats_data.T) 
    print(f"\n----------------------------------")
    print(f"Friedman {metric_name}: {friedman_res}")
    from tabulate import tabulate
    rows = []
    for i in range(len(methods)):
        row = []
        row.append(i+1)
        for j in range(len(methods)):
            # pvalue = round(nemenyi_res[i][j] * 100) / 100
            pvalue = nemenyi_res[i][j]
            pvalue_exp = int(f"{pvalue:.0e}".split("e")[1].replace("-0", "-").replace("+0", ""))
            if pvalue_exp == 0:
                row.append("")
            elif pvalue <= 0.05:
                row.append(f"\\textbf{{{pvalue_exp}}}")
            else:
                row.append(f"{pvalue_exp}")
        rows.append(row)

    with open(f"{out_folder}/{metric_name}-friedman.tex", "w") as stats_file:
        s = tabulate(rows, headers=["", *[i + 1 for i in range(len(methods))]], showindex=False, tablefmt="latex", numalign="left", stralign="left")
        s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("hf\_we\_topk\_10", "hf$^{10}_{we}$").replace("hf\_we\_", "hf$_{we}$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("lllllllllllllllllllllllll", "l|cccccccccccccccccccccccc").replace("rand", "\\hline rand")
        print(f"% Friedman {metric_name}: {friedman_res}", file = stats_file)
        print(s, file = stats_file)

    # print(tabulate(rows, , tablefmt="grid", numalign="center", stralign="center"))
    pass 

def run_wilcoxon_tests(metric_name: str = "best_accuracy_1",
                       datasets = benchmark,
                        out_folder = "data/roberta/filter-30",
                        res_suffix = "bl"):

    all_df = get_all_df(base_path = out_folder, datasets = datasets,
                        res_suffix= res_suffix, keep_only = [metric_name])

    metrics_per_ds = all_df.pivot(index=["infl_method", "agg_method", "module"], columns=["task", "seed"], values=metric_name)

    idx = pd.MultiIndex.from_tuples(metrics_per_ds.columns)

    metrics_per_ds.columns = idx

    ranks = metrics_per_ds.rank(axis = 0, ascending=False, method="average") # method="dense")

    metrics_per_ds['rank'] = ranks.mean(axis=1).round(1)
    metrics_per_ds['rank_std'] = ranks.std(axis=1).round(1)
    
    metrics_per_ds = metrics_per_ds.sort_values(by=['rank', 'rank_std'], ascending=[True, True]).drop(columns=["rank", 'rank_std'])    

    methods = [ (m if m in ['denoise', 'rand'] else f'{m}, all') if l == "" else f"{m}, {l}" for m, _, l in metrics_per_ds.index.to_list()]

    stats_data = metrics_per_ds.dropna(axis=1, how='any').to_numpy()

    wilcoxon_res = {}
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            w = sci_stats.wilcoxon(stats_data[i], stats_data[j])
            wilcoxon_res.setdefault(i, {}).setdefault(j, w.pvalue)
            wilcoxon_res.setdefault(j, {}).setdefault(i, w.pvalue)

    from tabulate import tabulate
    rows = []
    for i in range(len(methods)):
        row = []
        row.append(i+1)
        for j in range(len(methods)):
            # pvalue = round(nemenyi_res[i][j] * 100) / 100
            if i == j:
                pvalue = 1.0
            else:
                pvalue = wilcoxon_res[i][j]
            pvalue_exp = int(f"{pvalue:.0e}".split("e")[1].replace("-0", "-").replace("+0", ""))
            if pvalue_exp == 0:
                row.append("")
            elif pvalue <= 0.05:
                row.append(f"\\textbf{{{pvalue_exp}}}")
            else:
                row.append(f"{pvalue_exp}")
        rows.append(row)

    with open(f"{out_folder}/{metric_name}-wilcoxon.tex", "w") as stats_file:
        s = tabulate(rows, headers=["", *[i + 1 for i in range(len(methods))]], showindex=False, tablefmt="latex", numalign="left", stralign="left")
        s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("hf\_we\_topk\_10", "hf$^{10}_{we}$").replace("hf\_we\_", "hf$_{we}$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("lllllllllllllllllllllllll", "l|cccccccccccccccccccccccc").replace("rand", "\\hline rand")
        print(s, file = stats_file)

    # print(tabulate(rows, , tablefmt="grid", numalign="center", stralign="center"))
    pass 

def run_spearman_tests(metric_name: str = "best_accuracy_1", ndr_delta = None,
                        out_folder = "data/roberta/filter-30", 
                        datasets = benchmark,
                        ndr_metric_name = "noise_30",
                        suffix="",
                        res_suffix = "bl"):

    all_df = get_all_df(base_path = out_folder, datasets = datasets,
                        res_suffix= res_suffix, keep_only = [metric_name, ndr_metric_name])
    if ndr_delta is not None:
        all_df = all_df[(all_df["infl_method"] != "rand") & (all_df["infl_method"] != "denoise")]

    metrics_per_ds = all_df.pivot(index=["infl_method", "agg_method", "module"], columns=["task", "seed"], values=[metric_name, ndr_metric_name])
    metrics_per_ds = metrics_per_ds.dropna(axis=1, how='any')
    metric_columns = {(ds, seed) for _, ds, seed in metrics_per_ds.columns[metrics_per_ds.columns.get_level_values(0) == metric_name]}
    ndr_columns = {(ds, seed) for _, ds, seed in metrics_per_ds.columns[metrics_per_ds.columns.get_level_values(0) == ndr_metric_name]}
    common_columns = set.intersection(metric_columns, ndr_columns)

    # rho = {}
    # pvalues = {}
    series1_lists = {}
    series2_lists = {}
    for ds, seed in common_columns:
        series1 = metrics_per_ds.loc[:, (metric_name, ds, seed)].to_numpy()
        series2 = metrics_per_ds.loc[:, (ndr_metric_name, ds, seed)].to_numpy()
        if ndr_delta is not None:
            ndr_idxs = np.argsort(series2)[::-1]
            selected_idxs = [ndr_idxs[0]] 
            prev_ndr = series2[selected_idxs[-1]]
            for i in range(1, len(ndr_idxs)):
                cur_ndr = series2[ndr_idxs[i]]
                if  prev_ndr - cur_ndr >= ndr_delta:
                    selected_idxs.append(ndr_idxs[i])
                    prev_ndr = cur_ndr
            series1 = series1[selected_idxs]
            series2 = series2[selected_idxs]

        series1_lists.setdefault(ds, []).extend(series1)
        series2_lists.setdefault(ds, []).extend(series2)

        # s = sci_stats.spearmanr(series1, series2)
        # rho.setdefault(ds, []).append(s.correlation)
        # pvalues.setdefault(ds, []).append(s.pvalue)

    rows = []
    # header_row = [""]
    # for d in datasets:
    #     header_row.append(d)
    # header_row.append("Total")
    # rows.append(header_row)


    rho_row = ["Spearman $\\rho$"]
    p_values = {}
    for d in datasets:
        series1 = series1_lists[d]
        series2 = series2_lists[d]
        # series1 = np.concatenate(series1_list)
        # series2 = np.concatenate(series2_list)
        s = sci_stats.spearmanr(series1, series2)
        rho_row.append(f"{s.correlation}")
        p_values[d] = s.pvalue
    # all_rho_plain = [v for vl in rho.values() for v in vl]
    # rho_mean = np.mean(all_rho_plain)
    # rho_std = np.std(all_rho_plain)
    # rho_mean = round(np.mean(rho_mean) * 10) / 10
    # rho_std = round(np.std(rho_std) * 10) / 10
    # rho_row.append(f"{mean_rho} $\pm$ {std_rho}")
    rows.append(rho_row)

    pvalue_row = ["p-value"]
    for d in datasets:
        ds_pvalue = p_values[d]
        # r = sci_stats.combine_pvalues(ds_pvalues)
        # pvalue_exp = int(f"{r.pvalue:.0e}".split("e")[1].replace("-0", "-").replace("+0", ""))
        pvalue_row.append(f"{ds_pvalue:.0e}")
    
    # r = sci_stats.combine_pvalues([v for vl in pvalues.values() for v in vl])
    # pvalue_exp = int(f"{r.pvalue:.0e}".split("e")[1].replace("-0", "-").replace("+0", ""))
    # pvalue_row.append(f"{r.pvalue:.0e}")
    rows.append(pvalue_row)

    with open(f"{out_folder}/tables/{metric_name}-spearman{suffix}.tex", "w") as stats_file:
        s = tabulate(rows, headers=["", *datasets], showindex=False, tablefmt="latex", numalign="center", stralign="center")
        s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("hf\_we\_topk\_10", "hf$^{10}_{we}$").replace("hf\_we\_", "hf$_{we}$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("lllllllllllllllllllllllll", "l|cccccccccccccccccccccccc").replace("rand", "\\hline rand")
        print(s, file = stats_file)

    # print(tabulate(rows, , tablefmt="grid", numalign="center", stralign="center"))
    pass 

def run_concat_spearman_test(metric_name: str = "best_accuracy_1",
                        out_folder = "data/roberta/filter-30", 
                        datasets = benchmark,
                        ndr_metric_name = "noise_30",
                        res_suffix = "bl",
                        suffix=""):

    all_df = get_all_df(base_path = out_folder, datasets = datasets,
                        res_suffix= res_suffix, keep_only = [metric_name, ndr_metric_name])

    metrics_per_ds = all_df.pivot(index=["infl_method", "agg_method", "module"], columns=["task", "seed"], values=[metric_name, ndr_metric_name])
    metrics_per_ds = metrics_per_ds.dropna(axis=1, how='any')
    metric_columns = {(ds, seed) for _, ds, seed in metrics_per_ds.columns[metrics_per_ds.columns.get_level_values(0) == metric_name]}
    noise_30_columns = {(ds, seed) for _, ds, seed in metrics_per_ds.columns[metrics_per_ds.columns.get_level_values(0) == 'noise_30']}
    common_columns = set.intersection(metric_columns, noise_30_columns)

    ds_series1 = {}
    ds_series2 = {}

    for ds, seed in common_columns:
        series1 = metrics_per_ds.loc[:, (metric_name, ds, seed)].to_numpy()
        series2 = metrics_per_ds.loc[:, ('noise_30', ds, seed)].to_numpy()   
        ds_series1.setdefault(ds, []).append(series1) 
        ds_series2.setdefault(ds, []).append(series2)


    for ds in datasets:
        ds_series1[ds] = np.concatenate(ds_series1[ds])
        ds_series2[ds] = np.concatenate(ds_series2[ds])

    rho = {}
    pvalues = {}
    for ds in datasets:
        series1 = ds_series1[ds]
        series2 = ds_series2[ds]
        s = sci_stats.spearmanr(series1, series2)
        rho[ds] = s.correlation
        pvalues[ds] = s.pvalue

    return rho, pvalues 

def create_tun2_metric_table(metric_name: str = "best_accuracy_1", prec = 2, 
                        out_folder = "data/roberta/filter-30", datasets = benchmark,
                        highlight_max = True, mul = 100,
                        with_row_id = False, res_suffix = "bl"):

    all_df = get_all_df(base_path = out_folder, datasets = datasets,
                            res_suffix = res_suffix, keep_only = [metric_name])
    
    int_df_path = os.path.join(out_folder, "metrics", f"intranks-{metric_name}.pcl")
    int_df = pd.read_pickle(int_df_path)
    int_df.reset_index(inplace=True)
    int_df.drop(columns=["module"], inplace=True)
    int_df.rename(columns={"infl": "infl_method", "agg": "agg_method", "layer": "module"}, inplace=True)
    int_df.set_index(["infl_method", "agg_method", "module"], inplace=True)

    metrics_per_ds = all_df.pivot(index=["infl_method", "agg_method", "module"], columns=["task", "seed"], values=metric_name)

    idx = pd.MultiIndex.from_tuples(metrics_per_ds.columns)

    metrics_per_ds.columns = idx
    # metrics_per_ds = metrics_per_ds.round(prec)

    # ranks = metrics_per_ds.round(prec).rank(axis = 0, ascending=False, method="average") # method="dense")
    # ranks = metrics_per_ds.rank(axis = 0, ascending=False, method="average") # method="dense")

    # suffix = ""
    # if ds_ranks:
    #     # ranks.columns = idx
    #     # metrics_per_ds = ranks.groupby(level=0, axis=1).mean()
    #     metrics_per_ds = ranks
    #     suffix = "-rank"


    # metrics_per_ds.to_csv(f"{out_folder}/{metric_name}{suffix}-all.csv")

    metric_by_ds_mean = metrics_per_ds.groupby(level=0, axis=1).mean()
    metric_by_ds_std = metrics_per_ds.groupby(level=0, axis=1).std()
    metric_by_ds_std.columns = [c + "_std" for c in metric_by_ds_std.columns]
    metric_by_ds = pd.merge(metric_by_ds_mean, metric_by_ds_std, left_index=True, right_index=True)
    
    metric_by_ds['rank'] = int_df.loc[metric_by_ds.index]['rank']
    metric_by_ds['win_rate'] = int_df.loc[metric_by_ds.index]['win_rate']
    metric_by_ds['wstd'] = int_df.loc[metric_by_ds.index]['wstd']
        
    # metric_by_ds['rank_std'] = ranks.std(axis=1)
    metric_by_ds = metric_by_ds.sort_values(by=['rank', 'win_rate', 'wstd'], ascending=[True, False, True])

    metric_by_ds = metric_by_ds[["rank", "win_rate", "wstd", *[de for d in datasets for de in [d, d + "_std"]]]]
    # metric_by_ds.to_csv(f"{out_folder}/{metric_name}{suffix}-avg.csv")

    filtered_df = metric_by_ds.loc[metric_by_ds.index.get_level_values('infl_method') != 'denoise']
    if highlight_max:
        values_to_highlight = filtered_df[datasets].to_numpy().max(axis=0)
    else:
        values_to_highlight = filtered_df[datasets].to_numpy().min(axis=0)
    
    values_to_highlight = [round(v * (10 ** prec) * mul) / (10 ** prec) for v in values_to_highlight]

    rows = []
    for row_id, row in enumerate(metric_by_ds.reset_index().to_dict(orient="records")):
        new_row = {}
        method = row["infl_method"]        
        layer = row["module"]
        if with_row_id:
            new_row["id"] = (row_id + 1)
        method_rename = {"datainf": "DataInf", "hf": "TracIn", "cos": "Cosine", "denoise": "Full", "rand": "Random", "hf_we_": "TracIn$_{we}$", "hf_we_topk_10": "TracIn$^{10}_{we}$"}
        new_row["Method"] = method_rename[method]
        new_row["Layer"] = "all" if (method not in ["denoise", "rand"]) and layer == "" else layer 
        new_row["Rank"] = row["rank"]
        new_row["Win Rate"] = f"{row['win_rate']:.2f} {{\\footnotesize $\\pm$ {row['wstd']:.2f}}}"
        for did, d in enumerate(datasets):
            should_highlight = False
            if d == "rank":
                m = round(row[d] * 10) / 10
                m_std = round(row[d + "_std"] * 10) / 10
            else:
                m = round(row[d] * (10 ** prec) * mul) / (10 ** prec)
                if method != "denoise":
                    should_highlight = m == values_to_highlight[did]
                m_std = round(row[d + "_std"] * (10 ** prec) * mul) / (10 ** prec)
            m = f"{m:.1f}"
            m_std = f"{m_std:.1f}"

            d_name = d.upper()
            if should_highlight:
                new_row[d_name] = f"\\textbf{{{m}}} {{\\footnotesize $\pm$ {m_std} }}"
            else:
                new_row[d_name] = f"{m} {{\\footnotesize $\pm$ {m_std}}}"
        rows.append(new_row)

    with open(f"{out_folder}/tables/{metric_name}-{res_suffix}-avg.tex", "w") as stats_file:
        s = tabulate(rows, headers = "keys", showindex=False, tablefmt="latex")
        s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("lllllllllll", "ll|ccccccccc").replace("rand", "\\hline rand")
        print(s, file = stats_file)

    pass

# def draw_tun2_metric(base_path: str, task:str, metric_name: str, selected_methods: dict = {}, 
#                  draw_diff = False, suffix = ""):
#     infile = os.path.join(base_path, f"{task}-bl.jsonlist")
#     with open(infile, 'r') as f:
#         json_lines = f.readlines()
#     all_metrics = [json.loads(l) for l in json_lines]
#     method_metrics = defaultdict(list)
#     for metrics in all_metrics:
#         infl_method = metrics['config']['infl_method']
#         agg_method = metrics['config']['agg_method']                
#         module_name = metrics['config']['module_name']
#         key = (infl_method, agg_method, module_name)
#         if draw_diff:
#             before_metric_values = metrics['first_finetune'][metric_name] 
#             after_metric_values = metrics[metric_name]
#             metric_values = [a - b for a, b in zip(after_metric_values, before_metric_values)]
#         else:
#             metric_values = metrics[metric_name]        
#         if key not in selected_methods:
#             continue
#         method_metrics[key].append(metric_values)

#     # max_sz = max(len(l) for l in method_metrics.values())

#     # method_metrics_flat = {k: [v3 for v2 in v for v3 in (v2[:10] + [np.nan] * max(0, 10 - len(v2)))] for k, v in method_metrics.items()}
#     # method_metric_ranks = get_avg_ranks(method_metrics_flat)

#     # setup_names = list(setup_score_values.keys())

#     # method_names = sorted(method_metrics.keys(), key = method_metric_ranks.get)
#     method_names = list(selected_methods.keys())
#     plt.ioff()

#     handles_ = []

#     for i, (infl_method, agg_method, module_name) in enumerate(method_names):
#         metrics = method_metrics[(infl_method, agg_method, module_name)]
#         metric_values = np.array(metrics) * 100
#         mean = np.mean(metric_values, axis=0)
#         confidence_level = 0.95
#         degrees_freedom = metric_values.shape[0] - 1
#         sample_standard_error = stats.sem(metric_values, axis=0)
#         confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
#         min_v = confidence_interval[0]
#         max_v = confidence_interval[1]
#         method_settings = selected_methods[(infl_method, agg_method, module_name)]
#         # default_args = dict(marker='o', markersize=4, linestyle='none', linewidth=1, color = method_settings['color'])
#         default_args = dict(marker='o', markersize=0, linewidth=0.9, color = method_settings['color'])
#         draw_full = False
#         shift = ((i - len(method_metrics) // 2) * 0.025)
#         if infl_method == 'denoise':
#             default_args['linestyle']='--'
#             default_args['markersize'] = 0
#             draw_full = True 
#             shift = 0
#         if infl_method == 'rand':
#             default_args['linestyle']='-.'
#             default_args['markersize'] = 0
#             draw_full = True 
#             shift = 0
#         # xs = np.arange(len(mean)) + 1
#         xs = np.arange(len(mean)) + 1 + shift # Shift x-coordinates slightly
#         line = plt.plot(xs, mean, zorder=1, **default_args)
#         if draw_full:
#             plt.fill_between(xs, min_v, max_v, alpha=.05, color = line[0].get_color(), linewidth=0)
#         else:
#             plt.errorbar(xs, mean, yerr=[mean - min_v, max_v - mean], fmt='none', ecolor=line[0].get_color(), capsize=1, linewidth=0.5, zorder=0)
#         handles_.append((line[0], method_settings['legend_name'], method_settings['legend_order']))
    
#     handles_.sort(key = lambda x: x[2])
#     ordered_handles = [h[0] for h in handles_]
#     ordered_labels = [h[1] for h in handles_]
#     plt.xlabel('Epoch', fontsize=20)
#     plt.ylabel('Accuracy, \\%', fontsize=20)
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     plt.legend(ordered_handles, ordered_labels, fontsize=15)
#     # plt.title(f'{task.upper()} 70\\% filtered finetuning', fontsize=20)
#     plt.title(f'{task.upper()}', fontsize=20)
#     plt.tight_layout()
#     outfile = os.path.join(base_path, "plots", f"{task}-{metric_name}-{suffix}.pdf")
#     plt.savefig(outfile)  
#     plt.clf()  

def create_tun2_agg_metrics_table(metric_name: str = "best_accuracy_1", prec = 1, 
                        out_folder = "data/roberta", datasets = benchmark,
                        highlight_max = True, ds_ranks = False, mul = 100,
                        run_ids = [0,1,2,3,4,5,6,7,8,9], agg_metrics = {"mean": "Mean", "rank-c": "Rank", "vote2-c": "Vote"},
                        res_suffix = "all"): 
    ''' Like create_tun2_metric_table, but for many agg metrics '''

    all_df = get_all_df(base_path = out_folder, datasets = datasets,
                            res_suffix = res_suffix, keep_only=[metric_name])
    
    int_df_path = os.path.join(out_folder, "metrics", f"intranks-{metric_name}.pcl")
    int_df = pd.read_pickle(int_df_path)
    int_df.reset_index(inplace=True)
    int_df.drop(columns=["module"], inplace=True)
    int_df.rename(columns={"infl": "infl_method", "agg": "agg_method", "layer": "module"}, inplace=True)
    int_df.set_index(["infl_method", "agg_method", "module"], inplace=True)

    agg_method_keys = list(agg_metrics.keys())
    agg_method_keys.append('')

    all_df = all_df[all_df["agg_method"].isin(agg_method_keys) & all_df["seed"].isin(run_ids)]

    metrics_per_ds = all_df.pivot(index=["infl_method", "agg_method", "module"], columns=["task", "seed"], values=metric_name)

    idx = pd.MultiIndex.from_tuples(metrics_per_ds.columns)

    metrics_per_ds.columns = idx
    # metrics_per_ds = metrics_per_ds.round(prec)

    # ranks = metrics_per_ds.round(prec).rank(axis = 0, ascending=False, method="average") # method="dense")
    ranks = metrics_per_ds.rank(axis = 0, ascending=False, method="average") # method="dense")

    suffix = ""
    if ds_ranks:
        # ranks.columns = idx
        # metrics_per_ds = ranks.groupby(level=0, axis=1).mean()
        metrics_per_ds = ranks
        suffix = "-rank"


    # metrics_per_ds.to_csv(f"{out_folder}/{metric_name}{suffix}-all.csv")

    metric_by_ds_mean = metrics_per_ds.groupby(level=0, axis=1).mean()
    metric_by_ds_std = metrics_per_ds.groupby(level=0, axis=1).std()
    metric_by_ds_std.columns = [c + "_std" for c in metric_by_ds_std.columns]
    metric_by_ds = pd.merge(metric_by_ds_mean, metric_by_ds_std, left_index=True, right_index=True)
    metric_by_ds['rank'] = int_df.loc[metric_by_ds.index]['rank']
    metric_by_ds['win_rate'] = int_df.loc[metric_by_ds.index]['win_rate']
    metric_by_ds['wstd'] = int_df.loc[metric_by_ds.index]['wstd']
    metric_by_ds = metric_by_ds.sort_values(by=['rank', 'win_rate', 'wstd'], ascending=[True, False, True])

    metric_by_ds = metric_by_ds[["rank", "win_rate", "wstd", *[de for d in datasets for de in [d, d + "_std"]]]]
    # metric_by_ds.to_csv(f"{out_folder}/{metric_name}{suffix}-avg.csv")

    filtered_df = metric_by_ds.loc[metric_by_ds.index.get_level_values('infl_method') != 'denoise']
    if highlight_max:
        values_to_highlight = filtered_df[datasets].to_numpy().max(axis=0)
    else:
        values_to_highlight = filtered_df[datasets].to_numpy().min(axis=0)
    
    values_to_highlight = [round(v * (10 ** prec) * mul) / (10 ** prec) for v in values_to_highlight]

    rows = []
    for row_id, row in enumerate(metric_by_ds.reset_index().to_dict(orient="records")):
        new_row = {}
        method = row["infl_method"]                    
        layer = row["module"]
        method_rename = {"datainf": "DataInf", "hf": "TracIn", "cos": "Cosine", "denoise": "Full", "rand": "Random", "hf_we_": "TracIn$_{we}$", "hf_we_topk_10": "TracIn$^{10}_{we}$"}
        new_row["Method"] = method_rename[method]
        if len(agg_metrics) > 0:
            agg_method = row["agg_method"]
            new_row["Agg"] = agg_metrics.get(agg_method, '')
        new_row["Layer"] = "all" if (method not in ["denoise", "rand"]) and layer == "" else layer 
        new_row["Rank"] = row["rank"]
        new_row["Win Rate"] = f"{row['win_rate']:.2f} {{\\footnotesize $\\pm$ {row['wstd']:.2f}}}"
        for did, d in enumerate(datasets):
            should_highlight = False
            m = round(row[d] * (10 ** prec) * mul) / (10 ** prec)
            if method != "denoise":
                should_highlight = m == values_to_highlight[did]
            m_std = round(row[d + "_std"] * (10 ** prec) * mul) / (10 ** prec)
            m = f"{m:.1f}"
            m_std = f"{m_std:.1f}"

            d_name = d.upper()

            if should_highlight:
                new_row[d_name] = f"\\textbf{{{m}}} {{\\footnotesize $\pm$ {m_std} }}"
            else:
                new_row[d_name] = f"{m} {{\\footnotesize $\pm$ {m_std}}}"
        rows.append(new_row)

    with open(f"{out_folder}/tables/tun2-{metric_name}{suffix}.tex", "w") as stats_file:
        s = tabulate(rows, headers = "keys", showindex=False, tablefmt="latex_raw")
        print(s, file = stats_file)

    pass

def create_tun2_agg_diffs_table(metric_name: str = "best_accuracy_1", prec = 1, 
                        out_folder = "data/roberta", datasets = benchmark,
                        run_ids = [0,1,2,3,4,5,6,7,8,9], agg_metrics = {"mean": "Mean", "rank-c": "Rank", "vote2-c": "Vote"},
                        res_suffix = "all"): 
    ''' Like create_tun2_metric_table, but for many agg metrics '''

    all_df = get_all_df(base_path = out_folder, datasets = datasets,
                            res_suffix = res_suffix, keep_only=[metric_name])
    
    agg_method_keys = list(agg_metrics.keys())
    # agg_method_keys.append('')

    all_df = all_df[all_df["agg_method"].isin(agg_method_keys) & all_df["seed"].isin(run_ids)]
    all_df.set_index(["agg_method", "infl_method", "module", "task", "seed"], inplace=True)

    # metrics_per_ds = all_df.pivot(index=["infl_method", "module", "seed"], columns=["task", "agg_method"], values=metric_name)


    # idx = pd.MultiIndex.from_tuples(metrics_per_ds.columns)

    # metrics_per_ds.columns = idx

    base_level_df = all_df.loc['mean'].reset_index().pivot(index=["infl_method", "module"], columns=["task", "seed"], values=metric_name)
    rank_df = all_df.loc['rank-c'].reset_index().pivot(index=["infl_method", "module"], columns=["task", "seed"], values=metric_name)
    vote_df = all_df.loc['vote2-c'].reset_index().pivot(index=["infl_method", "module"], columns=["task", "seed"], values=metric_name)
    diff_rank_df = rank_df - base_level_df
    diff_vote_df = vote_df - base_level_df
    diff_rank_avgs = diff_rank_df.groupby(level=0, axis=1).mean()
    diff_rank_stds = diff_rank_df.groupby(level=0, axis=1).std()
    diff_vote_avgs = diff_vote_df.groupby(level=0, axis=1).mean()
    diff_vote_stds = diff_vote_df.groupby(level=0, axis=1).std()

    diff_rank_avgs0 = (diff_rank_avgs * 100).round(prec)
    diff_vote_avgs0 = (diff_vote_avgs * 100).round(prec)

    diff_rank_avgs0.to_csv(f"{out_folder}/tables/tmp-rank-diff.csv")  
    diff_vote_avgs0.to_csv(f"{out_folder}/tables/tmp-vote-diff.csv")  

    pass 
    # NOTE: stats tests here

    # rank_corr = []
    # vote_corr = []
    # for infl_method, module in base_level_df.index:
    #     rank_method_dict = {}
    #     vote_method_dict = {}
    #     for task in base_level_df.columns.get_level_values(0).unique():
    #         base_series = base_level_df.loc[(infl_method, module)][task].to_numpy()
    #         rank_series = rank_df.loc[(infl_method, module)][task].to_numpy()
    #         vote_series = vote_df.loc[(infl_method, module)][task].to_numpy()
    #         wilcoxon_p = sci_stats.wilcoxon(base_series, rank_series).pvalue
    #         rank_method_dict[task] = wilcoxon_p
    #         wilcoxon_p = sci_stats.wilcoxon(base_series, vote_series).pvalue
    #         vote_method_dict[task] = wilcoxon_p
    #     rank_corr.append(rank_method_dict)
    #     vote_corr.append(vote_method_dict)
    # rank_pvalues = pd.DataFrame(rank_corr, index=base_level_df.index)
    # vote_pvalues = pd.DataFrame(vote_corr, index=base_level_df.index)
    # pass
    
    pass 


def draw_all_tun2_metric(base_path: str, tasks = benchmark, metric_name: str = "accuracy", figsize = (8, 3),
                        selected_methods: dict = {}, num_in_row = 4, draw_diff = False, suffix = ""):
    
    tasks_metrics = {}
    for task in tasks:
        infile = os.path.join(base_path, "metrics", f"{task}-bl.jsonlist")
        with open(infile, 'r') as f:
            json_lines = f.readlines()
        all_metrics = [json.loads(l) for l in json_lines]
        method_metrics = defaultdict(list)
        for metrics in all_metrics:
            infl_method = metrics['config']['infl_method']
            agg_method = metrics['config']['agg_method']                
            module_name = metrics['config']['module_name']
            key = (infl_method, agg_method, module_name)
            if draw_diff:
                before_metric_values = metrics['first_finetune'][metric_name] 
                after_metric_values = metrics[metric_name]
                metric_values = [a - b for a, b in zip(after_metric_values, before_metric_values)]
            else:
                metric_values = metrics[metric_name]        
            if key not in selected_methods:
                continue
            method_metrics[key].append(metric_values)
        tasks_metrics[task] = method_metrics

    # max_sz = max(len(l) for l in method_metrics.values())

    # method_metrics_flat = {k: [v3 for v2 in v for v3 in (v2[:10] + [np.nan] * max(0, 10 - len(v2)))] for k, v in method_metrics.items()}
    # method_metric_ranks = get_avg_ranks(method_metrics_flat)

    # setup_names = list(setup_score_values.keys())

    # method_names = sorted(method_metrics.keys(), key = method_metric_ranks.get)
    plt.ioff()

    method_names = list(selected_methods.keys())

    import math
    num_rows = math.ceil(len(benchmark) / num_in_row)

    fig, axes = plt.subplots(num_rows, num_in_row, figsize=figsize)
    # plt.subplots_adjust(wspace=0.1, hspace=-0.1)
    handles_ = []

    for i, task in enumerate(benchmark):
        ax = axes[i // num_in_row, i % num_in_row]
        ax.set_title(task.upper(), fontsize=8, pad=0)

        method_metrics = tasks_metrics[task]

        for key in method_names:
            metrics = method_metrics[key]
            metric_values = np.array(metrics) * 100
            mean = np.mean(metric_values, axis=0)
            confidence_level = 0.95
            degrees_freedom = metric_values.shape[0] - 1
            sample_standard_error = stats.sem(metric_values, axis=0)
            confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
            min_v = confidence_interval[0]
            max_v = confidence_interval[1]
            method_settings = selected_methods[key]
            # default_args = dict(marker='o', markersize=4, linestyle='none', linewidth=1, color = method_settings['color'])
            default_args = dict(marker='o', markersize=0, linewidth=1.5, color = method_settings['color'])
            # shift = ((i - len(method_metrics) // 2) * 0.025)
            show_err = True 
            if key[0] == 'denoise':
                default_args['linestyle']='--'
                default_args['linewidth']=0.9
                default_args['markersize'] = 0
                show_err = False 
            if key[0] == 'rand':
                default_args['linestyle']='-.'
                default_args['linewidth']=0.9
                default_args['markersize'] = 0
                show_err = False
            # xs = np.arange(len(mean)) + 1
            xs = np.arange(len(mean)) + 1
            line = ax.plot(xs, mean, zorder=1, **default_args)
            if show_err:
                ax.fill_between(xs, min_v, max_v, alpha=.1, color = line[0].get_color(), linewidth=0)
            if i == 0:
                handles_.append((line[0], method_settings['legend_name'], method_settings['legend_order']))

        ax.set_xticks([2,4,6,8,10])
        if (i // num_in_row) != (num_rows - 1):  # Not in the last row
            ax.set_xticklabels([])
            ax.xaxis.set_tick_params(pad=0, length=1)
        else:            
            ax.set_xticklabels([2,4,6,8,10])
            ax.xaxis.set_tick_params(pad=1, length=1)
        ax.yaxis.set_tick_params(pad=1, length=1)

        if (i // num_in_row) == (num_rows - 1):
            ax.set_xlabel('Epoch', fontsize=8, labelpad = 0)

        # if (i % num_in_row) == 0:
        #     ax.set_ylabel('Accuracy \\%', fontsize=10, labelpad = 2)

        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
          
    
    handles_.sort(key = lambda x: x[2])
    ordered_handles = [h[0] for h in handles_]
    ordered_labels = [h[1] for h in handles_]

    fig.text(0.00, 0.5, f'Accuracy,\\%', ha='left', va='center', rotation='vertical', fontsize=8)
    # plt.xlabel('Epoch', fontsize=20s)
    # plt.ylabel('Accuracy, \\%', fontsize=20)
    fig.legend(ordered_handles, ordered_labels, loc='lower center', fontsize=8,
                ncol=len(ordered_handles), borderaxespad = 0,  # Arrange all legend items in one row
                bbox_to_anchor=(0.5, 0)  # Adjust position (centered below the grid)
        )
    # plt.title(f'{task.upper()} 70\\% filtered finetuning', fontsize=20)
    # plt.title(f'{task.upper()}', fontsize=20)
    fig.tight_layout(rect=[0.015, 0.1, 1, 1], pad = 0, h_pad=0.1, w_pad=0.1)
    outfile = os.path.join(base_path, "plots", f"fig-{metric_name}-{suffix}.pdf")
    fig.savefig(outfile)  
    plt.close(fig)  

agg_methods = {
    "mean": mean_matrix_score,
    "mean-c": partial(mean_matrix_score, use_correct = True),
    # "mean-i": partial(mean_matrix_score, use_correct = False),
    "rank": rank_matrix_score,
    "rank-c": partial(rank_matrix_score, use_correct = True), 
    # "rank-i": partial(rank_matrix_score, use_correct = False),
    # "vote": partial(rank_matrix_score, rank_score_fn = vote_matrix_score),
    # "vote-c": partial(rank_matrix_score, rank_score_fn = vote_matrix_score, use_correct = True),
    
    "vote2": partial(rank_matrix_score, rank_score_fn = vote2_matrix_score),
    "vote2-c": partial(rank_matrix_score, rank_score_fn = vote2_matrix_score, use_correct = True),

    "vote2-c-10": partial(rank_matrix_score, 
                          rank_score_fn = partial(vote2_matrix_score, 
                                                  filter_perc=0.1), 
                          use_correct = True),

    "vote2-c-20": partial(rank_matrix_score, 
                          rank_score_fn = partial(vote2_matrix_score, 
                                                  filter_perc=0.2), 
                          use_correct = True),

    "vote2-c-30": partial(rank_matrix_score, 
                          rank_score_fn = partial(vote2_matrix_score, 
                                                  filter_perc=0.3), 
                          use_correct = True),

    "vote2-c-40": partial(rank_matrix_score, 
                          rank_score_fn = partial(vote2_matrix_score, 
                                                  filter_perc=0.4), 
                          use_correct = True),          

    "vote2-c-50": partial(rank_matrix_score, 
                          rank_score_fn = partial(vote2_matrix_score, 
                                                  filter_perc=0.5), 
                          use_correct = True),   

    "vote2-c-60": partial(rank_matrix_score, 
                          rank_score_fn = partial(vote2_matrix_score, 
                                                  filter_perc=0.6), 
                          use_correct = True),

    "vote2-c-70": partial(rank_matrix_score, 
                          rank_score_fn = partial(vote2_matrix_score, 
                                                  filter_perc=0.7), 
                          use_correct = True),                          

    "vote2-c-80": partial(rank_matrix_score, 
                          rank_score_fn = partial(vote2_matrix_score, 
                                                  filter_perc=0.8), 
                          use_correct = True),  

    "vote2-c-90": partial(rank_matrix_score, 
                          rank_score_fn = partial(vote2_matrix_score, 
                                                  filter_perc=0.9), 
                          use_correct = True),                            

    "vote2-c-100": partial(rank_matrix_score, 
                          rank_score_fn = partial(vote2_matrix_score, 
                                                  filter_perc=1.0), 
                          use_correct = True),                                                    

    "rmin": partial(rank_matrix_score, rank_score_fn = min_matrix_score),
    # "rmin-c": partial(rank_matrix_score, rank_score_fn = min_matrix_score, use_correct = True),


    "median": median_matrix_score,
    # "maj": maj_matrix_score,
    # "cset": cset_matrix_score,
    # "cset-c": partial(cset_matrix_score, use_correct = True),
    # "cset-2": partial(cset_matrix_score, both_sides = True),
    # "min": mean_min_matrix_score,
    # "min-20": partial(mean_min_matrix_score, min_ratio=0.2),
    
    "cmean": confident_matrix_score,
    
    # "crank": partial(confident_matrix_score, base_method_fn=rank_matrix_score),
    # # "ccset": partial(confident_matrix_score, base_method_fn=commonset_matrix_score),
    # # "ccsset": partial(confident_matrix_score, n_confident = 100, base_method_fn=partial(commonsubset_matrix_score, vote_ratio = 0.3)),
    # "dir": dir_matrix_score,
    # "cset": partial(commonset_matrix_score, vote_ratio = 0.2),
    # "csset": partial(commonsubset_matrix_score, vote_ratio = 0.3),
    # # "csmi": partial(csmi_matrix_score, vote_ratio = 0.5, descending = False)


    "mean_10": partial(mean_matrix_score, trim_ratio=0.1),
    "mean_50": partial(mean_matrix_score, trim_ratio=0.5),
    "commonset-10": partial(commonset_matrix_score, vote_ratio = 0.1),
    "commonset-30": partial(commonset_matrix_score, vote_ratio = 0.3),
    "commonset-40": partial(commonset_matrix_score, vote_ratio = 0.4),
    "commonset-80": partial(commonset_matrix_score, vote_ratio = 0.8),
    "commonsubset-10": partial(commonsubset_matrix_score, vote_ratio = 0.1, descending = False),
    "commonsubset-30": partial(commonsubset_matrix_score, vote_ratio = 0.3, descending = False),
    "commonsubset-40": partial(commonsubset_matrix_score, vote_ratio = 0.4, descending = False),
    "commonsubset-80": partial(commonsubset_matrix_score, vote_ratio = 0.8, descending = False),
    "commonsubset-40r": partial(commonsubset_matrix_score, vote_ratio = 0.4, same_class = True, descending = True),
    "commonsubset-40rr": partial(commonsubset_matrix_score, vote_ratio = 0.4, same_class = True, descending = False),
    # "silhouette": cluster_matrix_score
}

def compute_ndr_metrics_table(base_dir_path: str, task='qnli', 
                                    group_file: str = "./groups.json",
                                    infl_methods = ['hf'],
                                    agg_method_names: list[str] = ["mean"],
                                    include_total = True, levels = [30],
                                    m_prefix = "m_bl", i_prefix="i_bl", ndr_prefix = "ndr_bl",
                                    save_df = True, device = "cuda",
                                    noise_hist_bins = 10):

    if group_file != '' and group_file is not None:
        import re
        if os.path.isabs(group_file):
            group_file_full = group_file
        else:
            group_file_full = os.path.join(base_dir_path, group_file)
        with open(group_file_full, "r") as f:
            module_groups_regex = json.load(f)
        module_groups_patterns = {name: re.compile(pattern) for name, pattern in module_groups_regex.items()}
    else:
        module_groups_patterns = {}

    task_in_dir = os.path.join(base_dir_path, task)

    metric_by_method = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is list of dict of metrics
    # f30_score_by_infl = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is filtered noise in first 30 percent
    # curves_by_methods_and_layer = defaultdict(list) # key is (infl_method, agg_method, nn_module), value is list of lists of agg_method filtering scores

    noise_list_dict, trainset_labels_dict, inflset_labels_dict = load_ds_info(task_in_dir)

    inflset_logits_dict = load_m_info(task_in_dir, m_prefix)

    for file_name_with_ext in os.listdir(task_in_dir):
        if not file_name_with_ext.startswith(i_prefix):
            continue
        print(f"Processing {file_name_with_ext}")
        file_name = file_name_with_ext.split('.')[0]
        fine_name_parts = file_name[len(i_prefix):].split('_')[1:]
        *method_parts, _, run_id_str = fine_name_parts
        run_id = int(run_id_str)
        infl_method = '_'.join(method_parts)
        if infl_method not in infl_methods:
            continue
        file_path = os.path.join(task_in_dir, file_name_with_ext)
        matrix_dict = torch.load(file_path)
        module_names = list(matrix_dict.keys())
        module_and_group_names = list(module_names)
        # transpose all matrices 
        for module_name in module_names:
            matrix_dict[module_name] = matrix_dict[module_name].t().to(device) # first dim is train_sample now and second is infl val sample
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
            group_module_ids = torch.tensor(group_modules[group_name], device = device)
            module_interactions.append(all_interactions[:, group_module_ids])
            del group_module_ids

        if include_total:
            module_interactions.insert(0, all_interactions)
            module_and_group_names.insert(0, 'total')
        
        noise_list = noise_list_dict[run_id_str]
        num_noise = sum(noise_list)
        num_clean = len(noise_list) - num_noise
        filter_idxs = torch.tensor([ round(level * len(noise_list) / 100) for level in levels], device = device)
        # first_30_idx = round(filter_perc * len(noise_list))
        ideal_area = num_noise / 2 + num_clean

        noise_mask = torch.tensor(noise_list, device = device)
        noise_mask_cpu = noise_mask.cpu().float()
        trainset_labels = torch.tensor(trainset_labels_dict[run_id_str], device = device)
        inflset_labels = torch.tensor(inflset_labels_dict[run_id_str], device = device)
        inflset_logits = inflset_logits_dict[run_id_str].to(device)
        inflset_preds = torch.argmax(inflset_logits, dim = -1)
        correct_infl_preds = inflset_preds == inflset_labels
        
        scores = torch.zeros((len(agg_method_names), len(module_interactions), all_interactions.shape[0]), device = device)

        for agg_method_id, agg_method_name in enumerate(agg_method_names):
            for matrix_id, inf_matrix in enumerate(module_interactions):
                agg_method_fn = agg_methods[agg_method_name] 
                new_scores = agg_method_fn(inf_matrix, noise_mask = noise_mask, 
                                       trainset_labels = trainset_labels, inflset_labels = inflset_labels, 
                                       inflset_logits = inflset_logits, 
                                       correct_infl_preds = correct_infl_preds, task = task, run_id = run_id_str)
                scores[agg_method_id, matrix_id] = new_scores
                del new_scores
                torch.cuda.empty_cache() 

        if noise_hist_bins is not None:
            hists, bin_sizes = compute_ndr_histogram(scores, noise_mask_cpu, bins = noise_hist_bins)            
        else:
            hists = None
            bin_sizes = None
                
        del trainset_labels, inflset_labels
                
        train_ids = torch.argsort(scores, dim = -1)

        noise_tensor = noise_mask[train_ids]

        del noise_mask, noise_mask_cpu

        noise_detection_curves = torch.cumsum(noise_tensor, dim = -1, dtype = torch.float)
        noise_detection_curves /= num_noise

        auc_ndrs = noise_detection_curves.sum(dim = -1) 
        auc_ndrs /= ideal_area

        ndr_at_levels = noise_detection_curves[:, :, filter_idxs]

        auc_ndrs_cpu = auc_ndrs.cpu()
        ndr_at_levels_cpu = ndr_at_levels.cpu()
        scores_cpu = scores.cpu()
        del auc_ndrs, ndr_at_levels, noise_detection_curves, scores, train_ids

        for agg_method_id, agg_method_name in enumerate(agg_method_names):
            for module_id, module_name in enumerate(module_and_group_names):
                one_metrics = {level: ndr_at_levels_cpu[agg_method_id, module_id, level_i].item()  for level_i, level in enumerate(levels) }
                one_metrics["auc_ndr"] = auc_ndrs_cpu[agg_method_id, module_id].item()                
                one_metrics["scores"] = scores_cpu[agg_method_id, module_id].tolist()
                one_metrics["noise_mask"] = noise_list
                one_metrics["run_id"] = run_id
                if hists is not None:
                    for bin_id in range(noise_hist_bins):
                        one_metrics[f"hist_y_{bin_id}"] = hists[agg_method_id, module_id, bin_id].item()
                        one_metrics[f"hist_x_{bin_id}"] = bin_sizes[agg_method_id, module_id, bin_id].item()
                metric_by_method[(infl_method, agg_method_name, module_name)].append(one_metrics)
        torch.cuda.empty_cache() 

    # first_30_mean_std = {key: (np.mean(first_30_values), np.std(first_30_values)) for key, first_30_values in f30_score_by_infl.items() }
    # first_30_ranks = get_avg_ranks(f30_score_by_infl, ascending=False)

    # auc_rocs_mean_std = {key: (np.mean(auc_ndr), np.std(auc_ndr)) for key, auc_ndr in auc_ndr_by_infl.items() }
    # auc_ndr_ranks = get_avg_ranks(auc_ndr_by_infl, ascending=False)

    # sorted_method_keys = sorted(first_30_ranks.keys(), key = lambda x: (first_30_ranks[x], x))

    rows = []
    # for sk in sorted_method_keys:
    for key, metrics_list_of_dict in metric_by_method.items():
        infl_method, agg_method_name, module_name = key
        if module_name in module_groups_patterns or module_name == 'total':
            module_layer, module_simple_name = module_name, "all"
        else:
            module_layer, module_simple_name = get_simple_module_and_layer_name(module_name)
        for metrics_dict in metrics_list_of_dict:
            # auc_ndr = auc_ndr_by_infl[key][run_id]
            # f30_mean, f30_std = first_30_mean_std[sk]
            # auc_ndr_mean, auc_ndr_std = auc_rocs_mean_std[sk]
            # f30_rank = first_30_ranks[sk]
            # auc_ndr_rank = auc_ndr_ranks[sk]
            row_data = {
                "task": task,
                "infl": infl_method,
                "agg": agg_method_name,
                "layer": module_layer,
                "module": module_simple_name,
                **metrics_dict
            }
            rows.append(row_data)

    df = pd.DataFrame(rows).set_index(["task", "infl", "agg", "layer", "module", "run_id"])    

    if save_df:
        outfile = os.path.join(base_dir_path, f"{ndr_prefix}_{task}.pcl")
        df.to_pickle(outfile)
        print(f"Saved ndr metrics to {outfile}")

    return df

def process_ndr_table(base_path: str, tasks: list[str] = benchmark, output_ranks = False, with_row_id = False,
                        metric_name = 30, ndr_prefix = "ndr_bl", layers = None,
                        best_group_by = None, custom_suffix = "",
                        agg_method_names = None, infl_method_names = None): 
    #NOTE: metric_name in ["f30", "auc_ndr"]):

    dfs = []
    for task in tasks:
        df = pd.read_pickle(os.path.join(base_path, "ndr", f"{ndr_prefix}_{task}.pcl"))
        df.drop(columns=["scores", "noise_mask"], inplace=True)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=False)

    if infl_method_names is not None:
        df = df.loc[df.index.get_level_values('infl').isin(infl_method_names)]
        pass

    if agg_method_names is not None:
        df = df.loc[df.index.get_level_values('agg').isin(agg_method_names)]
        pass

    if layers is not None:
        df = df.loc[df.index.get_level_values('layer').isin(layers)]
        df = df.loc[df.index.get_level_values('module') == "all"]
        pass


    metric_df = df.reset_index().pivot(index=["infl", "agg", "layer", "module"], columns=["task", "run_id"], values=[metric_name])

    ranks = metric_df.rank(axis = 0, ascending=False, method="average")
    
    suffix = ""
    if output_ranks:
        metric_df = ranks
        suffix = "-rank"

    metric_by_ds_mean = metric_df.groupby(level=1, axis=1).mean()
    metric_by_ds_std = metric_df.groupby(level=1, axis=1).std()
    metric_by_ds_std.columns = [c + "_std" for c in metric_by_ds_std.columns]
    metric_by_ds = pd.merge(metric_by_ds_mean, metric_by_ds_std, left_index=True, right_index=True)
    rank_columns = ["rank", "rank_std"]
    metric_by_ds["rank"] = ranks.mean(axis=1)
    metric_by_ds["rank_std"] = ranks.std(axis=1)

    if best_group_by is not None:
        best_idxs = metric_by_ds.groupby(level=best_group_by, axis=0)['rank'].idxmin()
        metric_by_ds = metric_by_ds.loc[best_idxs]
        metric_df = metric_df.loc[best_idxs]
        ranks = metric_df.rank(axis = 0, ascending=False, method="average")
        metric_by_ds["rank"] = ranks.mean(axis=1)
        metric_by_ds["rank_std"] = ranks.std(axis=1)
    
    metric_by_ds = metric_by_ds.sort_values(by=['rank', 'rank_std'], ascending=[True, True])

    metric_by_ds = metric_by_ds[[*[de for d in tasks for de in [d, d + "_std"]], *rank_columns]]
    # metric_by_ds.to_csv(f"{base_path}/{metric_name}{suffix}-avg.csv")

    values_to_highlight = metric_by_ds[tasks].to_numpy().max(axis=0)

    rows = []
    for row_id, row in enumerate(metric_by_ds.reset_index().to_dict(orient="records")):
        new_row = {}
        infl_method = row["infl"]
        agg_method = row["agg"]
        layer = row["layer"]
        module = row["module"].replace("embed_tokens", "WE")
        if with_row_id:
            new_row["id"] = (row_id + 1)
        new_row["infl"] = infl_method
        new_row["agg"] = agg_method
        find_layer = re.match(r"layers\s(\d+)\s", module)
        if find_layer is not None:
            layer = find_layer.group(1)
            module = module.replace(find_layer.group(0), "")
        new_row["layer"] = "WE" if module == "WE" else layer
        new_row["module"] = module
        for did, d in enumerate([*tasks, "rank"]):
            should_highlight = False
            if d == "rank":
                m = round(row[d] * 10) / 10
                m_std = round(row[d + "_std"] * 10) / 10
            else:
                should_highlight = row[d] == values_to_highlight[did]
                m = round(row[d] * 1000) / 10
                m_std = round(row[d + "_std"] * 1000) / 10
            m = str(m).rstrip("0").rstrip(".").lstrip("0").replace("-0.", "-.")
            m_std = str(m_std).rstrip("0").rstrip(".").lstrip("0")

            if should_highlight:
                if m_std == "":
                    new_row[d] = f"\\textbf{{{m}}}"
                else:
                    new_row[d] = f"\\textbf{{{m}}} $\pm$ {m_std}"
            else:
                if m_std == "":
                    new_row[d] = m
                else:
                    new_row[d] = f"{m} $\pm$ {m_std}"
        rows.append(new_row)

    with open(f"{base_path}/tables/ndr-{metric_name}{suffix}{custom_suffix}-avg.tex", "w") as stats_file:
        s = tabulate(rows, headers = "keys", showindex=False, tablefmt="latex")
        s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("hf\_we\_topk\_10", "hf$^{10}_{we}$").replace("hf\_we\_", "hf$_{we}$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("lllllllllll", "ll|ccccccccc").replace("rand", "\\hline rand")\
            .replace("_10", "$^{10}$").replace("_50", "$^{50}$")\
            .replace("commonset", "cset").replace("-20", "$^{20}$").replace("-30", "$^{30}$")\
            .replace("commonsubset", "csset").replace("out_proj", "proj")\
            .replace('self_attn ', '')\
            .replace('v_proj', 'value').replace('q_proj', 'query')
        print(s, file = stats_file)

    pass

# df1 = all_score_df.loc[:,:,"total","all"]

# df1 *= 100

# df2 = get_confidence_interval(df1)

# df3 = df2.reset_index()

# df3["agg"] = df3["agg"].str.extract(r"(\d+)$").astype(int)

# df4 = df3.sort_values(by=["infl", "agg"])

# df4.to_csv(base_path + "/ndr/d4.csv")



def draw_vote_k_ndr(base_path: str, 
                        methods: list[str] = ["hf","cos","datainf"],
                        method_names: list[str] = ["TracIn","Cosine","DataInf"],
                        tasks: list[str] = benchmark, 
                        modules: list[str] = [
                            "value B",
                            "query B",
                            "value A",
                            "query A",
                        ],
                        module_names: list[str] = [
                            "Value B",
                            "Query B",
                            "Value A",
                            "Query A",
                        ],
                        metric_name = 30, 
                        num_layers=24,
                        ndr_prefix = "ndr_vote_k", 
                        custom_suffix = "vote_k",
                        agg_methods=[
                            "vote2-c-10",
                            "vote2-c-20",
                            "vote2-c-30",
                            "vote2-c-40",
                            "vote2-c-50",
                            "vote2-c-60",
                            "vote2-c-70",
                            "vote2-c-80",
                            "vote2-c-90",
                            "vote2-c-100"
                        ], figsize=(8, 6)): 
    #NOTE: metric_name in ["f30", "auc_ndr"]):

    layers = list(range(num_layers))
    layers_str = [str(l) for l in layers]

    # df = load_df(base_path, tasks, use_ndr = True, ndr_prefix = ndr_prefix, metric_name=metric_name, 
    #             agg_method_names=agg_methods, infl_method_names=methods)

    cache_path = os.path.join(base_path, f"cached_layer_ndr_{custom_suffix}.pcl")

    if os.path.exists(cache_path):
        score_df = pd.read_pickle(cache_path) 
    else:
        df = load_df(base_path, tasks, use_ndr = True, ndr_prefix = ndr_prefix, metric_name=metric_name, 
                    agg_method_names=agg_methods, infl_method_names=methods)
    
        all_score_df = df.pivot(index=["infl", "agg", "layer", "module"], columns=["task", "seed"], values=metric_name)
        score_df = all_score_df[all_score_df.index.get_level_values("layer").isin(layers_str)]
        score_df *= 100.0
        score_df.to_pickle(cache_path)

    conf_int_df = get_confidence_interval(score_df)

    # conf_int_df = conf_int_df.copy()
    # conf_int_df.index = conf_int_df.index.set_levels(
    #     conf_int_df.index.levels[conf_int_df.index.names.index("agg")]
    #     .str.extract(r'(\d+)$')[0].astype(int),
    #     level="agg"
    # )
    # conf_int_df.index = conf_int_df.index.set_levels(
    #     conf_int_df.index.get_level_values("layer").astype(int),
    #     level="layer"
    # )    


    very_min = conf_int_df.min(axis=1).min()
    very_max = conf_int_df.max(axis=1).max()    

    plt.ioff()
    fig, axes = plt.subplots(len(modules), len(methods)) #, figsize=figsize, subplot_kw={'projection': '3d'})
    fig.subplots_adjust(wspace=0, hspace=0.0, left=0.0, right=1.0, bottom=0.0, top=1.0)
    # ordered_handles = []
    # ordered_labels = []

    for i, module in enumerate(modules):
        module_name = module_names[i]
        module_df = conf_int_df.loc[:,:,:,module]

        fig.text(
            0.0,  # x near left edge
            1.0 - (i + 0.5) / len(modules),  # y per row
            module_name,
            ha='left', va='center', rotation='vertical', fontsize=10
        )

        for j, method in enumerate(methods):
            method_name = method_names[j]
            if i == 0:
                fig.text(
                    (j + 0.5) / len(methods),  # x position
                    1.0,                     # y near top
                    method_name,
                    ha='center', va='top', fontsize=10
                )

            method_df = module_df.loc[method, :, :].reset_index()
            method_df["agg"]   = method_df["agg"].str.extract(r"(\d+)$").astype(int)
            method_df["layer"] = method_df["layer"].astype(int) + 1            

            pivot = method_df.pivot(index="layer", columns="agg", values="mean")
            X, Y = np.meshgrid(pivot.columns.to_numpy(), pivot.index.to_numpy())
            Z = pivot.values

            ax = axes[i, j]
            # X = method_df["agg"].values
            # Y = method_df["layer"].values + 1
            # Z = method_df["mean"].values

            # ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5,
            #                 alpha=0.2, rstride=4, cstride=1)          

            # ax.contourf(X, Y, Z, zdir='z', offset=very_min, cmap='coolwarm')
            # ax.contourf(X, Y, Z, zdir='x', offset=0, cmap='coolwarm')
            # ax.contourf(X, Y, Z, zdir='y', offset=num_layers + 1, cmap='coolwarm')

            ax.pcolormesh(X, Y, Z, cmap='coolwarm', shading='auto')  

            ax.yaxis.set_label_position("right")
            ax.set_xticks([])
            ax.set_yticks([])            


            # ax.set(xlim=(0, 110), ylim=(0, num_layers + 1), zlim=(very_min, very_max))
            ax.set(xlim=(10, 100), ylim=(1, num_layers)) #, zlim=(very_min, very_max))
            ax.set_xlabel('Vote k, \\%', fontsize=6, labelpad=0) #-10)
            ax.set_ylabel('Layer', fontsize=6, labelpad=0) #, labelpad=-10)
            # ax.set_zlabel('NDR, \\%', fontsize=6) #, labelpad=-10)
            ax.tick_params(axis='x', labelsize=6) #, pad=-6)
            ax.tick_params(axis='y', labelsize=6) #, pad=-6)
            for spine in ax.spines.values():
                spine.set_visible(False)            
            # ax.tick_params(axis='z', labelsize=6) #, pad=-6)
            # ax.set_proj_type('ortho')   # optional: makes spacing more compact
            # ax.margins(0)

    # fig.legend(ordered_handles, ordered_labels, loc='lower center', fontsize=8,
    #             ncol=len(ordered_handles), borderaxespad = 0,  # Arrange all legend items in one row
    #             bbox_to_anchor=(0.5, 0)  # Adjust position (centered below the grid)
    #     )        
    fig.tight_layout(w_pad=0.0, h_pad=0.0, pad=0.0, rect=[0, 0, 1, 1])
    fig.savefig(os.path.join(base_path, "plots", f"layers-ndr-hm-{custom_suffix}.pdf"))
    plt.close(fig)
    pass


def process_ndr_table2(base_path: str, tasks: list[str],
                        infl_methods = [ 'datainf', 'hf', 'cos'],
                        layers = [], agg_name = "mean",
                        ndr_prefix = "ndr_bl", num_bins = 5, 
                        hist_w_delta = 1, hist_w = 10, hist_w_min = 0.5,
                        hist_gap = 0.2,
                        suffix = ""): 
    
    task_rows = []

    for task in tasks:

        df = pd.read_pickle(os.path.join(base_path, f"{ndr_prefix}_{task}.pcl"))

        full_infl_methods = [mx for m in infl_methods for mg in [["hf_we_", "hf_we_topk_10", "hf"] if m == "hf" else [m]] for mx in mg]

        selected_df = df.loc[task, full_infl_methods, agg_name, layers, 'all']
        df = selected_df.reset_index().drop(columns=['task', 'agg', 'module'])
        df2 = df.groupby(["infl", "layer"]).mean()

        max_noise_filtering = df2[[c for c in df2.columns if type(c) is str and c.startswith("hist_y_") ]].max().max()
        

        datainf_ids = [('datainf', l) for l in layers if l != 'WE']
        hf_ids = [('hf_we_', 'WE'), ('hf_we_topk_10', 'WE'), *(('hf', l) for l in layers)]
        cos_ids = [('cos', l) for l in layers]
        
        datainf_df = df2.loc[datainf_ids]
        hf_df = df2.loc[hf_ids]
        cos_df = df2.loc[cos_ids]

        method_names = ['DataInf', 'TracIn', 'Cosine']
        subtables = []    

        for m_df_i, m_df in enumerate([datainf_df, hf_df, cos_df]):

            head_row = f"{task.upper()} & \multicolumn{{3}}{{l}}{{{method_names[m_df_i]}}}"

            rows = []

            for row_id, row in enumerate(m_df.reset_index().to_dict(orient="records")):
                layer = row['layer']
                if layer == 'WE' and m_df_i == 1 and row_id == 0:
                    layer = "TracIn$_{we}$"
                if layer == 'WE' and m_df_i == 1 and row_id == 1:
                    layer = "TracIn$^{10}_{we}$"
                ndr_30 = round(row[30] * 100)
                auc_ndr = row["auc_ndr"]
                hist_y = []
                hist_x = []
                for i in range(num_bins):
                    y = row[f"hist_y_{i}"]
                    x = row[f"hist_x_{i}"]
                    hist_y.append(y)
                    hist_x.append(x)

                min_x = np.min(hist_x)
                max_x = np.max(hist_x)
                if min_x == max_x: 
                    ws = np.ones_like(hist_x)
                ws = (np.array(hist_x) - min_x) * hist_w_delta / (max_x - min_x) + hist_w_min
                ws_sum = np.sum(ws)
                ws = ws / ws_sum * hist_w
                hs = np.array(hist_y)
                # max_h = np.max(hs)
                # hs = hs / max_noise_filtering * 100
                hs = hs * 100 / max_noise_filtering
                bars = []
                cur_x = 0

                max_density = np.max(hs / np.array(hist_x))

                for i, (w, y) in enumerate(zip(ws, hs)):
                    density = round(60 + ((y / hist_x[i]) / max_density) * 40)
                    density = 100 if density > 95 else density
                    bars.append(f"\\fill[gray!{density}] ({cur_x},0) rectangle +({w},{y});")
                    cur_x += w 
                    cur_x += hist_gap
                # bars.append(f"\\path [draw=white] (0, 0) -- ({cur_x - hist_gap}, 0);")
                all_bars = "\n".join(bars)
                tikz_bars = f"\\begin{{tikzpicture}}[x=3pt,y=0.15pt,baseline=(current bounding box.center)]\n{all_bars}\n\\end{{tikzpicture}}\n"
                line = f"{layer} & {ndr_30:.0f} & {auc_ndr:.2f} & {tikz_bars}"
                rows.append(line)

            subtables.extend([head_row, *rows])
            # subtable = head_row + "\\\\ \\hline\n" + "\\\\\n".join(rows) + "\\\\ \\hline\n"
            # subtables.append(subtable)

        header_main = "Layer & NDR & AUC & Noise"
        full_table = [header_main, *subtables]
        task_rows.append(full_table)

    all_rows = [ " & ".join(cols) for cols in zip(*task_rows) ]
    head_of_heads, others = all_rows[0], all_rows[1:]

    full_table = head_of_heads + "\\\\ \\hline\n" + "\\\\\n".join(others) + "\\\\ \\hline\n"

    with open(f"{base_path}/ndr2-{suffix}.tex", "w") as stats_file:
        print(full_table, file = stats_file)

    pass

def draw_one_noise_distr(ax, plot_data, bar_w_delta = 0.1, bar_w = 10, min_w = 0.5, max_y = 100, auc_ndr_max = 1):
    auc_ndr = plot_data["auc_ndr"]
    ndr_30 = plot_data[30]
    ys = plot_data[[c for c in plot_data.index if type(c) is str and c.startswith("hist_y_")]].to_numpy()
    ws = plot_data[[c for c in plot_data.index if type(c) is str and c.startswith("hist_x_")]].to_numpy()    
    ws *= bar_w
    ws = np.where(ws < min_w, min_w, ws)
    ws_sum = np.sum(ws)
    ws = ws / ws_sum * bar_w
    xs = np.insert(np.cumsum(ws), 0, 0)[:-1]
    xs += np.arange(len(xs)) * bar_w_delta
    ys *= 100
    ax.set_ylim(0, max_y)
    ax.bar(xs, ys, ws, align='edge', color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)  # Remove tick marks
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    auc_ndr_str = f"{auc_ndr:.2f}".lstrip("0")
    if auc_ndr == auc_ndr_max:
        auc_ndr_str = f"\\textbf{{{auc_ndr_str}}}"
    ax.text(0.98, 0.90, f"AUC={auc_ndr_str}", ha='right', va='top', fontsize=8, transform=ax.transAxes, color='black')

    pass

def reset_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)  # Remove tick marks
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    pass

def draw_noise_distr(base_path: str, tasks: list[str] = benchmark,
                        infl_methods = [ 'datainf', 'hf', 'cos'],
                        layers = [], agg_name = "mean", figsize = (8, 8),
                        ndr_prefix = "ndr_bl", num_bins = 10, no_left_no_bottom = False,
                        suffix = ""): 
    ''' Draws hists: y-ax is dataset, method, x-ax - layers '''

    full_infl_methods = [mx for m in infl_methods for mg in [["hf_we_", "hf_we_topk_10", "hf"] if m == "hf" else [m]] for mx in mg]

    dfs = []
    for task in tasks:
        df = pd.read_pickle(os.path.join(base_path, "ndr", f"{ndr_prefix}_{task}.pcl"))
        df.drop(columns=["scores", "noise_mask"], inplace=True)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=False)

    df = df.loc[tasks, full_infl_methods, agg_name, layers, 'all']
    df = df[[30, "auc_ndr", *(f"hist_x_{i}" for i in range(num_bins) ), *(f"hist_y_{i}" for i in range(num_bins) )]]

    df = df.reset_index().drop(columns=['agg', 'module', 'run_id'])

    grouped_df = df.groupby(["task", "infl", "layer"])
    mean_df = grouped_df.mean()
    min_df = grouped_df.min()
    max_df = grouped_df.max()

    plt.ioff()
    simple_methods = ['datainf', 'hf', 'cos']
    if 'WE' in layers:
        column_methods = [
            ['', '', '', *(['datainf'] * (len(layers) - 1))], #datainf
            ['hf_we_', 'hf_we_topk_10', *(['hf'] * len(layers))], #hf
            ['', '', *(['cos'] * len(layers))], #cos
        ]
        column_layers = ['WE', 'WE', *layers]
    else:
        column_methods = [
            ['datainf'] * len(layers),
            ['hf'] * len(layers),
            ['cos'] * len(layers)
        ]
        column_layers = layers
    fig, axes = plt.subplots(len(tasks) * len(column_methods), len(column_layers), figsize=figsize)

    for i, task in enumerate(tasks):
        task_df = mean_df.loc[task]
        ys_max = task_df[[c for c in task_df.columns if type(c) is str and c.startswith("hist_y_")]].max().max() * 100
        ys_max += 5
        for j, methods in enumerate(column_methods):
            mthd = simple_methods[j]
            if mthd == 'hf':
                mthd = ['hf_we_', 'hf_we_topk_10', 'hf']
            else:
                mthd = [mthd]
            auc_ndr_max = mean_df.loc[task].loc[mthd]['auc_ndr'].to_numpy().max()
            for k, (method, layer) in enumerate(zip(methods, column_layers)):
                ax = axes[i * len(column_methods) + j, k]
                if method == '':
                    reset_plot(ax)
                    continue
                plot_data = mean_df.loc[task, method, layer]
                draw_one_noise_distr(ax, plot_data, max_y=ys_max, auc_ndr_max = auc_ndr_max)
    
    # fig.legend(ordered_handles, ordered_labels, loc='lower center', fontsize=10,
    #             ncol=len(ordered_handles),  # Arrange all legend items in one row
    #             bbox_to_anchor=(0.5, -0.01)  # Adjust position (centered below the grid)
    #     )
    # plt.title(f'{task.upper()} 70\\% filtered finetuning', fontsize=20)
    # plt.title(f'{task.upper()}', fontsize=20)
    # fig.tight_layout(rect=[0, 0.05, 1, 1], h_pad=0.2, w_pad=0.2)

    left_delta = 0.02
    right_delta = 0.02
    top_delta = 0.01
    bottom_delta = 0.005
    if no_left_no_bottom:
        right_delta = 0
        left_delta = 0.01
        top_delta = 0.02 
        bottom_delta = 0.01
    # fig_delta = 0.01 if no_left_no_bottom else 0.02
    x_fig_sz = 1 - left_delta - right_delta
    y_fig_sz = 1 - top_delta - bottom_delta

    for i in range(1, len(tasks)):
        y_pos = bottom_delta + y_fig_sz * (i / len(tasks))
        fig.add_artist(plt.Line2D([0, 1], [y_pos, y_pos], color='black', linewidth=1))

    for i, task in enumerate(tasks):    
        # y_start = i / len(tasks)
        y_middle = (1 - top_delta) - ((i + 0.5) / len(tasks)) * y_fig_sz
        y_start = (1 - top_delta) - (i / len(tasks)) * y_fig_sz
        y_width = y_fig_sz / len(tasks)
        if not no_left_no_bottom:
            fig.text(0.00, y_middle, f'{task.upper()}', ha='left', va='center', rotation='vertical', fontsize=10)
        for j, method in enumerate(['DataInf', 'TracIn', 'Cosine']):
            m_width = y_width / 3
            m_middle = y_start - (j + 0.5) * m_width
            m_start = y_start - j * m_width

            if no_left_no_bottom:
                fig.text(0, m_middle, f'{method}', ha='left', va='center', rotation=90, fontsize=10)
                if j != 0:
                    fig.add_artist(plt.Line2D([0, 1], [m_start, m_start], color='gray', linewidth=0.5, linestyle='--'))            

            else:
                fig.text(1, m_middle, f'{method}', ha='right', va='center', rotation=270, fontsize=7)

            if j != 0:
                y_line = y_start - j * m_width

                for i in range(1, len(tasks)):
                    fig.add_artist(plt.Line2D([left_delta, 1], [y_line, y_line], color='gray', linewidth=0.5, linestyle='--'))

    layer_names = ['TracIn$_{we}$','TracIn$^{10}_{we}$', *layers]
    for i, layer_name in enumerate(layer_names):
        
        w = x_fig_sz / len(layer_names)
        w_middle = left_delta + (i + 0.5) * w
        fig.text(w_middle, 1, layer_name, ha='center', va='top', fontsize=8)
        # if not no_left_no_bottom:
        #     fig.text(w_middle, 0.00, layer_name, ha='center', va='bottom', fontsize=8)
        if i > 0:
            x_line = left_delta + i * w
            fig.add_artist(plt.Line2D([x_line, x_line], [0, 1], color='gray', linewidth=0.5, linestyle='--'))

    # fig.subplots_adjust(hspace=0, wspace=0, top=1, bottom=0, left=0, right=1)
    fig.tight_layout(pad = 0, rect=(left_delta, bottom_delta, 1 - right_delta, 1 - top_delta))
    outfile = os.path.join(base_path, "plots", f"noise-{suffix}.pdf")
    fig.savefig(outfile)  
    plt.close(fig)  

    pass



def draw_ndr_curve(ys: np.ndarray, xs:np.ndarray, ylegend:list[str], title:str, outfile: str,
                    xaxis_line = 30, noise_ratio = 20): 
    ''' 
        ys - 3d, method * levels * run_ids
        xs - measure points, len(xs) == len(values)
        len(ylegend) == len(method)

        xs and ys are in range [0, 100]
    '''
    plt.ioff()
    xs = np.concatenate([[0], xs, [100]])
    for method_id, method in enumerate(ylegend):
        y = ys[method_id].T
        y_mean = np.nanmean(y, axis=0)
        confidence_level = 0.95
        degrees_freedom = y.shape[0] - 1
        sample_standard_error = stats.sem(y, axis=0, nan_policy = 'omit')
        confidence_interval = stats.t.interval(confidence_level, degrees_freedom, y_mean, sample_standard_error)
        min_v = confidence_interval[0]
        max_v = confidence_interval[1]
        y_mean = np.concatenate([[0], y_mean, [100]])
        min_v = np.concatenate([[0], min_v, [100]])
        max_v = np.concatenate([[0], max_v, [100]])
        plt.plot(xs, y_mean, label=method)
        plt.fill_between(xs, min_v, max_v, alpha=.1, linewidth=0)
    plt.axvline(x=xaxis_line, color='r', linestyle='--', linewidth=1)
    plt.axhline(y=100, color='gray', linestyle='--', linewidth=0.5)
    # best ndr asymptote
    plt.plot([0, noise_ratio], [0, 100], color='gray', linestyle='--', linewidth=0.5)

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('Data inspected (\\%)', fontsize=20)
    plt.ylabel('Detection Rate (\\%)', fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile)  
    plt.clf()      

def draw_ndr_curves(base_path: str, tasks: list[str] = benchmark, levels = [5,10,15,20,25,30,35,40,45,50,60,70,80,90], 
                        ndr_prefix = "ndr_bl",
                        selected_methods: dict[str, str] = {
                            "hf:mean:0:query A": "hf, mean, L0, query A", 
                            "hf:mean:total:all": "hf, mean, total",  
                            "datainf:mean:total:all": "datainf, mean, total",
                        }): 
    #NOTE: metric_name in ["f30", "auc_ndr"]):

    selected_methods = {tuple(m.split(":")):v  for m, v in selected_methods.items()}
    selected_method_ids = list(selected_methods.keys())

    for task in tasks:

        df = pd.read_pickle(os.path.join(base_path, f"{ndr_prefix}_{task}.pcl"))
        df = df.reset_index().drop(columns=["task", "auc_ndr"])
        metric_df = df.pivot(index=["infl", "agg", "layer", "module"], columns=["run_id"], values=levels)
        selected_df = metric_df.loc[selected_method_ids]
        ys = selected_df.to_numpy().reshape(len(selected_method_ids), len(levels), -1) * 100.0

        outfile = os.path.join(base_path, "plots", f"ndr-curve-{task}.pdf")
        draw_ndr_curve(ys, levels, list(selected_methods.values()), f"{task.upper()}", outfile)


    pass

def draw_tun2_bar_metric(base_path: str, tasks:list[str] = benchmark, metric_name: str = "accuracy",
                         selected_methods: dict = {}, from_method = None, suffix = "", title = None):
    all_task_means = []
    all_task_std = []
    categories = [v for k, v in selected_methods.items() if k != from_method] 
    for task in tasks:
        task_means = []
        task_std = []
        infile = os.path.join(base_path, "metrics", f"{task}-bl.jsonlist")
        with open(infile, 'r') as f:
            json_lines = f.readlines()
        all_metrics = [json.loads(l) for l in json_lines]
        method_metrics = defaultdict(list)
        for metrics in all_metrics:
            infl_method = metrics['config']['infl_method']
            agg_method = metrics['config']['agg_method']                
            module_name = metrics['config']['module_name']
            key = (infl_method, agg_method, module_name)
            if key not in selected_methods:
                continue

            # before_metric_values = metrics['first_finetune'][metric_name] 
            # after_metric_values = metrics[metric_name]
            # metric_value = np.max([a - b for a, b in zip(after_metric_values, before_metric_values)])

            before_metric_value = np.max(metrics['first_finetune'][metric_name] )
            after_metric_value = np.max(metrics[metric_name])
            metric_value = after_metric_value - before_metric_value

            method_metrics[key].append(metric_value)

        delta = None 
        if from_method is not None:
            delta = np.array(method_metrics[from_method])

        for method_id, key in enumerate(selected_methods.keys()):
            if key == from_method:
                continue
            metrics = method_metrics[key]
            if delta is not None:
                metric_values = (np.array(metrics) - delta[:len(metrics)]) * 100
            else:
                metric_values = np.array(metrics) * 100
            # mean = np.mean(metric_values)            
            # std = np.std(metric_values)
            mean = np.mean(metric_values)            
            std = 0

            task_means.append(mean)
            task_std.append(std)
        all_task_means.append(task_means)
        all_task_std.append(task_std)

    all_task_means = np.array(all_task_means)
    all_task_std = np.array(all_task_std)

    xs = np.arange(len(tasks)) * 2.5

    bar_width = 0.3

    plt.ioff()

    for i, category in enumerate(categories):
        plt.bar(xs + i * bar_width, 
                all_task_means[:, i], 
                # yerr=all_task_std[:, i], 
                width=bar_width, 
                # error_kw=dict(lw=0.5, capsize=1, capthick=1),
                label=category)

    # plt.xlabel('Datasets', fontsize=20)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    plt.ylabel('$\\Delta$ Acc, \\%', fontsize=15)
    plt.xticks(xs + bar_width * (len(categories) - 1) / 2, tasks, rotation=45, fontsize=10)  # Center group labels
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    # plt.title(f'{task.upper()} 70\\% filtered finetuning', fontsize=20)
    if title:
        plt.title(title, fontsize=15)
    # plt.tight_layout()
    outfile = os.path.join(base_path, "plots", f"{metric_name}{suffix}-diffs.pdf")
    plt.savefig(outfile)  
    plt.clf()  


def draw_all_tun2_box_metric(base_path: str, tasks:list[str] = benchmark, metric_name: str = "best_accuracy_1", figsize = (8, 5),
                             agg_method = "mean", num_in_rows = 4,
                             layers: list[str] = [], suffix = "",
                             res_suffix = "bl"):

    df = get_all_df(base_path, tasks, res_suffix=res_suffix, keep_only=[metric_name])
    df = df[df["agg_method"] == agg_method].drop(columns=["agg_method"])
    metrics_per_ds = df.pivot(index=["task", "seed"], columns=["infl_method", "module"], values=metric_name)
    metrics_per_ds = metrics_per_ds * 100.0
    # metric_by_ds_mean = metrics_per_ds.groupby(level=0, axis=1).mean() * 100

    # datainf_df = metrics_per_ds.loc[('datainf', agg_method, layers)]
    # hf0_df = metrics_per_ds.loc[('hf_we_', agg_method, ['WE'])]
    # hf1_df = metrics_per_ds.loc[('hf_we_topk_10', agg_method, ['WE'])]
    # hf2_df = metrics_per_ds.loc[('hf', agg_method, layers)]
    # hf_df = pd.concat([hf0_df, hf1_df, hf2_df], axis=0)
    # cos_df = metrics_per_ds.loc[('cos', agg_method, layers)]
    # pass

    # datainf_data = datainf_df.to_numpy()
    # import math 
    # datainf_min = math.floor(datainf_data.min() / 5) * 5
    # datainf_max = math.ceil(datainf_data.max() / 5) * 5
    # hf_data = hf_df.to_numpy()
    # hf_min = math.floor(hf_data.min() / 5) * 5
    # hf_max = math.ceil(hf_data.max() / 5) * 5
    # cos_data = cos_df.to_numpy()
    # cos_min = math.floor(cos_data.min() / 5) * 5
    # cos_max = math.ceil(cos_data.max() / 5) * 5

    # categories_list = [[v for k, v in ms.items()] for ms in selected_methods] 
        

    # bar_width = 0.3

    # xs = 3 * (np.arange(len(tasks)) + 1)
    plt.ioff()    

    # methods = [datainf_data, hf_data, cos_data] 
    # methods_name = ['DataInf', 'TracIn', 'Cosine']
    # method_mins = [datainf_min, hf_min, cos_min]
    # method_maxs = [datainf_max, hf_max, cos_max]
    
    import math
    num_rows = math.ceil(len(tasks) / num_in_rows)
    fig, axes = plt.subplots(num_rows, num_in_rows, figsize=figsize)

    width = 1
    for i, task in enumerate(tasks):
        ax = axes[i // num_in_rows, i % num_in_rows]
        ax.set_title(task.upper(), fontsize=10, pad = 2)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        # first column 
        # if i % num_in_rows == 0:
        #     ax.set_ylabel('Accuracy \\%', fontsize=15, labelpad = 2)
        
        data = metrics_per_ds.loc[task].T

        datainf_np = data.loc[[('datainf', l) for l in layers if l != 'WE']].to_numpy().tolist()
        datainf_np = [[y for y in d if not np.isnan(y)] for d in datainf_np]
        hf0_np = data.loc[('hf_we_', 'WE')].to_numpy().tolist()
        hf1_np = data.loc[('hf_we_topk_10', 'WE')].to_numpy().tolist()
        hf2_np = data.loc[[('hf', l) for l in layers]].to_numpy().tolist()
        hf_np = [hf0_np, hf1_np, *hf2_np]
        hf_np = [[y for y in d if not np.isnan(y)] for d in hf_np]
        cos_np = data.loc[[('cos', l) for l in layers]].to_numpy().tolist()
        cos_np = [[y for y in d if not np.isnan(y)] for d in cos_np]

        colors = plt.cm.Set3.colors 

        def set_props(bplot, delta):
            for i, patch in enumerate(bplot['boxes']):
                patch.set_facecolor(colors[(i + delta) % len(colors)])
                patch.set_edgecolor('black')
                patch.set_linewidth(0.5)
            for whisker in bplot['whiskers']:
                whisker.set_linewidth(0.5)
                whisker.set_color('black')
            for cap in bplot['caps']:
                cap.set_linewidth(0.5)
                cap.set_color('black')

            for line in bplot['medians']:
                line.set_linewidth(0)
                line.set_color('black')                

        bplot1 = ax.boxplot(datainf_np, positions=np.arange(1, len(datainf_np) +  1), widths=width, patch_artist=True, showmeans=True, meanline=True, whis=0, showfliers=False, meanprops={'color': 'black', 'linewidth': 1})
        set_props(bplot1, 3)

        ax.plot(np.arange(1, len(datainf_np) +  1), [np.mean(d) for d in datainf_np], color='black', linestyle='--', linewidth=0.5)

        ax.axvline(x=len(datainf_np) + 1.5, color='gray', linestyle='--', linewidth=0.5)

        bplot2 = ax.boxplot(hf_np, positions= len(datainf_np) + 2 + np.arange(1, len(hf_np) +  1), widths=width, patch_artist=True, showmeans=True, meanline=True, whis=0, showfliers=False, meanprops={'color': 'black', 'linewidth': 1})
        set_props(bplot2, 0)

        ax.plot(len(datainf_np) + 2 + np.arange(1, len(hf_np) +  1), [np.mean(d) for d in hf_np], color='black', linestyle='--', linewidth=0.5)

        ax.axvline(x=len(datainf_np) + 2 + len(hf_np) + 1.5, color='gray', linestyle='--', linewidth=0.5)

        bplot3 = ax.boxplot(cos_np, positions= len(datainf_np) + 2 + len(hf_np) + 2 + np.arange(1, len(cos_np) +  1), widths=width, patch_artist=True, showmeans=True, meanline=True, whis=0, showfliers=False, meanprops={'color': 'black', 'linewidth': 1})
        set_props(bplot3, 2)

        ax.plot(len(datainf_np) + 2 + len(hf_np) + 2 + np.arange(1, len(cos_np) +  1), [np.mean(d) for d in cos_np], color='black', linestyle='--', linewidth=0.5)

        ax.set_xticks([(len(datainf_np) + 1) / 2, len(datainf_np) + 2 + (len(hf_np) + 1) / 2, len(datainf_np) + 2 + len(hf_np) + 2 + (len(cos_np) + 1) / 2])
        ax.set_xticklabels([])

        # last row 
        if i // num_in_rows == num_rows - 1:
            ax.set_xticklabels(['DataInf', 'TracIn', 'Cosine'], fontsize=8)

        ax.yaxis.set_tick_params(pad=1)
        

        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)            

        

    # for i, category in enumerate(categories):
    #     plt.bar(xs + i * bar_width, 
    #             all_task_means[:, i], 
    #             # yerr=all_task_std[:, i], 
    #             width=bar_width, 
    #             # error_kw=dict(lw=0.5, capsize=1, capthick=1),
    #             label=category)

    # plt.xlabel('Datasets', fontsize=20)
    # plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    # plt.ylabel('$\\Delta$ Acc, \\%', fontsize=15)

    # fig.legend(['', '', *layers], loc='lower center', fontsize=10,
    #             ncol=len(layers),  # Arrange all legend items in one row
    #             bbox_to_anchor=(0.5, -0.01)  # Adjust position (centered below the grid)
    #     )
    # plt.title(f'{task.upper()} 70\\% filtered finetuning', fontsize=20)
    # plt.title(f'{task.upper()}', fontsize=20)

    colors = [colors[i] for i in range(2 + len(layers))]
    labels = ['TracIn$_{we}$', 'TracIn$^{10}_{we}$', *layers]

    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=color, label=title) for color, title in zip(colors, labels)]

    fig.legend(handles = legend_handles, loc='lower center', fontsize=10,
                ncol=len(legend_handles),  # Arrange all legend items in one row
                bbox_to_anchor=(0.5, -0.01),
                columnspacing = 0.75  # Adjust position (centered below the grid)
        )

    fig.text(0.01, 0.5, 'Accuracy \\%', ha='center', va='center', rotation='vertical', fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 1], h_pad=0.2, w_pad=0.2)

    # plt.tight_layout()
    outfile = os.path.join(base_path, "plots", f"box-{metric_name}{suffix}.pdf")
    fig.savefig(outfile)  
    plt.close(fig)  
    pass

def where_is_the_noise(base_dir_path: str, task: str, infl_method: str, 
                       module_pattern: str, bins = 10,
                       i_prefix = 'i_b', m_prefix = "m_b", device = "cuda"):
    ''' Works with infl matrix, pick corresponding layers, does simple aggregation 
        and builds the histograms of noise distribution for each val sample '''
    module_pattern = re.compile(module_pattern)
    task_in_dir = os.path.join(base_dir_path, task)

    histograms_by_run_ids = defaultdict(list) # run_id -> histogram

    noise_list_dict, trainset_labels_dict, inflset_labels_dict = load_ds_info(task_in_dir)

    inflset_logits_dict = load_m_info(task_in_dir, m_prefix)

    for file_name_with_ext in os.listdir(task_in_dir):
        if not file_name_with_ext.startswith(i_prefix):
            continue
        file_name = file_name_with_ext.split('.')[0]
        fine_name_parts = file_name[len(i_prefix):].split('_')[1:]
        *method_parts, _, run_id_str = fine_name_parts
        # run_id = int(run_id_str)
        file_infl_method = '_'.join(method_parts)
        if file_infl_method != infl_method:
            continue
        file_path = os.path.join(task_in_dir, file_name_with_ext)
        matrix_dict = torch.load(file_path)
        file_module_names = list(matrix_dict.keys())
        filtered_module_names = [name for name in file_module_names if module_pattern.match(name)]
        if len(filtered_module_names) == 0:
            continue
        # transpose all matrices 
        filtered_matrix_dict = {}
        for module_name in filtered_module_names:
            filtered_matrix_dict[module_name] = matrix_dict[module_name].to(dtype=torch.float, device = device) # first dim is train_sample now and second is infl val sample
        del matrix_dict
        all_interactions = torch.stack([filtered_matrix_dict[module_name] for module_name in filtered_module_names], dim = 1)
        for int_matrix in filtered_matrix_dict.values():
            del int_matrix
        # now the dimensions is train_sample * module * infl_val_sample

        # create module views
        module_interactions = torch.mean(all_interactions, dim = 1).numpy()
        del all_interactions
        
        # infl_predictions = torch.argmax(inflset_logits_dict[run_id_str], dim=-1).to(device = device).numpy()
        ilogits = inflset_logits_dict[run_id_str].to(dtype=torch.float, device = device).numpy()
        infl_labels = np.array(inflset_labels_dict[run_id_str])
        infl_predictions1 = (ilogits[:, 1] > ilogits[:, 0]).astype(int)
        gold_logits = ilogits[np.arange(ilogits.shape[0]), infl_labels]
        logit_dist = gold_logits - ilogits[np.arange(ilogits.shape[0]), 1 - infl_labels]
        n_confident = min(50, np.sum(logit_dist > 0))
        most_confident_infl_ids = np.argsort(logit_dist)[-n_confident:]
        best_logit_dist = logit_dist[most_confident_infl_ids]
        # trainset_labels = np.array(trainset_labels_dict[run_id_str])
        infl_error_mask1 = (infl_predictions1 != infl_labels)
        # if mid_ground < 1 - probably the NN is badly trained
        infl_error_mask2 = np.ones_like(infl_error_mask1)
        infl_error_mask2[most_confident_infl_ids] = False
        # train_mask = (trainset_labels != infl_labels)
        noise_list = noise_list_dict[run_id_str]
        noise_mask = np.array(noise_list, dtype=float)
        histograms = np.zeros((module_interactions.shape[0], bins), dtype=float)     
        bin_sizes_all = np.zeros((module_interactions.shape[0], bins), dtype=float)     

        for infl_id in range(module_interactions.shape[0]):
            bin_edges = np.quantile(module_interactions[infl_id], np.linspace(0, 1, bins + 1))
            count_in_bin = np.sum((module_interactions[infl_id] >= bin_edges[0]) & (module_interactions[infl_id] <= bin_edges[1]))
            hist, _ = np.histogram(module_interactions[infl_id], bins = bin_edges, weights=noise_mask, density=False)
            histograms[infl_id] = hist
            bin_sizes_all[infl_id] = bin_edges[1:] - bin_edges[:-1]

        hist_mask = ~(histograms[:, 0] > 2 * histograms[:, -1])
        ratio_in_errors1 = np.sum(hist_mask & infl_error_mask1) / np.sum(hist_mask)
        ratio_in_errors2 = np.sum(hist_mask & infl_error_mask2) / np.sum(hist_mask)
        sel_infl_pred = ilogits[hist_mask]
        sel_infl_labels = infl_labels[hist_mask]
        sel_infl_pred_no_err = ilogits[hist_mask & ~infl_error_mask1]
        sel_infl_labels_no_err = infl_labels[hist_mask & ~infl_error_mask1]
        hist_no_err = histograms[hist_mask & ~infl_error_mask1]
        bin_sizes_no_err = bin_sizes_all[hist_mask & ~infl_error_mask1]
        print(f"Ratios {ratio_in_errors1:.2f} {ratio_in_errors2:.2f} {np.sum(~infl_error_mask2)}")
        histograms_by_run_ids[run_id_str].append(histograms)

        torch.cuda.empty_cache() 

    return histograms_by_run_ids

    # first_30_mean_std = {key: (np.mean(first_30_values), np.std(first_30_values)) for key, first_30_values in f30_score_by_infl.items() }
    # first_30_ranks = get_avg_ranks(f30_score_by_infl, ascending=False)

    # auc_rocs_mean_std = {key: (np.mean(auc_ndr), np.std(auc_ndr)) for key, auc_ndr in auc_ndr_by_infl.items() }
    # auc_ndr_ranks = get_avg_ranks(auc_ndr_by_infl, ascending=False)

    # sorted_method_keys = sorted(first_30_ranks.keys(), key = lambda x: (first_30_ranks[x], x))

#NOTE: not finished
# def draw_cancellation(task: str, base_path:str):
#     ds_path = os.path.join(base_path, f"{task}-bl.jsonlist")
#     with open(ds_path, 'r') as f:
#         json_lines = f.readlines()
#     all_metrics = [json.loads(l) for l in json_lines]
    
#     for metrics in all_metrics:
#         print(metrics)
#         pass

network_layers = {
    "roberta": ['WE', '00-05', '06-11', '12-17', '18-23', 'CL'],
    "llama": ['WE', '00-03', '04-07', '08-11', '12-15', 'CL'],
    "mistral": ['WE', '00-07', '08-15', '16-23', '24-31', 'CL'],
    "qwen": ['WE', '00-06', '07-13', '14-20', '21-27', 'CL'],
}

network_modules = {
    "roberta": {"query A": "Query A", "query B": "Query B", 
                "value A": "Value A", "value B": "Value B"},
    "llama": {"self_attn q_proj A": "Query A", "self_attn q_proj B": "Query B", 
                "self_attn v_proj A": "Value A", "self_attn v_proj B": "Value B"},
    "qwen": {"self_attn q_proj A": "Query A", "self_attn q_proj B": "Query B", 
                "self_attn v_proj A": "Value A", "self_attn v_proj B": "Value B"},
    "mistral": {"self_attn q_proj A": "Query A", "self_attn q_proj B": "Query B", 
                "self_attn v_proj A": "Value A", "self_attn v_proj B": "Value B"},                                
}

network_best_ndr = {
    "roberta": "value B",
    "mistral": "self_attn v_proj B",
    "qwen": "self_attn v_proj B",
    "llama": "self_attn v_proj B",
}

network_layer_count = {
    "roberta": 24,
    "llama": 16,
    "mistral": 32,
    "qwen": 28
}

def ndr_test():
    run_id = 2
    base_path = "data/roberta/ndr/test"
    ndr_path = os.path.join(base_path, f"ndr_bl_qnli.pcl")
    df = pd.read_pickle(ndr_path)
    scores_path = os.path.join(base_path, f"s_bl_{run_id}.pt")
    scores_dict = torch.load(scores_path)
    pass

# import gc
def compute_corr_matrix(base_path: str, tasks = benchmark,
                        ndr_prefix = "ndr_bl", agg_method = "mean",
                        infl_method_names = ['cos', 'hf', 'datainf'],
                        selected_layers = ['WE'],
                        run_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        noise_only = True):
    ''' For each of tasks computes noise correlation from ndr recorded scores 
        to establish how similar the distribution of noise across the modules by different methods
    '''
    task_dfs = {}
    for task in tasks:
        dfs = task_dfs.setdefault(task, {})
        df = pd.read_pickle(os.path.join(base_path, "ndr", f"{ndr_prefix}_{task}.pcl"))
        keys = set((i, l) for (t, i, a, l, m, r) in df.index if a == agg_method and m == 'all')
        for run_id in run_ids:
            for i, l in keys:
                df0 = df.loc[(task, i, agg_method, l, 'all', run_id)]
                scores = np.array(df0['scores'])
                if noise_only:
                    noise_mask = np.array(df0['noise_mask'])
                    scores = scores[noise_mask]
                dfs.setdefault((i, l), []).extend(scores)
        del df                   
        # gc.collect()
    for task in task_dfs:
        dfs = task_dfs[task]
        data = list(dfs.values())
        keys = list(dfs.keys())
        df = pd.DataFrame(data, index =  pd.MultiIndex.from_tuples(keys))
        task_dfs[task] = df
        pass


    inf_method_layers = []
    for infl_method in infl_method_names:
        for layer in selected_layers:
            if infl_method == 'hf' and layer == 'WE':
                inf_method_layers.append(('hf_we_', layer))
                inf_method_layers.append(('hf_we_topk_10', layer))
                inf_method_layers.append((infl_method, layer))
            elif infl_method == 'datainf' and layer == 'WE':
                continue
            else:
                inf_method_layers.append((infl_method, layer))

    per_task_data = {}
    for task in tasks:
        setup_data = per_task_data.setdefault(task, {})
        df = task_dfs[task]
        for infl_method, layer in inf_method_layers: 
            setup_name = (infl_method, layer)
            infl_df = df.loc[setup_name].to_numpy()
            setup_data.setdefault(setup_name, []).extend(infl_df)

    prefix = "noise" if noise_only else "score"
    for task in tasks:
        setup_data = per_task_data[task]
        scores_df = pd.DataFrame(setup_data)
        scores_corr = scores_df.corr(method='spearman')
        scores_corr.to_pickle(os.path.join(base_path, "ndr", f"{prefix}-corr-{task}.pcl"))
        pass
    pass

def compute_all_corr_matrix(base_path: str, tasks = benchmark,
                        ndr_prefix = "ndr_bl", agg_method = "mean",
                        infl_method_names = ['cos', 'hf', 'datainf'],
                        selected_layers = ['WE'],
                        run_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        noise_only = True):
    ''' For each of tasks computes noise correlation from ndr recorded scores 
        to establish how similar the distribution of noise across the modules by different methods
    '''
    # dfs = [ pd.read_pickle(os.path.join(base_path, "ndr", f"{ndr_prefix}_{task}.pcl")) for task in tasks ]
    # df = pd.concat(dfs, ignore_index=False)
    setup_data = {}

    task_dfs = {}
    for task in tasks:
        dfs = task_dfs.setdefault(task, {})
        df = pd.read_pickle(os.path.join(base_path, "ndr", f"{ndr_prefix}_{task}.pcl"))
        keys = set((i, l) for (t, i, a, l, m, r) in df.index if a == agg_method and m == 'all')
        for run_id in run_ids:
            for i, l in keys:
                df0 = df.loc[(task, i, agg_method, l, 'all', run_id)]
                scores = np.array(df0['scores'])
                if noise_only:
                    noise_mask = np.array(df0['noise_mask'])
                    scores = scores[noise_mask]
                dfs.setdefault((i, l), []).extend(scores)
        del df            

    inf_method_layers = []
    for infl_method in infl_method_names:
        for layer in selected_layers:
            if infl_method == 'hf' and layer == 'WE':
                inf_method_layers.append(('hf_we_topk_10', layer))
                inf_method_layers.append(('hf_we_', layer))
                inf_method_layers.append((infl_method, layer))
            elif infl_method == 'datainf' and layer == 'WE':
                continue
            else:
                inf_method_layers.append((infl_method, layer))
    for task in tasks:
        df = task_dfs[task]
        for infl_method, layer in inf_method_layers: 
            setup_name = (infl_method, layer)
            # for run_id in run_ids:
            noise_scores = df[setup_name]
            setup_data.setdefault(setup_name, []).extend(noise_scores)

    prefix = "noise" if noise_only else "score"
    scores_df = pd.DataFrame(setup_data)
    scores_corr = scores_df.corr(method='spearman')
    scores_corr.to_pickle(os.path.join(base_path, "ndr", f"{prefix}-corr-all.pcl"))
    pass

def average_correlation_matrices(corr_matrices):
    fisher_transformed = [np.arctanh(corr) for corr in corr_matrices]
    fisher_mean = np.mean(fisher_transformed, axis=0)
    averaged_corr = np.tanh(fisher_mean)
    return pd.DataFrame(averaged_corr, index=corr_matrices[0].index, columns=corr_matrices[0].columns)

def draw_one_corr_heatmap(scores_corr: pd.DataFrame, task: str, base_path: str, prefix: str):
    import seaborn as sns
    ticklabels = ["TI$^{10}_{we}$" if m == "hf_we_topk_10" else "TI$_{we}$" if m == "hf_we_" else l for m, l in scores_corr.index ]
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(scores_corr, annot=True, fmt=".1f", cmap='coolwarm', cbar=False,
                        center = 0.5, vmin=0, vmax = 1, 
                        annot_kws={"color": "black", "fontsize": 12},
                        xticklabels=ticklabels, yticklabels=ticklabels)
    ax.set_xlabel("")
    ax.set_ylabel("")   
    ax.invert_yaxis()  
    boundary_positions = [6, 14]  # Positions where you want the borders
    for pos in boundary_positions:
        ax.hlines(pos, *ax.get_xlim(), colors="black", linewidth=2)  # Horizontal line
        ax.vlines(pos, *ax.get_ylim(), colors="black", linewidth=2)  # Vertical line

    ax.text(-0.6, 3, f"Cosine", ha='center', va='center', rotation='vertical', fontsize=12, color='black')
    ax.text(-0.6, 10, f"TracIn", ha='center', va='center', rotation='vertical', fontsize=12, color='black')
    ax.text(-0.6, 16.5, f"DataInf", ha='center', va='center', rotation='vertical', fontsize=12, color='black')

    ax.text(3, -0.7, f"Cosine", ha='center', va='center', fontsize=12, color='black')
    ax.text(10, -0.7, f"TracIn", ha='center', va='center', fontsize=12, color='black')
    ax.text(16.5, -0.7, f"DataInf", ha='center', va='center', fontsize=12, color='black')

    ax.tick_params(axis='x', labelsize=10, rotation=0)  # Set font size for x-axis tick labels
    # ax.set_xticklabels(ax.get_xticks(), verticalalignment='center', fontsize=10)
    ax.tick_params(axis='y', labelsize=10, rotation = 90)  # Set font size for y-axis tick labels

    if task != "avg" and task != "all":
        plt.title(f"{task.upper()}")
    plt.tight_layout(pad=0, rect=[0.01, 0.01, 1, 1])
    plt.savefig(os.path.join(base_path, "plots", f"{prefix}-corr-{task}.pdf"))
    plt.clf()

def draw_corr_heatmap(base_path: str, tasks = benchmark, noise_only = True):
    import seaborn as sns
    plt.ioff()
    corr_matrices = []
    prefix = "noise" if noise_only else "score"
    for task in tasks:
        scores_corr = pd.read_pickle(os.path.join(base_path, "ndr", f"{prefix}-corr-{task}.pcl"))
        draw_one_corr_heatmap(scores_corr, task, base_path, prefix = prefix)
        corr_matrices.append(scores_corr)
        pass
    avg_corr = average_correlation_matrices(corr_matrices)
    draw_one_corr_heatmap(avg_corr, "avg", base_path, prefix = prefix)
    pass

def draw_all_corr_heatmap(base_path: str, noise_only = True):
    import seaborn as sns
    prefix = "noise" if noise_only else "score"
    plt.ioff()
    scores_corr = pd.read_pickle(os.path.join(base_path, "ndr", f"{prefix}-corr-all.pcl"))
    draw_one_corr_heatmap(scores_corr, "all", base_path, prefix = prefix)
    pass

def draw_one_model_corr_heatmap(model_name, scores_corr: pd.DataFrame, ax,
                                no_y_marks = False):
    import seaborn as sns
    ticklabels = ["TI$^{10}_{we}$" if m == "hf_we_topk_10" else "TI$_{we}$" if m == "hf_we_" else l for m, l in scores_corr.index ]
    if no_y_marks:
        yticklabels = []
    else:
        yticklabels = ticklabels
    ax = sns.heatmap(scores_corr, annot=True, fmt=".1f", cmap='coolwarm', cbar=False,
                        center = 0.5, vmin=0, vmax = 1, 
                        annot_kws={"color": "black", "fontsize": 12},
                        xticklabels=ticklabels, yticklabels=yticklabels, ax = ax)
    ax.set_xlabel("")
    ax.set_ylabel("")   
    ax.invert_yaxis()  
    boundary_positions = [6, 14]  # Positions where you want the borders
    for pos in boundary_positions:
        ax.hlines(pos, *ax.get_xlim(), colors="white", linewidth=1)  # Horizontal line
        ax.vlines(pos, *ax.get_ylim(), colors="white", linewidth=1)  # Vertical line

    if not no_y_marks:
        ax.text(-0.7, 3, f"Cosine", ha='center', va='center', rotation='vertical', fontsize=16, color='black')
        ax.text(-0.7, 10, f"TracIn", ha='center', va='center', rotation='vertical', fontsize=16, color='black')
        ax.text(-0.7, 16.5, f"DataInf", ha='center', va='center', rotation='vertical', fontsize=16, color='black')
        
    ax.text(3, -0.7, f"Cosine", ha='center', va='center', fontsize=16, color='black')
    ax.text(10, -0.7, f"TracIn", ha='center', va='center', fontsize=16, color='black')
    ax.text(16.5, -0.7, f"DataInf", ha='center', va='center', fontsize=16, color='black')

    ax.tick_params(axis='x', length = 0, labelsize=8, rotation=0, pad=0)  # Set font size for x-axis tick labels
    ax.tick_params(axis='y', length = 0, labelsize=8, rotation = 90, pad=0)  # Set font size for y-axis tick labels

    ax.set_title(model_name, fontsize = 20, pad=0)

def draw_all_models_corr_heatmap(base_paths: list[tuple[str, str]], noise_only = True):
    import seaborn as sns
    prefix = "noise" if noise_only else "score"
    plt.ioff()
    fig, axes = plt.subplots(1, len(base_paths), figsize=(16, 6)) #, gridspec_kw={'wspace': 0.2, 'hspace': 0})
    for model_i, (model_name, base_path) in enumerate(base_paths):
        ax =  axes[model_i]
        scores_corr = pd.read_pickle(os.path.join(base_path, "ndr", f"{prefix}-corr-all.pcl"))
        draw_one_model_corr_heatmap(model_name, scores_corr, ax, no_y_marks=model_i > 0)
    fig.tight_layout(w_pad=1, h_pad=0, pad=0, rect=[0.005, 0, 1, 1])
    fig.savefig(os.path.join(base_paths[0][1], "..", "plots", f"{prefix}-corr-all-models.pdf"))
    plt.close(fig)
    pass

def draw_perf_diffs(metric_name: str = "best_accuracy_1",
                        out_folder = "data/roberta", datasets = benchmark,
                        run_ids = [0,1,2,3,4,5,6,7,8,9], layers = [],
                        res_suffix = "all", pvalue = 0.05): 
    ''' Like create_tun2_metric_table, but for many agg metrics '''
    layers.append('')

    all_df = get_all_df(base_path = out_folder, datasets = datasets, 
                            res_suffix = res_suffix, keep_only=[metric_name])
    
    agg_method_keys = ["mean", "rank-c", "vote2-c"]
    # agg_method_keys.append('')

    all_df = all_df[all_df["agg_method"].isin(agg_method_keys) & all_df["seed"].isin(run_ids) & all_df["module"].isin(layers)]
    all_df.loc[all_df['infl_method'] == 'hf_we_', ['infl_method', 'module']] = ('hf', 'TI$_{we}$')
    all_df.loc[all_df['infl_method'] == 'hf_we_topk_10', ['infl_method', 'module']] = ('hf', 'TI$^{10}_{we}$')
    all_df.set_index(["agg_method", "infl_method", "module", "task", "seed"], inplace=True)

    # metrics_per_ds = all_df.pivot(index=["infl_method", "module", "seed"], columns=["task", "agg_method"], values=metric_name)


    # idx = pd.MultiIndex.from_tuples(metrics_per_ds.columns)

    # metrics_per_ds.columns = idx

    base_level_df = all_df.loc['mean'].reset_index().pivot(index=["module"], columns=["infl_method", "task", "seed"], values=metric_name)
    rank_df = all_df.loc['rank-c'].reset_index().pivot(index=["module"], columns=["infl_method", "task", "seed"], values=metric_name)
    vote_df = all_df.loc['vote2-c'].reset_index().pivot(index=["module"], columns=["infl_method", "task", "seed"], values=metric_name)
    diff_rank_df = rank_df - base_level_df
    diff_vote_df = vote_df - base_level_df
    diff_rank_avgs = diff_rank_df.groupby(level=[0,1], axis=1).mean()
    # diff_rank_stds = diff_rank_df.groupby(level=0, axis=1).std()
    diff_vote_avgs = diff_vote_df.groupby(level=[0,1], axis=1).mean()
    # diff_vote_stds = diff_vote_df.groupby(level=0, axis=1).std()
    column_order = [(i, t) for i in ['hf', 'cos', 'datainf'] for t in datasets]

    layers = [ *(['TI$_{we}$', 'TI$^{10}_{we}$'] if 'WE' in layers else []), *layers]
            
    diff_rank_avgs0 = diff_rank_avgs.loc[layers][column_order] * 100
    diff_vote_avgs0 = diff_vote_avgs.loc[layers][column_order] * 100

    pass 
    # NOTE: stats tests here

    rank_corr = []
    vote_corr = []
    column_index = set()
    for module in base_level_df.index:
        rank_method_dict = {}
        vote_method_dict = {}
        infl_method_dataset = set((x[0], x[1]) for x in base_level_df.columns)
        for (infl_method, task) in infl_method_dataset:            
            if (infl_method != 'hf' and module.startswith('TI')) or (infl_method == "datainf" and module == "WE"):
                continue
            column_index.add((infl_method, task))
            clms = [(infl_method, task, r) for r in run_ids]
            base_series = base_level_df.loc[module][clms].to_numpy()
            rank_series = rank_df.loc[module][clms].to_numpy()
            vote_series = vote_df.loc[module][clms].to_numpy()
            wilcoxon_p = sci_stats.wilcoxon(base_series, rank_series).pvalue
            rank_method_dict[(infl_method, task)] = wilcoxon_p
            wilcoxon_p = sci_stats.wilcoxon(base_series, vote_series).pvalue
            vote_method_dict[(infl_method, task)] = wilcoxon_p
        rank_corr.append(rank_method_dict)
        vote_corr.append(vote_method_dict)
    rank_pvalues = pd.DataFrame(rank_corr, index=base_level_df.index, columns=pd.MultiIndex.from_tuples(list(column_index), names = ['infl_method', 'task']))
    vote_pvalues = pd.DataFrame(vote_corr, index=base_level_df.index, columns=pd.MultiIndex.from_tuples(list(column_index), names = ['infl_method', 'task']))

    if len(run_ids) > 5:
        rank_pvalues_mask = rank_pvalues < pvalue
        vote_pvalues_mask = vote_pvalues < pvalue
    else:
        rank_pvalues_mask = rank_pvalues >= 0 
        vote_pvalues_mask = vote_pvalues >= 0
    pass
    
    pass 

    
    import seaborn as sns
    plt.ioff()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={'wspace': 0, 'hspace': 0})

    methods_dfs = [("Rank", diff_rank_avgs0, rank_pvalues_mask), ("Vote", diff_vote_avgs0, vote_pvalues_mask)]

    single_color_cmap = ListedColormap(["lightgray"])

    for method_i, (method_name, method_df, method_mask) in enumerate(methods_dfs):

        ax = axes[method_i]

        mask = method_df.isna()

        sign_mask0 = method_mask.loc[method_df.index][method_df.columns] & (method_df.abs() >= 1)
        sign_mask = mask | (~sign_mask0)
        insign_mask_base = mask | sign_mask0
        insign_mask0 = (~method_mask.loc[method_df.index][method_df.columns]) & (method_df.abs() >= 1)
        insign_mask = mask | (~insign_mask0)

        yticklabels = ['All' if l == '' else l for l in layers]
        xticklabels = [d.upper() for d in datasets] * 3
        # ticklabels = ["TracIn$^{10}_{we}$" if m == "hf_we_topk_10" else "TracIn$_{we}$" if m == "hf_we_" else l for m, l in scores_corr.index ]
        ax0 = sns.heatmap(method_df, annot=True, fmt=".1f", cmap='coolwarm', cbar=False, 
                            mask=sign_mask, annot_kws={"color": "black", "fontsize": 10},
                            center = 0, vmin = -10, vmax = 10,
                            xticklabels=xticklabels, yticklabels=yticklabels, ax = ax)
        sns.heatmap(method_df, annot=False, cmap=single_color_cmap, cbar=False, 
                            mask=insign_mask_base, annot_kws={"color": "black", "fontsize": 7},
                            center = 0, vmin = -10, vmax = 10,
                            xticklabels=xticklabels, yticklabels=yticklabels, ax = ax)   
        sns.heatmap(method_df, annot=True, cmap=single_color_cmap, cbar=False, 
                            mask=insign_mask, annot_kws={"color": "black", "fontsize": 7},
                            center = 0, vmin = -10, vmax = 10,
                            xticklabels=xticklabels, yticklabels=yticklabels, ax = ax)               
        ax0.set_title(method_name, fontsize=20, pad=10)
        ax0.xaxis.tick_top()
        ax0.set_xlabel("")
        ax0.set_ylabel("")   
        ax0.invert_yaxis()  
        # ax1.invert_yaxis()
        ax0.spines['bottom'].set_linewidth(0)
        ax0.spines['top'].set_linewidth(0)        
        # ax1.spines['bottom'].set_linewidth(0)
        # ax1.spines['top'].set_linewidth(0)
        ax0.tick_params(axis = "x", length = 0, labelsize=6, rotation=0, pad=0)
        ax0.tick_params(axis = "y", length = 0, labelsize=7, rotation=90, pad=0)
        if method_i > 0:
            ax0.tick_params(axis='y', left=False, labelleft=False)        
        # ax.invert_yaxis()  
        boundary_positions = [0, 8, 16]
        b_ys = [ (0, 9), (0, 9), (2, 9) ]
        for line_id, (b_y, pos) in enumerate(zip(b_ys, boundary_positions)):
            if line_id == 0 and method_i == 0:
                continue
            # ax.hlines(pos, *b_y, colors="black", linewidth=1)  # Horizontal line
            ax0.vlines(pos, *b_y, colors="black", linewidth=1)  # Vertical line

        # ax.text(-0.5, 3, f"Cosine", ha='center', va='center', rotation='vertical', fontsize=10, color='black')
        # ax.text(-0.5, 10, f"TracIn", ha='cefnter', va='center', rotation='vertical', fontsize=10, color='black')
        # ax.text(-0.5, 16.5, f"DataInf", ha='center', va='center', rotation='vertical', fontsize=10, color='black')

        ax0.text(4, -0.5, f"TracIn", ha='center', va='center', fontsize=16, color='black')
        ax0.text(12, 2-0.5, f"Cosine", ha='center', va='center', fontsize=16, color='black')
        ax0.text(20, 3-0.5, f"DataInf", ha='center', va='center', fontsize=16, color='black')

        # ax.tick_params(axis='x', labelsize=7, rotation=0)  # Set font size for x-axis tick labels
        # ax.tick_params(axis='y', labelsize=7, rotation = 90)  # Set font size for y-axis tick labels

        # if task != "avg" and task != "all":
        #     plt.title(f"{task.upper()}")
    fig.tight_layout(h_pad=0, w_pad=0, pad=0)
    fig.savefig(os.path.join(base_path, "plots", f"rank-vote-diffs.pdf"))
    plt.close(fig)
    pass

def draw_perf_diffs2(metric_name: str = "best_accuracy_1",
                        out_folder = "data/roberta", datasets = benchmark,
                        run_ids = [0,1,2,3,4,5,6,7,8,9], layers = [],
                        res_suffix = "all", pvalue = 0.05): 
    ''' Like create_tun2_metric_table, but for many agg metrics '''
    layers.append('')

    all_df = get_all_df(base_path = out_folder, datasets = datasets, 
                            res_suffix = res_suffix, keep_only=[metric_name])
    
    agg_method_keys = ["mean", "rank-c", "vote2-c"]
    # agg_method_keys.append('')

    all_df = all_df[all_df["agg_method"].isin(agg_method_keys) & all_df["seed"].isin(run_ids) & all_df["module"].isin(layers)]
    all_df.loc[all_df['infl_method'] == 'hf_we_', ['infl_method', 'module']] = ('hf', 'TI$_{we}$')
    all_df.loc[all_df['infl_method'] == 'hf_we_topk_10', ['infl_method', 'module']] = ('hf', 'TI$^{10}_{we}$')
    all_df.set_index(["agg_method", "infl_method", "module", "task", "seed"], inplace=True)

    # metrics_per_ds = all_df.pivot(index=["infl_method", "module", "seed"], columns=["task", "agg_method"], values=metric_name)


    # idx = pd.MultiIndex.from_tuples(metrics_per_ds.columns)

    # metrics_per_ds.columns = idx

    base_level_df = all_df.loc['mean'].reset_index().pivot(index=["task", "infl_method", "module"], columns=['seed'], values=metric_name)
    rank_df = all_df.loc['rank-c'].reset_index().pivot(index=["task", "infl_method", "module"], columns=["seed"], values=metric_name)
    vote_df = all_df.loc['vote2-c'].reset_index().pivot(index=["task", "infl_method", "module"], columns=["seed"], values=metric_name)
    diff_rank_df = rank_df.loc[base_level_df.index][base_level_df.columns] - base_level_df
    diff_vote_df = vote_df.loc[base_level_df.index][base_level_df.columns] - base_level_df
    diff_rank_df *= 100
    diff_vote_df *= 100
    diff_rank_means = diff_rank_df.mean(axis=1)
    diff_vote_means = diff_vote_df.mean(axis=1)
    diff_rank_mins = diff_rank_df.groupby(level=[0], axis=0).min()
    diff_rank_maxs = diff_rank_df.groupby(level=[0], axis=0).max()
    diff_vote_mins = diff_vote_df.groupby(level=[0], axis=0).min()
    diff_vote_maxs = diff_vote_df.groupby(level=[0], axis=0).max()

    diff_range = pd.concat([diff_rank_mins, diff_vote_mins, diff_rank_maxs, diff_vote_maxs], axis=1).abs().max(axis=1)
    diff_mins = pd.concat([diff_rank_mins, diff_vote_mins], axis=1).min(axis=1)
    diff_maxs = pd.concat([diff_rank_maxs, diff_vote_maxs], axis=1).max(axis=1)

    # diff_rank_stds = diff_rank_df.groupby(level=0, axis=1).std()
    # diff_vote_stds = diff_vote_df.groupby(level=0, axis=1).std()
    # column_order = [(i, t) for i in ['hf', 'cos', 'datainf'] for t in datasets]

    # layers = [ *(['TI$_{we}$', 'TI$^{10}_{we}$'] if 'WE' in layers else []), *layers]
            
    # diff_rank_avgs0 = diff_rank_avgs.loc[layers][column_order] * 100
    # diff_vote_avgs0 = diff_vote_avgs.loc[layers][column_order] * 100

    pass 
    # NOTE: stats tests here

    rank_corr = []
    vote_corr = []
    # column_index = set()
    for idx_key in base_level_df.index:
        # rank_method_dict = {}
        # vote_method_dict = {}
        # infl_method_dataset = set((x[0], x[1]) for x in base_level_df.columns)
        # for (infl_method, task) in infl_method_dataset:            
        #     if (infl_method != 'hf' and module.startswith('TI')) or (infl_method == "datainf" and module == "WE"):
        #         continue
        #     column_index.add((infl_method, task))
        #     clms = [(infl_method, task, r) for r in run_ids]
        base_series = base_level_df.loc[idx_key][run_ids].to_numpy()
        
        rank_series = rank_df.loc[idx_key][run_ids].to_numpy()
        vote_series = vote_df.loc[idx_key][run_ids].to_numpy()
        wilcoxon_p = sci_stats.wilcoxon(base_series, rank_series).pvalue
        rank_corr.append(wilcoxon_p)
        # rank_method_dict[(infl_method, task)] = wilcoxon_p
        wilcoxon_p = sci_stats.wilcoxon(base_series, vote_series).pvalue
        vote_corr.append(wilcoxon_p)
        # vote_method_dict[(infl_method, task)] = wilcoxon_p

    rank_pvalues = pd.DataFrame(rank_corr, index=base_level_df.index, columns=["pvalue"])
    vote_pvalues = pd.DataFrame(vote_corr, index=base_level_df.index, columns=["pvalue"])

    if len(run_ids) > 5:
        rank_pvalues_mask = rank_pvalues < pvalue
        vote_pvalues_mask = vote_pvalues < pvalue
    else:
        rank_pvalues_mask = rank_pvalues >= 0 
        vote_pvalues_mask = vote_pvalues >= 0
    pass
    
    pass 

    from matplotlib import cm
    # import seaborn as sns
    plt.ioff()
    fig, axes = plt.subplots(6, 8, figsize=(12, 5), gridspec_kw={'wspace': 0, 'hspace': 0})

    # methods_dfs = [("Rank", diff_rank_avgs0, rank_pvalues_mask), ("Vote", diff_vote_avgs0, vote_pvalues_mask)]

    # single_color_cmap = ListedColormap(["lightgray"])
    from itertools import product

    ifl_agg_methods = list(product(
                               [('Rank', diff_rank_df, diff_rank_means, rank_pvalues_mask), 
                                ('Vote', diff_vote_df, diff_vote_means, vote_pvalues_mask)],
                                ['hf', 'datainf', 'cos']))
    
    # colors = cm.get_cmap('tab10', 10)
    colors = plt.cm.Set1.colors 
    all_layers =  [ *(['TI$_{we}$', 'TI$^{10}_{we}$'] if 'WE' in layers else []), *layers]
    layer_colors = {l:colors[i] for i, l in enumerate(all_layers)}

    yd1 = 0.06
    yd2 = 0.96

    yc = (yd1 + yd2) / 2
    yc1 = (yc + yd1) / 2
    yc2 = (yc + yd2) / 2
    d = (yc - yc1) / 3 * 2

    handles = []
    for col_id, task in enumerate(datasets): # columns 
        for row_id, ((agg_name, method_df, method_means, method_mask), infl_method) in enumerate(ifl_agg_methods):
            ax = axes[row_id, col_id]
            ax_df = method_df.loc[(task, infl_method)]
            # ax_points = ax_df.to_numpy() #2d: module --> points
            # draw stripe
            stripe_height = 1
            ax.set_ylim(-stripe_height / 2, stripe_height / 2)
            # r = diff_range.loc[task]
            r_min = diff_mins.loc[task]
            r_max = diff_maxs.loc[task]
            ax.set_xlim(r_min, r_max)
            # x_min, x_max = ax.get_xlim()
            # stripe_width = x_max - x_min

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)            

            # Draw the x-axis at y=0
            # ax.hlines(y=0, xmin=-r, xmax=r, color='gray', linewidth=0.5, linestyle='-')
            
            ax.vlines(x=0, ymin=-0.5, ymax=0.5, color='black', linewidth=0.5, linestyle='--', zorder=0)
            ax.vlines(x=r_min, ymin=-0.5, ymax=0.5, color='black', linewidth=0.3, linestyle=':', zorder=0)
            ax.vlines(x=r_max, ymin=-0.5, ymax=0.5, color='black', linewidth=0.3, linestyle=':', zorder=0)

            # ax.vlines(x=-r, ymin=-0.2, ymax=0.2, color='black', linewidth=0.5, linestyle='--')
            # ax.vlines(x=r, ymin=-0.2, ymax=0.2, color='black', linewidth=0.5, linestyle='--')

            # if row_id == 0 or row_id == len(ifl_agg_methods) - 1:
            #     ax.text(-r, 0, f"{-r:.0f}", ha='left', va='center', fontsize=8, color='black', rotation="vertical")
            #     ax.text(r, 0, f"{r:.0f}", ha='right', va='center', fontsize=8, color='black', rotation="vertical")

            # ax.text(12, 2-0.5, f"Cosine", ha='center', va='center', fontsize=16, color='black')
            # ax.text(20, 3-0.5, f"DataInf", ha='center', va='center', fontsize=16, color='black')



            present_layers = [l for l in all_layers if l in ax_df.index]

            for i, layer in enumerate(present_layers):
                points = ax_df.loc[layer].to_numpy()
                is_sign = method_mask.loc[(task, infl_method, layer)].bool()
                mean = method_means.loc[(task, infl_method, layer)]
                y_pos = (0.1 / len(present_layers)) + (i - len(present_layers) / 2) * ((stripe_height - 0.2) / len(present_layers))
                y_poss = np.full_like(points, y_pos)
                # y_positions = np.random.uniform(-0.1, 0.1, size=len(points))
                alpha = (1 if is_sign and (abs(mean) >= 1) else 0.2)
                ax.plot([np.min(points), np.max(points)], [y_pos, y_pos], color='black', linewidth=0.3, alpha=alpha, zorder=0)
                points_ch = ax.scatter(points, y_poss, color=layer_colors[layer], 
                                       s=20, 
                                       edgecolors=('black' if is_sign and (abs(mean) >= 1) else 'none'), 
                                       alpha=alpha,
                                       linewidths=0.1
                                    ) #, clip_on=False)
                if row_id == 0 and col_id == 0:
                    handles.append((points_ch, layer))


            # Remove y-axis ticks and labels
            ax.tick_params(axis='both', which='both', length=0) 
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])

            # if row_id == 0: # add title to column
            #     ax.set_title(task.upper(), fontsize=16, pad=10)

            # Add x-axis labels
            # ax.set_xlabel('Spread of Points', fontsize=10)
            # ax.set_xticks(np.linspace(-stripe_width / 2, stripe_width / 2, 5))
            # ax.tick_params(axis='x', labelsize=8)

            # Add a legend
            # ax.legend(fontsize=8, loc='upper right')
            pass 
            # break

    fig.add_artist(plt.Line2D([0, 1], [yc, yc], color='black', linewidth=0.5))

    fig.text(0.00, yc1, f'Vote', ha='left', va='center', rotation='vertical', fontsize=12)
    fig.text(0.01, yc1 - d, f'Cosine', ha='left', va='center', rotation='vertical', fontsize=10)
    fig.text(0.01, yc1, f'DataInf', ha='left', va='center', rotation='vertical', fontsize=10)
    fig.text(0.01, yc1 + d, f'TracIn', ha='left', va='center', rotation='vertical', fontsize=10)
    fig.text(0.00, yc2, f'Rank', ha='left', va='center', rotation='vertical', fontsize=12)
    fig.text(0.01, yc2 - d, f'Cosine', ha='left', va='center', rotation='vertical', fontsize=10)
    fig.text(0.01, yc2, f'DataInf', ha='left', va='center', rotation='vertical', fontsize=10)
    fig.text(0.01, yc2 + d, f'TracIn', ha='left', va='center', rotation='vertical', fontsize=10)

    xd1 = 0.02

    for col_id, task in enumerate(datasets): # columns 
        r_min = diff_mins.loc[task]
        r_max = diff_maxs.loc[task]

        # ax = axes[0, col_id]
        # # pos = ax.get_position() 
        # # x0, y0, x1, y1 = pos.extents                 
        # axes_coord = ax.transData.transform((r_min, 0))
        # fig_x_min, _ = fig.transFigure.inverted().transform(axes_coord) #ax.transAxes.transform(axes_coord))

        fig_x_min = xd1 + (1 - xd1) * col_id / len(datasets)

        fig.text(fig_x_min + 0.002, yc, f"{r_min:.0f}", ha='left', va='top', fontsize=10, color='black')
        # axes_coord = ax.transData.transform((r_max, 0))
        # fig_x_max, _ = fig.transFigure.inverted().transform(axes_coord) #ax.transAxes.transform(axes_coord))
        fig_x_max = xd1 + (1 - xd1) * (col_id + 1) / len(datasets)
        fig.text(fig_x_max - 0.002, yc, f"{r_max:.0f}", ha='right', va='bottom', fontsize=10, color='black')
        pass

    # for method_i, (method_name, method_df, method_mask) in enumerate(methods_dfs):

    #     ax = axes[method_i]

    #     mask = method_df.isna()

    #     sign_mask0 = method_mask.loc[method_df.index][method_df.columns] & (method_df.abs() >= 1)
    #     sign_mask = mask | (~sign_mask0)
    #     insign_mask_base = mask | sign_mask0
    #     insign_mask0 = (~method_mask.loc[method_df.index][method_df.columns]) & (method_df.abs() >= 1)
    #     insign_mask = mask | (~insign_mask0)

    #     yticklabels = ['All' if l == '' else l for l in layers]
    #     xticklabels = [d.upper() for d in datasets] * 3
    #     # ticklabels = ["TracIn$^{10}_{we}$" if m == "hf_we_topk_10" else "TracIn$_{we}$" if m == "hf_we_" else l for m, l in scores_corr.index ]
    #     ax0 = sns.heatmap(method_df, annot=True, fmt=".1f", cmap='coolwarm', cbar=False, 
    #                         mask=sign_mask, annot_kws={"color": "black", "fontsize": 10},
    #                         center = 0, vmin = -10, vmax = 10,
    #                         xticklabels=xticklabels, yticklabels=yticklabels, ax = ax)
    #     sns.heatmap(method_df, annot=False, cmap=single_color_cmap, cbar=False, 
    #                         mask=insign_mask_base, annot_kws={"color": "black", "fontsize": 7},
    #                         center = 0, vmin = -10, vmax = 10,
    #                         xticklabels=xticklabels, yticklabels=yticklabels, ax = ax)   
    #     sns.heatmap(method_df, annot=True, cmap=single_color_cmap, cbar=False, 
    #                         mask=insign_mask, annot_kws={"color": "black", "fontsize": 7},
    #                         center = 0, vmin = -10, vmax = 10,
    #                         xticklabels=xticklabels, yticklabels=yticklabels, ax = ax)               
    #     ax0.set_title(method_name, fontsize=20, pad=10)
    #     ax0.xaxis.tick_top()
    #     ax0.set_xlabel("")
    #     ax0.set_ylabel("")   
    #     ax0.invert_yaxis()  
    #     # ax1.invert_yaxis()
    #     ax0.spines['bottom'].set_linewidth(0)
    #     ax0.spines['top'].set_linewidth(0)        
    #     # ax1.spines['bottom'].set_linewidth(0)
    #     # ax1.spines['top'].set_linewidth(0)
    #     ax0.tick_params(axis = "x", length = 0, labelsize=6, rotation=0, pad=0)
    #     ax0.tick_params(axis = "y", length = 0, labelsize=7, rotation=90, pad=0)
    #     if method_i > 0:
    #         ax0.tick_params(axis='y', left=False, labelleft=False)        
    #     # ax.invert_yaxis()  
    #     boundary_positions = [0, 8, 16]
    #     b_ys = [ (0, 9), (0, 9), (2, 9) ]
    #     for line_id, (b_y, pos) in enumerate(zip(b_ys, boundary_positions)):
    #         if line_id == 0 and method_i == 0:
    #             continue
    #         # ax.hlines(pos, *b_y, colors="black", linewidth=1)  # Horizontal line
    #         ax0.vlines(pos, *b_y, colors="black", linewidth=1)  # Vertical line

    #     # ax.text(-0.5, 3, f"Cosine", ha='center', va='center', rotation='vertical', fontsize=10, color='black')
    #     # ax.text(-0.5, 10, f"TracIn", ha='cefnter', va='center', rotation='vertical', fontsize=10, color='black')
    #     # ax.text(-0.5, 16.5, f"DataInf", ha='center', va='center', rotation='vertical', fontsize=10, color='black')

    #     ax0.text(4, -0.5, f"TracIn", ha='center', va='center', fontsize=16, color='black')
    #     ax0.text(12, 2-0.5, f"Cosine", ha='center', va='center', fontsize=16, color='black')
    #     ax0.text(20, 3-0.5, f"DataInf", ha='center', va='center', fontsize=16, color='black')

    #     # ax.tick_params(axis='x', labelsize=7, rotation=0)  # Set font size for x-axis tick labels
    #     # ax.tick_params(axis='y', labelsize=7, rotation = 90)  # Set font size for y-axis tick labels

    #     # if task != "avg" and task != "all":
    #     #     plt.title(f"{task.upper()}")
    
    # xd1 = 0

    for col_id, title in enumerate(datasets):
        x_pos = xd1 + (1 - xd1) * (col_id + 0.5) / len(datasets)  # Calculate the x position for each title
        fig.text(x_pos, 1, title.upper(), ha='center', va='top', fontsize=12)

    # plt.xlabel('Epoch', fontsize=20s)
    # plt.ylabel('Accuracy, \\%', fontsize=20)
    # ordered_handles = [h[0] for h in handles]

    from matplotlib.patches import Patch

    ordered_handles = [Patch(facecolor=layer_colors[layer], label=layer) for layer in all_layers]
    all_layer_names =  [ *(['TracIn$_{we}$', 'TracIn$^{10}_{we}$'] if 'WE' in layers else []), *layers]
    ordered_labels = ["All" if n == "" else n for n in all_layer_names]    
    fig.legend(ordered_handles, ordered_labels, loc='lower center', fontsize=10,
                ncol=len(ordered_handles), borderaxespad = 0,  # Arrange all legend items in one row
                bbox_to_anchor=(0.5, 0)  # Adjust position (centered below the grid)
        )
    # plt.title(f'{task.upper()} 70\\% filtered finetuning', fontsize=20)
    # plt.title(f'{task.upper()}', fontsize=20)
    # fig.tight_layout(rect=[0.015, 0.1, 1, 1], pad = 0, h_pad=0.1, w_pad=0.1)


    fig.tight_layout(rect=[xd1, yd1, 1, yd2], h_pad=0, w_pad=0, pad=0)
    fig.savefig(os.path.join(base_path, "plots", f"rank-vote-diffs2.pdf"))
    plt.close(fig)
    pass

def create_cancel_table(base_path:str, m_prefix: str = "m_bl", levels=[1], layers = ['WE', 'CL']):
    import math
    df = pd.read_pickle(os.path.join(base_path, "cancel", f"cancellation_{m_prefix}.pkl"))
    df_agg = df[['cancellation']].groupby(level=levels, axis=0).mean().rename(columns={'cancellation': 'c_mean'})
    df_agg['c_std'] = df[['cancellation']].groupby(level=levels, axis=0).std()
    df_agg['c_med_median'] = df[['median_cancellation']].groupby(level=levels, axis=0).median()
    # df_agg['c_med_std'] = df[['median_cancellation']].groupby(level=levels, axis=0).std()
    df_agg['c_min'] = df[['min_cancellation']].groupby(level=levels, axis=0).median()
    df_agg['c_max'] = df[['max_cancellation']].groupby(level=levels, axis=0).median()
    df_agg['num_params'] = df[['num_params']].groupby(level=levels, axis=0).median()
    rows = []
    for row_dict in df_agg.loc[layers].reset_index().to_dict(orient='index').values():
        row = []
        row.append(row_dict['layer'])
        row.append(f"{row_dict['c_mean']:.1f} \\footnotesize $\\pm$ {row_dict['c_std']:.1f}")
        row.append(f"{row_dict['c_med_median']:.1f}")
        row.append(f"{row_dict['c_min']:.1f}")
        row.append("$\\infty$" if math.isinf(row_dict['c_max']) else f"{row_dict['c_max']:.1e}")
        row.append(f"{row_dict['num_params']:.1e}")
        rows.append(row)
    out_path = os.path.join(base_path, "tables", f"cancellation.tex")
    with open(out_path, "w") as f:
        print(tabulate(rows, headers=["Layer", "Mean $\pm$ Std", "Median", "Min", "Max", "#Params"], tablefmt="latex_raw"), file=f)
    pass


def run_spearman_total(out_folder = "data/roberta", datasets = benchmark,
                        m_prefix = "m_bl", layers = ['WE', 'CL'], run_ids = [0,1,2,3,4,5,6,7,8,9],
                        suffix=""):

    c_df = pd.read_pickle(os.path.join(out_folder, "cancel", f"cancellation_{m_prefix}.pkl"))

    all_df = get_all_df(base_path = out_folder, datasets = datasets, res_suffix="all",
                        keep_only=["best_accuracy_1", "noise_30", "auc_ndr"])

    all_df.set_index(["infl_method", "agg_method", "task", "module", "seed"], inplace=True)

    # metrics_per_ds = all_df.pivot(index=["infl_method", "agg_method", "module"], columns=["task", "seed"], values=[metric_name, ndr_metric_name])
    # # metrics_per_ds = metrics_per_ds.dropna(axis=1, how='any')
    # metric_columns = {(ds, seed) for _, ds, seed in metrics_per_ds.columns[metrics_per_ds.columns.get_level_values(0) == metric_name]}
    # ndr_columns = {(ds, seed) for _, ds, seed in metrics_per_ds.columns[metrics_per_ds.columns.get_level_values(0) == ndr_metric_name]}
    # common_columns = set.intersection(metric_columns, ndr_columns)

    infl_methods = ["datainf", "hf_we_", "hf_we_topk_10", "hf", "cos"]
    infl_mapping = ["DataInf", "TracIn$_{we}$", "TracIn$^{10}_{we}$", "TracIn", "Cosine"]
    agg_methods = ["mean", "rank-c", "vote2-c"] 
    agg_mapping = ["Mean", "Rank", "Vote"]

    can_per_infl_method = defaultdict(list)
    mcan_per_infl_method = defaultdict(list)
    acc_per_infl_method = defaultdict(list)
    ndr_per_infl_method = defaultdict(list)
    auc_per_infl_method = defaultdict(list)
    for layer in layers:
        for task in datasets:
            for run_id in run_ids: 
                for infl_method in infl_methods:
                    if infl_method == 'datainf' and layer == 'WE':
                        continue 
                    if infl_method.startswith("hf_") and layer != "WE":
                        continue
                    cancellation = c_df.loc[task, layer, run_id]['cancellation']
                    median_cancellation = c_df.loc[task, layer, run_id]['median_cancellation']
                    for agg_method in agg_methods:
                        can_per_infl_method[(infl_method, agg_method)].append(cancellation)
                        mcan_per_infl_method[(infl_method, agg_method)].append(median_cancellation)
                        v = all_df.loc[(infl_method, agg_method, task, layer, run_id)]["best_accuracy_1"]
                        acc_per_infl_method[(infl_method, agg_method)].append(v)
                        v2 = all_df.loc[(infl_method, agg_method, task, layer, run_id)]['noise_30']
                        ndr_per_infl_method[(infl_method, agg_method)].append(v2)
                        v3 = all_df.loc[(infl_method, agg_method, task, layer, run_id)]['auc_ndr']
                        auc_per_infl_method[(infl_method, agg_method)].append(v3)
        
    rows = []
    for ai, agg_method in enumerate(agg_methods):
        for ii, infl_method in enumerate(infl_methods):
            series0 = acc_per_infl_method[(infl_method, agg_method)]
            series1 = can_per_infl_method[(infl_method, agg_method)]
            series2 = ndr_per_infl_method[(infl_method, agg_method)]
            series3 = auc_per_infl_method[(infl_method, agg_method)]
            series4 = mcan_per_infl_method[(infl_method, agg_method)]

            s1 = sci_stats.spearmanr(series0, series1)
            s4 = sci_stats.spearmanr(series0, series4)
            s2 = sci_stats.spearmanr(series0, series2)
            s3 = sci_stats.spearmanr(series0, series3)
            rows.append([
                agg_mapping[ai], infl_mapping[ii], 
                f"{s1.correlation:.1f}" + ("*" if s1.pvalue > 0.05 else ""),
                # f"{s4.correlation:.1f}" + ("**" if s4.pvalue < 0.05 else ""),
                f"{s2.correlation:.1f}" + ("*" if s2.pvalue > 0.05 else ""),
                f"{s3.correlation:.1f}" + ("*" if s3.pvalue > 0.05 else ""),
            ])

    headers = ["Agg", "Infl", "$C$", "NDR$_{30}$", "AUC$_{NDR}$"]
    with open(f"{out_folder}/tables/spearman-all.tex", "w") as stats_file:
        s = tabulate(rows, headers=headers, tablefmt="latex_raw", numalign="center", stralign="center")
        # s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("hf\_we\_topk\_10", "hf$^{10}_{we}$").replace("hf\_we\_", "hf$_{we}$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("lllllllllllllllllllllllll", "l|cccccccccccccccccccccccc").replace("rand", "\\hline rand")
        print(s, file = stats_file)

    pass 

def run_spearman_total2(out_folder = "data/roberta", datasets = benchmark,
                        m_prefix = "m_bl", layers = ['WE', 'CL'], run_ids = [0,1,2,3,4,5,6,7,8,9],
                        suffix=""):

    c_df = pd.read_pickle(os.path.join(out_folder, "cancel", f"cancellation_{m_prefix}.pkl"))

    all_df = get_all_df(base_path = out_folder, datasets = datasets, res_suffix="all",
                        keep_only=["best_accuracy_1", "noise_30", "auc_ndr"])

    all_df.set_index(["infl_method", "agg_method", "task", "module", "seed"], inplace=True)

    # metrics_per_ds = all_df.pivot(index=["infl_method", "agg_method", "module"], columns=["task", "seed"], values=[metric_name, ndr_metric_name])
    # # metrics_per_ds = metrics_per_ds.dropna(axis=1, how='any')
    # metric_columns = {(ds, seed) for _, ds, seed in metrics_per_ds.columns[metrics_per_ds.columns.get_level_values(0) == metric_name]}
    # ndr_columns = {(ds, seed) for _, ds, seed in metrics_per_ds.columns[metrics_per_ds.columns.get_level_values(0) == ndr_metric_name]}
    # common_columns = set.intersection(metric_columns, ndr_columns)

    # infl_methods = ["hf"] # ["datainf", "hf_we_", "hf_we_topk_10", "hf", "cos"]
    # # infl_mapping = ["DataInf", "TracIn$_{we}$", "TracIn$^{10}_{we}$", "TracIn", "Cosine"]

    can_per_infl_method = defaultdict(list)
    mcan_per_infl_method = defaultdict(list)
    acc_per_infl_method = defaultdict(list)
    for task in datasets:
        for layer in layers:
            for run_id in run_ids: 
                cancellation = c_df.loc[task, layer, run_id]['cancellation']
                median_cancellation = c_df.loc[task, layer, run_id]['median_cancellation']
                can_per_infl_method[layer].append(cancellation)
                mcan_per_infl_method[layer].append(median_cancellation)
                v = all_df.loc[("hf", "mean", task, layer, run_id)]["best_accuracy_1"]
                acc_per_infl_method[layer].append(v)                    
        
    rows = []
    for layer in layers:
        series0 = acc_per_infl_method[layer]
        series1 = can_per_infl_method[layer]
        series2 = mcan_per_infl_method[layer]

        s1 = sci_stats.spearmanr(series0, series1)
        s2 = sci_stats.spearmanr(series0, series2)
        rows.append([layer,
            f"{s1.correlation:.1f}" + ("*" if s1.pvalue > 0.05 else ""),
            # f"{s4.correlation:.1f}" + ("**" if s4.pvalue < 0.05 else ""),
            f"{s2.correlation:.1f}" + ("*" if s2.pvalue > 0.05 else "")
        ])

    headers = ["Layers", "$C$", "Median $C$"]
    with open(f"{out_folder}/tables/spearman-layers-all.tex", "w") as stats_file:
        s = tabulate(rows, headers=headers, tablefmt="latex_raw", numalign="center", stralign="center")
        # s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("hf\_we\_topk\_10", "hf$^{10}_{we}$").replace("hf\_we\_", "hf$_{we}$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("lllllllllllllllllllllllll", "l|cccccccccccccccccccccccc").replace("rand", "\\hline rand")
        print(s, file = stats_file)

    pass 

def load_df(base_path, tasks = benchmark, use_ndr = False,
                ndr_prefix = "ndr_bl", agg_method_names = None,
                infl_method_names = None, selected_layers = None,
                res_suffix = "all", metric_name = "best_accuracy_1"):
    # loading data 
    if use_ndr: 
        dfs = []
        for task in tasks:
            df = pd.read_pickle(os.path.join(base_path, "ndr", f"{ndr_prefix}_{task}.pcl"))
            df.drop(columns=["scores", "noise_mask"], inplace=True)
            df.reset_index(inplace=True)
            if agg_method_names is not None:
                df = df[df["agg"].isin(agg_method_names)]
            if infl_method_names is not None:
                df = df[df["infl"].isin(infl_method_names)]
            if selected_layers is not None:
                df = df[df["layer"].isin(selected_layers)]
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        df = df.rename(columns={"run_id": "seed"})
        df = df[["task", "infl", "agg", "layer", "module", "seed", metric_name]]

        def layer_module_normalize(row):
            layer = row["layer"]
            module = row["module"].replace("embed_tokens", "WE")
            find_layer = re.match(r"layers\s(\d+)\s", module)
            if find_layer is not None:
                layer = find_layer.group(1)
                module = module.replace(find_layer.group(0), "")
            row["layer"] = "WE" if module == "WE" else layer
            row["module"] = module
            return row

        df = df.apply(layer_module_normalize, axis=1)
        
    else: #use jsonlists 
        df = get_all_df(base_path = base_path, datasets = tasks,
                                res_suffix = res_suffix, keep_only=[metric_name])
        if agg_method_names is not None:
            df = df[df["agg_method"].isin(agg_method_names)]
        if infl_method_names is not None:
            df = df[df["infl_method"].isin(infl_method_names)]
        if selected_layers is not None:
            df = df[df["module"].isin(selected_layers)]        
        df = df.rename(columns={"infl_method": "infl", "agg_method": "agg", "module": "layer"})
        df["module"] = "all"
    return df

def setup_interactions(base_path: str, tasks=benchmark, 
                    metric_name = "best_accuracy_1",
                    agg_method_names = ["mean", "rank-c", "vote2-c"],
                    infl_method_names = ['cos', 'hf', 'datainf'],
                    selected_layers = None,
                    use_ndr = False, dom_threshold = 0.75,
                    ndr_prefix = "ndr_bl", res_suffix = "all",
                    metric_thershold = 0.00, rank_start = 0, with_mean = False,
                    suffix = ""):
    ''' From all run scores, computes the interaction matrix between setups  
        and their pareto fronts 
    '''
    df = load_df(base_path, tasks, use_ndr, ndr_prefix, metric_name=metric_name, 
                res_suffix=res_suffix,
                agg_method_names=agg_method_names, infl_method_names=infl_method_names,
                selected_layers=selected_layers)

    score_df = df.pivot(index=["infl", "agg", "layer", "module"], columns=["task", "seed"], values=metric_name)

    if with_mean:
        mean_df = score_df.mean(axis=1)
        std_df = score_df.std(axis=1)
    else:
        mean_df = None
        std_df = None

    run_index = score_df.columns

    rows = list(score_df.iterrows())
    dom = {}
    dom_full = {}
    wcount = {}
    lcount = {}

    for i in range(len(rows)):
        index_i, row_i = rows[i]
        dom.setdefault(index_i, {})[index_i] = 0
        dom_full.setdefault(index_i, {})[index_i] = 0
        wcount.setdefault(index_i, 0)
        lcount.setdefault(index_i, 0)
        for j in range(i+1, len(rows)):
            index_j, row_j = rows[j]
            wcount.setdefault(index_j, 0)
            lcount.setdefault(index_j, 0)
            i_count = (((row_i - row_j) > metric_thershold)).sum() 
            i_frac = i_count / len(run_index)
            j_count = (((row_j - row_i) > metric_thershold)).sum() 
            j_frac = j_count / len(run_index)
            dom.setdefault(index_i, {})[index_j] = i_frac if i_frac > dom_threshold else 0
            dom.setdefault(index_j, {})[index_i] = j_frac if j_frac > dom_threshold else 0
            dom_full.setdefault(index_i, {})[index_j] = i_frac
            dom_full.setdefault(index_j, {})[index_i] = j_frac
            wcount[index_i] += i_count
            wcount[index_j] += j_count
            lcount[index_i] += j_count
            lcount[index_j] += i_count
    
    dom_rows = [dom[i] for i in score_df.index]


    dom_rows_full = [dom_full[i] for i in score_df.index]
    
    int_df_full = pd.DataFrame(dom_rows_full, index=score_df.index, columns=score_df.index)

    wrate_df = int_df_full.mean(axis=1)
    wrate_std = int_df_full.std(axis=1)


    int_df = pd.DataFrame(dom_rows, index=score_df.index, columns=score_df.index)
    int_np = int_df.to_numpy()

    exclude_indices = []
    layer_ranks = np.zeros(int_np.shape[0], dtype=int)
    layer_ranks[:] = -1
    dom_num = np.zeros(int_np.shape[0], dtype=int)
    negdom_num = np.zeros(int_np.shape[0], dtype=int)

    domination_matrix = np.all(int_np[:, None] <= int_np, axis=2) & np.any(int_np[:, None] < int_np, axis=2)
    rank = rank_start
    while len(exclude_indices) < len(int_np):
        front_indices = get_pareto_front_indexes(int_np, exclude_indexes=exclude_indices)
        assert np.all(layer_ranks[front_indices] == -1), f"Repeating ranks"        
        dom_num[front_indices] = domination_matrix[:, front_indices].sum(axis=0)
        negdom_num[front_indices] = domination_matrix[front_indices, :].sum(axis=1)
        layer_ranks[front_indices] = rank
        exclude_indices.extend(front_indices)
        rank += 1

    assert np.all(layer_ranks != -1), f"Not all ranks assigned"

    infl_mapping = {"datainf": "DataInf", "hf_we_": "TracIn$_{we}$", "hf_we_topk_10": "TracIn$^{10}_{we}$", "hf": "TracIn", "cos": "Cosine", "denoise": "Full", "rand": "Random"}
    agg_mapping = {"mean": "Mean", "rank-c": "Rank", "vote2-c": "Vote", '': ''}
    rows = []
    all_modules = int_df.index.get_level_values(3).unique()
    all_are_all = np.all(all_modules == "all")
    data_for_df = []
    for i, index in enumerate(int_df.index):
        layer_rank = layer_ranks[i]
        win_rate = wrate_df[index]
        wstd = wrate_std[index]
        num_wins = wcount[index]
        num_losses = lcount[index]
        num_doms = dom_num[i]
        num_domed = negdom_num[i]

        new_row = []
        # new_row.append(i+1)
        new_row.append(infl_mapping[index[0]])
        new_row.append(agg_mapping[index[1]])
        new_row.append("All" if index[2] == '' else index[2])
        if not all_are_all:
            new_row.append(index[3])
        new_row.append(layer_rank)
        new_row.append(f"{win_rate:.2f} \\pm \\footnotesize {wstd:.2f}")
        new_row.append(num_wins)
        new_row.append(num_losses)
        new_row.append(num_doms)
        new_row.append(num_domed)
        # for x in row:
        #     new_row.append(f"{x:.1f}".lstrip('0'))
        new_row_res = {
            "rank": layer_rank,
            "win_rate": win_rate,
            "wstd": wstd,
            "num_wins": num_wins,
            "num_losses": num_losses,
            "num_doms": num_doms,
            "num_domed": num_domed
        }
        if with_mean:
            new_row_res["mean"] = mean_df[index]
            new_row_res["std"] = std_df[index]
            new_row.append(f"{mean_df[index]:.2f} \\pm \\footnotesize {std_df[index]:.2f}")
        rows.append((layer_rank, win_rate, wstd, num_wins - num_losses, new_row))
        data_for_df.append(new_row_res)
    rows.sort(key=lambda x: (x[0], -x[1], x[2], -x[3]))
    rows = [x[4] for x in rows]
    
    out_path = os.path.join(base_path, 'tables', f"intranks-{metric_name}{suffix}.tex")

    headers = ["Infl", "Agg", "Layer", "Module", "Rank", "Win rate", "Wins", "Loss", "Doms", "Domed"]
    if with_mean:
        headers.append("Value")
    if all_are_all:
        headers.remove("Module")
    with open(out_path, "w") as f:
        print(tabulate(rows, headers=headers, tablefmt="latex_raw"), file=f)

    data_df = pd.DataFrame(data_for_df, index=int_df.index)
    out_path = os.path.join(base_path, 'metrics', f"intranks-{metric_name}{suffix}.pcl")
    data_df.to_pickle(out_path)
    pass

def get_confidence_interval(data_df: pd.DataFrame, confidence_level=0.95):
    data_np_t = data_df.to_numpy().T
    data_mean = data_np_t.mean(axis=0)
    degrees_freedom = data_np_t.shape[0] - 1 # num of measures - 1
    sample_standard_error = stats.sem(data_np_t, axis=0)
    confidence_interval = stats.t.interval(confidence_level, degrees_freedom, data_mean, sample_standard_error)
    min_v = confidence_interval[0]
    max_v = confidence_interval[1]
    res_df = pd.DataFrame(data={"min_v": min_v, "mean": data_mean, "max_v": max_v}, index=data_df.index)
    return res_df


def layer_ndr_metric_graphs(base_path: str, tasks=benchmark, 
                    metric_name = 30, figsize = (8, 2),
                    infl_method_names = ['datainf', 'hf', 'cos'],
                    module_names = {"self_attn q_proj A": "Query A", "self_attn q_proj B": "Query B", 
                                    "self_attn v_proj A": "Value A", "self_attn v_proj B": "Value B"},
                    # module_names = {"self_attn q_proj A": "query A"},
                    module_layers = list(range(32)),
                    cl_modules = {"all": "CL"}, # makes sense for Roberta only
                    ndr_prefix = "ndr_bl", suffix = "",
                    max_of = "self_attn v_proj B",
                    y_title = "Layer-wise NDR, \\%"):
    ''' From all run scores, computes the interaction matrix between setups  
        and their pareto fronts 
    '''
    orig_infl_names = list(infl_method_names)

    cache_path = os.path.join(base_path, f"layer_ndr{suffix}.pcl")
    if os.path.exists(cache_path):
        score_df = pd.read_pickle(cache_path) 
    else:
        df = load_df(base_path, tasks, use_ndr = True, ndr_prefix = ndr_prefix, metric_name=metric_name, 
                    agg_method_names=['mean'], infl_method_names=infl_method_names)
    
        score_df = df.pivot(index=["infl", "agg", "layer", "module"], columns=["task", "seed"], values=metric_name)
        score_df.to_pickle(cache_path)

    score_df *= 100.0

    infl_method_names = orig_infl_names
    infl_method_names.remove("hf_we_")
    infl_method_names.remove("hf_we_topk_10")


    # mean_df = score_df.mean(axis=1)
    conf_int_df = get_confidence_interval(score_df)
    # std_df = score_df.std(axis=1)

    very_min = conf_int_df.min(axis=1).min()
    very_max = conf_int_df.max(axis=1).max()

    module_name_list = list(module_names.keys())


    ndr_per_infl_values = []
    infl_method_mapping = {"datainf": "DataInf", "hf_we_": "TracIn$_{we}$", "hf_we_topk_10": "TracIn$^{10}_{we}$", "hf": "TracIn", "cos": "Cosine"}
    for infl_method in infl_method_names:
        infl_name = infl_method_mapping[infl_method]
        infl_chart = {"id": infl_method, "name": infl_name}
        x_markers = infl_chart.setdefault("x_markers", [])
        end_x_marks = []
        # x_min_values = []
        # x_mean_values = []
        # x_max_values = []
        # x_markers = [] 
        # common_start = []
        # common_end = []
        we_value = None
        if infl_method == "hf": 
            start_y_min_values = infl_chart.setdefault("start_y_min_values", [])
            start_y_mean_values = infl_chart.setdefault("start_y_mean_values", [])
            start_y_max_values = infl_chart.setdefault("start_y_max_values", [])
            # selecting also hf_we_ and hf_we_topk_10
            we_value = conf_int_df.loc[("hf_we_topk_10", "mean", "WE", "all")]
            start_y_min_values.append(we_value["min_v"])
            start_y_mean_values.append(we_value["mean"])
            start_y_max_values.append(we_value["max_v"])
            x_markers.append("TI$^{10}_{we}$")

            we_value = conf_int_df.loc[("hf_we_", "mean", "WE", "all")]
            start_y_min_values.append(we_value["min_v"])
            start_y_mean_values.append(we_value["mean"])
            start_y_max_values.append(we_value["max_v"])
            x_markers.append("TI$_{we}$")        

            we_value = conf_int_df.loc[("hf", "mean", "WE", "all")]
            start_y_min_values.append(we_value["min_v"])
            start_y_mean_values.append(we_value["mean"])
            start_y_max_values.append(we_value["max_v"])
            x_markers.append("WE")
        elif infl_method == "cos":
            start_y_min_values = infl_chart.setdefault("start_y_min_values", [])
            start_y_mean_values = infl_chart.setdefault("start_y_mean_values", [])
            start_y_max_values = infl_chart.setdefault("start_y_max_values", [])
            we_value = conf_int_df.loc[("cos", "mean", "WE", "all")]
            start_y_min_values.append(we_value["min_v"])
            start_y_mean_values.append(we_value["mean"])
            start_y_max_values.append(we_value["max_v"])
            x_markers.append("WE")
        prev_we_value = we_value
        next_we_value = None
        for cl_module in cl_modules.keys():
            end_y_min_values = infl_chart.setdefault("end_y_min_values", [])
            end_y_mean_values = infl_chart.setdefault("end_y_mean_values", [])
            end_y_max_values = infl_chart.setdefault("end_y_max_values", [])
            we_value = conf_int_df.loc[(infl_method, "mean", "CL", cl_module)]
            end_y_min_values.append(we_value["min_v"])
            end_y_mean_values.append(we_value["mean"])
            end_y_max_values.append(we_value["max_v"])
            end_x_marks.append(cl_modules[cl_module])
            if next_we_value is None:
                next_we_value = we_value
        if prev_we_value is not None:
            for module_name in module_name_list:
                y_min_values = infl_chart.setdefault(f"{module_name}:min", [])
                y_mean_values = infl_chart.setdefault(f"{module_name}:mean", [])
                y_max_values = infl_chart.setdefault(f"{module_name}:max", [])
                y_min_values.append(prev_we_value["min_v"])
                y_mean_values.append(prev_we_value["mean"])
                y_max_values.append(prev_we_value["max_v"])                
        for module_layer in module_layers:
            layer_name = str(module_layer)
            x_markers.append(module_layer + 1)
            for module_name in module_name_list:
                y_min_values = infl_chart.setdefault(f"{module_name}:min", [])
                y_mean_values = infl_chart.setdefault(f"{module_name}:mean", [])
                y_max_values = infl_chart.setdefault(f"{module_name}:max", [])
                we_value = conf_int_df.loc[(infl_method, "mean", layer_name, module_name)]
                y_min_values.append(we_value["min_v"])
                y_mean_values.append(we_value["mean"])
                y_max_values.append(we_value["max_v"])
        if next_we_value is not None:
            for module_name in module_name_list:
                y_min_values = infl_chart.setdefault(f"{module_name}:min", [])
                y_mean_values = infl_chart.setdefault(f"{module_name}:mean", [])
                y_max_values = infl_chart.setdefault(f"{module_name}:max", [])
                y_min_values.append(next_we_value["min_v"])
                y_mean_values.append(next_we_value["mean"])
                y_max_values.append(next_we_value["max_v"]) 
        x_markers.extend(end_x_marks)
        ndr_per_infl_values.append(infl_chart)

    plt.ioff()
    fig, axes = plt.subplots(1, len(ndr_per_infl_values), figsize=figsize)
    ordered_handles = []
    ordered_labels = []
    for chart_id, chart_data in enumerate(ndr_per_infl_values):
        ax =  axes[chart_id]
        start_id = 1
        if "start_y_mean_values" in chart_data:
            start_means = chart_data["start_y_mean_values"]
            start_id += (len(start_means) - 1)
        if start_id > 1:
            adjustment = [1, 3.5, 6]
            start_id = adjustment[-1]
        else:
            adjustment = [1]
        x_max_pos = None
        for module_name in module_name_list:
            y_min_values = chart_data[f"{module_name}:min"]
            y_mean_values = chart_data[f"{module_name}:mean"]
            y_max_values = chart_data[f"{module_name}:max"]
            local_x = np.arange(len(y_mean_values)) + start_id
            if max_of == module_name:
                max_pos = np.argmax(y_mean_values)
                x_max_pos = local_x[max_pos]
            ln = ax.plot(local_x, y_mean_values, label=module_names[module_name], linewidth=1)
            if chart_id == 0:
                ordered_handles.append(ln[0])
                ordered_labels.append(module_names[module_name])
            ax.fill_between(local_x, y_min_values, y_max_values, color=ln[0].get_color(), alpha=0.2)
            # start_id += len(y_mean_values)
        start_id += (len(y_mean_values) - 1)
        if "end_y_mean_values" in chart_data:
            end_means = chart_data["end_y_mean_values"]
            local_x = np.arange(len(end_means)) + start_id
            ln = ax.plot(local_x, end_means, color="black", linewidth=1) 
            ax.fill_between(local_x, chart_data["end_y_min_values"], chart_data["end_y_max_values"], color=ln[0].get_color(), alpha=0.2)           
            # start_id += len(end_means)
        start_id = 1
        if "start_y_mean_values" in chart_data:
            start_means = chart_data["start_y_mean_values"]
            local_x = np.arange(len(start_means)) + start_id
            ln = ax.plot(adjustment, start_means, color="black", linewidth=1) #linestyle="--", linewidth=1.5)
            if len(adjustment) > 1 and chart_id == 1:
                ordered_handles.append(ln[0])
                ordered_labels.append("WE methods")
            ax.fill_between(adjustment, chart_data["start_y_min_values"], chart_data["start_y_max_values"], color=ln[0].get_color(), alpha=0.2)
            # start_id += len(start_means)            
        x_markers = chart_data["x_markers"]
        tick_poss = adjustment + list(np.arange(len(x_markers) - len(adjustment)) + 1 + adjustment[-1])
        ax.set_xlim(0.5, (adjustment[-1] + len(x_markers) - len(adjustment))+0.5)
        filtered_labels = []
        filtered_ticks = []
        for mid, m in enumerate(x_markers):
            try:
                num = int(m)
                if num % 3 == 0:
                    filtered_ticks.append(tick_poss[mid])
                    filtered_labels.append(str(num))
                else:
                    if len(filtered_labels) == 0:
                        filtered_ticks.append(tick_poss[mid])
                        filtered_labels.append(str(num))
                    # else:
                    #     filtered_labels.append("")
            except ValueError:
                filtered_ticks.append(tick_poss[mid])
                filtered_labels.append(m)
        ax.set_xticks(filtered_ticks)
        ax.set_xticklabels(filtered_labels, fontsize=7, verticalalignment='center')
        ax.tick_params(axis='x', length=1)
        ax.tick_params(axis='y', pad=0, length=1)
        if chart_id == 0:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
            ax.set_ylabel(y_title, fontsize=10, labelpad=0)
            ax.set_yticks([40,50,60,70,80,90])
        else:
            ax.set_yticks([])
        ax.set_yticklabels(ax.get_yticks(), fontsize=7) 
        ax.set_ylim(very_min, very_max)
        ax.set_xlabel("Layers", fontsize=8, labelpad=0)
        ax.set_title(chart_data["name"], fontsize=10, pad=0)
        for yhline in [40,50,60,70,80,90]:
            ax.axhline(y=yhline, color="gray", linewidth=0.5, linestyle="--")
        if x_max_pos is not None:
            ax.axvline(x=x_max_pos, color="gray", linewidth=0.5, linestyle="--")

    fig.legend(ordered_handles, ordered_labels, loc='lower center', fontsize=8,
                ncol=len(ordered_handles), borderaxespad = 0,  # Arrange all legend items in one row
                bbox_to_anchor=(0.5, 0)  # Adjust position (centered below the grid)
        )        
    fig.tight_layout(w_pad=0, h_pad=0, pad=0, rect=[0, 0.13, 1, 1])
    fig.savefig(os.path.join(base_path, "plots", f"layers-ndr-{suffix}.pdf"))
    plt.close(fig)
    pass

def mean_confidence(x, confidence_level = 0.95):
    mean = x.mean()
    n = len(x)
    df = n - 1
    se = stats.sem(x)  # same as std/sqrt(n)

    # t-based confidence interval
    ci_low, ci_high = stats.t.interval(confidence_level, df, loc=mean, scale=se)    
    return {
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high
    }   

def auc_recall_metric_graphs(metrics: list[str],
                    metric_names: list[str],
                    methods: list[str] = ["hf", "cos", "datainf", "outlier"],
                    method_names: list[str] = ["HF", "Cosine", "DataInf", "Outlier"],
                    metric: Literal["auc", "recall"] = "auc",
                    modules: list[str] = ["A q", "B q", "A v", "B v"],
                    module_names: list[str] = ["A q", "B q", "A v", "B v"],
                    y_title = "Layer-wise AUC, \\%", 
                    figsize = (8, 2),
                    out_dir="."):

    very_min = 100
    very_max = 0

    dfs = {}
    for metric_file in metrics:
        df = pd.read_csv(metric_file)
        dfs[metric_file] = df

    # include_repsim = False
    # if "repsim" in methods:
    #     include_repsim = True
    plt.ioff()
    fig, axes = plt.subplots(len(methods), len(metrics), figsize=figsize, squeeze=False)
    ordered_handles = []
    ordered_labels = []
    repsim_handle_init = []
    def default_setting(ax, method_name, metric_name, i, j, max_x):
        ax.set_xlim(0.5, max_x+0.5)

        x_values = np.arange(1, max_x + 1)
        xticks = [x for x in x_values if x % 3 == 0 or x == 1]  # always show layer 1 + every 3rd
        xlabels = [str(x) for x in xticks]

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=7, verticalalignment='center')
        ax.tick_params(axis='x', length=1)
        ax.tick_params(axis='y', pad=0, length=1)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))            
        # ax.set_ylabel(y_title, fontsize=10, labelpad=0)
        if j == 0:
            ax.set_ylabel(method_name, fontsize=10, labelpad=0)
        ylabels = [60,70,80,90]
        ax.set_yticks(ylabels)
        ax.set_yticklabels(ylabels, fontsize=7, verticalalignment='center')
        ax.set_ylim(very_min, very_max)
        # ax.set_xlabel("Layers", fontsize=8, labelpad=0)
        if i == 0:
            ax.set_title(metric_name, fontsize=10, pad=0)
        for yhline in [60,70,80,90]:
            ax.axhline(y=yhline, color="gray", linewidth=0.5, linestyle="--")
        # if x_max_pos is not None:
        #     ax.axvline(x=x_max_pos, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlim(0.5, max_x+0.5)

        x_values = np.arange(1, max_x + 1)
        xticks = [x for x in x_values if x % 3 == 0 or x == 1]  # always show layer 1 + every 3rd
        xlabels = [str(x) for x in xticks]

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=7, verticalalignment='center')
        ax.tick_params(axis='x', length=1)
        ax.tick_params(axis='y', pad=0, length=1)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))            
        # ax.set_ylabel(y_title, fontsize=10, labelpad=0)
        if j == 0:
            ax.set_ylabel(method_name, fontsize=10, labelpad=0)
        ylabels = [60,70,80,90]
        ax.set_yticks(ylabels)
        ax.set_yticklabels(ylabels, fontsize=7, verticalalignment='center')
        ax.set_ylim(very_min, very_max)
        # ax.set_xlabel("Layers", fontsize=8, labelpad=0)
        if i == 0:
            ax.set_title(metric_name, fontsize=10, pad=0)
        for yhline in [60,70,80,90]:
            ax.axhline(y=yhline, color="gray", linewidth=0.5, linestyle="--")
        # if x_max_pos is not None:
        #     ax.axvline(x=x_max_pos, color="gray", linewidth=0.5, linestyle="--")
    for i, method in enumerate(methods):
        method_name = method_names[i]
        if method != "repsim":        
            for j, metric_file in enumerate(metrics):
                ax =  axes[i,j]
                df = dfs[metric_file]
                df = df[df["method"] == method].copy()
                df['layer'] = df['layer'].astype(int)
                df['seed'] = df['seed'].astype(int)
                df[metric] = df[metric].astype(float)
                metric_name = metric_names[j]
                max_x = 0
                for k, module in enumerate(modules):
                    module_name = module_names[k]            
                    module_df0 = df[df["module"] == module]
                    per_seed = module_df0.groupby(["seed", "layer"])[metric].mean().reset_index()
                    module_df = per_seed.groupby(["layer"])[metric].agg(mean_confidence).apply(pd.Series).reset_index()
                    local_x = module_df["layer"] + 1
                    max_x = max(max_x, local_x.max())
                    y_mean_values = module_df["mean"] * 100
                    y_min_values = module_df["ci_low"] * 100
                    y_max_values = module_df["ci_high"] * 100
                    very_min = min(very_min, y_min_values.min())
                    very_max = max(very_max, y_max_values.max())
                    ln = ax.plot(local_x, y_mean_values, label=module_name, linewidth=1)
                    if (i == 0) and (j == 0):
                        ordered_handles.append(ln[0])
                        ordered_labels.append(module_name)
                    ax.fill_between(local_x, y_min_values, y_max_values, color=ln[0].get_color(), alpha=0.2)
                    default_setting(ax, method_name, metric_name, i, j, max_x)
        else:
            submethods = ["repsim-last", "repsim-mean"]
            submethod_names = ["RepSim Last", "RepSim Mean"]
            for j, metric_file in enumerate(metrics):
                ax =  axes[i,j]
                df = dfs[metric_file].copy()
                metric_name = metric_names[j]
                max_x = 0
                for k, submethod in enumerate(submethods):
                    submethod_name = submethod_names[k]          
                    if submethod == "repsim-last":
                        color = "black"
                        linestyle = "-"
                    elif submethod == "repsim-mean":
                        color = "black"
                        linestyle = "--"
                    else:
                        color = None         # let Matplotlib choose
                        linestyle = "-"                      
                    submethod_df0 = df[df["method"] == submethod].copy()                    
                    submethod_df0['layer'] = submethod_df0['layer'].astype(int)
                    submethod_df0['seed'] = submethod_df0['seed'].astype(int)
                    submethod_df0[metric] = submethod_df0[metric].astype(float)
                    per_seed = submethod_df0.groupby(["seed", "layer"])[metric].mean().reset_index()
                    submethod_df = per_seed.groupby(["layer"])[metric].agg(mean_confidence).apply(pd.Series).reset_index()
                    local_x = submethod_df["layer"] + 1
                    max_x = max(max_x, local_x.max())
                    y_mean_values = submethod_df["mean"] * 100
                    y_min_values = submethod_df["ci_low"] * 100
                    y_max_values = submethod_df["ci_high"] * 100
                    very_min = min(very_min, y_min_values.min())
                    very_max = max(very_max, y_max_values.max())
                    ln = ax.plot(local_x, y_mean_values, label=submethod_name, linewidth=1,
                                color=color,
                                linestyle=linestyle                                 
                                 )
                    if submethod not in repsim_handle_init:
                        ordered_handles.append(ln[0])
                        ordered_labels.append(submethod_name)
                        repsim_handle_init.append(submethod)
                    ax.fill_between(local_x, y_min_values, y_max_values, color=ln[0].get_color(), alpha=0.2)
                    default_setting(ax, method_name, metric_name, i, j, max_x)


    fig.legend(ordered_handles, ordered_labels, loc='lower center', fontsize=8,
                ncol=len(ordered_handles), borderaxespad = 0,  # Arrange all legend items in one row
                bbox_to_anchor=(0.5, 0)  # Adjust position (centered below the grid)
        )        
    fig.tight_layout(w_pad=0, h_pad=0, pad=0, rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(out_dir, f"layers-{metric}-ds.pdf"))
    plt.close(fig)
    pass


if __name__ == "__main__":

    # auc_recall_metric_graphs(
    #     metrics=[
    #         "./data/dev/qwen/sentense/metrics_sentense.csv",
    #     ],
    #     metric_names=[
    #         "Sentence AUC vs Layer, \\%", "Math AUC vs Layer, \\%", "Math w. Reason AUC vs Layer, \\%"
    #     ],
    #     methods=['hf','cos','datainf','outlier','kr-ekfac', 'repsim'],
    #     method_names=["TracIn", "Cosine", "DataInf", "Outlier Gradient", "EKFAC", 'RepSim'],
    #     metric='auc',
    #     modules=['A q', 'B q', 'A v', 'B v'],
    #     module_names=["Query A", "Query B", "Value A", "Value B"],
    #     y_title="Layer-wise AUC, \\%",
    #     figsize=(8, 7),
    #     out_dir="./data/dev/qwen/sentense"
    # )
    # pass

    # auc_recall_metric_graphs(
    #     metrics=[
    #         "./data/dev/qwen/sentense/metrics_sentense.csv",
    #     ],
    #     metric_names=[
    #         "Sentence Recall vs Layer, \\%", "Math Recall vs Layer, \\%", "Math w. Reason Recall vs Layer, \\%"
    #     ],
    #     methods=['hf','cos','datainf','outlier','kr-ekfac', 'repsim'],
    #     method_names=["TracIn", "Cosine", "DataInf", "Outlier Gradient", "EKFAC", 'RepSim'],
    #     metric='recall',
    #     modules=['A q', 'B q', 'A v', 'B v'],
    #     module_names=["Query A", "Query B", "Value A", "Value B"],
    #     y_title="Layer-wise Recall, \\%",
    #     figsize=(8, 7),
    #     out_dir="./data/dev/qwen/sentense"
    # )
    # pass

    # def util_cat(model:str, type: str):
    #     df1 = pd.read_csv(f"./data/{model}/ds-0/metrics_{type}.csv")
    #     df2 = pd.read_csv(f"./data/{model}/ds-0/metrics-rs_{type}.csv")
    #     df3 = pd.concat([df1, df2], axis=0)
    #     df3.to_csv(f"./data/{model}/ds-0/metrics-all_{type}.csv")    


    # auc_recall_metric_graphs(
    #     metrics=[
    #         "./data/qwen/ds/metrics_sentense.csv",
    #         "./data/qwen/ds/metrics_math.csv",
    #         "./data/qwen/ds/metrics_mathR.csv"
    #     ],
    #     metric_names=[
    #         "Sentence AUC vs Layer, \\%", "Math AUC vs Layer, \\%", "Math w. Reason AUC vs Layer, \\%"
    #     ],
    #     methods=['hf','cos','datainf','outlier','kr-ekfac', 'repsim'],
    #     method_names=["TracIn", "Cosine", "DataInf", "Outlier Gradient", "EKFAC", "RepSim"],
    #     metric='auc',
    #     modules=['A q', 'B q', 'A v', 'B v'],
    #     module_names=["Query A", "Query B", "Value A", "Value B"],
    #     y_title="Layer-wise AUC, \\%",
    #     figsize=(8, 7),
    #     out_dir="./data/qwen/ds"
    # )
    # pass

    # auc_recall_metric_graphs(
    #     metrics=[
    #         "./data/qwen/ds/metrics_sentense.csv",
    #         "./data/qwen/ds/metrics_math.csv",
    #         "./data/qwen/ds/metrics_mathR.csv"
    #     ],
    #     metric_names=[
    #         "Sentence Recall vs Layer, \\%", "Math Recall vs Layer, \\%", "Math w. Reason Recall vs Layer, \\%"
    #     ],
    #     methods=['hf','cos','datainf','outlier','kr-ekfac', 'repsim'],
    #     method_names=["TracIn", "Cosine", "DataInf", "Outlier Gradient", "EKFAC", "RepSim"],
    #     metric='recall',
    #     modules=['A q', 'B q', 'A v', 'B v'],
    #     module_names=["Query A", "Query B", "Value A", "Value B"],
    #     y_title="Layer-wise Recall, \\%",
    #     figsize=(8, 7),
    #     out_dir="./data/qwen/ds"
    # )
    # pass

    # auc_recall_metric_graphs(
    #     metrics=[
    #         "./data/mistral/ds/metrics_sentense.csv",
    #         "./data/mistral/ds/metrics_math.csv",
    #         "./data/mistral/ds/metrics_mathR.csv"
    #     ],
    #     metric_names=[
    #         "Sentence AUC vs Layer, \\%", "Math AUC vs Layer, \\%", "Math w. Reason AUC vs Layer, \\%"
    #     ],
    #     methods=['hf','cos','datainf','outlier','kr-ekfac', 'repsim'],
    #     method_names=["TracIn", "Cosine", "DataInf", "Outlier Gradient", "EKFAC", "RepSim"],
    #     metric='auc',
    #     modules=['A q', 'B q', 'A v', 'B v'],
    #     module_names=["Query A", "Query B", "Value A", "Value B"],
    #     y_title="Layer-wise AUC, \\%",
    #     figsize=(8, 7),
    #     out_dir="./data/mistral/ds"
    # )
    # pass

    # auc_recall_metric_graphs(
    #     metrics=[
    #         "./data/mistral/ds/metrics_sentense.csv",
    #         "./data/mistral/ds/metrics_math.csv",
    #         "./data/mistral/ds/metrics_mathR.csv"
    #     ],
    #     metric_names=[
    #         "Sentence Recall vs Layer, \\%", "Math Recall vs Layer, \\%", "Math w. Reason Recall vs Layer, \\%"
    #     ],
    #     methods=['hf','cos','datainf','outlier','kr-ekfac', 'repsim'],
    #     method_names=["TracIn", "Cosine", "DataInf", "Outlier Gradient", "EKFAC", "RepSim"],
    #     metric='recall',
    #     modules=['A q', 'B q', 'A v', 'B v'],
    #     module_names=["Query A", "Query B", "Value A", "Value B"],
    #     y_title="Layer-wise Recall, \\%",
    #     figsize=(8, 7),
    #     out_dir="./data/mistral/ds"
    # )
    # pass
    
    # ndr_test()
    # pass

    # with open("data/llama/metrics/need_fix/mrpc-vote.jsonlist", "r") as f:
    #     json_lines = f.readlines()
    # all_metrics = [json.loads(l) for l in json_lines]
    # counts = {}
    # for metrics in all_metrics:
    #     c = metrics['config']
    #     counts.setdefault((c["infl_method"], c["module_name"]), []).append(c["seed"])
    # cnts = {s:c for s,c in counts.items() if len(c) < 10}

    pass


    # network = "mistral"
    network="roberta"
    # network="qwen"
    group_file = "./groups.json"
    base_path = f"data/{network}"
    selected_layers = network_layers[network]
    noise_only = False

    model_num_layers = {
        "roberta": 24,
        "qwen": 28
    }

    infl_method_names = ["hf_we_", "hf_we_topk_10",  "hf", "cos", "datainf", "denoise", "rand"]
    # agg_method_names = ["mean", "rank-c", "vote2-c", '']
    # setup_interactions(base_path, use_ndr=False, metric_name="best_accuracy_1",
    #                     agg_method_names = agg_method_names, dom_threshold = 0.75,
    #                     infl_method_names=infl_method_names)

    infl_method_names = ["datainf", "hf_we_", "hf_we_topk_10",  "hf", "cos"] #, "denoise", "rand"]
    agg_method_names = ["mean", '']
    res_suffix = "bl"
    # setup_interactions(base_path, use_ndr=False, metric_name="best_accuracy_1",
    #                     agg_method_names = agg_method_names, dom_threshold = 0.5,
    #                     infl_method_names=infl_method_names, suffix="-mean",
    #                     res_suffix=res_suffix)    
    selected_network_modules = network_modules[network]
    # layer_ndr_metric_graphs(base_path, metric_name=30,
    #                     infl_method_names=infl_method_names, suffix="",
    #                     module_layers = list(range(network_layer_count[network])),
    #                     module_names = selected_network_modules,
    #                     max_of = network_best_ndr[network],figsize=(8, 1.5))
    pass 

    # for i, columns in enumerate([["datainf"], ["hf_we_", "hf_we_topk_10",  "hf"], ["cos"]]):
    #     setup_interactions(base_path, use_ndr=False, metric_name="best_accuracy_1",
    #                         agg_method_names = ["mean"], selected_layers = selected_layers,
    #                         infl_method_names=columns, dom_threshold = 0.5,
    #                         res_suffix=res_suffix, suffix = f'-MN{i}')
    pass
    
    # dfs = []
    # for i in range(3):
    #     in_path = os.path.join(base_path, 'metrics', f"intranks-best_accuracy_1-MN{i}.pcl")
    #     df = pd.read_pickle(in_path)
    #     dfs.append(df)
    # pass
        
    # pass
    # df_cancellations(base_path, ["qnli"], run_ids = [42], selected_layers = ["WE", "00-05", 'CL'])
    # create_cancel_table(base_path, layers = selected_layers)
    pass
    # compute_corr_matrix(base_path, selected_layers=selected_layers, noise_only = noise_only)
    #                     # tasks = ["qnli", "mrpc", "sst2", "qqp", "cola", "stsb"])
    # compute_all_corr_matrix(base_path, selected_layers=selected_layers, noise_only = noise_only)
                            # tasks = ["qnli", "mrpc", "sst2", "qqp", "cola", "stsb"])
    # draw_corr_heatmap(base_path, noise_only = noise_only) #tasks = ["qnli", "mrpc", "sst2", "qqp", "cola", "stsb"])
    # draw_all_corr_heatmap(base_path, noise_only = noise_only)
    base_paths = [
        ('Roberta-Large', 'data/roberta'),
        ('Llama-3.2 1B', 'data/llama'),
        ('Mistral 7B', 'data/mistral'),
        ('Qwen-2.5 1.5B', 'data/qwen'),
    ]
    # draw_all_models_corr_heatmap(base_paths, noise_only = noise_only)
    pass

    # create_tun2_metric_table(metric_name="noise_30", ds_ranks=False, mul = 100, highlight_max = True, out_folder=base_path,
    #                             with_row_id = False, prec = 1)
    # pass

    # create_tun2_metric_table(metric_name="best_accuracy_1", mul = 100, 
    #                          highlight_max = True, out_folder=base_path,
    #                             with_row_id = False, prec=1,
    #                             res_suffix="all")
    # pass

    # run_spearman_total2(out_folder=base_path, layers = selected_layers) #, run_ids = [0,1,2,3,4])
    pass

    # run_spearman_total(out_folder=base_path, layers = selected_layers) #, run_ids = [0,1,2,3,4])
    # create_tun2_agg_metrics_table(out_folder=base_path, res_suffix = res_suffix)

    # draw_perf_diffs2(out_folder = base_path, layers = selected_layers,
    #                 pvalue = 0.1) # run_ids = [0,1,2,3,4])
    pass

    # create_tun2_agg_diffs_table(metric_name="best_accuracy_1", ds_ranks=False, mul = 100, 
    #                          highlight_max = True, out_folder=base_path, prec=1, run_ids = [0,1,2,3,4])
    # pass

    # run_friedman_tests(metric_name="best_accuracy_1", out_folder=base_path)
    # pass 

    # run_wilcoxon_tests(metric_name="best_accuracy_1", out_folder=base_path)
    # pass 

    # run_spearman_tests(metric_name="best_accuracy_1", ndr_metric_name="noise_30", 
    #                    out_folder=base_path, suffix="-30") #, ndr_delta = 0.1)
    pass 

    # where_is_the_noise(base_path, task='qnli', infl_method='datainf', 
    #                    module_pattern=".*\\.layers\\.([4-7])\\..*\\.lora_(A)\\..*",
    #                 #    module_pattern=".*\\.layer\\.(1[8-9]|2[0-3])\\..*\\.lora_(A|B)\\..*",
    #                    device = 'cpu')
    # pass
    # agg_method_names = ["mean", "rank", "rmin", "vote"]
    # agg_method_names = ["rank", "rank-c", "mean", "mean-c", "cset", "cset-c", "vote2", "vote2-c"]
    agg_method_names = ["vote2-c-10", "vote2-c-20", "vote2-c-30", "vote2-c-40", "vote2-c-50", "vote2-c-60", "vote2-c-70", 'vote2-c-80', 'vote2-c-90', 'vote2-c-100']
    # agg_method_names = ['cset-c']
    dss = ["mrpc", "qnli", "sst2", "qqp", "cola", "mnli", "rte", "stsb"]

    process_ndr_table(base_path, tasks=dss, with_row_id=False, custom_suffix = "-agga", 
                      best_group_by=["infl", "agg"], 
                    #   layers=selected_layers, 
                      ndr_prefix="ndr_agga",
                    #   agg_method_names=agg_method_names
                      )
    pass 

    draw_vote_k_ndr(
        base_path, 
        methods = ["hf","cos","datainf"],
        method_names = ["TracIn","Cosine","DataInf"],
        tasks = benchmark, 
        modules = [
            # "value B",
            # "query B",
            # "value A",
            # "query A",
            'self_attn v_proj B',
            'self_attn q_proj B', 
            'self_attn v_proj A',
            'self_attn q_proj A', 
        ],
        module_names = [
            "Value B",
            "Query B",
            "Value A",
            "Query A",
        ],
        metric_name = 30, 
        ndr_prefix = "ndr_vote_k", 
        custom_suffix = "vote_k",
        agg_methods=[
            "vote2-c-10",
            "vote2-c-20",
            "vote2-c-30",
            "vote2-c-40",
            "vote2-c-50",
            "vote2-c-60",
            "vote2-c-70",
            "vote2-c-80",
            "vote2-c-90",
            "vote2-c-100"
        ], figsize=(2 * 3, 2 * 4),
        num_layers = model_num_layers[network]
    )
    # # dss = ["mrpc", "qnli", "sst2", "qqp"]
    # dss = ["mrpc"]
    # base_infl_path = os.path.join(base_path, "infl-tensors")
    # group_file_2 = os.path.abspath(os.path.join(base_path, "groups.json"))
    # for ds in dss:
    #     compute_ndr_metrics_table(base_infl_path, task=ds, 
    #                             group_file=group_file_2, levels=[5,10,15,20,25,30,35,40,45,50,60,70,80,90],
    #                             infl_methods = ['hf', 'cos', 'datainf', 'hf_we_', 'hf_we_topk_10'],
    #                             agg_method_names=agg_method_names)
    pass 

    # roberta_layers = ['WE', '00-05', '06-11', '12-17', '18-23', 'CL']
    # llama_layers = ['WE', '00-03', '04-07', '08-11', '12-15', 'CL']
    # mistral_layers = ['WE', '00-07', '08-15', '16-23', '24-31', 'CL']
    # selected_layers = mistral_layers
    # for am in ['mean', 'rank', 'vote', 'rmin']:
    #     for infl_ms in [['datainf'], ['hf', 'hf_we_', 'hf_we_topk_10'], ['cos']]:
    #         process_ndr_table(base_path, tasks=benchmark, with_row_id=False,
    #                             layers=selected_layers, 
    #                             infl_method_names=infl_ms,
    #                             agg_method_names=[am, f'{am}-c'], custom_suffix=f"-{infl_ms[0]}-{am}-s")    
    # process_ndr_table(base_path, tasks=dss, with_row_id=False, custom_suffix = "-vote-k", 
    #                   best_group_by=["infl", "agg"], layers=selected_layers, ndr_prefix="ndr_vote_k",
    #                   agg_method_names=agg_method_names)
    pass
    # process_ndr_table(base_path, tasks=benchmark, with_row_id=False,
    #                     layers=selected_layers, agg_method_names=agg_method_names)
    # process_ndr_table(base_path, tasks=benchmark, with_row_id=False, custom_suffix = "-hf",
    #                     agg_method_names=['vote2-c'],
    #                     infl_method_names=['hf'])
    # draw_noise_distr(base_path, tasks=['qnli'], layers = selected_layers, suffix="qnli-n-sm", agg_name="mean",
    #                         figsize = (8,1.5), no_left_no_bottom = True)
    # pass
    # draw_noise_distr(base_path, tasks=benchmark, layers = selected_layers, suffix="all", agg_name="mean",
    #                     figsize=(8, 10))
    pass
    # draw_noise_distr(base_path, tasks=benchmark, layers = selected_layers, suffix="rankc", agg_name="rank-c")
    # draw_noise_distr(base_path, tasks=benchmark, layers = selected_layers, suffix="votec", agg_name="vote-c")
    pass

    # roberta_selected_methods = {
    #         "datainf:commonset-20:CL:all": "datainf, cset$^{20}$, CL", 
    #         "hf:commonset-20:total:all": "hf, cset$^{20}$",
    #         "cos:rank:23:value B": "cos, rank, L23, value B", 
    #         # "hf:mean:total:all": "hf, mean, total",  
    #     }
    # llama_selected_methods = {
    #         "cos:rank::layers 9 self_attn v_proj B": "cos, rank, L8, value B", 
    #         "datainf:rank::layers 7 self_attn v_proj B": "datainf, rank, L7, value B",
    #         "hf:rank:04-07:all": "hf, rank, L4-7", 
    #         # "hf:mean:total:all": "hf, mean, total",  
    #     }    
    # draw_ndr_curves(base_path, selected_methods=llama_selected_methods)
    # pass 

    # draw_tun2_bar_metric(base_path, metric_name="accuracy",
    #                      selected_methods={
    #                         # ('denoise', '', ''): "denoise",
    #                         ('hf_we_', 'mean', 'WE'): "hf$_{we}$",
    #                         # ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$^{10}_{we}$', 'legend_order': 1},
    #                         ('hf', 'mean', 'WE'): "hf, WE",
    #                         # ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
    #                         # ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
    #                         ('cos', 'mean', '12-17'): "cos, 12-17",
    #                         ('datainf', 'mean', '18-23'): "datainf, 18-23",
    #                         # ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                         ('rand', '', ''): "rand",
    #                     },
    #                     from_method=("rand", "", ""),
    #                     suffix = "-rand",
    #                     title = "Accuracy difference from random")
    pass

    # draw_tun2_bar_metric(base_path, metric_name="accuracy",
    #                      selected_methods={
    #                         # ('denoise', '', ''): "denoise",
    #                         ('hf_we_', 'mean', 'WE'): "hf$_{we}$",
    #                         # ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$^{10}_{we}$', 'legend_order': 1},
    #                         ('hf', 'mean', 'WE'): "hf, WE",
    #                         # ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
    #                         # ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
    #                         ('cos', 'mean', '12-17'): "cos, 12-17",
    #                         ('datainf', 'mean', '18-23'): "datainf, 18-23",
    #                         # ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #                         ('rand', '', ''): "rand",
    #                     },
    #                     # from_method=("rand", "", ""),
    #                     # suffix = "-rand",
    #                     title = "Accuracy difference from first finetune")    

    roberta_selected_methods = {
        ('denoise', '', ''): {'color': 'gray', 'legend_name': 'Full', 'legend_order': -1},
        # ('hf_we_', 'mean', 'WE'): {'color': '#33e0ff', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
        # ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$^{10}_{we}$', 'legend_order': 1},
        ('hf', 'mean', 'WE'): {'color': 'blue', 'legend_name': 'TracIn, WE', 'legend_order': 2},
        # ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
        # ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
        ('cos', 'mean', '12-17'): {'color': 'green', 'legend_name': 'Cosine, 12-17', 'legend_order': 5},
        ('datainf', 'mean', '18-23'): {'color': 'red', 'legend_name': 'DataInf, 18-23', 'legend_order': 6},
        # ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
        ('rand', '', ''): {'color': 'gray', 'legend_name': 'Random', 'legend_order': 8},
    }

    llama_selected_methods = {
        ('denoise', '', ''): {'color': 'gray', 'legend_name': 'Full', 'legend_order': -1},
        # ('hf_we_', 'mean', 'WE'): {'color': '#33e0ff', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
        # ('hf_we_topk_10', 'mean', 'WE'): "hf$^{10}_{we}$",
        ('hf', 'mean', '04-07'): {'color': 'blue', 'legend_name': 'TracIn, 04-07', 'legend_order': 2},
        # ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
        # ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
        ('cos', 'mean', '04-07'): {'color': 'green', 'legend_name': 'Cosine, 04-07', 'legend_order': 5},
        ('datainf', 'mean', '04-07'): {'color': 'red', 'legend_name': 'DataInf, 04-07', 'legend_order': 6},
        # ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
        ('rand', '', ''): {'color': 'gray', 'legend_name': 'Random', 'legend_order': 8},
    }

    # draw_tun2_bar_metric(base_path, metric_name="accuracy",
    #                      selected_methods=llama_selected_methods,
    #                     from_method=("rand", "", ""),
    #                     suffix = "-rand",
    #                     title = "Accuracy difference from random")

    # draw_tun2_bar_metric(base_path, metric_name="accuracy",
    #                      selected_methods=llama_selected_methods,
    #                     # from_method=("rand", "", ""),
    #                     # suffix = "-rand",
    #                     title = "Accuracy difference from first finetune")        
    # pass

    
    mistral_selected_methods = {
        ('denoise', '', ''): {'color': 'gray', 'legend_name': 'Full', 'legend_order': -1},
        # ('hf_we_', 'mean', 'WE'): {'color': '#33e0ff', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
        # ('hf_we_topk_10', 'mean', 'WE'): "hf$^{10}_{we}$",
        ('hf_we_topk_10', 'mean', 'WE'): {'color': 'blue', 'legend_name': 'TracIn$^{10}_{we}$, WE', 'legend_order': 2},
        # ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
        # ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
        ('cos', 'mean', '08-15'): {'color': 'green', 'legend_name': 'Cosine, 08-15', 'legend_order': 5},
        ('datainf', 'mean', '08-15'): {'color': 'red', 'legend_name': 'DataInf, 08-15', 'legend_order': 6},
        # ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
        ('rand', '', ''): {'color': 'gray', 'legend_name': 'Random', 'legend_order': 8},        
    }

    qwen_selected_methods = {
        ('denoise', '', ''): {'color': 'gray', 'legend_name': 'Full', 'legend_order': -1},
        # ('hf_we_', 'mean', 'WE'): {'color': '#33e0ff', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
        # ('hf_we_topk_10', 'mean', 'WE'): "hf$^{10}_{we}$",
        ('hf', 'mean', '07-13'): {'color': 'blue', 'legend_name': 'TracIn, 07-13', 'legend_order': 2},
        # ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
        # ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
        ('cos', 'mean', '07-13'): {'color': 'green', 'legend_name': 'Cosine, 07-13', 'legend_order': 5},
        ('datainf', 'mean', '07-13'): {'color': 'red', 'legend_name': 'DataInf, 07-13', 'legend_order': 6},
        # ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
        ('rand', '', ''): {'color': 'gray', 'legend_name': 'Random', 'legend_order': 8},        
    }    

    draw_all_tun2_metric(base_path, selected_methods=llama_selected_methods, suffix="-sm",
                            figsize=(8, 2))
    # roberta_layers = ['WE', '00-05', '06-11', '12-17', '18-23', 'CL']
    # llama_layers = ['WE', '00-03', '04-07', '08-11', '12-15', 'CL']
    # mistral_layers = ['WE', '00-07', '08-15', '16-23', '24-31', 'CL']
    # draw_all_tun2_box_metric(base_path, layers = selected_layers, suffix = "-sm",
    #                         c)
    pass


    # for task in benchmark:
    #     draw_tun2_metric(base_path, task, "accuracy", suffix = "best", selected_methods={
    #         ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': -1},
    #         # ('hf_we_', 'mean', 'WE'): {'color': '#33e0ff', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #         ('hf_we_topk_10', 'mean', 'WE'): {'color': '#33e0ff', 'legend_name': 'hf$^{10}_{we}$', 'legend_order': 1},
    #         ('hf', 'mean', '04-07'): {'color': 'blue', 'legend_name': 'hf, 04-07', 'legend_order': 2},
    #         # ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
    #         # ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
    #         ('cos', 'mean', '04-07'): {'color': 'green', 'legend_name': 'cos, 04-07', 'legend_order': 5},
    #         ('datainf', 'mean', '00-03'): {'color': 'red', 'legend_name': 'datainf, 00-03', 'legend_order': 6},
    #         # ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #         ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': 8},
    #     })    
    pass
    
