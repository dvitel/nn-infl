from collections import defaultdict
from functools import partial
import json
import os
import pandas as pd
import re
from typing import Optional
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

benchmark = ["qnli", "mrpc", "sst2", "qqp", "cola", "mnli", "rte", "stsb"]

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

def commonsubset_matrix_score(int_matrix: torch.Tensor, *, noise_mask: torch.Tensor, 
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

def commonset_matrix_score(int_matrix: torch.Tensor, *, noise_mask: torch.Tensor, vote_ratio = 0.2, noise_ratio = 0.3, **_) -> torch.Tensor:
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

def get_pareto_front_indexes_neg(int_matrix: torch.Tensor) -> np.ndarray:
    ''' Get the pareto front from a population. 
        NOTE: greater is better here. Invert your fitness if it is the opposite.
    '''
    # mask = np.ones(int_matrix.shape[0], dtype=bool)
    # mask[exclude_indexes] = False
    # index_remap = np.where(mask)[0]
    domination_matrix = torch.all(int_matrix[:, None] >= int_matrix, dim=2) & torch.any(int_matrix[:, None] > int_matrix, axis=2)
    indexes = torch.where(~torch.any(domination_matrix, axis=1))[0]
    return indexes

def pareto_matrix_score(int_matrix: torch.Tensor, *, noise_mask: torch.Tensor, noise_ratio = 0.3, **_) -> torch.Tensor:
    ''' 3-D tensor: train sample * module * infl sample '''
    total_int_matrix = int_matrix.mean(dim=-2)
    total_int_matrix = total_int_matrix[:, :10]
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
def get_df_from_file(metric_file: str, module_groups):    

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
        modules_pattern = module_groups.get(module_name, None)

        if modules_pattern is None:
            cancel = 0
        else:
            cancel_map = r["first_finetune"]["cancel_norm"]

            cancel = 0
            cancel_count = 0
            for m in cancel_map.keys():
                if modules_pattern.match(m):
                    cancel += cancel_map[m]
                    cancel_count ++ 1 
            
            cancel = 0 if cancel_count == 0 else (cancel / cancel_count)

        row = {
               "task": task,
               "filter_perc": filter_perc,
               "infl_method": infl_method,
               "agg_method": r["config"]["agg_method"],
               "module": module_name,
               
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
                "cancel": cancel,

               "seed": seed

               }
        rows.append(row)

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
                tmp_file = "tmp.pkl", module_groups = {}):
    # if os.path.exists(tmp_file):
    #     df1 = pd.read_pickle(tmp_file)
    # else:
    plain_dataframes = []
    for dataset in datasets:
        file = os.path.join(base_path, f"{dataset}-bl.jsonlist")
        df = get_df_from_file(file, module_groups)
        plain_dataframes.append(df)
    
    df1 = pd.concat(plain_dataframes, ignore_index=True)
    # df1.to_pickle(tmp_file)
    return df1

def rank_checkpoints(base_path = "data/roberta-2", metric = "noise_30",
                        datasets = ["mrpc", "qnli", "sst2", "qqp"],
                        checkpoints = ["b", "l", "bl"]):
    df1 = get_all_df(base_path = base_path, datasets = datasets,
                        checkpoints = checkpoints)
    key_columns = ["checkpoint", "task", "infl_method", "agg_method", "module", "seed"]

    df2 = df1[(df1["agg_method"] != "")]
    df3 = df2[key_columns + [metric]]
    df4 = df3.pivot(index=["task", "infl_method", "agg_method", "module", "seed"], columns="checkpoint", values=metric)
    df4 = df4[~df4.isna().any(axis=1)]
    df5: DataFrame = df4[df4[checkpoints].gt(0).any(axis=1)]

    stats_data = df5.T.to_numpy()

    friedman_res = sci_stats.friedmanchisquare(*stats_data)

    import scikit_posthocs as sci_posthocs

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

def create_dataset_table():

    with open("data/roberta/groups.json", "r") as f:
        module_patterns = json.load(f)
        
    module_patterns = {m: re.compile(p) for m, p in module_patterns.items()}

    datasets = ["mrpc", "qnli", "sst2", "qqp", "cola", "mnli", "rte", "stsb"]
    for dataset in datasets:    
        df = get_all_df(datasets = [dataset], module_groups=module_patterns)
        sorted_df = get_agg_df(df)
        # sorted_df2 = sorted_df.reset_index()

        sorted_df2 = sorted_df.sort_values(by="best_accuracy_1_mean", ascending=False)

        columns_to_include = ["best_accuracy_1", "best_infl_accuracy_1", "accuracy_rand_delta", "noise_30", "auc_ndr", "infl_loss_delta", "cancel", "logits_change"]
        rows = []
        for row in sorted_df2.to_dict(orient="records"):
            new_row = []
            infl_method = row["infl_method"]
            module = row["module"]
            new_row.append(infl_method)
            new_row.append(module)
            for c in columns_to_include:
                v_mean = row[c + "_mean"]
                v_std = row[c + "_std"]
                if c in ["best_accuracy_1", "best_infl_accuracy_1", "accuracy_rand_delta", "noise_30", "auc_ndr"]:
                    v_mean = round(v_mean * 100) 
                    v_std = round(v_std * 100)
                else:
                    if v_mean == 0 and c == "cancel":
                        v_mean = ""
                        v_std = "0"
                    else:
                        v_mean = round(v_mean * 100) / 100
                        v_mean = str(v_mean)
                        if v_mean.startswith("0."):
                            v_mean = "." + v_mean[2:]
                        if v_mean.startswith("-0."):
                            v_mean = "-." + v_mean[3:]
                        if  v_std == 0:
                            v_std = "0"
                        else:
                            v_std = round(v_std * 100) / 100
                            v_std = str(v_std)
                            if v_std.startswith("0."):
                                v_std = "." + v_std[2:] 
                if (v_std == 0) or (v_std == "0"):
                    new_row.append(f"{v_mean}")
                else:
                    new_row.append(f"{v_mean} $\pm$ {v_std}")
            rows.append(new_row)

        column_names = ["method", "layers", "$Acc_t$", "$Acc_v$", "$\Delta Acc^{rnd}_t$", "$NDR_{30}$", "$AUC_{NDR}$", "$\Delta loss_v$", "Cancel", "$\Delta logits_t $"]
        with open(f"{dataset}.tex", "w") as stats_file:
            s = tabulate(rows, headers = column_names, showindex=False, tablefmt="latex", floatfmt=".3f")
            s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("hf\_we\_topk\_10", "$hf^{10}_{we}$").replace("hf\_we\_", "$hf_{we}$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("llllllllll", "ll|cccccccc")
            print(s, file = stats_file)
    pass

def run_friedman_tests(metric_name: str = "best_accuracy_1",
                        out_folder = "data/roberta/filter-30"):

    datasets = ["qnli", "mrpc", "sst2", "qqp", "cola", "mnli", "rte", "stsb"]

    all_df = get_all_df(base_path = out_folder, datasets = datasets)

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
                        out_folder = "data/roberta/filter-30"):

    datasets = ["qnli", "mrpc", "sst2", "qqp", "cola", "mnli", "rte", "stsb"]

    all_df = get_all_df(base_path = out_folder, datasets = datasets)

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

def run_spearman_tests(metric_name: str = "best_accuracy_1",
                        out_folder = "data/roberta/filter-30"):

    datasets = ["qnli", "mrpc", "sst2", "qqp", "cola", "mnli", "rte", "stsb"]

    all_df = get_all_df(base_path = out_folder, datasets = datasets)

    metrics_per_ds = all_df.pivot(index=["infl_method", "agg_method", "module"], columns=["task", "seed"], values=[metric_name, 'noise_30'])
    metrics_per_ds = metrics_per_ds.dropna(axis=1, how='any')
    metric_columns = {(ds, seed) for _, ds, seed in metrics_per_ds.columns[metrics_per_ds.columns.get_level_values(0) == metric_name]}
    noise_30_columns = {(ds, seed) for _, ds, seed in metrics_per_ds.columns[metrics_per_ds.columns.get_level_values(0) == 'noise_30']}
    common_columns = set.intersection(metric_columns, noise_30_columns)

    rho = {}
    pvalues = {}
    for ds, seed in common_columns:
        series1 = metrics_per_ds.loc[:, (metric_name, ds, seed)].to_numpy()
        series2 = metrics_per_ds.loc[:, ('noise_30', ds, seed)].to_numpy()
        s = sci_stats.spearmanr(series1, series2)
        rho.setdefault(ds, []).append(s.correlation)
        pvalues.setdefault(ds, []).append(s.pvalue)

    rows = []
    # header_row = [""]
    # for d in datasets:
    #     header_row.append(d)
    # header_row.append("Total")
    # rows.append(header_row)

    rho_row = ["Spearman $\\rho$"]
    for d in datasets:
        ds_rho = rho[d]
        mean_rho = round(np.mean(ds_rho) * 10) / 10
        std_rho = round(np.std(ds_rho) * 10) / 10
        rho_row.append(f"{mean_rho} $\pm$ {std_rho}")
    all_rho_plain = [v for vl in rho.values() for v in vl]
    rho_mean = np.mean(all_rho_plain)
    rho_std = np.std(all_rho_plain)
    rho_mean = round(np.mean(rho_mean) * 10) / 10
    rho_std = round(np.std(rho_std) * 10) / 10
    rho_row.append(f"{mean_rho} $\pm$ {std_rho}")
    rows.append(rho_row)

    pvalue_row = ["p-value"]
    for d in datasets:
        ds_pvalues = pvalues[d]
        r = sci_stats.combine_pvalues(ds_pvalues)
        # pvalue_exp = int(f"{r.pvalue:.0e}".split("e")[1].replace("-0", "-").replace("+0", ""))
        pvalue_row.append(f"{r.pvalue:.0e}")
    
    r = sci_stats.combine_pvalues([v for vl in pvalues.values() for v in vl])
    # pvalue_exp = int(f"{r.pvalue:.0e}".split("e")[1].replace("-0", "-").replace("+0", ""))
    pvalue_row.append(f"{r.pvalue:.0e}")
    rows.append(pvalue_row)

    with open(f"{out_folder}/{metric_name}-spearman.tex", "w") as stats_file:
        s = tabulate(rows, headers=["", *datasets, "Total"], showindex=False, tablefmt="latex", numalign="center", stralign="center")
        s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("hf\_we\_topk\_10", "hf$^{10}_{we}$").replace("hf\_we\_", "hf$_{we}$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("lllllllllllllllllllllllll", "l|cccccccccccccccccccccccc").replace("rand", "\\hline rand")
        print(s, file = stats_file)

    # print(tabulate(rows, , tablefmt="grid", numalign="center", stralign="center"))
    pass 

def create_tun2_metric_table(metric_name: str = "best_accuracy_1", prec = 2, 
                        out_folder = "data/roberta/filter-30",
                        highlight_max = True, ds_ranks = False, mul = 100,
                        with_row_id = False):
    with open("data/roberta/groups.json", "r") as f:
        module_patterns = json.load(f)
        
    module_patterns = {m: re.compile(p) for m, p in module_patterns.items()}

    datasets = ["qnli", "mrpc", "sst2", "qqp", "cola", "mnli", "rte", "stsb"]

    all_df = get_all_df(base_path = out_folder, datasets = datasets, module_groups=module_patterns)

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
    metric_by_ds['rank'] = ranks.mean(axis=1)
    metric_by_ds['rank_std'] = ranks.std(axis=1)
    metric_by_ds = metric_by_ds.sort_values(by=['rank', 'rank_std'], ascending=[True, True])

    metric_by_ds = metric_by_ds[[*[de for d in datasets for de in [d, d + "_std"]], "rank", "rank_std"]]
    # metric_by_ds.to_csv(f"{out_folder}/{metric_name}{suffix}-avg.csv")

    filtered_df = metric_by_ds.loc[metric_by_ds.index.get_level_values('infl_method') != 'denoise']
    if highlight_max:
        values_to_highlight = filtered_df[datasets].to_numpy().max(axis=0)
    else:
        values_to_highlight = filtered_df[datasets].to_numpy().min(axis=0)

    rows = []
    for row_id, row in enumerate(metric_by_ds.reset_index().to_dict(orient="records")):
        new_row = {}
        method = row["infl_method"]
        layer = row["module"]
        if with_row_id:
            new_row["id"] = (row_id + 1)
        new_row["method"] = method 
        new_row["layer"] = "all" if (method not in ["denoise", "rand"]) and layer == "" else layer 
        for did, d in enumerate([*datasets, "rank"]):
            should_highlight = False
            if d == "rank":
                m = round(row[d] * 10) / 10
                m_std = round(row[d + "_std"] * 10) / 10
            else:
                if method != "denoise":
                    should_highlight = row[d] == values_to_highlight[did]
                m = round(row[d] * (10 ** prec) * mul) / (10 ** prec)
                m_std = round(row[d + "_std"] * (10 ** prec) * mul) / (10 ** prec)
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

    with open(f"{out_folder}/{metric_name}{suffix}-avg.tex", "w") as stats_file:
        s = tabulate(rows, headers = "keys", showindex=False, tablefmt="latex")
        s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("hf\_we\_topk\_10", "hf$^{10}_{we}$").replace("hf\_we\_", "hf$_{we}$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("lllllllllll", "ll|ccccccccc").replace("rand", "\\hline rand")
        print(s, file = stats_file)

    pass

def draw_tun2_metric(base_path: str, task:str, metric_name: str, selected_methods: dict = {}, 
                 draw_diff = False, suffix = ""):
    infile = os.path.join(base_path, f"{task}-bl.jsonlist")
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

    # max_sz = max(len(l) for l in method_metrics.values())

    # method_metrics_flat = {k: [v3 for v2 in v for v3 in (v2[:10] + [np.nan] * max(0, 10 - len(v2)))] for k, v in method_metrics.items()}
    # method_metric_ranks = get_avg_ranks(method_metrics_flat)

    # setup_names = list(setup_score_values.keys())

    # method_names = sorted(method_metrics.keys(), key = method_metric_ranks.get)
    method_names = list(selected_methods.keys())
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
        method_settings = selected_methods[(infl_method, agg_method, module_name)]
        # default_args = dict(marker='o', markersize=4, linestyle='none', linewidth=1, color = method_settings['color'])
        default_args = dict(marker='o', markersize=0, linewidth=0.9, color = method_settings['color'])
        draw_full = False
        shift = ((i - len(method_metrics) // 2) * 0.025)
        if infl_method == 'denoise':
            default_args['linestyle']='--'
            default_args['markersize'] = 0
            draw_full = True 
            shift = 0
        if infl_method == 'rand':
            default_args['linestyle']='-.'
            default_args['markersize'] = 0
            draw_full = True 
            shift = 0
        # xs = np.arange(len(mean)) + 1
        xs = np.arange(len(mean)) + 1 + shift # Shift x-coordinates slightly
        line = plt.plot(xs, mean, zorder=1, **default_args)
        if draw_full:
            plt.fill_between(xs, min_v, max_v, alpha=.05, color = line[0].get_color(), linewidth=0)
        else:
            plt.errorbar(xs, mean, yerr=[mean - min_v, max_v - mean], fmt='none', ecolor=line[0].get_color(), capsize=1, linewidth=0.5, zorder=0)
        handles_.append((line[0], method_settings['legend_name'], method_settings['legend_order']))
    
    handles_.sort(key = lambda x: x[2])
    ordered_handles = [h[0] for h in handles_]
    ordered_labels = [h[1] for h in handles_]
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Accuracy, \\%', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(ordered_handles, ordered_labels, fontsize=15)
    # plt.title(f'{task.upper()} 70\\% filtered finetuning', fontsize=20)
    plt.title(f'{task.upper()}', fontsize=20)
    plt.tight_layout()
    outfile = os.path.join(base_path, "plots", f"{task}-{metric_name}-{suffix}.pdf")
    plt.savefig(outfile)  
    plt.clf()  

agg_methods = {
    "rank": rank_matrix_score, 
    "mean": mean_matrix_score,
    "pareto": pareto_matrix_score,
    "dir": dir_matrix_score,
    "commonset-20": partial(commonset_matrix_score, vote_ratio = 0.2),
    "commonsubset-40": partial(commonsubset_matrix_score, vote_ratio = 0.4),
    # "csmi": partial(csmi_matrix_score, vote_ratio = 0.5, descending = False)


    "mean_10": partial(mean_matrix_score, trim_ratio=0.1),
    "mean_50": partial(mean_matrix_score, trim_ratio=0.5),
    "commonset1": partial(commonset_matrix_score, vote_ratio = 0.1),
    "commonset-80": partial(commonset_matrix_score, vote_ratio = 0.8),
    "commonset3": partial(commonset_matrix_score, vote_ratio = 0.3),
    "commonset4": partial(commonset_matrix_score, vote_ratio = 0.4),
    "commonsubset1": partial(commonsubset_matrix_score, vote_ratio = 0.1, descending = False),
    "commonsubset2": partial(commonsubset_matrix_score, vote_ratio = 0.2, descending = False),
    "commonsubset3": partial(commonsubset_matrix_score, vote_ratio = 0.3, descending = False),
    "commonsubset-60": partial(commonsubset_matrix_score, vote_ratio = 0.6, descending = False),
    "commonsubset-40r": partial(commonsubset_matrix_score, vote_ratio = 0.4, same_class = True, descending = True),
    "commonsubset-40rr": partial(commonsubset_matrix_score, vote_ratio = 0.4, same_class = True, descending = False),
    "silhouette": cluster_matrix_score
}

def compute_ndr_metrics_table(base_dir_path: str, task='qnli', 
                                    group_file: str = "./groups.json",
                                    infl_methods = ['hf'],
                                    agg_method_names: list[str] = ["mean"],
                                    include_total = True, levels = [30],
                                    m_prefix = "m_bl", i_prefix="i_bl", ndr_prefix = "ndr_bl",
                                    save_df = True, device = "cuda"):

    if group_file != '' and group_file is not None:
        import re
        with open(os.path.join(base_dir_path, group_file), "r") as f:
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
        trainset_labels = torch.tensor(trainset_labels_dict[run_id_str], device = device)
        inflset_labels = torch.tensor(inflset_labels_dict[run_id_str], device = device)
        inflset_logits = inflset_logits_dict[run_id_str].to(device)
        
        scores = torch.zeros((len(agg_methods), len(module_interactions), all_interactions.shape[0]), device = device)

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

        ndr_at_levels = noise_detection_curves[:, :, filter_idxs]

        auc_ndrs_cpu = auc_ndrs.cpu()
        ndr_at_levels_cpu = ndr_at_levels.cpu()
        del auc_ndrs, ndr_at_levels, noise_detection_curves, scores, train_ids

        for agg_method_id, agg_method_name in enumerate(agg_method_names):
            for module_id, module_name in enumerate(module_and_group_names):
                one_metrics = {level: ndr_at_levels_cpu[agg_method_id, module_id, level_i].item()  for level_i, level in enumerate(levels) }
                one_metrics["auc_ndr"] = auc_ndrs_cpu[agg_method_id, module_id].item()
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
        for run_id, metrics_dict in enumerate(metrics_list_of_dict):
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
                "run_id": run_id,
                **metrics_dict
            }
            rows.append(row_data)

    df = pd.DataFrame(rows).set_index(["task", "infl", "agg", "layer", "module", "run_id"])    

    if save_df:
        outfile = os.path.join(base_dir_path, f"{ndr_prefix}_{task}.pcl")
        df.to_pickle(outfile)
        print(f"Saved ndr metrics to {outfile}")

    return df

def process_ndr_table(base_path: str, tasks: list[str], output_ranks = False, with_row_id = False,
                        metric_name = 30, ndr_prefix = "ndr_bl"): 
    #NOTE: metric_name in ["f30", "auc_ndr"]):

    dfs = [ pd.read_pickle(os.path.join(base_path, f"{ndr_prefix}_{task}.pcl")) for task in tasks ]
    df = pd.concat(dfs, ignore_index=False)

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
        module = row["module"]
        if with_row_id:
            new_row["id"] = (row_id + 1)
        new_row["infl"] = infl_method
        new_row["agg"] = agg_method
        new_row["layer"] = layer
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

    with open(f"{base_path}/ndr-{metric_name}{suffix}-avg.tex", "w") as stats_file:
        s = tabulate(rows, headers = "keys", showindex=False, tablefmt="latex")
        s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("hf\_we\_topk\_10", "hf$^{10}_{we}$").replace("hf\_we\_", "hf$_{we}$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("lllllllllll", "ll|ccccccccc").replace("rand", "\\hline rand")
        print(s, file = stats_file)

    pass

def draw_ndr_curve(ys: np.ndarray, xs:np.ndarray, ylegend:list[str], title:str, outfile: str,
                    xaxis_line = 30, noise_ratio = 20): 
    ''' 
        ys - 3d, method * run_id * values 
        xs - measure points, len(xs) == len(values)
        len(ylegend) == len(method)

        xs and ys are in range [0, 100]
    '''
    plt.ioff()
    for method_id, method in enumerate(ylegend):
        y = ys[method_id]
        y_mean = np.mean(y, axis=0)
        confidence_level = 0.95
        degrees_freedom = y.shape[0] - 1
        sample_standard_error = stats.sem(y, axis=0)
        confidence_interval = stats.t.interval(confidence_level, degrees_freedom, y_mean, sample_standard_error)
        min_v = confidence_interval[0]
        max_v = confidence_interval[1]
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

# def draw_ndr_curves(base_path: str, tasks: list[str], levels = [10,20,30,40,50,60,70,80,90], 
#                         ndr_prefix = "ndr_bl",
#                         selected_methods: list[str] = [],
#                         best_method = True, best_by_infl = True): 
#     #NOTE: metric_name in ["f30", "auc_ndr"]):

#     for task in tasks:

#         df = pd.read_pickle(os.path.join(base_path, f"{ndr_prefix}_{task}.pcl"))

#         # metric_df = df.reset_index().pivot(index=["infl", "agg", "layer", "module"], columns=["task", "run_id"], values=[metric_name])

#     metric_by_ds_mean = metric_df.groupby(level=1, axis=1).mean()
#     metric_by_ds_std = metric_df.groupby(level=1, axis=1).std()
#     metric_by_ds_std.columns = [c + "_std" for c in metric_by_ds_std.columns]
#     metric_by_ds = pd.merge(metric_by_ds_mean, metric_by_ds_std, left_index=True, right_index=True)
#     rank_columns = ["rank", "rank_std"]
#     metric_by_ds["rank"] = ranks.mean(axis=1)
#     metric_by_ds["rank_std"] = ranks.std(axis=1)
#     metric_by_ds = metric_by_ds.sort_values(by=['rank', 'rank_std'], ascending=[True, True])

#     metric_by_ds = metric_by_ds[[*[de for d in tasks for de in [d, d + "_std"]], *rank_columns]]
#     # metric_by_ds.to_csv(f"{base_path}/{metric_name}{suffix}-avg.csv")

#     values_to_highlight = metric_by_ds[tasks].to_numpy().max(axis=0)


#     rows = []
#     for row_id, row in enumerate(metric_by_ds.reset_index().to_dict(orient="records")):
#         new_row = {}
#         infl_method = row["infl"]
#         agg_method = row["agg"]
#         layer = row["layer"]
#         module = row["module"]
#         if with_row_id:
#             new_row["id"] = (row_id + 1)
#         new_row["infl"] = infl_method
#         new_row["agg"] = agg_method
#         new_row["layer"] = layer
#         new_row["module"] = module
#         for did, d in enumerate([*tasks, "rank"]):
#             should_highlight = False
#             if d == "rank":
#                 m = round(row[d] * 10) / 10
#                 m_std = round(row[d + "_std"] * 10) / 10
#             else:
#                 should_highlight = row[d] == values_to_highlight[did]
#                 m = round(row[d] * 1000) / 10
#                 m_std = round(row[d + "_std"] * 1000) / 10
#             m = str(m).rstrip("0").rstrip(".").lstrip("0").replace("-0.", "-.")
#             m_std = str(m_std).rstrip("0").rstrip(".").lstrip("0")

#             if should_highlight:
#                 if m_std == "":
#                     new_row[d] = f"\\textbf{{{m}}}"
#                 else:
#                     new_row[d] = f"\\textbf{{{m}}} $\pm$ {m_std}"
#             else:
#                 if m_std == "":
#                     new_row[d] = m
#                 else:
#                     new_row[d] = f"{m} $\pm$ {m_std}"
#         rows.append(new_row)

#     with open(f"{base_path}/ndr-{metric_name}{suffix}-avg.tex", "w") as stats_file:
#         s = tabulate(rows, headers = "keys", showindex=False, tablefmt="latex")
#         s = s.replace("\\textbackslash{}", "\\").replace("\\$", "$").replace("hf\_we\_topk\_10", "hf$^{10}_{we}$").replace("hf\_we\_", "hf$_{we}$").replace("\\_", "_").replace("\{", "{").replace("\}", "}").replace("\^{}", "^").replace("lllllllllll", "ll|ccccccccc").replace("rand", "\\hline rand")
#         print(s, file = stats_file)

#     pass


if __name__ == "__main__":

    base_path = "data/llama"
    group_file = "./groups.json"

    # create_tun2_metric_table(metric_name="noise_30", ds_ranks=False, mul = 100, highlight_max = True, out_folder=base_path,
    #                             with_row_id = False, prec = 1)
    # pass

    # create_tun2_metric_table(metric_name="best_infl_accuracy_1", ds_ranks=False, mul = 100, 
    #                          highlight_max = True, out_folder=base_path,
    #                             with_row_id = False, prec=1)
    # pass

    # run_friedman_tests(metric_name="best_accuracy_1", out_folder=base_path)
    # pass 

    # run_wilcoxon_tests(metric_name="best_accuracy_1", out_folder=base_path)
    # pass 

    # run_spearman_tests(metric_name="best_accuracy_1", out_folder=base_path)
    # pass 

    # compute_ndr_metrics_table(base_path, task='qnli', 
    #                           group_file=group_file, levels=[10, 20, 30, 40, 50, 60, 70, 80, 90],
    #                           infl_methods = ['hf', 'cos', 'datainf', 'hf_we_', 'hf_we_topk_10'],
    #                           agg_method_names=["mean", "rank"])
    # compute_ndr_metrics_table(base_path, task='mrpc', 
    #                           group_file=group_file, levels=[10, 20, 30, 40, 50, 60, 70, 80, 90],
    #                           infl_methods = ['hf', 'cos', 'datainf', 'hf_we_', 'hf_we_topk_10'],
    #                           agg_method_names=["mean", "rank"])
    # process_ndr_table(base_path, tasks=['qnli', 'mrpc'], with_row_id=True)
    # pass

    
    # for task in benchmark:
    #     draw_tun2_metric(base_path, task, "accuracy", suffix = "best", selected_methods={
    #         ('denoise', '', ''): {'color': 'gray', 'legend_name': 'denoise', 'legend_order': -1},
    #         ('hf_we_', 'mean', 'WE'): {'color': '#33e0ff', 'legend_name': 'hf$_{we}$', 'legend_order': 0},
    #         # ('hf_we_topk_10', 'mean', 'WE'): {'color': '#7f7f7f', 'legend_name': 'hf$^{10}_{we}$', 'legend_order': 1},
    #         ('hf', 'mean', 'WE'): {'color': 'blue', 'legend_name': 'hf, WE', 'legend_order': 2},
    #         # ('hf', 'mean', '00-05'): {'color': '#2ca02c', 'legend_name': 'hf, 00-05', 'legend_order': 3},
    #         # ('hf', 'mean', '06-11'): {'color': '#d62728', 'legend_name': 'hf, 06-11', 'legend_order': 4},
    #         ('cos', 'mean', '12-17'): {'color': 'green', 'legend_name': 'cos, 12-17', 'legend_order': 5},
    #         ('datainf', 'mean', '18-23'): {'color': 'red', 'legend_name': 'datainf, 18-23', 'legend_order': 6},
    #         # ('hf', 'mean', 'CL'): {'color': '#9467bd', 'legend_name': 'hf, CL', 'legend_order': 7},
    #         ('rand', '', ''): {'color': 'gray', 'legend_name': 'rand', 'legend_order': 8},
    #     })


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
    
