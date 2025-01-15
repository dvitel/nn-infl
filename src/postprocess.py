from collections import defaultdict
import json
import os
import pickle
import re

from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import torch
from datasets import load_from_disk

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

def draw_ft2_metric(task: str, infile: str, outfile: str, metric = 'accuracy'):
    with open(infile, 'r') as f:
        json_lines = f.readlines()
    all_metrics = [json.loads(l) for l in json_lines]
    method_metrics = defaultdict(list)
    for metrics in all_metrics:
        metric_values = metrics['metrics'][metric]
        infl_method = metrics['finetune2']['infl_method']
        filter_method = metrics['finetune2']['filter_method']
        if filter_method  == 'infl':
            filter_method = infl_method
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
        plt.plot(np.arange(1, 11), mean, label=method, marker='o', markersize=5, linewidth=1)
        plt.fill_between(np.arange(1, 11), min_v, max_v, alpha=.05, linewidth=0)
    
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

if __name__ == "__main__":
    for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
        draw_mislabel_detection_rate(task=ds, res_folder = "./data/infl", module_pattern = '', 
                                    datasets_folder = './data/datasets', out = f"./data/mdr/{ds}.png")
        
    for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
        draw_mislabel_detection_rate(task=ds, res_folder = "./data/infl", module_pattern = '.*\.layer\.[0-7]\..*\.lora_(A|B)\..*', 
                                    datasets_folder = './data/datasets', out = f"./data/mdr-0/{ds}.png", title="Layers 0-7")

    for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
        draw_mislabel_detection_rate(task=ds, res_folder = "./data/infl", module_pattern = '.*\.layer\.([8-9]|1[0-5])\..*\.lora_(A|B)\..*', 
                                    datasets_folder = './data/datasets', out = f"./data/mdr-1/{ds}.png", title="Layers 8-15")

    for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
        draw_mislabel_detection_rate(task=ds, res_folder = "./data/infl", module_pattern = '.*\.layer\.(1[6-9]|2[0-3])\..*\.lora_(A|B)\..*', 
                                    datasets_folder = './data/datasets', out = f"./data/mdr-2/{ds}.png", title="Layers 16-23")
        
    for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
        draw_mislabel_detection_rate(task=ds, res_folder = "./data/infl", module_pattern = '.*\.classifier\..*', 
                                    datasets_folder = './data/datasets', out = f"./data/mdr-3/{ds}.png", title="Classifier")        

    # for ds in ['mrpc', 'qnli', 'qqp', 'sst2']:
    #     draw_ft2_metric(task = ds, infile = f'./data/ft2-infl/{ds}.jsonlist', outfile = f'./data/accuracy/{ds}.png', metric = 'accuracy')
    # list_modules('./data/infl/mrpc/i_datainf_mrpc_0.pt', '') #'.*\.layer\.(1[6-9]|2[0-3])\..*\.lora_(A|B)\..*')