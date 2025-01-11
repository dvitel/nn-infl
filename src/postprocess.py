import os
import pickle
import re

from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

def draw_curve(res_folder = "./data/raw/", res_filer='infl_qnli_', out = "./data/auc/qnli.png", module=''):
    # res_pattern = re.compile(res_filer)
    data_detection_per_method = {}
    for file in os.listdir(res_folder):
        if file.startswith(res_filer) == False:  
            continue 

        res_file = os.path.join(res_folder, file)      
        with open(res_file, 'rb') as file:
            res = pickle.load(file)

        noise_index = res['noise_index']
        influences = res['influences']
        infl_methods = list(influences.keys())
        for method in infl_methods:
            infls = influences[method][module]
            n_train = len(infls)
            detection_rate_list=[]
            low_quality_to_high_quality=np.argsort(infls)[::-1]
            for ind in range(1, len(low_quality_to_high_quality)+1):
                detected_samples = set(low_quality_to_high_quality[:ind]).intersection(noise_index)
                detection_rate = 100*len(detected_samples)/len(noise_index)
                detection_rate_list.append(detection_rate)
            data_detection_per_method.setdefault(method, []).append(detection_rate_list)
    for method, detection_rate_lists in data_detection_per_method.items():
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
    plt.ioff()
    plt.xlabel('Data inspected (%)')
    plt.ylabel('Detection Rate (%)')
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.legend(fontsize='small')
    plt.title('Mislabeled Data Detection', fontsize=15)
    plt.tight_layout()
    plt.savefig(out)  
    plt.clf()  

if __name__ == "__main__":
    # draw_curve(res_filer='infl_qnli_', out = "./data/auc/qnli.png")    
    draw_curve(res_filer='infl_sst2_', out = "./data/auc/sst2.png")