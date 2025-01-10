import pickle

from matplotlib import pyplot as plt
import numpy as np

with open('results_0.pkl', 'rb') as file:
    res = pickle.load(file)

plt.figure(figsize=(5,4))
for method in res['influence']:
    n_train = len(res['influence'][method])
    noise_index = res['noise_index']
    detection_rate_list=[]
    low_quality_to_high_quality=np.argsort(res['influence'][method])[::-1]
    for ind in range(1, len(low_quality_to_high_quality)+1):
        detected_samples = set(low_quality_to_high_quality[:ind]).intersection(noise_index)
        detection_rate = 100*len(detected_samples)/len(noise_index)
        detection_rate_list.append(detection_rate)
    # print(f"Method: {method}, Detection Rate: {detection_rate_list[-1]}%")
    plt.plot(100*np.arange(len(low_quality_to_high_quality))/n_train, 
             detection_rate_list,
             label=method)
plt.xlabel('Data inspected (%)', fontsize=18)
plt.ylabel('Detection Rate (%)', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=15)
plt.title('Mislabeled Data Detection', fontsize=15)
plt.savefig('detection_rate.png')    