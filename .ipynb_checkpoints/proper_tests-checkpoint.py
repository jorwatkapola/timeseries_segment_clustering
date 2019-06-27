import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import importlib
import time
import datetime
import segment_cluster as sc
import pandas as pd

importlib.reload(sc)
sys.stdout.flush()
np.random.seed(0)

k_clusters = [10, 50, 100, 200]
seg_lens = [10, 50, 100, 150, 200]

def plot_time_series(time_series_file):
    fig, axes = plt.subplots(nrows=5, ncols=1)
    for index, time_series in enumerate(time_series_file[0:5]):
        axes[index].plot(time_series)
        axes[index].get_xaxis().set_visible(False)
    axes[-1].get_xaxis().set_visible(True)
    fig.tight_layout()
    fig.show()
    return
    
def validation_and_analysis(ordinary_file, outlier_file, k_clusters, seg_lens):
    file_name, validation_results = sc.validate_algorithm(ordinary_file, outlier_file, k_clusters, seg_lens, save_results=True)
    summary = sc.analyse(file_name, k_clusters, seg_lens, save_histograms=True, save_grid=True)
#     valid1_summary = pd.DataFrame(summary[2,1:,1:]/1000, 
#                                   index=["K"+str(param) for param in summary[2,1:,0]], 
#                                   columns=["K"+str(param) for param in summary[2,0,1:]])
#     print("file_name: {}".format(file_name))
#     print("F1 metric for different sets of hyperparameters. \nF1 values between 0-1, the higher the better")
#     print(valid1_summary.round(2))
#     return summary

ordinary_file=np.loadtxt("data/synthetic_rhos_v2.csv", delimiter=',')

csv_list = ["data/synthetic_flats.csv", "data/synthetic_boxes.csv", "data/synthetic_boxes_thick.csv", "data/synthetic_boxes_thin.csv", "data/synthetic_sines.csv", "data/synthetic_sines_low.csv", "data/synthetic_sines_long.csv", "data/synthetic_sines_short.csv", "data/synthetic_sines_low_long.csv", "data/synthetic_sines_low_short.csv"]

for csv_file in csv_list:
    outlier_file = np.loadtxt(csv_file, delimiter=',')
    validation_and_analysis(ordinary_file[0:200], outlier_file[0:50], k_clusters, seg_lens)
    
csv_list = ["data/synthetic_flats.csv", "data/synthetic_boxes.csv", "data/synthetic_sines.csv"]

for csv_file in csv_list:
    validation_and_analysis(outlier_file[0:200], ordinary_file[0:50], k_clusters, seg_lens)