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

    
def validation_and_analysis(ordinary_file, outlier_file, k_clusters, seg_lens):
    file_name, validation_results = sc.validate_algorithm(ordinary_file, outlier_file, k_clusters, seg_lens, save_results=True)
    #summary = sc.analyse(file_name, k_clusters, seg_lens, save_histograms=True, save_grid=True)


ordinary_file=np.loadtxt("data/synthetic_rhos_v2.csv", delimiter=',')

csv_list = ["data/synthetic_flats.csv", "data/synthetic_boxes.csv", "data/synthetic_boxes_thick.csv", "data/synthetic_sines.csv", "data/synthetic_sines_low.csv", "data/synthetic_sines_long.csv", "data/synthetic_sines_short.csv", "data/synthetic_sines_low_long.csv", "data/synthetic_sines_low_short.csv"]

for csv_file in csv_list:
    outlier_file = np.loadtxt(csv_file, delimiter=',')
    validation_and_analysis(ordinary_file[0:200], outlier_file[0:50], k_clusters, seg_lens)
    
csv_list = ["data/synthetic_flats.csv", "data/synthetic_boxes.csv", "data/synthetic_sines.csv"]

for csv_file in csv_list:
    outlier_file = np.loadtxt(csv_file, delimiter=',')
    validation_and_analysis(outlier_file[0:200], ordinary_file[0:50], k_clusters, seg_lens)