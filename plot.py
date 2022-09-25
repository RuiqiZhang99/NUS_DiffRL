import os
import json
import matplotlib
import matplotlib.pyplot as plt
import tensorboard
import numpy as np

task_name = 'ant'
step_loss_paths = []
local_dir_path = os.path.dirname(os.path.abspath(__file__))
tmp_dir_path = local_dir_path + '/logs'


file_paths = []
def all_files_path(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if 'episode_loss_his.npy' in file_path:
                file_paths.append(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            all_files_path(dir_path)

all_files_path(tmp_dir_path)
