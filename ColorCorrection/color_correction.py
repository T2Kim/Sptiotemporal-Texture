import os
import sys
from glob import glob
import time
import subprocess
import json

def run_extract_undistort(src_dir, FOV):
    proc = subprocess.Popen(['python', 'undistort.py', '--src_dir', src_dir, '--FOV', FOV])
    return proc

def run_color_correction(root_dir, atals_dir, tar_idx):
    proc = subprocess.Popen(['TextureStitching.exe', \
                            '--in', root_dir + 'mesh.ply',\
                            root_dir + 'atlas/' + atals_dir + 'video/Frame_' + str(tar_idx).zfill(6) + '.png',\
                            root_dir + 'atlas/' + atals_dir + 'video/Mask_' + str(tar_idx).zfill(6) + '.png',\
                            '--out', root_dir + 'atlas/' + atals_dir + 'video/Smooth_' + str(tar_idx).zfill(6) + '.png'])
    return proc
                            

with open('../TextureMappingNonRigid/conf.json', 'rt', encoding='UTF8') as f:
    conf = json.load(f)

case_arr = [("case_main_test")]

for case in case_arr:
    data_case = conf[case]
    root_dir = data_case['main']['data_root_path']
    atals_dir = data_case['main']['atlas_path'] + '/'
    file_list = glob(root_dir + "atlas/" + atals_dir + "/video/Frame_*.png")

    count = len(file_list)
    end_idx = count
    now_idx = 0
    thread_num = 3
    whole_time = 0
    print(count,"Scenes exist!!!")

    while(now_idx + thread_num < end_idx):
        start = time.time()
        processes = []
        for i in range(thread_num):
            proc= run_color_correction(root_dir, atals_dir, now_idx + i)
            processes.append(proc)

        for proc in processes:
            if(proc is not None):
                proc.communicate()
        end = time.time()
        print(now_idx , '~', now_idx + thread_num, "take", end - start, 'seconds')
        whole_time += end - start
        now_idx += thread_num


    start = time.time()
    processes = []
    for i in range(end_idx - now_idx):
        proc= run_color_correction(root_dir, atals_dir, now_idx + i)
        processes.append(proc)

    for proc in processes:
        if(proc is not None):
            proc.communicate()
    end = time.time()
    print(now_idx , '~', now_idx + thread_num, "take", end - start, 'seconds')
    whole_time += end - start

    print(count, "Scenes take", whole_time, 'seconds')