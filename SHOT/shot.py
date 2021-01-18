
import json
import os
import sys
import time
import subprocess


def run_color_correction(root_dir, tar_idx):
    proc = subprocess.Popen(['./SHOT.exe', \
                           '-i',root_dir,\
                            '-n',str(tar_idx),\
                            '-e', "off"])
    return proc

with open('../TextureMappingNonRigid/conf.json', 'rt', encoding='UTF8') as f:
    conf = json.load(f)
    
case_array = ['case_main_test']


for case in case_array:
    start = time.time()

    data_case = conf[case]
    data_base = data_case['main']['data_root_path']
    mesh_num = data_case['main']['end_idx']
    
    
    proc= run_color_correction(data_base, mesh_num)
    if(proc is not None):
        proc.communicate()
    end = time.time()
    print(case, "take", end - start, 'seconds')