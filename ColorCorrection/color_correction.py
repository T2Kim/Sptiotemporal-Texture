import os
import sys
from glob import glob
import time
import subprocess

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
                            

# case_arr = [("./200307_hyomin/", "new_all2/")]
case_arr = [("./210111_hyomin/", "test/")]
# case_arr = [("./200304_police/", "test/"), ("./200627_shark_tilt/", "ttt/"), ("./200630_mario2/", "ttt1/"), ("./200629_sister3/", "ttt2/"), ("./200307_hyomin/", "swtest/")]
# case_arr = [("./gunhee/", "new_all/"), ("./200304_police/", "new_all/"), ("./200627_shark_tilt/", "new_all/"), ("./200630_mario2/", "new_all/"), ("./200629_sister3/", "new_all/"), ("./200307_hyomin/", "new_all/")]
# case_arr = [("./200629_sister3/", "finalfinal/")]
# case_arr = [("./gunhee/", "global/"), ("./gunhee/", "multi/")]
# case_arr = [("./gunhee/", "final2/")]
# case_arr = [("./200627_shark_tilt/", "final/")]
# case_arr = [("./200307_hyomin/", "finalfinal_pong2_3/")]
# case_arr = [("./200307_hyomin/", "swtest/"), ("./200307_hyomin/", "swtest_nosw/"), ("./200307_hyomin/", "swtest_notemp/")]
# case_arr = [("./200307_hyomin/", "swtest/")]
# case_arr = [("./200630_mario2/", "ttt1/")]
# case_arr = [("./200618_haeun_move2/", "ttt/")]

for case in case_arr:
    root_dir = case[0]
    atals_dir = case[1]
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