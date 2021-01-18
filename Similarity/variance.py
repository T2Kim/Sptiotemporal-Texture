import os
from glob import glob
import json

import numpy as np
import cv2
import openmesh as om
import csv
data_base = "./dataset/"


def cal_ssim(bbase):
    from skimage.measure import compare_ssim as ssim
    from scipy.ndimage import uniform_filter, gaussian_filter

    img_files_path = glob(bbase + "/sub_atlas/texel/*.png")
    mask_files_path = glob(bbase + "/sub_atlas/mask/*.png")
    out_base = bbase + "ssim/"
    if not os.path.exists(out_base):
        os.mkdir(out_base)

    img_now = None
    img_pre = None

    filtered_mask_now = None
    filtered_mask_pre = None

    side = 500

    win_size = 13
    filter_func = uniform_filter
    filter_args = {'size': win_size}

    min_ssim = np.ones((side, side))

    valid_count = np.zeros((side, side))
    valid_sum = np.zeros((side, side))

    for i in range(len(img_files_path)):
        img_now = cv2.imread(img_files_path[i])
        img_now = cv2.cvtColor(img_now, cv2.COLOR_BGR2GRAY)
        img_now = img_now.astype(np.float64)
        img_now /= 255.0
        img_now = cv2.resize(img_now, (side, side))

        mask_now = cv2.imread(mask_files_path[i])
        mask_now = cv2.cvtColor(mask_now, cv2.COLOR_BGR2GRAY)
        mask_now = mask_now.astype(np.float64)
        mask_now /= 255.0
        mask_now = cv2.resize(mask_now, (side, side))
        mask_now = np.where(mask_now > 0.0, 1.0, 0.0)
        # filtered_mask_now = filter_func(mask_now, **filter_args)
        # filtered_mask_now = np.where(filtered_mask_now < 1.0, 0.0, 1.0)
        filtered_mask_now = cv2.GaussianBlur(mask_now, (11, 11), 1.5)
        filtered_mask_now = np.where(filtered_mask_now < 0.9999, 0.0, 1.0)

        if i > 0: 
            v, r = ssim(img_pre, img_now, win_size = win_size, full = True, gaussian_weights = True)
            print(v)
            r = np.where(filtered_mask_pre + filtered_mask_now < 2, 1.0, r)
            valid_count = np.where(filtered_mask_pre + filtered_mask_now < 2, valid_count, valid_count + 1.0)
            valid_sum += np.where(filtered_mask_pre + filtered_mask_now < 2, 0, r)
            min_ssim = np.minimum(min_ssim, r)
            r = (r * 255.0).astype(np.uint8)
            r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(out_base + str(i).zfill(6) + ".png",r)


        filtered_mask_pre = filtered_mask_now
        img_pre = img_now

    
    valid_count = np.where(valid_count == 0, len(img_files_path), valid_count)
    avg_ssim = valid_sum / valid_count

    avg_ssim = (avg_ssim * 255.0).astype(np.uint8)
    avg_ssim = cv2.cvtColor(avg_ssim, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(bbase + "avg_ssim.png",avg_ssim)

    min_ssim = (min_ssim * 255.0).astype(np.uint8)
    min_ssim = cv2.cvtColor(min_ssim, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(bbase + "min_ssim.png",min_ssim)

def classify_case(case_base, bbase, mesh_path):
    in_base = bbase + "ssim/"
    ssim_files_path = glob(bbase + "ssim/*.png")
    img_now = None
    img_pre = None
    side = 1000

    mesh = om.read_trimesh(mesh_path, vertex_tex_coord=True, face_texture_index=True)
    texcoord_array = mesh.vertex_texcoords2D().astype(np.float32)
    face_array = mesh.face_vertex_indices().astype(np.uint32)

    image = -np.ones((side, side), np.int32)

    cal_table = np.zeros((side, side))
    cal_table1 = np.ones((side, side))
    cal_table_tempo = np.zeros((side, side, len(ssim_files_path)))
    result = np.zeros((side, side))

    for i in range(len(ssim_files_path)):
        img_now = cv2.imread(ssim_files_path[i])
        img_now = cv2.cvtColor(img_now, cv2.COLOR_BGR2GRAY)
        img_now = img_now.astype(np.float64)
        img_now /= 255.0
        img_now = cv2.resize(img_now, (side, side))
        cal_table1 = np.minimum(img_now, cal_table1)
        cal_table_tempo[:,:,i] = img_now
        if i > 0:
            diff = np.abs(img_now - img_pre)
            cal_table = np.where(diff > 0.05, cal_table + 1, cal_table)


        img_pre = img_now

    
    cal_table2 = cv2.imread(bbase + "min_ssim.png")
    cal_table2 = cv2.cvtColor(cal_table2, cv2.COLOR_BGR2GRAY)
    cal_table2 = cal_table2.astype(np.float64)
    cal_table2 /= 255.0
    cal_table2 = cv2.resize(cal_table2, (side, side))

    # result = np.where(cal_table > 10, 1.0, 0.0)
    # result *= 255
    # cv2.imshow("111", result)

    # result1 = np.where(cal_table1 < 0.95, 1.0, 0.0)
    # result1 *= 255
    # cv2.imshow("222", result1)
    
    result2 = np.where(cal_table2 < 0.95, 1.0, 0.0)
    # result22 = result2 * 255
    # cv2.imshow("333", result22)
    #cv2.waitKey(0)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    result2 = cv2.dilate(result2, kernel2)
    result2 *= 255
    # cv2.imshow("444", result2)
    # cv2.waitKey(0)

    for i in range(len(ssim_files_path)):
        cal_table_tempo[:,:,i] = np.where(cal_table_tempo[:,:,i] < 0.98, 0.0, 1.0)
    tri_set_tempo = []
    
    for k, face in enumerate(face_array):
        pt1 = (texcoord_array[face[0], 0]* side, (1 - texcoord_array[face[0], 1])* side)
        pt2 = (texcoord_array[face[1], 0]* side, (1 - texcoord_array[face[1], 1])* side)
        pt3 = (texcoord_array[face[2], 0]* side, (1 - texcoord_array[face[2], 1])* side)

        triangle_cnt = np.array( [pt1, pt2, pt3] , np.int32)

        cv2.drawContours(image, [triangle_cnt], 0, k, -1)
    
    image = np.where(result2 < 1.0, -1, image)
    tri_set = np.unique(image.reshape(side * side))
    print(tri_set)

    for i in range(len(ssim_files_path)):
        image = -np.ones((side, side), np.int32)
        for k, face in enumerate(face_array):
            pt1 = (texcoord_array[face[0], 0]* side, (1 - texcoord_array[face[0], 1])* side)
            pt2 = (texcoord_array[face[1], 0]* side, (1 - texcoord_array[face[1], 1])* side)
            pt3 = (texcoord_array[face[2], 0]* side, (1 - texcoord_array[face[2], 1])* side)

            triangle_cnt = np.array( [pt1, pt2, pt3] , np.int32)

            cv2.drawContours(image, [triangle_cnt], 0, k, -1)
        
        image = np.where(cal_table_tempo[:,:,i] < 1.0, -1, image)
        tri_set_tempo.append([np.unique(image.reshape(side * side))])
        if i == 0:
            tri_set_tempo.append([np.unique(image.reshape(side * side))])


    # image = -np.ones((side, side), np.int32)
    # for k in tri_set:
    #     # print(k)
    #     if k < 0:
    #         continue
    #     face = face_array[k]
    #     pt1 = (texcoord_array[face[0], 0]* side, (1 - texcoord_array[face[0], 1])* side)
    #     pt2 = (texcoord_array[face[1], 0]* side, (1 - texcoord_array[face[1], 1])* side)
    #     pt3 = (texcoord_array[face[2], 0]* side, (1 - texcoord_array[face[2], 1])* side)

    #     triangle_cnt = np.array( [pt1, pt2, pt3] , np.int32)

    #     cv2.drawContours(image, [triangle_cnt], 0, float(k), -1)

    
    # image += 1
    # image = image.astype(np.float32)
    # image /= np.max(image)
    # cv2.imshow("image", image)
    # cv2.waitKey()
    temp_table = np.zeros((len(ssim_files_path) + 1, len(face_array)))
    for i in range(len(ssim_files_path) + 1):
        for kkk in tri_set_tempo[i]:
            temp_table[i, kkk] = 1.0

    temp_window = 3
    temp_table_final = np.zeros((len(ssim_files_path) + 1, len(face_array)))
    temp_table_final += temp_table
    for i in range(temp_window):
        temp_table_final[i+1:, :] += temp_table[:-(i+1), :]
        temp_table_final[:-(i+1), :] += temp_table[i+1:, :]
    temp_table_final = np.where(temp_table_final < 1.0 + temp_window * 2, 0.0, 1.0)
    
    f = open(case_base + "var_table.csv", "w")
    for i in range(len(face_array)):
        if np.isin(i, tri_set):
            f.write(str(10.0))
        else:
            f.write(str(1.0))
        f.write(", ")
    for l in range(len(ssim_files_path) + 1):
        f.write("\n")
        for i in range(len(face_array)):
            f.write(str(temp_table[l,i]))
            if i < len(face_array) - 1 or l < len(ssim_files_path):
                f.write(", ")
    f.close()

def drawgraph():
    bbase = "./dataset/multi/"
    in_base = bbase + "ssim/"
    ssim_files_path = glob(in_base + "*.png")

    img_g_atlas = cv2.imread("./dataset/multi.png", cv2.IMREAD_UNCHANGED)
    side = 1000
    filtered_mask_pre = img_g_atlas[:,:,3]
    img_g_atlas = cv2.cvtColor(img_g_atlas, cv2.COLOR_BGRA2BGR)
    img_g_atlas = img_g_atlas.astype(np.float64)
    filtered_mask_pre = filtered_mask_pre.astype(np.float64)
    img_g_atlas /= 255.0
    img_g_atlas = cv2.resize(img_g_atlas, (side, side))

    # rec_pos = (425, 600) # 등
    # rec_pos = (200, 600) # 팔
    rec_pos = (290, 935) # 입
    # rec_pos = (240, 280)
    rec_rad = (10, 10)

    img_g_atlas_rec = cv2.rectangle(img_g_atlas, (rec_pos[1] - rec_rad[1], rec_pos[0] - rec_rad[0]), (rec_pos[1] + rec_rad[1], rec_pos[0] + rec_rad[0]), (0,0,1), 3)
    cv2.imshow("111", img_g_atlas_rec)
    cv2.waitKey(0)
    for ssim_file in ssim_files_path:
        ssim_map = cv2.imread(ssim_file)
        ssim_map = cv2.cvtColor(ssim_map, cv2.COLOR_BGR2GRAY)
        ssim_map = ssim_map.astype(np.float64)
        ssim_map /= 255.0
        target_map = ssim_map[rec_pos[0] - rec_rad[0]:rec_pos[0] + rec_rad[0], rec_pos[1] - rec_rad[1]:rec_pos[1] + rec_rad[1]]
        print(np.sum(target_map))

def drawgraph_set():
    bbase = "./dataset/multi/"
    in_base = bbase + "ssim/"
    ssim_files_path = glob(in_base + "*.png")
    
    case_arr = [((425, 600), (1,0,0)), ((190, 600), (0,1,0)), ((290, 935), (0,0,1)), ((140, 280), (1,1,0)), ((240, 935), (0,1,1))]

    img_g_atlas = cv2.imread("./dataset/multi.png", cv2.IMREAD_UNCHANGED)
    side = 1000
    filtered_mask_pre = img_g_atlas[:,:,3]
    img_g_atlas = cv2.cvtColor(img_g_atlas, cv2.COLOR_BGRA2BGR)
    img_g_atlas = img_g_atlas.astype(np.float64)
    filtered_mask_pre = filtered_mask_pre.astype(np.float64)
    img_g_atlas /= 255.0
    img_g_atlas = cv2.resize(img_g_atlas, (side, side))

    # rec_pos = (425, 600) # 등
    # rec_pos = (200, 600) # 팔
    # rec_pos = (290, 935) # 입
    # rec_pos = (240, 280)
    rec_rad = (10, 10)

    for case in case_arr:
        img_g_atlas = cv2.rectangle(img_g_atlas, (case[0][1] - rec_rad[1], case[0][0] - rec_rad[0]), (case[0][1] + rec_rad[1], case[0][0] + rec_rad[0]), case[1], 3)
    cv2.imshow("111", img_g_atlas)
    cv2.imwrite(bbase + "colored.png", img_g_atlas * 255)
    cv2.waitKey(0)

    f = open(bbase + "ssim_table.csv", "w")
    for i, ssim_file in enumerate(ssim_files_path):
        ssim_map = cv2.imread(ssim_file)
        ssim_map = cv2.cvtColor(ssim_map, cv2.COLOR_BGR2GRAY)
        ssim_map = ssim_map.astype(np.float64)
        ssim_map /= 255.0
        ssim_map = cv2.GaussianBlur(ssim_map, (11, 11), 1.5)
        for j, case in enumerate(case_arr):
            if j == 0:
                f.write(str(i))
            f.write(",")
            target_map = ssim_map[case[0][0] - rec_rad[0]:case[0][0] + rec_rad[0], case[0][1] - rec_rad[1]:case[0][1] + rec_rad[1]]
            # f.write(str(np.min(target_map)))
            f.write(str(np.average(target_map)))
        f.write("\n")
    f.close()

case_array = ['case_main_test']

with open('../TextureMappingNonRigid/conf.json', 'rt', encoding='UTF8') as f:
    conf = json.load(f)

for case in case_array:

    data_case = conf[case]
    data_base = data_case['main']['data_root_path']
    test_case = data_case['main']['unit_test_path']
    mesh_path = data_base + data_case['main']['tex_mesh_path']

    b_base = data_base + "unit_test/" + test_case + "/"
    cal_ssim(b_base)
    classify_case(data_base, b_base, mesh_path)

