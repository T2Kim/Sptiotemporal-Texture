import os
from glob import glob
import json

import numpy as np
import cv2
import openmesh as om
import csv
data_base = "./dataset/"

def calculate_variance_(images, masks):
    result00 = np.zeros(images[0, :, :, :].shape, dtype=np.float32)
    result01 = np.zeros(images[0, :, :, :].shape, dtype=np.float32)
    result1 = np.zeros(masks[0, :, :].shape, dtype=np.float32)
    for i, m in zip(images, masks):
        i = cv2.cvtColor(i, cv2.COLOR_LAB2BGR)
        m = np.where(m > 0.0, 1, 0.0)
        result00 += np.square(i)
        result01 += i
        result1 += m
    tt = result00[:,:,0]
    ta = result01[:,:,0]
    print(np.max(tt))
    print(np.max(ta))
    # result1 += 0.0001
    for k in range(3):
        # aa = np.divide(result00[:, :, k], result1)
        # bb = np.divide(result01[:, :, k], result1)
        # result00[:, :, k] = aa
        # result01[:, :, k] = bb
        result00[:, :, k] = np.divide(result00[:, :, k], result1, out=np.zeros_like(result00[:, :, k]), where=result1!=0)
        result01[:, :, k] = np.divide(result01[:, :, k], result1, out=np.zeros_like(result01[:, :, k]), where=result1!=0)

    tt1 = result00[:,:,0]
    ta1 = result01[:,:,0]
    print(np.max(tt1))
    print(np.max(ta1))
    result01 = np.square(result01)

    tt1 = result00[:,:,0]
    ta1 = result01[:,:,0]
    tt2 = result00[:,:,1]
    ta2 = result01[:,:,1]
    tt3 = result00[:,:,2]
    ta3 = result01[:,:,2]
    print(np.min(tt1 - ta1))
    print(np.min(tt2 - ta2))
    print(np.min(tt3 - ta3))
    print(np.max(tt1 - ta1))
    print(np.max(tt2 - ta2))
    print(np.max(tt3 - ta3))
    result = np.zeros(images[0, :, :, :].shape, dtype=np.float32)
    # result = np.sqrt(result00 - result01) * 255.0
    result = np.abs(result00 - result01) * 255.0
    result = result.astype(np.uint8)
    
    # result = np.var(images, axis=0)
    mask = np.where(result1 > 0.0, 255, result1)
    mask = mask.astype(np.uint8)
    print(np.max(result))
    print(np.min(result))
    # cv2.imshow("1111", result)
    cv2.imwrite(data_base + "result.png", result)
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(data_base + "result_gray.png", result_gray)
    result_graya = cv2.cvtColor(cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2BGRA)
    result_graya[:, :, 3] = mask
    cv2.imwrite(data_base + "result_graya.png", result_graya)
    # cv2.waitKey()

def calculate_variance__(images, masks):
    result00 = np.zeros(images[0, :, :, :].shape, dtype=np.float32)
    result01 = np.zeros(images[0, :, :, :].shape, dtype=np.float32)
    result1 = np.zeros(masks[0, :, :].shape, dtype=np.float32)
    for i, m in zip(images, masks):
        m = np.where(m > 0.0, 1, 0.0)
        result01 += i
        result1 += m
    for k in range(3):
        result01[:, :, k] = np.divide(result01[:, :, k], result1, out=np.zeros_like(result01[:, :, k]), where=result1!=0)
        
    # cv2.imshow("000", cv2.resize(result01, (400, 400)))
    # cv2.waitKey(0)
    for i in images:
        result00 += np.square(i - result01)
    for k in range(3):
        result00[:, :, k] = np.divide(result00[:, :, k], result1, out=np.zeros_like(result00[:, :, k]), where=result1!=0)

    tt1 = result00[:,:,0]
    ta1 = result01[:,:,0]
    print(np.max(tt1))
    print(np.max(ta1))
    # result01 = np.square(result01)

    tt1 = result00[:,:,0]
    ta1 = result01[:,:,0]
    tt2 = result00[:,:,1]
    ta2 = result01[:,:,1]
    tt3 = result00[:,:,2]
    ta3 = result01[:,:,2]
    print(np.min(tt1 - ta1))
    print(np.min(tt2 - ta2))
    print(np.min(tt3 - ta3))
    print(np.max(tt1 - ta1))
    print(np.max(tt2 - ta2))
    print(np.max(tt3 - ta3))
    result = np.zeros(images[0, :, :, :].shape, dtype=np.float32)
    result = np.sqrt(result00) * 255.0
    result = result.astype(np.uint8)
    
    # result = np.var(images, axis=0)
    mask = np.where(result1 > 0.0, 255, result1)
    mask = mask.astype(np.uint8)
    print(np.max(result))
    print(np.min(result))
    # cv2.imshow("1111", result)
    cv2.imwrite(data_base + "result.png", result)
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(data_base + "result_gray.png", result_gray)
    result_graya = cv2.cvtColor(cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2BGRA)
    result_graya[:, :, 3] = mask
    cv2.imwrite(data_base + "result_graya.png", result_graya)
    # cv2.waitKey()


def diff_sum(images, masks):
    result00 = np.zeros(images[0, :, :, :].shape, dtype=np.float32)
    result01 = np.zeros(images[0, :, :, :].shape, dtype=np.float32)
    result1 = np.zeros(masks[0, :, :].shape, dtype=np.float32)
    mask = np.zeros(masks[0, :, :].shape, dtype=np.float32)
    tmpimg = np.zeros(images[0, :, :, :].shape, dtype=np.float32)
    result_img = np.zeros(images[0, :, :, :].shape, dtype=np.float32)
    for i in range(len(images)):
        masks[i] = np.where(masks[i] > 0.0, 1, 0.0)
    for i in range(len(images)-1):
        tmpimg = np.zeros(images[0, :, :, :].shape, dtype=np.float32)
        tmpimg -= np.where(images[i, :, :, :] != 0, images[i+1, :, :, :], 0)
        tmpimg += np.where(images[i+1, :, :, :] != 0, images[i, :, :, :], 0)
        tmpimg = abs(tmpimg)
        # cv2.imshow("111", cv2.resize(tmpimg, (400, 400)))
        # cv2.waitKey(0)
        
        result_img += tmpimg
        mask += np.where(masks[i+1, :, :] != 0, masks[i, :, :], 0)
        
    for k in range(3):
        result_img[:, :, k] = np.divide(result_img[:, :, k], mask, out=np.zeros_like(result_img[:, :, k]), where=mask!=0)
        
    result_img = result_img * 255.0
    result_img = result_img.astype(np.uint8)
    
    # result = np.var(images, axis=0)
    mask = np.where(mask > 0.0, 255, result1)
    mask = mask.astype(np.uint8)
    # cv2.imshow("1111", result)
    cv2.imwrite(data_base + "result.png", result_img)
    result_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(data_base + "result_gray.png", result_gray)
    result_graya = cv2.cvtColor(cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2BGRA)
    result_graya[:, :, 3] = mask
    cv2.imwrite(data_base + "result_graya.png", result_graya)
    # cv2.waitKey()


def calculate_variance(images, masks):
    result = np.zeros(masks[0, :, :].shape, dtype=np.float32)
    result1 = np.zeros(masks[0, :, :].shape, dtype=np.float32)

    n, h, w, c = images.shape
    for m in masks:
        result1 += m
    mask = np.where(result1 > 0.0, 255, result1)
    mask = mask.astype(np.uint8)

    for i in range(h):
        for j in range(w):
            if mask[i, j] == 0:
                continue
            a = images[:, i, j, 0]
            b = images[:, i, j, 1]
            e, v = np.linalg.eig(np.cov(a, b))
            result[i, j] = max(e)
    result *= 255.0
    result = result.astype(np.uint8)
    result_graya = cv2.cvtColor(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2BGRA)
    result_graya[:, :, 3] = mask
    cv2.imwrite(data_base + "result_graya.png", result_graya)
    # cv2.waitKey()


def cal_ssim(img1, img2):
    
    K = [0.01, 0.03]
    L = 255
    
    M,N = np.shape(img1)

    C1 = (K[0]*L)**2
    C2 = (K[1]*L)**2
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    
    mu1 = cv2.GaussianBlur(img1, (3, 3), 1.5)
    mu2 = cv2.GaussianBlur(img2, (3, 3), 1.5)
    
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    
    
    sigma1_sq = cv2.GaussianBlur(img1*img1, (3, 3), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2*img2, (3, 3), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1*img2, (3, 3), 1.5) - mu1_mu2
   
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    mssim = np.mean(ssim_map)
    return mssim,ssim_map

def test1():
    images = []
    masks = []

    if os.path.exists(data_base + "img_dummy.npz"):
        dummy_savez = np.load(data_base + "img_dummy.npz")
        images = dummy_savez['i']
        masks = dummy_savez['m']
    else:
        img_files_path = glob("./dataset/texel/*.png")
        mask_files_path = glob("./dataset/mask/*.png")
        for f, m in zip(img_files_path, mask_files_path):
            img = cv2.imread(f)
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = cv2.resize(img, dsize=(1000, 1000), interpolation=cv2.INTER_CUBIC)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            img = img.astype(np.float32)
            img /= 255.0
            images.append(img)

            mask = cv2.imread(f)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask, dsize=(1000, 1000), interpolation=cv2.INTER_CUBIC)
            mask = mask.astype(np.float32)
            mask /= 255.0
            masks.append(mask)


        np.savez(data_base + "img_dummy.npz", i=images, m=masks)

    diff_sum(np.array(images), np.array(masks))

def classify_case():
    bbase = "./dataset/inter5/"
    in_base = bbase + "ssim/"
    ssim_files_path = glob(bbase + "ssim/*.png")
    mesh_path = "./dataset/200307_hyomin_tex.obj"
    img_now = None
    img_pre = None
    side = 1000

    mesh = om.read_trimesh("./dataset/200307_hyomin_tex.obj", vertex_tex_coord=True, face_texture_index=True)
    texcoord_array = mesh.vertex_texcoords2D().astype(np.float32)
    face_array = mesh.face_vertex_indices().astype(np.uint32)

    image = -np.ones((side, side), np.int32)

    cal_table = np.zeros((side, side))
    cal_table1 = np.ones((side, side))
    result = np.zeros((side, side))

    for i in range(len(ssim_files_path)):
        img_now = cv2.imread(ssim_files_path[i])
        img_now = cv2.cvtColor(img_now, cv2.COLOR_BGR2GRAY)
        img_now = img_now.astype(np.float64)
        img_now /= 255.0
        img_now = cv2.resize(img_now, (side, side))
        cal_table1 = np.minimum(img_now, cal_table1)
        if i > 0:
            diff = np.abs(img_now - img_pre)
            cal_table = np.where(diff > 0.05, cal_table + 1, cal_table)


        img_pre = img_now

    
    cal_table2 = cv2.imread(bbase + "min_ssim.png")
    cal_table2 = cv2.cvtColor(cal_table2, cv2.COLOR_BGR2GRAY)
    cal_table2 = cal_table2.astype(np.float64)
    cal_table2 /= 255.0
    cal_table2 = cv2.resize(cal_table2, (side, side))

    result = np.where(cal_table > 10, 1.0, 0.0)
    result *= 255
    cv2.imshow("111", result)

    result1 = np.where(cal_table1 < 0.95, 1.0, 0.0)
    result1 *= 255
    cv2.imshow("222", result1)
    
    result2 = np.where(cal_table2 < 0.75, 1.0, 0.0)
    result2 *= 255
    cv2.imshow("333", result2)
    # cv2.waitKey(0)
    
    for k, face in enumerate(face_array):
        pt1 = (texcoord_array[face[0], 0]* side, (1 - texcoord_array[face[0], 1])* side)
        pt2 = (texcoord_array[face[1], 0]* side, (1 - texcoord_array[face[1], 1])* side)
        pt3 = (texcoord_array[face[2], 0]* side, (1 - texcoord_array[face[2], 1])* side)

        triangle_cnt = np.array( [pt1, pt2, pt3] , np.int32)

        cv2.drawContours(image, [triangle_cnt], 0, k, -1)
    
    image = np.where(result2 < 1.0, -1, image)
    tri_set = np.unique(image.reshape(side * side))
    print(tri_set)

    image = -np.ones((side, side), np.int32)
    for k in tri_set:
        # print(k)
        if k < 0:
            continue
        face = face_array[k]
        pt1 = (texcoord_array[face[0], 0]* side, (1 - texcoord_array[face[0], 1])* side)
        pt2 = (texcoord_array[face[1], 0]* side, (1 - texcoord_array[face[1], 1])* side)
        pt3 = (texcoord_array[face[2], 0]* side, (1 - texcoord_array[face[2], 1])* side)

        triangle_cnt = np.array( [pt1, pt2, pt3] , np.int32)

        cv2.drawContours(image, [triangle_cnt], 0, float(k), -1)

    
    image += 1
    image = image.astype(np.float32)
    image /= np.max(image)
    cv2.imshow("image", image)
    cv2.waitKey()
    
    f = open(bbase + "var_table.csv", "w")
    for i in range(len(face_array)):
        if np.isin(i, tri_set):
            f.write(str(20.0))
        else:
            f.write(str(1.0))
        f.write(", ")
    f.write("\n")
    for i in range(len(face_array)):
        f.write(str(1.0))
        if i < len(face_array) - 1:
            f.write(", ")
    f.close()


def test_ssim():
    from skimage.measure import compare_ssim as ssim
    from scipy.ndimage import uniform_filter, gaussian_filter

    bbase = "./dataset/inter5/"
    img_files_path = glob(bbase + "texel/*.png")
    mask_files_path = glob(bbase + "mask/*.png")
    out_base = bbase + "ssim/"

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
            # v, r = cal_ssim(img_pre, img_now)
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
    cv2.imwrite(out_base + "avg_ssim.png",avg_ssim)

    min_ssim = (min_ssim * 255.0).astype(np.uint8)
    min_ssim = cv2.cvtColor(min_ssim, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(out_base + "min_ssim.png",min_ssim)

def cal_ssim(bbase):
    from skimage.measure import compare_ssim as ssim
    from scipy.ndimage import uniform_filter, gaussian_filter

    img_files_path = glob(bbase + "video/Frame_*.png")
    mask_files_path = glob(bbase + "video/Mask_*.png")
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
            # v, r = cal_ssim(img_pre, img_now)
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


def test_ssim_global_atlas():
    from skimage.measure import compare_ssim as ssim
    from scipy.ndimage import uniform_filter, gaussian_filter

    bbase = "./dataset/all/"
    img_files_path = glob(bbase + "texel/*.png")
    mask_files_path = glob(bbase + "mask/*.png")
    out_base = bbase + "ssim2/"

    img_now = None
    img_pre = None

    filtered_mask_now = None
    filtered_mask_pre = None

    side = 1000

    img_pre = cv2.imread("./dataset/multi.png", cv2.IMREAD_UNCHANGED)
    print(img_pre.shape)
    filtered_mask_pre = img_pre[:,:,3]
    img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGRA2GRAY)
    img_pre = img_pre.astype(np.float64)
    filtered_mask_pre = filtered_mask_pre.astype(np.float64)
    img_pre /= 255.0
    img_pre = cv2.resize(img_pre, (side, side))
    filtered_mask_pre /= 255.0
    filtered_mask_pre = cv2.resize(filtered_mask_pre, (side, side))
    filtered_mask_pre = np.where(filtered_mask_pre > 0.0, 1.0, 0.0)
    #cv2.imshow("000", filtered_mask_pre)
    filtered_mask_pre = cv2.GaussianBlur(filtered_mask_pre, (7, 7), 1.5)
    #cv2.imshow("001", filtered_mask_pre)
    filtered_mask_pre = np.where(filtered_mask_pre < 0.9999, 0.0, 1.0)
    #cv2.imshow("111", filtered_mask_pre)

    win_size = 7
    filter_func = uniform_filter
    filter_args = {'size': win_size}

    result = np.zeros((side, side))
    result_min = np.ones((side, side))
    masksum = np.zeros((side, side))
    result = result.astype(np.float64)
    result_min = result_min.astype(np.float64)
    masksum = masksum.astype(np.float64)

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
        filtered_mask_now = cv2.GaussianBlur(mask_now, (7, 7), 1.5)
        # filtered_mask_now = filter_func(mask_now, **filter_args)
        # filtered_mask_now = np.where(filtered_mask_now < 1.0, 0.0, 1.0)
        filtered_mask_now = np.where(filtered_mask_now < 0.9999, 0.0, 1.0)
        #cv2.imshow("222", filtered_mask_now)
        #cv2.waitKey(0)

        v, r = ssim(img_pre, img_now, full = True, gaussian_weights = True)
        # v, r = cal_ssim(img_pre, img_now)
        print(v)
        r = np.where(filtered_mask_pre + filtered_mask_now < 2, 1.0, r)
        result_min = np.minimum(r, result_min)
        r = np.where(filtered_mask_pre + filtered_mask_now < 2, 0.0, r)
        masksum += np.where(filtered_mask_pre + filtered_mask_now < 2, 0.0, 1.0)
        result += r
        r = (r * 255.0).astype(np.uint8)
        r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(out_base + str(i).zfill(6) + ".png",r)
    
    masksum = np.where(masksum < 1, 1.0, masksum)
    result = np.divide(result, masksum)
    result = (result * 255.0).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(out_base + "result.png",result)

    # result_min = np.where(result_min < 0.5, 1.0, result_min)
    result_min = (result_min * 255.0).astype(np.uint8)
    result_min = cv2.cvtColor(result_min, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(out_base + "result_min.png",result_min)

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
    # filtered_mask_pre = img_g_atlas[:,:,3]
    # img_g_atlas = cv2.cvtColor(img_g_atlas, cv2.COLOR_BGRA2BGR)
    img_g_atlas = img_g_atlas.astype(np.float64)
    # filtered_mask_pre = filtered_mask_pre.astype(np.float64)
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


# case_array = ['case_police']
# # case_array = [('case_hand_v3', 1.0, 0.05, 3, 2)]

# with open('../TextureMappingNonRigid/conf.json', 'rt', encoding='UTF8') as f:
#     conf = json.load(f)

# for case in case_array:

#     data_case = conf[case]
#     data_base = data_case['main']['data_root_path']
#     test_case = data_case['main']['atlas_path']

#     b_base = data_base + "atlas/" + test_case + "/"
#     cal_ssim(b_base)

if __name__ == "__main__":
    # test_ssim_global_atlas()
    # test_ssim()
    # cal_ssim()
    # classify_case()
    drawgraph_set()
