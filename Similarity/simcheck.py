from rigid_alignment import *
from renderer import Renderer

from threading import Thread
import os
from glob import glob
import openmesh as om
import json
import cv2

import csv

CHECK = False
LOCAL_CHECK = True

# region load data
with open('../TextureMappingNonRigid/conf.json', 'rt', encoding='UTF8') as f:
    conf = json.load(f)
# data_case = conf['case_hyomin_07']
data_case = conf['case_mario5_short']
# data_case = conf['case_sister2_short']
# data_case = conf['case_wonjong']
# data_case = conf['case_haeun_18_move2']
# data_case = conf['case_police']
data_base = data_case['main']['data_root_path']
count = 0
pcd_set = []
indices = []
nor_set = []
cen_set = []
if os.path.exists(data_base + "mesh_dummy.npz"):
    dummy_savez = np.load(data_base + "mesh_dummy.npz")
    indices = dummy_savez['i']
    pcd_set = dummy_savez['p']
    nor_set = dummy_savez['n']
    cen_set = dummy_savez['c']
else:
    mesh_path = glob(data_base + 'mesh/*.off')
    mesh = om.read_trimesh(mesh_path[0])
    indices = mesh.face_vertex_indices().astype(np.uint32)
    for f in mesh_path:
        print(count)
        mesh = om.read_trimesh(f)
        mesh.update_vertex_normals()
        pcd_set.append(mesh.points())
        nor_set.append(mesh.vertex_normals())
        cen_set.append(np.average(mesh.points(), axis=(0)))
        count += 1
    np.savez(data_base + "mesh_dummy.npz", i=indices, p=pcd_set, n=nor_set, c=cen_set)
# endregion

if LOCAL_CHECK:
    img_path = glob(data_base + 'stream/color/*.png')
    mesh_num = len(pcd_set)
    vertex_num = len(pcd_set[0])

    renderer = Renderer(data_case['mapper4D']['depth_intrinsic'])
    renderer.addProgram('default0')

    simtable = np.fromfile(data_base + 'local_sim_table.dat', dtype=np.float32)
    print(simtable.shape)
    print(np.mean(simtable))
    simtable = simtable.reshape((vertex_num, mesh_num, mesh_num))


    for pcd in pcd_set:
        renderer.addModel(pcd, nor_set[0], indices)
    while(True):
        while_exit = False
        for k in range(mesh_num):
            if while_exit:
                break
            renderer.updateModel(None, nor_set[k], None, k)
            model_img = renderer.render(k, r_mode='default0')
            model_img = model_img.astype(np.uint8)
            cv2.imshow('111', model_img)
            cv2.waitKey(1)
            count = 0
            for t in range(mesh_num):
                print(str(k) + " : " + str(t))
                color = simtable[:, k, t].reshape((vertex_num, 1))
                print(np.max(color))
                # color *= 10
                renderer.updateModel(None, color, None, t)
                tar_img = renderer.render(t, r_mode='default0')
                tar_img = tar_img.astype(np.uint8)
                cv2.imshow('222', tar_img)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
                if key == 27:
                    while_exit = True
                    break
                # if key == ord(' '):
                #     cv2.imwrite("./" + str(count) + ".png", img)
                #     count += 1
        if while_exit:
            break

elif CHECK:
    img_path = glob(data_base + 'stream/color/*.png')
    renderer = Renderer(data_case['mapper4D']['depth_intrinsic'])
    renderer.addProgram('default0')
    for pcd in pcd_set:
        renderer.addModel(pcd, nor_set[0], indices)

    sim = np.loadtxt(data_base + '/sim_table_tt.csv', delimiter=",", dtype=np.float32)
    for k, sim_row in enumerate(sim):
        print(k)
        min2max_idx = sim_row.argsort()
        model_img = renderer.render(k, r_mode='default0')
        count = 0
        # for idx in min2max_idx[::-1]:
        for idx in min2max_idx:
            print(": " + str(idx))
            img = cv2.imread(img_path[idx])
            # img = cv2.resize(img, (500, 500), interpolation = cv2.INTER_CUBIC)

            cv2.imshow('111', model_img)
            # cv2.imshow('222', cv2.resize(img, (800, 800)))
            cv2.imshow('222', img)
            key = cv2.waitKey(0)
            if key == 27:
                break
            # if key == ord(' '):
            #     cv2.imwrite("./" + str(count) + ".png", img)
            #     count += 1
            

else:
    T = np.zeros((len(pcd_set), len(pcd_set), 4, 4))

    # threads = [None] * len(pcd_set) * len(pcd_set)
    '''
    for i in range(len(pcd_set)):
        print(i)
        for j in range(len(pcd_set)):
            if i < j:
                print(str(i) + " -- " + str(j))
                #threads[i * len(pcd_set) + j] = Thread(target=align_np2, args=(pcd_set[i], pcd_set[j], cen_set[i], cen_set[j], nor_set[i], nor_set[j], T[i, j, :, :]))
                #threads[i * len(pcd_set) + j].start()
                T[i, j, :, :] = align_np(pcd_set[i], pcd_set[j], cen_set[i], cen_set[j], nor_set[i], nor_set[j])
                T[j, i, :, :] = np.linalg.inv(T[i, j])
    '''
    T = align_np_set_gpu(np.array(pcd_set), np.array(cen_set), np.array(nor_set))
    # for i in range(len(pcd_set)):
    #     for j in range(len(pcd_set)):
    #         if i < j:
    #             threads[i * len(pcd_set) + j].join()
    #             T[j, i, :, :] = np.linalg.inv(T[i, j])

    renderer = Renderer(data_case['mapper4D']['depth_intrinsic'])
    renderer.addProgram('depth0')
    renderer.addProgram('depth1')


    for pcd in pcd_set:
        renderer.addModel(pcd, nor_set[0], indices)


    # tar_img_path = glob(data_base + 'stream/depth/*.png')
    # I = []
    # for img_path in tar_img_path:
    #     I.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
    f = open(data_base + "sim_table.csv", "w")
    for i in range(len(pcd_set)):
        print(i)
        for j in range(len(pcd_set)):
            src_ren = renderer.render(i, modelRT=T[i, j, :, :], r_mode='depth0')
            tar_ren = renderer.render(j, r_mode='depth1')
            src_ren = src_ren.astype(np.float32)
            tar_ren = tar_ren.astype(np.float32)
            src_ren /= 1000.0
            tar_ren /= 1000.0
            diff_ren = abs(src_ren - tar_ren)
            mask_ren = np.where(diff_ren > 0.0, 1.0, 0.0)
            diff_ren = np.where(diff_ren > 0.1, 0.1, diff_ren)
            masksum = np.sum(mask_ren)
            if masksum == 0:
                f.write(str(0.0))
            else:
                f.write(str(np.sum(diff_ren) / np.sum(mask_ren)))
            if j != len(pcd_set) - 1 or i != len(pcd_set) - 1:
                f.write(",")
        if i != len(pcd_set) - 1:
            f.write("\n")
    f.close()
