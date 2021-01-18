from rigid_alignment import *
from renderer import Renderer

from threading import Thread
import os
from glob import glob
import openmesh as om
import json
import cv2

import csv


# region load data
with open('../TextureMappingNonRigid/conf.json', 'rt', encoding='UTF8') as f:
    conf = json.load(f)
data_case = conf['case_main_test']
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

T = align_np_set_gpu(np.array(pcd_set), np.array(cen_set), np.array(nor_set))

renderer = Renderer(data_case['mapper4D']['depth_intrinsic'])
renderer.addProgram('depth0')
renderer.addProgram('depth1')


for pcd in pcd_set:
    renderer.addModel(pcd, nor_set[0], indices)

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
