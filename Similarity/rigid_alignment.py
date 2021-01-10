import numpy as np
import open3d as o3d
import random
import copy
import threading
from multiprocessing import Pool
from T2Utils import watchModule

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

""" rigid alignment for deforming mesh
"""
'''
# TODO: GPU implement -> RANSAC
'''

VISUALIZE = False    
# region TIMING
TIMING_ALL = False
TIMING_ALIGN = True
TIMING_RANSAC = False
TIMING_ICP = False
# endregion


# region o3d visualization
def visualize(visualizer, pcd):
    visualizer.update_geometry(pcd)
    visualizer.poll_events()
    visualizer.update_renderer()


def draw_registration_result(source, target, transformation):
    if VISUALIZE:
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])
# endregion


# region conversion function
def Rt2T(R, t):
    return np.array([[R[0, 0], R[0, 1], R[0, 2], t[0]],
                     [R[1, 0], R[1, 1], R[1, 2], t[1]],
                     [R[2, 0], R[2, 1], R[2, 2], t[2]],
                     [0, 0, 0, 1]])


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0])


def q2R(q):
    # q: a + bi + cj + dk
    # q[0 ~ 4]: a, b, c, d
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    return np.array([[1.0 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                    [2 * qx * qy + 2 * qz * qw, 1.0 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
                    [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1.0 - 2 * qx ** 2 - 2 * qy ** 2]])


def pair2q(p_src, p_tar):
    q = np.array([0.0, 0.0, 0.0, 0.0])
    q[1:] = np.cross(p_src, p_tar)
    q[0] = np.sqrt(np.linalg.norm(p_src) ** 2 * np.linalg.norm(p_tar) ** 2) + np.dot(p_src, p_tar)
    return q


def pair2R(p_src, p_tar):
    v = np.cross(p_src, p_tar)
    c = np.dot(p_src, p_tar)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.identity(3) + vx + np.dot(vx, vx) / (1 + c)


def plane2R(data_src, data_dst):
    if np.shape(data_src) != (2, 3) or np.shape(data_dst) != (2, 3):
        raise TypeError('need two vector pairs')
    tan_src = data_src[0, :] / np.linalg.norm(data_src[0, :])
    tan_dst = data_dst[0, :] / np.linalg.norm(data_dst[0, :])
    R0 = pair2R(tan_src, tan_dst)
    nor_src = np.cross(data_src[0, :], data_src[1, :])
    nor_dst = np.cross(data_dst[0, :], data_dst[1, :])
    nor_src = nor_src / np.linalg.norm(nor_src)
    nor_dst = nor_dst / np.linalg.norm(nor_dst)
    rot_nor_src = np.dot(R0, nor_src.T).T
    # rot_nor_dst = np.dot(R0, nor_dst.T).T
    rot_nor_dst = nor_dst
    R1 = pair2R(rot_nor_src, rot_nor_dst)
    return np.dot(R1, R0)


def plane2q(data_src, data_dst):
    if np.shape(data_src) != (2, 3) or np.shape(data_dst) != (2, 3):
        raise TypeError('need two vector pairs')
    tan_src = data_src[0, :]
    tan_dst = data_dst[0, :]
    q0 = pair2q(tan_src, tan_dst)
    q0_conj = -q0
    q0_conj[0] = -q0_conj[0]

    nor_src = np.cross(data_src[0, :], data_src[1, :])
    nor_dst = np.cross(data_dst[0, :], data_dst[1, :])
    nor_src = nor_src / np.linalg.norm(nor_src)
    nor_dst = nor_dst / np.linalg.norm(nor_dst)
    rot_nor_src = quaternion_multiply(quaternion_multiply(q0, np.concatenate(([0], nor_src))), q0_conj)[1:]
    rot_nor_dst = quaternion_multiply(quaternion_multiply(q0, np.concatenate(([0], nor_dst))), q0_conj)[1:]
    q1 = pair2q(rot_nor_src, rot_nor_dst)
    return quaternion_multiply(q0, q1)
# endregion


# region RANSAC
def calculate_rot(data_src, data_dst):
    if data_src is None or data_dst is None:
        return np.identity(3)
    # return q2R(plane2q(data_src, data_dst))
    return plane2R(data_src, data_dst)

def rot_inlier_check(data_src, data_dst, R, threshold):
    errors = np.linalg.norm((np.dot(R, data_src.T) - data_dst.T), axis=0)
    inlier_list = np.where(errors < threshold)[0]
    curr_inliers = len(inlier_list)
    return inlier_list, curr_inliers


def rot_inlier_check_mt(data_src, data_dst, R, threshold, inliers_set, k):
    errors = np.linalg.norm((np.dot(R, data_src.T) - data_dst.T), axis=0)
    inliers_set[k] = len(np.where(errors < threshold)[0])

@watchModule(TIMING_RANSAC, TIMING_ALL)
def RANSAC_mt(iteration, selectnum, threshold, data_src, data_dst, estim_func, eval_func):
    if np.shape(data_src) != np.shape(data_dst):
        raise TypeError('incompatible dataset')
    pair_count = len(data_src)
    bset_F = estim_func(None, None)
    inliers_set = np.zeros(iteration, dtype=np.uint32)
    F_set = np.zeros((iteration, 3, 3))


    for k in range(iteration):
        idx_list = []
        while len(idx_list) < selectnum:
            tmp_idx = random.randint(0, pair_count - 1)
            if tmp_idx not in idx_list:
                idx_list.append(tmp_idx)
        tmp_data1 = np.array([data_src[i] for i in idx_list])
        tmp_data2 = np.array([data_dst[i] for i in idx_list])

        F_set[k, :, :] = estim_func(tmp_data1, tmp_data2)
    
    threads = [None] * iteration
    for i in range(len(threads)):
        threads[i] = threading.Thread(target=eval_func, args=(data_src, data_dst, F_set[k, :, :], threshold, inliers_set, k))
        threads[i].start()
    for i in range(len(threads)):
        threads[i].join()

    best_idx = np.argmax(inliers_set)

    return F_set[best_idx]


@watchModule(TIMING_RANSAC, TIMING_ALL)
def RANSAC(iteration, selectnum, threshold, data_src, data_dst, estim_func, eval_func):
    if np.shape(data_src) != np.shape(data_dst):
        raise TypeError('incompatible dataset')
    pair_count = len(data_src)
    bset_F = estim_func(None, None)
    max_inliers = 0

    for k in range(iteration):
        idx_list = []
        while len(idx_list) < selectnum:
            tmp_idx = random.randint(0, pair_count - 1)
            if tmp_idx not in idx_list:
                idx_list.append(tmp_idx)
        tmp_data1 = np.array([data_src[i] for i in idx_list])
        tmp_data2 = np.array([data_dst[i] for i in idx_list])

        tmp_F = estim_func(tmp_data1, tmp_data2)
        inlier_list, curr_inliers = eval_func(
            data_src, data_dst, tmp_F, threshold)

        if curr_inliers > max_inliers:
            # print(bset_F)
            bset_F = tmp_F
            max_inliers = curr_inliers

    return bset_F, inlier_list


mod = SourceModule("""
    __device__ void cross3x1(float3* out, float3 s_vec, float3 d_vec){
        out[0].x = s_vec.y * d_vec.z - s_vec.z * d_vec.y;
        out[0].y = s_vec.z * d_vec.x - s_vec.x * d_vec.z;
        out[0].z = s_vec.x * d_vec.y - s_vec.y * d_vec.x;
    }
    __device__ void dot3x3_scalar(float3* out, float3* in1, float3* in2, float c){
        out[0].x = in1[0].x * in2[0].x + in1[0].y * in2[1].x + in1[0].z * in2[2].x;
        out[0].y = in1[0].x * in2[0].y + in1[0].y * in2[1].y + in1[0].z * in2[2].y;
        out[0].z = in1[0].x * in2[0].z + in1[0].y * in2[1].z + in1[0].z * in2[2].z;

        out[1].x = in1[1].x * in2[0].x + in1[1].y * in2[1].x + in1[1].z * in2[2].x;
        out[1].y = in1[1].x * in2[0].y + in1[1].y * in2[1].y + in1[1].z * in2[2].y;
        out[1].z = in1[1].x * in2[0].z + in1[1].y * in2[1].z + in1[1].z * in2[2].z;

        out[2].x = in1[2].x * in2[0].x + in1[2].y * in2[1].x + in1[2].z * in2[2].x;
        out[2].y = in1[2].x * in2[0].y + in1[2].y * in2[1].y + in1[2].z * in2[2].y;
        out[2].z = in1[2].x * in2[0].z + in1[2].y * in2[1].z + in1[2].z * in2[2].z;

        out[0].x *= c;
        out[0].y *= c;
        out[0].z *= c;
        out[1].x *= c;
        out[1].y *= c;
        out[1].z *= c;
        out[2].x *= c;
        out[2].y *= c;
        out[2].z *= c;
    }
    __device__ void pair2R(float3* out, float3 s_vec, float3 d_vec){
        float3 cross_v;
        float dot_v;
        cross3x1(&cross_v, s_vec, d_vec);
        dot_v = s_vec.x * d_vec.x + s_vec.y * d_vec.y + s_vec.z * d_vec.z;
        float3 vx[3], vxvx_c1[3];
        vx[0].x = 0;
        vx[0].y = -cross_v.z;
        vx[0].z = cross_v.y;
        vx[1].x = cross_v.z;
        vx[1].y = 0;
        vx[1].z = -cross_v.x;
        vx[2].x = -cross_v.y;
        vx[2].y = cross_v.x;
        vx[2].z = 0;
        dot3x3_scalar(vxvx_c1, vx, vx, 1.0 / (1.0 + dot_v));
        out[0].x = 1.0 + vx[0].x + vxvx_c1[0].x;
        out[0].y = vx[0].y + vxvx_c1[0].y;
        out[0].z = vx[0].z + vxvx_c1[0].z;
        out[1].x = vx[1].x + vxvx_c1[1].x;
        out[1].y = 1.0 + vx[1].y + vxvx_c1[1].y;
        out[1].z = vx[1].z + vxvx_c1[1].z;
        out[2].x = vx[2].x + vxvx_c1[2].x;
        out[2].y = vx[2].y + vxvx_c1[2].y;
        out[2].z = 1.0 + vx[2].z + vxvx_c1[2].z;
    }
    __device__ void calcR(float3* out, float3 s_vec1, float3 s_vec2, float3 d_vec1, float3 d_vec2){
        float s_len1 = sqrt(s_vec1.x*s_vec1.x + s_vec1.y*s_vec1.y + s_vec1.z*s_vec1.z);
        float s_len2 = sqrt(s_vec2.x*s_vec2.x + s_vec2.y*s_vec2.y + s_vec2.z*s_vec2.z);
        s_vec1.x /= s_len1;
        s_vec1.y /= s_len1;
        s_vec1.z /= s_len1;
        s_vec2.x /= s_len2;
        s_vec2.y /= s_len2;
        s_vec2.z /= s_len2;
        float3 R0[3], R1[3];
        pair2R(R0, s_vec1, s_vec2);
        float3 n_d, n_s;
        float3 rot_n_s;
        cross3x1(&n_s, s_vec1, s_vec2);
        cross3x1(&n_d, d_vec1, d_vec2);

        float n_len1 = sqrt(n_s.x*n_s.x + n_s.y*n_s.y + n_s.z*n_s.z);
        float n_len2 = sqrt(n_d.x*n_d.x + n_d.y*n_d.y + n_d.z*n_d.z);

        n_s.x /= n_len1;
        n_s.y /= n_len1;
        n_s.z /= n_len1;
        n_d.x /= n_len2;
        n_d.y /= n_len2;
        n_d.z /= n_len2;

        rot_n_s.x = R0[0].x * n_s.x + R0[0].y * n_s.y + R0[0].z * n_s.z;
        rot_n_s.y = R0[1].x * n_s.x + R0[1].y * n_s.y + R0[1].z * n_s.z;
        rot_n_s.z = R0[2].x * n_s.x + R0[2].y * n_s.y + R0[2].z * n_s.z;
        pair2R(R1, rot_n_s, n_d);
        dot3x3_scalar(out, R1, R0, 1.0);
    }

    __global__ void RANSAC_gpu(float3 *out, float3 *pcd_set, uint2* random_idx, int set_len, int pcd_len)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        if (i > set_len - 1 || j > set_len - 1)
            return;
        int maxinliers = 0;
        for(int k = 0; k < 100; k++){
            unsigned int idx1 = random_idx[k].x;
            unsigned int idx2 = random_idx[k].y;

            float3 R[3];
            calcR(R, pcd_set[i * pcd_len + idx1], pcd_set[i * pcd_len + idx2], pcd_set[j * pcd_len + idx1], pcd_set[j * pcd_len + idx2]);
            int count = 0;
            for(int vi = 0; vi < pcd_len; vi++){
                float3 temp_vec = pcd_set[i * pcd_len + vi];
                float3 rot_temp_vec;
                rot_temp_vec.x = R[0].x * temp_vec.x + R[0].y * temp_vec.y + R[0].z * temp_vec.z;
                rot_temp_vec.y = R[1].x * temp_vec.x + R[1].y * temp_vec.y + R[1].z * temp_vec.z;
                rot_temp_vec.z = R[2].x * temp_vec.x + R[2].y * temp_vec.y + R[2].z * temp_vec.z;
                rot_temp_vec.x -= pcd_set[j * pcd_len + vi].x;
                rot_temp_vec.y -= pcd_set[j * pcd_len + vi].y;
                rot_temp_vec.z -= pcd_set[j * pcd_len + vi].z;
                if(0.0025 > (rot_temp_vec.x * rot_temp_vec.x + rot_temp_vec.y * rot_temp_vec.y + rot_temp_vec.z * rot_temp_vec.z))
                    count++;
            }
            if (maxinliers < count){
                maxinliers = count;
                out[(i * set_len + j) * 3] = R[0];
                out[(i * set_len + j) * 3 + 1] = R[1];
                out[(i * set_len + j) * 3 + 2] = R[2];
            }
        }
    }
""")

# endregion


# region ICP
# TODO: implement ICP?
@watchModule(TIMING_ICP, TIMING_ALL)
def rigidICP(src, dst, thresh, init, metric):
    return o3d.registration.registration_icp(src, dst, thresh, init, metric)
# endregion


@watchModule(TIMING_ALIGN, TIMING_ALL)
def align(mesh_src, mesh_dst, center_src, center_tar, ransac_iter=50, thresh_rough=0.05, thresh_ICP=0.02):
    # region DOCSTRING
    """align rigid alignment for deforming mesh

    Args:
        mesh_src: source mesh (o3d mesh)
        mesh_dst: target mesh (o3d mesh)
        ransac_iter: RANSAC iteration (int) default=1000
        thresh_rough: rough rotation -> RANSAC error threshold (float, meter) default=0.05
        thresh_ICP: ICP neighbor threshold (float, meter) default=0.02

    Raises:
        TypeError: two meshes are incompatible

    Returns:
        numpy matrix, array: R, t -> R dot src + t = tar
    """
    # endregion

    if len(mesh_src.vertices) != len(mesh_dst.vertices):
        raise TypeError('incompatible meshes')

    resultT = Rt2T(np.identity(3), np.array([0.0, 0.0, 0.0]))

    draw_registration_result(mesh_src, mesh_dst, resultT)

    # region compute rough Rt
    R, inlier_list = RANSAC(ransac_iter, 2, thresh_rough,
                            np.asarray(mesh_src.vertices) - center_src,
                            np.asarray(mesh_dst.vertices) - center_tar,
                            calculate_rot, rot_inlier_check)

    resultT = np.dot(Rt2T(np.identity(3), center_tar), np.dot(Rt2T(R, np.array((0, 0, 0))), Rt2T(np.identity(3), -center_src)))
    # endregion

    draw_registration_result(mesh_src, mesh_dst, resultT)

    # region ICP
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = mesh_src.vertices
    pcd_src.colors = mesh_src.vertex_colors
    pcd_src.normals = mesh_src.vertex_normals
    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = mesh_dst.vertices
    pcd_dst.colors = mesh_dst.vertex_colors
    pcd_dst.normals = mesh_dst.vertex_normals
    reg_p2l = rigidICP(pcd_src, pcd_dst, thresh_ICP, resultT,
        o3d.registration.TransformationEstimationPointToPlane())

    resultT = reg_p2l.transformation
    # endregion

    draw_registration_result(mesh_src, mesh_dst, resultT)

    return resultT


@watchModule(TIMING_ALIGN, TIMING_ALL)
def align_np(pcd_src, pcd_dst, center_src, center_dst, norm_src, norm_dst, ransac_iter=50, thresh_rough=0.05, thresh_ICP=0.02):
    # region DOCSTRING
    """align rigid alignment for deforming mesh numpy version

    Args:
        mesh_src: source mesh (o3d mesh)
        mesh_dst: target mesh (o3d mesh)
        ransac_iter: RANSAC iteration (int) default=1000
        thresh_rough: rough rotation -> RANSAC error threshold (float, meter) default=0.05
        thresh_ICP: ICP neighbor threshold (float, meter) default=0.02

    Raises:
        TypeError: two meshes are incompatible

    Returns:
        numpy matrix, array: R, t -> R dot src + t = tar
    """
    # endregion

    if len(pcd_src) != len(pcd_dst):
        raise TypeError('incompatible meshes')

    resultT = Rt2T(np.identity(3), np.array([0.0, 0.0, 0.0]))

    # region compute rough Rt
    R, inlier_list = RANSAC(ransac_iter, 2, thresh_rough,
                            pcd_src - center_src,
                            pcd_dst - center_dst,
                            calculate_rot, rot_inlier_check)

    resultT = np.dot(Rt2T(np.identity(3), center_dst), np.dot(Rt2T(R, np.array((0, 0, 0))), Rt2T(np.identity(3), -center_src)))
    # endregion

    # region ICP
    pcd_src_o3d = o3d.geometry.PointCloud()
    pcd_src_o3d.points = o3d.utility.Vector3dVector(pcd_src)
    pcd_src_o3d.normals = o3d.utility.Vector3dVector(norm_src)
    pcd_dst_o3d = o3d.geometry.PointCloud()
    pcd_dst_o3d.points = o3d.utility.Vector3dVector(pcd_dst)
    pcd_dst_o3d.normals = o3d.utility.Vector3dVector(norm_dst)
    reg_p2l = rigidICP(pcd_src_o3d, pcd_dst_o3d, thresh_ICP, resultT,
        o3d.registration.TransformationEstimationPointToPlane())

    resultT = reg_p2l.transformation
    # endregion

    return resultT


@watchModule(TIMING_ALIGN, TIMING_ALL)
def align_np_set_gpu(pcd_set, center_set, norm_set, ransac_iter=1000, thresh_rough=0.05, thresh_ICP=0.02):
    # region DOCSTRING
    """align rigid alignment for deforming mesh numpy gpu version

    Args:
        mesh_src: source mesh (o3d mesh)
        mesh_dst: target mesh (o3d mesh)
        ransac_iter: RANSAC iteration (int) default=1000
        thresh_rough: rough rotation -> RANSAC error threshold (float, meter) default=0.05
        thresh_ICP: ICP neighbor threshold (float, meter) default=0.02

    Raises:
        TypeError: two meshes are incompatible

    Returns:
        numpy matrix, array: R, t -> R dot src + t = tar
    """
    # endregion
    print("rough start")
    # data allocation
    shifted_pcd_set = copy.deepcopy(pcd_set)
    for i in range(len(pcd_set)):
        shifted_pcd_set[i, :, :] -= center_set[i, :]
    shifted_pcd_set = shifted_pcd_set.astype(np.float32)
    pcd_gpu = cuda.mem_alloc(shifted_pcd_set.nbytes)
    cuda.memcpy_htod(pcd_gpu, shifted_pcd_set)        

    # random number generator
    random_idx = np.zeros((100, 2))
    random_idx = random_idx.astype(np.int32)
    for k in range(100):
        idx_list = []
        while len(idx_list) < 2:
            tmp_idx = random.randint(0, len(pcd_set[0, :, :]) - 1)
            if tmp_idx not in idx_list:
                idx_list.append(tmp_idx)
        random_idx[k, :] = np.array(idx_list)
    random_idx_gpu = cuda.mem_alloc(random_idx.nbytes)
    cuda.memcpy_htod(random_idx_gpu, random_idx)        

    # result allocation
    resultR_array = np.zeros((len(pcd_set), len(pcd_set), 3, 3))
    resultR_array = resultR_array.astype(np.float32)
    resultR_gpu = cuda.mem_alloc(resultR_array.nbytes)
    cuda.memcpy_htod(resultR_gpu, resultR_array)


    l1_grid = (int((len(pcd_set) + 31) / 32.), int((len(pcd_set) + 31) / 32.))
    l1_block = (32,32, 1)    
    RANSAC_gpu = mod.get_function("RANSAC_gpu")
    RANSAC_gpu(resultR_gpu, pcd_gpu, random_idx_gpu, np.int32(len(pcd_set)), np.int32(len(pcd_set[0, :, :])), block=l1_block, grid=l1_grid)

    cuda.memcpy_dtoh(resultR_array, resultR_gpu)
    
    resultT_array = np.zeros((len(pcd_set), len(pcd_set), 4, 4))
    print("icp start")
    for i in range(len(pcd_set)):
        for j in range(len(pcd_set)):
            if i < j:
                print(str(i) + ", " + str(j))
                resultT = np.dot(Rt2T(np.identity(3), center_set[j, :]), np.dot(Rt2T(resultR_array[i, j, :, :], np.array((0, 0, 0))), Rt2T(np.identity(3), -center_set[i, :])))

                # region ICP
                pcd_src_o3d = o3d.geometry.PointCloud()
                pcd_src_o3d.points = o3d.utility.Vector3dVector(pcd_set[i, :, :])
                pcd_src_o3d.normals = o3d.utility.Vector3dVector(norm_set[i, :, :])
                pcd_dst_o3d = o3d.geometry.PointCloud()
                pcd_dst_o3d.points = o3d.utility.Vector3dVector(pcd_set[j, :, :])
                pcd_dst_o3d.normals = o3d.utility.Vector3dVector(norm_set[j, :, :])
                reg_p2l = rigidICP(pcd_src_o3d, pcd_dst_o3d, thresh_ICP, resultT,
                    o3d.registration.TransformationEstimationPointToPlane())
                draw_registration_result(pcd_src_o3d, pcd_dst_o3d, resultT)
                # print(resultR_array[i, j, :, :])

                resultT_array[i, j, :, :] = reg_p2l.transformation
                resultT_array[j, i, :, :] = np.linalg.inv(reg_p2l.transformation)
                draw_registration_result(pcd_src_o3d, pcd_dst_o3d, reg_p2l.transformation)
                # endregion
            if i == j:
                resultT_array[i, j, :, :] = np.identity(4)

    return resultT_array



@watchModule(TIMING_ALIGN, TIMING_ALL)
def align_np_mt(pcd_src, pcd_dst, center_src, center_dst, norm_src, norm_dst, ransac_iter=1000, thresh_rough=0.05, thresh_ICP=0.02):
    # region DOCSTRING
    """align rigid alignment for deforming mesh numpy version

    Args:
        mesh_src: source mesh (o3d mesh)
        mesh_dst: target mesh (o3d mesh)
        ransac_iter: RANSAC iteration (int) default=1000
        thresh_rough: rough rotation -> RANSAC error threshold (float, meter) default=0.05
        thresh_ICP: ICP neighbor threshold (float, meter) default=0.02

    Raises:
        TypeError: two meshes are incompatible

    Returns:
        numpy matrix, array: R, t -> R dot src + t = tar
    """
    # endregion

    if len(pcd_src) != len(pcd_dst):
        raise TypeError('incompatible meshes')

    resultT = Rt2T(np.identity(3), np.array([0.0, 0.0, 0.0]))

    # region compute rough Rt
    R = RANSAC_mt(ransac_iter, 2, thresh_rough,
                            pcd_src - center_src,
                            pcd_dst - center_dst,
                            calculate_rot, rot_inlier_check_mt)
    # ransac_thread.join()
     
    resultT = np.dot(Rt2T(np.identity(3), center_dst), np.dot(Rt2T(R, np.array((0, 0, 0))), Rt2T(np.identity(3), -center_src)))
    # endregion

    # region ICP
    pcd_src_o3d = o3d.geometry.PointCloud()
    pcd_src_o3d.points = o3d.utility.Vector3dVector(pcd_src)
    pcd_src_o3d.normals = o3d.utility.Vector3dVector(norm_src)
    pcd_dst_o3d = o3d.geometry.PointCloud()
    pcd_dst_o3d.points = o3d.utility.Vector3dVector(pcd_dst)
    pcd_dst_o3d.normals = o3d.utility.Vector3dVector(norm_dst)
    reg_p2l = rigidICP(pcd_src_o3d, pcd_dst_o3d, thresh_ICP, resultT,
        o3d.registration.TransformationEstimationPointToPlane())

    resultT = reg_p2l.transformation
    # endregion

    return resultT



if __name__=="__main__":
    import openmesh as om

    meshNmae1 = "./Frame_007.off"
    meshNmae2 = "./Frame_133.off"

    # FIXME: move to main part
    # if not os.path.exists(meshNmae1) or not os.path.exists(meshNmae2):
    #         raise IOError('mesh is not exist')

    mesh_src = o3d.io.read_triangle_mesh(meshNmae1)
    mesh_dst = o3d.io.read_triangle_mesh(meshNmae2)
    mesh_src.compute_vertex_normals()
    mesh_dst.compute_vertex_normals()

    center_src = np.average(np.asarray(mesh_src.vertices), axis=(0))
    center_tar = np.average(np.asarray(mesh_dst.vertices), axis=(0))

    trans = align(mesh_src, mesh_dst, center_src, center_tar)
    print(trans)

    pcd_set = []
    nor_set = []
    cen_set = []
    mesh1 = om.read_trimesh(meshNmae1)
    mesh2 = om.read_trimesh(meshNmae2)
    mesh1.update_vertex_normals()
    mesh2.update_vertex_normals()
    pcd_set.append(mesh1.points())
    pcd_set.append(mesh2.points())
    nor_set.append(mesh1.vertex_normals())
    nor_set.append(mesh2.vertex_normals())
    cen_set.append(np.average(mesh1.points(), axis=(0)))
    cen_set.append(np.average(mesh2.points(), axis=(0)))

    trans_array = align_np_set_gpu(np.array(pcd_set), np.array(cen_set), np.array(nor_set))
    print(trans_array)

