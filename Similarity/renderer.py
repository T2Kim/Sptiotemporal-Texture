from OpenGL.GL import *
import glfw
import glm

from GLmesh import *
from GLpcd import *
import ShaderLoader

import numpy as np
from dataclasses import dataclass
import re

# TODO: render silhouette

VISUALIZE = False
import cv2

class Renderer:
    # TODO: imp delete function
    @dataclass
    class Program:
        fbo: GL_UNSIGNED_INT
        shader: GL_UNSIGNED_INT
        MVP_loc: GL_UNSIGNED_INT
        MV_loc: GL_UNSIGNED_INT
    
    def __init__(self, config):
        self.name = "original"
        # region camera parameter
        self.g_Width = config['width']
        self.g_Height = config['height']
        fx = config['fx']
        fy = config['fy']
        cx = config['cx']
        cy = config['cy']
        g_nearPlane = 0.1
        g_farPlane = 10.0
        self.M_proj = np.array([2 * fx / self.g_Width, 0, 1 - 2 * cx / self.g_Width, 0,
                    0, 2 * fy / self.g_Height, (2 * cy / self.g_Height - 1), 0,
                    0, 0, (g_farPlane + g_nearPlane) / (g_nearPlane - g_farPlane),(2 * g_farPlane * g_nearPlane) / (g_nearPlane - g_farPlane),
                    0, 0, -1, 0]).reshape(4, 4)
        # endregion
        # region opengl init
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, VISUALIZE)
        self.window = glfw.create_window(self.g_Width, self.g_Height, "vis window", None, None)
        glfw.make_context_current(self.window)


        # endregion
        self.obj_num = 0
        self.program = {}
        self.object = []
        self.curr_mode = ""


    def addProgram(self, r_mode="default", base_folder="./"):
        if r_mode in self.program:
            print("already compiled shader: " + base_folder + "shader/" + r_mode)
            self.curr_mode = r_mode
            return None
        self.curr_mode = r_mode
        if self.__extractCase() is "":
            print("undefined case in " + self.name + " renderer, go to child")
            return ""
        self.program[self.curr_mode] = self.Program(0, 0, 0, 0)
        self.program[self.curr_mode].fbo = self.__GenFBO()
        self.program[self.curr_mode].shader = ShaderLoader.compile_shader(base_folder + "shader/" + self.__extractCase() + "_vs.shader", 
                                            base_folder + "shader/" + self.__extractCase() + "_fs.shader")
        self.program[self.curr_mode].MVP_loc = glGetUniformLocation(self.program[self.curr_mode].shader, "MVP")
        self.program[self.curr_mode].MV_loc = glGetUniformLocation(self.program[self.curr_mode].shader, "MV")
        # self.program[name].cam_pos_loc = glGetUniformLocation(self.program[name].shader, "cam_pos")
        
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(100.0)
        return None


    def addModel(self, vertex, color, indices):
        # TODO: add PointCloud case
        if indices is not None:
            self.object.append(GLmesh())
            self.object[self.obj_num].setMesh(np.concatenate((vertex, color), axis=1).astype(np.float32), indices)
        else:
            self.object.append(GLpcd())
            self.object[self.obj_num].setPoints(np.concatenate((vertex, color), axis=1).astype(np.float32))

        # if model_type is "trimesh":
        #     mesh = om.read_trimesh(file_path)
        #     mesh.update_vertex_normals()
        #     point_array = mesh.points()
        #     face_array = mesh.face_vertex_indices().astype(np.uint32)
        #     color_array = abs(mesh.vertex_normals())
        #     if mesh.has_vertex_colors():
        #         color_array = mesh.vertex_colors()
        #     pcd = np.concatenate((point_array, color_array[:, :3]), axis=1).astype(np.float32)
        #     self.object.append(GLmesh())
        #     self.object[self.obj_num].setMesh(pcd, face_array)
        self.obj_num += 1


    def updateModel(self, vertex, color, indices, tar_obj_num):
        # self.object[tar_obj_num].free()
        if color.shape[1] == 1:
            color = np.repeat(color, 3, axis=1)

        if vertex is None:
            self.object[tar_obj_num].setColor(color.astype(np.float32))
            return

        if indices is not None:
            self.object[tar_obj_num].setMesh(np.concatenate((vertex, color), axis=1).astype(np.float32), indices)
        else:
            self.object[tar_obj_num].setPoints(np.concatenate((vertex, color), axis=1).astype(np.float32))


    def render(self, obj_idx=0, cam_pos=np.identity(4), modelRT=np.identity(4), r_mode=""):
        if r_mode != "":
            self.curr_mode = r_mode
        if self.program[self.curr_mode] is None:
            print("complie the shader before render!!!")
            return
        if self.obj_num <= obj_idx:
            print(obj_idx + "th model is not loaded")
            return

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.program[self.curr_mode].fbo)
        glUseProgram(self.program[self.curr_mode].shader)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # glClearColor(0.0, 0.0, 0.0, 0.0)
        # glClearDepth(100.0)
        

        cv2gl = np.eye(4)
        cv2gl[1,1] = -1
        cv2gl[2,2] = -1
        MVP = np.dot(np.dot(self.M_proj, np.dot(cv2gl, modelRT)), np.linalg.inv(cam_pos))
        MV = np.dot(modelRT, np.linalg.inv(cam_pos))
        glUniformMatrix4fv(self.program[self.curr_mode].MVP_loc, 1, GL_TRUE, MVP)
        glUniformMatrix4fv(self.program[self.curr_mode].MV_loc, 1, GL_TRUE, MV)
        self.object[obj_idx].display()
        
        # cv2.imwrite("./depth.png", self.__Frame2np())
        if VISUALIZE:
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
            try:
                glBlitFramebuffer(0,0,self.g_Width,self.g_Height,0,0,self.g_Width,self.g_Height, GL_COLOR_BUFFER_BIT, GL_NEAREST)
            except:
                pass
            glfw.swap_buffers(self.window)
        #return self.program[self.curr_mode].fbo #0 #self.__Frame2np()
        return self.__Frame2np()


    def __extractCase(self):
        default_case = re.compile("default*")
        depth_case = re.compile("depth*")
        edge_case = re.compile("edge*")
        if default_case.match(self.curr_mode) is not None:
            return "default"
        elif depth_case.match(self.curr_mode) is not None:
            return "depth"
        elif edge_case.match(self.curr_mode) is not None:
            return "edge"
        else:
            return ""


    def __GenFBO(self):
        if self.__extractCase() is "default":
            fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.g_Width, self.g_Height, 0, GL_RGBA, GL_FLOAT, None)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
            glBindTexture(GL_TEXTURE_2D, 0)
            
            DrawBuffers = list([GL_COLOR_ATTACHMENT0])
            glDrawBuffers(1, DrawBuffers)
            
            depthrenderbuffer = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.g_Width, self.g_Height)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer)
            glBindRenderbuffer(GL_RENDERBUFFER, 0)

            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
            glEnable(GL_DEPTH_CLAMP)
            glEnable(GL_DITHER)
            glShadeModel(GL_SMOOTH)
            glEnable(GL_CULL_FACE)
            glFrontFace(GL_CCW)
            glCullFace(GL_BACK)
            
            return fbo
        elif self.__extractCase() is "depth":
            fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, self.g_Width, self.g_Height, 0, GL_RED_INTEGER, GL_INT, None)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
            glBindTexture(GL_TEXTURE_2D, 0)
            
            DrawBuffers = list([GL_COLOR_ATTACHMENT0])
            glDrawBuffers(1, DrawBuffers)
            
            depthrenderbuffer = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.g_Width, self.g_Height)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer)
            glBindRenderbuffer(GL_RENDERBUFFER, 0)

            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
            glEnable(GL_DEPTH_CLAMP)
            glEnable(GL_DITHER)
            glShadeModel(GL_SMOOTH)
            glEnable(GL_CULL_FACE)
            glFrontFace(GL_CCW)
            glCullFace(GL_BACK)

            glEnable(GL_PROGRAM_POINT_SIZE)
            
            return fbo
        elif self.__extractCase() is "edge":
            fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.g_Width, self.g_Height, 0, GL_RGBA, GL_FLOAT, None)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
            glBindTexture(GL_TEXTURE_2D, 0)
            
            DrawBuffers = list([GL_COLOR_ATTACHMENT0])
            glDrawBuffers(1, DrawBuffers)
            
            depthrenderbuffer = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.g_Width, self.g_Height)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer)
            glBindRenderbuffer(GL_RENDERBUFFER, 0)

            glEnable(GL_DEPTH_TEST)
            # glDisable(GL_DEPTH_TEST)
            # glDepthFunc(GL_LESS)
            # glEnable(GL_DEPTH_CLAMP)
            # glEnable(GL_DITHER)
            # glShadeModel(GL_SMOOTH)
            # glEnable(GL_CULL_FACE)
            # glFrontFace(GL_CCW)
            # glCullFace(GL_BACK)
            
            return fbo
        print("not implemented mode yet")
        return 0

    
    def __Frame2np(self):
        if self.__extractCase() is "default":
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.program[self.curr_mode].fbo)
            vd_pixel = glReadPixels(0,0,self.g_Width, self.g_Height, GL_RGBA, GL_FLOAT)
            vd_pixel = vd_pixel.astype(np.float32)
            vd_pixel = np.reshape(vd_pixel, (self.g_Height, self.g_Width, 4))
            vd_pixel = np.flipud(vd_pixel)
            # vd_pixel *= 255.0
            return vd_pixel
        if self.__extractCase() is "depth":
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.program[self.curr_mode].fbo)
            vd_pixel = glReadPixels(0,0,self.g_Width, self.g_Height, GL_RED_INTEGER, GL_INT)
            vd_pixel = vd_pixel.astype(np.uint16)
            vd_pixel = np.reshape(vd_pixel, (self.g_Height, self.g_Width))
            vd_pixel = np.flipud(vd_pixel)
            return vd_pixel
        if self.__extractCase() is "edge":
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.program[self.curr_mode].fbo)
            vd_pixel = glReadPixels(0,0,self.g_Width, self.g_Height, GL_RGBA, GL_FLOAT)
            vd_pixel = vd_pixel.astype(np.float32)
            vd_pixel = np.reshape(vd_pixel, (self.g_Height, self.g_Width, 4))
            vd_pixel = np.flipud(vd_pixel)
            vd_pixel = vd_pixel[...,[2,1,0,3]]
            # vd_pixel[:,:,:3] *= 0.5
            # vd_pixel[:,:,:3] += 0.5
            vd_pixel *= 255.0
            return vd_pixel
        print("not implemented mode yet")
        return 0


def PixelToPoint_IR(pixel, conf):
    point = []
    point.append((pixel[0] - conf['cx']) / conf['fx'] * pixel[2])
    point.append((pixel[1] - conf['cy']) / conf['fy'] * pixel[2])
    point.append(pixel[2])
    return point


if __name__=="__main__":
    
    import os
    import openmesh as om
    import open3d as o3d
    import json
    import cv2
    import numpy as np
    with open('../TextureMappingNonRigid/conf.json', 'rt', encoding='UTF8') as f:
        conf = json.load(f)

    '''
    case_base = conf['case_hand_v3']
    # case_base = conf['case_template']
    renderer = Renderer(case_base['mapper4D']['color_intrinsic'])
    d_conf = case_base['mapper4D']['depth_intrinsic']
    d2c = case_base['mapper4D']['d_c_extrinsic']
    d2c.extend([0,0,0,1])
    d2c = np.array(d2c).reshape((4,4))
    renderer.addProgram('depth0')

    img = cv2.imread('Frame_000000.png', cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32)
    img /= 1000.0

    # img = np.array(img.reshape(d_conf['height'] * d_conf['width'], 1))
    points_template = []
    for i in range(d_conf['height']):
        for j in range(d_conf['width']):
            points_template.append(PixelToPoint_IR([j, i, 1.0], d_conf))
    
    points_template = np.array(points_template)
    points_template = points_template.reshape(d_conf['height'], d_conf['width'], 3)
    points = np.zeros_like(points_template)
    points[:, :, 0] = np.multiply(points_template[:, :, 0], img) 
    points[:, :, 1] = np.multiply(points_template[:, :, 1], img) 
    points[:, :, 2] = np.multiply(points_template[:, :, 2], img) 
    points = points.reshape(d_conf['height'] * d_conf['width'], 3)
    renderer.addModel(points, points, None)
    #ddd = renderer.render(0, modelRT=d2c, r_mode='depth0')
    ddd = renderer.render(0, modelRT=np.identity(4), r_mode='depth0')
    
    cv2.imwrite("./333.png", ddd)
    '''
    
    data_case = conf['case_hyomin_07']
    renderer = Renderer(data_case['mapper4D']['depth_intrinsic'])
    renderer.addProgram('depth0')

    # mesh = om.read_trimesh("./Frame_007.off")
    # mesh.update_vertex_normals()
    # point_array = mesh.points()
    # face_array = mesh.face_vertex_indices().astype(np.uint32)
    # color_array = mesh.vertex_normals()
    # if mesh.has_vertex_colors():
    #     color_array = mesh.vertex_colors()
    # renderer.addModel(point_array, color_array, face_array)

    data_base = data_case['main']['data_root_path']
    tex_mesh = data_case['main']['tex_mesh_path']
    
    # for i in range(27):
    #     mesh = om.read_trimesh(data_base + "mesh/Frame_" + str(i * 10).zfill(3) +".off")
    #     mesh.update_vertex_normals()
    #     point_array = mesh.points()
    #     face_array = mesh.face_vertex_indices().astype(np.uint32)
    #     color_array = mesh.vertex_normals()
    #     renderer.addModel(point_array, color_array, face_array)
    # mesh = om.read_trimesh(data_base + tex_mesh)
    # mesh.update_vertex_normals()
    # point_array = mesh.points()
    # face_array = mesh.face_vertex_indices().astype(np.uint32)
    # color_array = mesh.vertex_normals()
    # renderer.addModel(point_array, color_array, face_array)
    # for i in range(28):
    #     cv2.imwrite("./for_skel/" + "Frame_" + str(i * 10).zfill(3) +".png", renderer.render(i, modelRT=np.identity(4), r_mode='depth0').astype(np.uint16))



    mesh = om.read_trimesh("./Frame_133.off")
    mesh.update_vertex_normals()
    point_array = mesh.points()
    face_array = mesh.face_vertex_indices().astype(np.uint32)
    color_array = mesh.vertex_normals()
    if mesh.has_vertex_colors():
        color_array = mesh.vertex_colors()
    renderer.addModel(point_array, color_array, face_array)

    aaa = np.array([[-0.97255938, -0.00370338, -0.2326253 ,  0.3672922 ],
                    [-0.09772753,  0.91388712,  0.39403002, -0.75124078],
                    [ 0.21113402,  0.40595149, -0.88917142,  3.58760582],
                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    # aaa = np.identity(4)

    for i in range(8):
        print(i)
        # renderer.render(0, modelRT=aaa, r_mode='edge0')
        cv2.imshow("111", renderer.render(0, modelRT=aaa, r_mode='depth0'))
        cv2.waitKey(0)
    
