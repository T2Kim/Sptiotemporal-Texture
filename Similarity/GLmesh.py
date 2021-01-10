import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import cv2

class GLmesh:
    def __init__(self):
        self.valid = True
    
    def setMesh(self, points, indices):
        self.count = len(points)
        points = np.reshape(points, points.size)
        indices = np.reshape(indices, indices.size).astype(np.uint32)
        self.idx_count = len(indices)
        self.VAO = glGenVertexArrays(1)
        self.vertexbuffer = glGenBuffers(1)
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexbuffer)
        glBufferData(GL_ARRAY_BUFFER, points.itemsize * len(points), points, GL_STATIC_DRAW)
        self.elembuffer = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.elembuffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.itemsize * len(indices), indices, GL_STATIC_DRAW)

        # points
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, points.itemsize * 6, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # color
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, points.itemsize * 6, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    def setColor(self, color):
        self.count = len(color)
        color = np.reshape(color, color.size)
        self.vertexbuffer_color = glGenBuffers(1)
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexbuffer_color)
        glBufferData(GL_ARRAY_BUFFER, color.itemsize * len(color), color, GL_STATIC_DRAW)

        # color
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, color.itemsize * 3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)
        
    def setMesh_tex(self, points, indices, texPath):
        self.count = len(points)
        points = np.reshape(points, points.size)
        indices = np.reshape(indices, indices.size).astype(np.uint32)
        self.idx_count = len(indices)
        self.VAO = glGenVertexArrays(1)
        self.vertexbuffer = glGenBuffers(1)
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexbuffer)
        glBufferData(GL_ARRAY_BUFFER, points.itemsize * len(points), points, GL_STATIC_DRAW)
        self.elembuffer = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.elembuffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.itemsize * len(indices), indices, GL_STATIC_DRAW)

        # points
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, points.itemsize * 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # color
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, points.itemsize * 8, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        # tex_coord
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, points.itemsize * 8, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)
        glBindVertexArray(0)
        
        texture_im = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_im)
        # Set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # Set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # load image
        image = cv2.imread(texPath)
        h, w, c = image.shape
        if c == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            h, w, c = image.shape
        image = np.flipud(image)
        img_data = np.reshape(np.array(image / 255.0, np.float32), h*w*c)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_FLOAT, img_data)
        glEnable(GL_TEXTURE_2D)

    def display(self):
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, self.idx_count, GL_UNSIGNED_INT, None)

