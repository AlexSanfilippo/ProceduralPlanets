import numpy as np

# python
import math
import numpy as np
from OpenGL.GL import glBindVertexArray, glDrawElements, GL_TRIANGLES, GL_UNSIGNED_INT
from OpenGL.GL import glDeleteVertexArrays, glDeleteBuffers
from OpenGL.GL import glUseProgram

DEFAULT_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
out vec3 vertexColor;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    vertexColor = color;
}
"""
DEFAULT_FRAGMENT_SHADER = """
#version 330 core  
in vec3 vertexColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vertexColor, 1.0);
}
"""

class Icosphere:
    """
    Generate an icosphere mesh.

    Usage:
        ico = Icosphere(subdivisions=2)
        vertices, indices = ico.vertices, ico.indices
    Vertex layout: [x, y, z, r, g, b] (float32)
    Indices: uint32 flat array of triangle indices
    """

    def __init__(self, subdivisions=2, shader=None, position=(0,0,0), rotation=(0,0,0), scale=1.0):
        self.subdivisions = max(0, int(subdivisions))
        self.vertices, self.indices = self._build_icosphere(self.subdivisions)
        self.VAO, self.VBO, self.EBO = self.setup_buffers()
        # self.vertex_shader = DEFAULT_VERTEX_SHADER
        # self.fragment_shader = DEFAULT_FRAGMENT_SHADER
        self.shader = shader
        self.position = position
        self.rotation = rotation
        self.scale = scale

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def _build_icosphere(self, subdivisions):
        # Create base icosahedron
        t = (1.0 + math.sqrt(5.0)) / 2.0

        verts = [
            np.array([-1,  t,  0], dtype=np.float64),
            np.array([ 1,  t,  0], dtype=np.float64),
            np.array([-1, -t,  0], dtype=np.float64),
            np.array([ 1, -t,  0], dtype=np.float64),

            np.array([ 0, -1,  t], dtype=np.float64),
            np.array([ 0,  1,  t], dtype=np.float64),
            np.array([ 0, -1, -t], dtype=np.float64),
            np.array([ 0,  1, -t], dtype=np.float64),

            np.array([ t,  0, -1], dtype=np.float64),
            np.array([ t,  0,  1], dtype=np.float64),
            np.array([-t,  0, -1], dtype=np.float64),
            np.array([-t,  0,  1], dtype=np.float64),
        ]

        # Normalize base vertices to lie on unit sphere
        verts = [self._normalize(v) for v in verts]

        faces = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ]

        # midpoint cache to avoid duplicating vertices: key = tuple(sorted(i,j))
        midpoint_cache = {}

        def midpoint(i1, i2):
            key = (min(i1, i2), max(i1, i2))
            if key in midpoint_cache:
                return midpoint_cache[key]
            v = self._normalize((verts[i1] + verts[i2]) * 0.5)
            verts.append(v)
            idx = len(verts) - 1
            midpoint_cache[key] = idx
            return idx

        # Subdivide faces
        for _ in range(subdivisions):
            new_faces = []
            for tri in faces:
                i0, i1, i2 = tri
                a = midpoint(i0, i1)
                b = midpoint(i1, i2)
                c = midpoint(i2, i0)
                new_faces.extend([
                    (i0, a, c),
                    (i1, b, a),
                    (i2, c, b),
                    (a, b, c),
                ])
            faces = new_faces

        # Build final vertex array with color per vertex
        # Color chosen from position mapped to [0,1]
        packed_vertices = []
        for v in verts:
            pos = np.array(v, dtype=np.float32)
            color = (pos * 0.5) + 0.5  # map [-1,1] -> [0,1]
            packed_vertices.extend([pos[0], pos[1], pos[2], color[0], color[1], color[2]])

        vertices_array = np.array(packed_vertices, dtype=np.float32)
        indices_flat = np.array([i for tri in faces for i in tri], dtype=np.uint32)

        return vertices_array, indices_flat

    def setup_buffers(self):
        """
        Setup OpenGL buffers for the icosphere.
        Returns VAO, VBO, EBO
        """
        from OpenGL.GL import (
            glGenVertexArrays, glGenBuffers,
            glBindVertexArray, glBindBuffer,
            glBufferData, glEnableVertexAttribArray,
            glVertexAttribPointer,
            GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER,
            GL_STATIC_DRAW, GL_FLOAT, GL_FALSE
        )
        from ctypes import c_void_p

        VAO = glGenVertexArrays(1)
        VBO = glGenBuffers(1)
        EBO = glGenBuffers(1)

        glBindVertexArray(VAO)

        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, c_void_p(0))
        # Color attribute
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, c_void_p(3 * 4))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        return VAO, VBO, EBO

    def draw(self,view_projection_matrix):
        """
        Draw the icosphere. Binds the instance's shader before drawing.
        """
        glUseProgram(self.shader)
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def cleanup(self):
        """
        Cleanup OpenGL buffers.
        """
        glDeleteVertexArrays(1, [self.VAO])
        glDeleteBuffers(1, [self.VBO])
        glDeleteBuffers(1, [self.EBO])
