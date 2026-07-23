"""
Billboarded quads rendered via GPU instancing.
Each billboard is a textured quad that always faces the camera.
"""

import ctypes
import numpy as np
from OpenGL.GL import *
from engine.shader_program import create_shader
from engine.texture_loader import load_texture


class Billboard:
    """
    Draws multiple billboarded quads using GPU instancing.

    Each billboard is a textured quad that rotates to always face the camera.
    All billboards are drawn in a single instanced draw call.

    Parameters
    ----------
    texture_path : str
        Path to the texture file (relative to working directory).
    num_billboards : int
        Number of billboards to draw.
    scale : float
        Half-width/half-height of each quad (in world units).
    positions : np.ndarray
        Array of shape (num_billboards, 3) containing world-space positions.
        dtype should be float32.
    """

    _shader = None  # Compiled shader, shared across all instances

    def __init__(self, texture_path: str, num_billboards: int, scale: float,
                 positions: np.ndarray):
        self.num_billboards = int(num_billboards)
        self.scale = float(scale)

        # Ensure positions is the right shape and type
        positions = np.asarray(positions, dtype=np.float32)
        if positions.shape != (num_billboards, 3):
            raise ValueError(f"positions must have shape ({num_billboards}, 3), got {positions.shape}")
        self.positions = positions

        # Lazily compile the shader once for all instances
        if Billboard._shader is None:
            Billboard._shader = create_shader(
                vertex_file='engine/shaders/billboard.vs',
                fragment_file='engine/shaders/billboard.fs',
            )

        # Load texture
        self.texture = glGenTextures(1)
        load_texture(texture_path, self.texture)

        # Create VAO/VBO/EBO for the quad template
        self._setup_geometry()

        # Create instance buffer for positions
        self._setup_instance_buffer()

    def _setup_geometry(self):
        """
        Create a single quad with 4 vertices ([-1,-1], [1,-1], [1,1], [-1,1]).
        This quad will be instanced and rotated by the vertex shader to face the camera.
        """
        # Quad vertices in local space (before billboard rotation)
        # Each vertex is (local_x, local_y, tex_u, tex_v)
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,   # bottom-left
             1.0, -1.0, 1.0, 0.0,   # bottom-right
             1.0,  1.0, 1.0, 1.0,   # top-right
            -1.0,  1.0, 0.0, 1.0,   # top-left
        ], dtype=np.float32)

        indices = np.array([
            0, 1, 2, 0, 2, 3
        ], dtype=np.uint32)

        self.index_count = len(indices)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)

        glBindVertexArray(self.vao)

        # Upload vertex data
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # Upload index data
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Vertex attributes: 2 floats (local_x, local_y) + 2 floats (tex_u, tex_v) = 4 floats per vertex
        stride = 4 * ctypes.sizeof(ctypes.c_float)

        # location 0: local position (2 floats)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

        # location 1: texture coordinates (2 floats)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))

        glBindVertexArray(0)

    def _setup_instance_buffer(self):
        """Create the instance buffer for billboard positions."""
        self.instance_vbo = glGenBuffers(1)
        glBindBuffer(GL_COPY_WRITE_BUFFER, self.instance_vbo)
        glBufferData(GL_COPY_WRITE_BUFFER, self.positions.nbytes, self.positions, GL_DYNAMIC_DRAW)

        # Bind the instance buffer to the VAO
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)

        # location 2: instance position (3 floats)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * ctypes.sizeof(ctypes.c_float),
                              ctypes.c_void_p(0))
        glVertexAttribDivisor(2, 1)  # Advance once per instance

        glBindVertexArray(0)

    def update_positions(self, positions: np.ndarray):
        """
        Update the billboard positions.

        Parameters
        ----------
        positions : np.ndarray
            Array of shape (num_billboards, 3), dtype float32.
        """
        positions = np.asarray(positions, dtype=np.float32)
        if positions.shape != (self.num_billboards, 3):
            raise ValueError(f"positions must have shape ({self.num_billboards}, 3), got {positions.shape}")
        self.positions = positions

        # Update the instance buffer
        glBindBuffer(GL_COPY_WRITE_BUFFER, self.instance_vbo)
        glBufferSubData(GL_COPY_WRITE_BUFFER, 0, positions.nbytes, positions)

    def draw(self, view, projection):
        """
        Draw all billboards.

        Parameters
        ----------
        view : 4x4 matrix
            The view matrix (used to compute camera orientation for billboarding).
        projection : 4x4 matrix
            The projection matrix.
        """
        shader = Billboard._shader
        glUseProgram(shader)

        # Convert to float32 if needed
        view_mat = np.array(view, dtype=np.float32).reshape(4, 4)
        proj_mat = np.array(projection, dtype=np.float32).reshape(4, 4)

        # Pass matrices to shader
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view_mat)
        glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, proj_mat)
        glUniform1f(glGetUniformLocation(shader, "scale"), self.scale)

        # Bind texture to unit 0
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glUniform1i(glGetUniformLocation(shader, "u_texture"), 0)

        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Draw the instanced billboards
        glBindVertexArray(self.vao)
        glDrawElementsInstanced(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None,
                                self.num_billboards)
        glBindVertexArray(0)

        glDisable(GL_BLEND)

    def cleanup(self):
        """Free GPU resources."""
        glDeleteBuffers(1, [self.vbo])
        glDeleteBuffers(1, [self.ebo])
        glDeleteBuffers(1, [self.instance_vbo])
        glDeleteVertexArrays(1, [self.vao])
        glDeleteTextures([self.texture])

