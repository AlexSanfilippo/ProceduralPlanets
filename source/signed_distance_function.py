"""
Signed-Distance-Function primitives rendered via ray-marching on a fullscreen quad.
"""

import ctypes
import numpy as np
from OpenGL.GL import *
from engine.shader_program import create_shader


class Sphere:
    """
    Draws a sphere using a ray-marched SDF in a fragment shader.

    Parameters
    ----------
    position : tuple/list of 3 floats
        World-space center of the sphere.
    radius : float
        Radius of the sphere.
    color : tuple/list of 3 floats
        RGB colour (0-1 range).
    light_pos : tuple/list of 3 floats, optional
        Position of the point light used for shading (default: [1305, 0, 0]).
    """

    _shader = None           # shared across all instances
    _quad_vao = None
    _quad_vbo = None

    def __init__(self, position, radius, color, light_pos=None):
        self.position = list(position)
        self.radius = float(radius)
        self.color = list(color)
        self.light_pos = list(light_pos) if light_pos else [1305.0, 0.0, 0.0]

        # lazily compile the shader once for all instances
        if Sphere._shader is None:
            Sphere._shader = create_shader(
                vertex_file='engine/shaders/sdf_sphere.vs',
                fragment_file='engine/shaders/sdf_sphere.fs',
            )

        # lazily create a fullscreen-quad VAO once for all instances
        if Sphere._quad_vao is None:
            self._init_quad()

    # ------------------------------------------------------------------
    @staticmethod
    def _init_quad():
        """Create a simple fullscreen triangle-strip quad (NDC -1..1)."""
        vertices = np.array([
            -1.0, -1.0, 0.0,
             1.0, -1.0, 0.0,
            -1.0,  1.0, 0.0,
             1.0,  1.0, 0.0,
        ], dtype=np.float32)

        Sphere._quad_vao = glGenVertexArrays(1)
        Sphere._quad_vbo = glGenBuffers(1)

        glBindVertexArray(Sphere._quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, Sphere._quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # location 0 -> vec3 a_position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))

        glBindVertexArray(0)

    # ------------------------------------------------------------------
    def draw(self, view, projection, camera_pos):
        """
        Ray-march the SDF sphere.

        Parameters
        ----------
        view : 4x4 numpy/pyrr matrix
            The current view matrix.
        projection : 4x4 numpy/pyrr matrix
            The current projection matrix.
        camera_pos : list/tuple of 3 floats
            World-space camera position.
        """
        shader = Sphere._shader
        glUseProgram(shader)

        # compute inverse matrices for ray construction
        view_mat = np.array(view, dtype=np.float32).reshape(4, 4)
        proj_mat = np.array(projection, dtype=np.float32).reshape(4, 4)
        inv_view = np.linalg.inv(view_mat)
        inv_proj = np.linalg.inv(proj_mat)

        glUniformMatrix4fv(glGetUniformLocation(shader, "inv_view"), 1, GL_FALSE, inv_view)
        glUniformMatrix4fv(glGetUniformLocation(shader, "inv_proj"), 1, GL_FALSE, inv_proj)
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view_mat)
        glUniformMatrix4fv(glGetUniformLocation(shader, "proj"), 1, GL_FALSE, proj_mat)
        glUniform3fv(glGetUniformLocation(shader, "cam_pos"), 1, list(camera_pos))

        glUniform3fv(glGetUniformLocation(shader, "sphere_center"), 1, self.position)
        glUniform1f(glGetUniformLocation(shader, "sphere_radius"), self.radius)
        glUniform3fv(glGetUniformLocation(shader, "sphere_color"), 1, self.color)

        glUniform3fv(glGetUniformLocation(shader, "light_pos"), 1, self.light_pos)

        # draw the fullscreen quad; the fragment shader does the ray-marching
        glBindVertexArray(Sphere._quad_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)


class SphereTransparent(Sphere):
    """
    A transparent variant of the SDF Sphere that shades by water depth.

    Parameters
    ----------
    position : tuple/list of 3 floats
        World-space center of the sphere.
    radius : float
        Radius of the sphere.
    color : tuple/list of 3 floats
        RGB colour (0-1 range). (kept for API compatibility; colour is now depth-driven)
    transparency : float
        Base transparency level between 0 (fully opaque) and 1 (fully invisible).
    max_depth : float
        World-space depth at which the water is fully opaque dark-blue.
    light_pos : tuple/list of 3 floats, optional
        Position of the point light used for shading.
    """

    _transparent_shader = None  # separate shader shared across transparent instances

    def __init__(self, position, radius, color, transparency=0.5, max_depth=30.0,
                 near_plane=0.1, far_plane=30000.0, light_pos=None):
        # Skip Sphere.__init__'s shader compilation; we use our own shader
        self.position = list(position)
        self.radius = float(radius)
        self.color = list(color)
        self.transparency = float(max(0.0, min(1.0, transparency)))
        self.max_depth = float(max_depth)
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)
        self.light_pos = list(light_pos) if light_pos else [1305.0, 0.0, 0.0]

        # compile the transparent shader once
        if SphereTransparent._transparent_shader is None:
            SphereTransparent._transparent_shader = create_shader(
                vertex_file='engine/shaders/sdf_sphere.vs',
                fragment_file='engine/shaders/sdf_ocean.fs',
            )

        # reuse the parent's shared quad VAO
        if Sphere._quad_vao is None:
            self._init_quad()

    def draw(self, view, projection, camera_pos, depth_texture=None, scene_color_texture=None, screen_size: tuple = None):
        """
        Ray-march the transparent SDF sphere with depth-based colour blending.
        Composites the scene colour from the FBO with the water effect.

        Parameters
        ----------
        depth_texture : int or None
            OpenGL texture ID of the scene depth texture from the opaque pass FBO.
        scene_color_texture : int or None
            OpenGL texture ID of the scene color texture from the opaque pass FBO.
        screen_size : tuple of (width, height)
            Pixel dimensions of the viewport, needed to sample the depth texture.
        """
        shader = SphereTransparent._transparent_shader
        glUseProgram(shader)

        if screen_size is None:
            screen_size = (1728, 972)

        view_mat = np.array(view, dtype=np.float32).reshape(4, 4)
        proj_mat = np.array(projection, dtype=np.float32).reshape(4, 4)
        inv_view = np.linalg.inv(view_mat)
        inv_proj = np.linalg.inv(proj_mat)

        glUniformMatrix4fv(glGetUniformLocation(shader, "inv_view"), 1, GL_FALSE, inv_view)
        glUniformMatrix4fv(glGetUniformLocation(shader, "inv_proj"), 1, GL_FALSE, inv_proj)
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"),     1, GL_FALSE, view_mat)
        glUniformMatrix4fv(glGetUniformLocation(shader, "proj"),     1, GL_FALSE, proj_mat)
        glUniform3fv(glGetUniformLocation(shader, "cam_pos"),        1, list(camera_pos))

        glUniform3fv(glGetUniformLocation(shader, "sphere_center"),  1, self.position)
        glUniform1f(glGetUniformLocation(shader, "sphere_radius"),   self.radius)
        glUniform3fv(glGetUniformLocation(shader, "sphere_color"),   1, self.color)
        glUniform1f(glGetUniformLocation(shader, "transparency"),    self.transparency)
        glUniform1f(glGetUniformLocation(shader, "max_depth"),       self.max_depth)
        glUniform1f(glGetUniformLocation(shader, "near_plane"),      self.near_plane)
        glUniform1f(glGetUniformLocation(shader, "far_plane"),       self.far_plane)
        glUniform3fv(glGetUniformLocation(shader, "light_pos"),      1, self.light_pos)
        glUniform2f(glGetUniformLocation(shader, "screen_size"),     float(screen_size[0]), float(screen_size[1]))

        # Bind scene color texture to texture unit 0
        glActiveTexture(GL_TEXTURE0)
        if scene_color_texture is not None:
            glBindTexture(GL_TEXTURE_2D, scene_color_texture)
        glUniform1i(glGetUniformLocation(shader, "scene_color_tex"), 0)

        # Bind scene depth texture to texture unit 1
        glActiveTexture(GL_TEXTURE1)
        if depth_texture is not None:
            glBindTexture(GL_TEXTURE_2D, depth_texture)
        glUniform1i(glGetUniformLocation(shader, "depth_tex"), 1)

        # No blending needed — the shader composites scene + water internally
        glDisable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)

        glBindVertexArray(Sphere._quad_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

        glEnable(GL_DEPTH_TEST)
        glActiveTexture(GL_TEXTURE0)


class Atmosphere(SphereTransparent):
    """
    Atmosphere shell rendered via SDF with path-length-based transparency.
    Short rays (looking down) are nearly invisible; long rays (horizon) show colour.

    Parameters
    ----------
    min_depth : float
        Path length below which atmosphere is fully transparent.
    max_depth : float
        Path length at which atmosphere reaches full opacity.
    """

    _atmosphere_shader = None

    def __init__(self, position, radius, color, transparency=0.0, min_depth=0.0,
                 max_depth=100.0, near_plane=0.1, far_plane=30000.0, light_pos=None):
        # Skip parent shader init — we use our own shader
        self.position = list(position)
        self.radius = float(radius)
        self.color = list(color)
        self.transparency = float(max(0.0, min(1.0, transparency)))
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)
        self.light_pos = list(light_pos) if light_pos else [1305.0, 0.0, 0.0]

        if Atmosphere._atmosphere_shader is None:
            Atmosphere._atmosphere_shader = create_shader(
                vertex_file='engine/shaders/sdf_sphere.vs',
                fragment_file='engine/shaders/atmosphere.fs',
            )

        if Sphere._quad_vao is None:
            self._init_quad()

    def draw(self, view, projection, camera_pos, depth_texture=None, scene_color_texture=None, screen_size=None):
        shader = Atmosphere._atmosphere_shader
        glUseProgram(shader)

        if screen_size is None:
            screen_size = (1728, 972)

        view_mat = np.array(view, dtype=np.float32).reshape(4, 4)
        proj_mat = np.array(projection, dtype=np.float32).reshape(4, 4)
        inv_view = np.linalg.inv(view_mat)
        inv_proj = np.linalg.inv(proj_mat)

        glUniformMatrix4fv(glGetUniformLocation(shader, "inv_view"), 1, GL_FALSE, inv_view)
        glUniformMatrix4fv(glGetUniformLocation(shader, "inv_proj"), 1, GL_FALSE, inv_proj)
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"),     1, GL_FALSE, view_mat)
        glUniformMatrix4fv(glGetUniformLocation(shader, "proj"),     1, GL_FALSE, proj_mat)
        glUniform3fv(glGetUniformLocation(shader, "cam_pos"),        1, list(camera_pos))

        glUniform3fv(glGetUniformLocation(shader, "sphere_center"),  1, self.position)
        glUniform1f(glGetUniformLocation(shader, "sphere_radius"),   self.radius)
        glUniform3fv(glGetUniformLocation(shader, "sphere_color"),   1, self.color)
        glUniform1f(glGetUniformLocation(shader, "transparency"),    self.transparency)
        glUniform1f(glGetUniformLocation(shader, "min_depth"),       self.min_depth)
        glUniform1f(glGetUniformLocation(shader, "max_depth"),       self.max_depth)
        glUniform1f(glGetUniformLocation(shader, "near_plane"),      self.near_plane)
        glUniform1f(glGetUniformLocation(shader, "far_plane"),       self.far_plane)
        glUniform3fv(glGetUniformLocation(shader, "light_pos"),      1, self.light_pos)
        glUniform2f(glGetUniformLocation(shader, "screen_size"),     float(screen_size[0]), float(screen_size[1]))

        glActiveTexture(GL_TEXTURE0)
        if scene_color_texture is not None:
            glBindTexture(GL_TEXTURE_2D, scene_color_texture)
        glUniform1i(glGetUniformLocation(shader, "scene_color_tex"), 0)

        glActiveTexture(GL_TEXTURE1)
        if depth_texture is not None:
            glBindTexture(GL_TEXTURE_2D, depth_texture)
        glUniform1i(glGetUniformLocation(shader, "depth_tex"), 1)

        glDisable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)

        glBindVertexArray(Sphere._quad_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

        glEnable(GL_DEPTH_TEST)
        glActiveTexture(GL_TEXTURE0)
