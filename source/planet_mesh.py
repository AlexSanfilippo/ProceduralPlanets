import glfw
import glm
import numpy as np
import pyrr
from OpenGL.GL import *
from noise import pnoise3
from pyglm.glm import vec3, cross, normalize, rotate, mat4

from noise_generators import fbm_terrain_3d, fbm_noise


class Line:
    """
    Simple line segment between two points. Not indexed.
    - shader_program: GLuint for compiled/linked shader program (passed to __init__).
    - draw(view_matrix): view_matrix should be convertible to a (4,4) float32 numpy array.
    """

    def __init__(self, shader_program, start=vec3(0.0, 0.0, 0.0), end=vec3(1.0, 1.0, 1.0), color_start=vec3(0.0, 1.0, 0.0), color_end=vec3(1.0, 0.0, 0.0), projection=None):
        self.shader_program = shader_program
        self.start = start
        self.end = end
        self.projection = projection if projection is not None else pyrr.matrix44.create_perspective_projection_matrix(45.0, 1.0, 0.1, 100.0, dtype=np.float32)
        self.vertices = np.array([start.x, start.y, start.z, color_start.x,color_start.y,color_start.z, end.x, end.y, end.z,color_end.x,color_end.y,color_end.z,], dtype=np.float32)
        self.setup_buffers()

    def cleanup(self):
        glDeleteVertexArrays(1, [self.VAO])
        glDeleteBuffers(1, [self.VBO])

    def setup_buffers(self):
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # Each vertex: position (3 floats) + color (3 floats) = 6 floats
        stride = 6 * self.vertices.itemsize

        # position -> location 0 (vec3)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        # color -> location 1 (vec3)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * self.vertices.itemsize))

        glBindVertexArray(0)

    def draw(self, view_matrix):
        model = np.identity(4, dtype=np.float32)
        view = np.array(view_matrix, dtype=np.float32).reshape((4, 4))

        glUseProgram(self.shader_program)

        loc_model = glGetUniformLocation(self.shader_program, "model")
        if loc_model != -1:
            glUniformMatrix4fv(loc_model, 1, GL_FALSE, model)

        loc_view = glGetUniformLocation(self.shader_program, "view")
        if loc_view != -1:
            glUniformMatrix4fv(loc_view, 1, GL_FALSE, view)

        loc_proj = glGetUniformLocation(self.shader_program, "projection")
        if loc_proj != -1:
            glUniformMatrix4fv(loc_proj, 1, GL_FALSE, self.projection)

        glBindVertexArray(self.VAO)
        glDrawArrays(GL_LINES, 0, 2)
        glBindVertexArray(0)

        glUseProgram(0)

class TriangleIndexed:
    """
    Triangle drawn with glDrawElements.
    - shader_program: GLuint for compiled/linked shader program (passed to __init__).
    - draw(view_matrix): view_matrix should be convertible to a (4,4) float32 numpy array.
    """

    def __init__(self, shader_program, position=vec3(0.0, 0.0, 0.0), scale=1.0, projection=None):
        self.shader_program = shader_program
        self.position = position
        self.scale = scale
        self.projection = projection if projection is not None else pyrr.matrix44.create_perspective_projection_matrix(45.0, 1.0, 0.1, 100.0, dtype=np.float32)
        # Interleaved vertex data: [pos.x, pos.y, pos.z, normal.x, normal.y, normal.z] * 3
        self.vertices = self.generate_vertices(position=position, scale=scale)

        # Indices for a single triangle
        self.indices = np.array([0, 1, 2], dtype=np.uint32)
        self.index_count = self.indices.size
        self.model_matrix = self.get_model_matrix()
        self.setup_buffers()

    def cleanup(self):
        """
        Delete OpenGL buffers.
        """
        glDeleteVertexArrays(1, [self.VAO])
        glDeleteBuffers(1, [self.VBO])
        glDeleteBuffers(1, [self.EBO])

    def generate_vertices(self, position, scale):
        """
        Generate vertex data for the triangle given position and scale.
        position: pyglm.vec3
        scale: float
        """
        # Define triangle vertices in local space
        local_vertices = np.array([
            [0.0,  1.0, 0.0],  # Top vertex
            [-1.0, -1.0, 0.0],  # Bottom left vertex
            [1.0, -1.0, 0.0],  # Bottom right vertex
        ], dtype=np.float32)

        # Define normals for each vertex (facing +Z)
        normals = np.array([
            [0.0, 0.0, 1.0],  # Normal for top vertex
            [0.0, 0.0, 1.0],  # Normal for bottom left vertex
            [0.0, 0.0, 1.0],  # Normal for bottom right vertex
        ], dtype=np.float32)

        # Apply scale and position to vertices
        transformed_vertices = (local_vertices * scale) + np.array([position.x, position.y, position.z], dtype=np.float32)

        # Interleave position and normal data
        interleaved_data = np.hstack((transformed_vertices, normals)).flatten().astype(np.float32)

        return interleaved_data


    def setup_buffers(self):
        """
        Set up VAO, VBO, EBO for the triangle.
        """
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)


        # Vertex attributes
        stride = 6 * self.vertices.itemsize  # 6 floats per vertex
        # position -> location 0 (vec3)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        # normal -> location 1 (vec3)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * self.vertices.itemsize))

        # Unbind VAO (EBO stays bound to VAO state)
        glBindVertexArray(0)

    def get_model_matrix(self):
        return np.identity(4, dtype=np.float32)

    def draw(self, view_matrix):
        """
        Draw the triangle. view_matrix should be a 4x4 matrix (numpy array or convertible).
        The shader program is used; if uniforms named 'model' or 'view' exist they will be set.
        """
        # Prepare matrices
        model = self.model_matrix
        view = np.array(view_matrix, dtype=np.float32).reshape((4, 4))

        glUseProgram(self.shader_program)

        # Set model uniform if present
        loc_model = glGetUniformLocation(self.shader_program, "model")
        if loc_model != -1:
            glUniformMatrix4fv(loc_model, 1, GL_FALSE, model)

        # Set view uniform if present
        loc_view = glGetUniformLocation(self.shader_program, "view")
        if loc_view != -1:
            glUniformMatrix4fv(loc_view, 1, GL_FALSE, view)

        loc_proj = glGetUniformLocation(self.shader_program, "projection")
        if loc_proj != -1:
            glUniformMatrix4fv(loc_proj, 1, GL_FALSE, self.projection)

        # Bind VAO and draw
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glBindVertexArray(0)

        glUseProgram(0)


class Cube(TriangleIndexed):
    """
    Cube drawn with glDrawElements. Overrides vertex generation and indices to produce
    a cube with per-face normals (24 vertices) and 36 indices (6 faces * 2 triangles).
    """

    def __init__(self, shader_program, position=vec3(0.0, 0.0, 0.0), scale=1.0, projection=None):
        self.shader_program = shader_program
        self.position = position
        self.scale = scale
        self.projection = projection if projection is not None else pyrr.matrix44.create_perspective_projection_matrix(45.0, 1.0, 0.1, 100.0, dtype=np.float32)

        # Build interleaved vertex data (position + normal) and index array
        self.vertices = self.generate_vertices(position, scale)
        # indices filled inside generate_vertices path or here after vertices built
        # generate_vertices returns interleaved array; we also compute indices below
        # (keeps API similar to TriangleIndexed)
        self.setup_buffers()

    def generate_vertices(self, position, scale):
        """
        Create interleaved position (vec3) + normal (vec3) data for a cube centered at
        `position` with side length `scale`. Returns a flattened float32 numpy array.
        Also sets self.indices and self.index_count.
        """
        h = float(scale) * 0.5
        px, py, pz = float(position.x), float(position.y), float(position.z)

        # Define faces: for each face we supply 4 vertices (quad) in CCW order when looking at the face
        # and a normal for the whole face.
        faces = [
            # front (+Z)
            ([
                (-h,  h,  h),
                (-h, -h,  h),
                ( h, -h,  h),
                ( h,  h,  h),
            ], (0.0, 0.0, 1.0)),
            # back (-Z)
            ([
                ( h,  h, -h),
                ( h, -h, -h),
                (-h, -h, -h),
                (-h,  h, -h),
            ], (0.0, 0.0, -1.0)),
            # left (-X)
            ([
                (-h,  h, -h),
                (-h, -h, -h),
                (-h, -h,  h),
                (-h,  h,  h),
            ], (-1.0, 0.0, 0.0)),
            # right (+X)
            ([
                ( h,  h,  h),
                ( h, -h,  h),
                ( h, -h, -h),
                ( h,  h, -h),
            ], (1.0, 0.0, 0.0)),
            # top (+Y)
            ([
                (-h,  h, -h),
                (-h,  h,  h),
                ( h,  h,  h),
                ( h,  h, -h),
            ], (0.0, 1.0, 0.0)),
            # bottom (-Y)
            ([
                (-h, -h,  h),
                (-h, -h, -h),
                ( h, -h, -h),
                ( h, -h,  h),
            ], (0.0, -1.0, 0.0)),
        ]

        verts = []
        indices = []
        for face_idx, (quad, normal) in enumerate(faces):
            base = face_idx * 4
            # append 4 vertices for this face
            for vx, vy, vz in quad:
                verts.extend([vx + px, vy + py, vz + pz, normal[0], normal[1], normal[2]])
            # two triangles per face (CCW)
            indices.extend([base, base + 1, base + 2, base + 2, base + 3, base])

        interleaved = np.array(verts, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)
        self.index_count = int(self.indices.size)
        return interleaved


class Icosphere(TriangleIndexed):
    """
    Icosphere mesh (position + normal per vertex) using indexed drawing.
    - subdivisions: number of recursive subdivisions (>= 0)
    - scale: radius of the sphere
    - position: pyglm.vec3 translation applied to final vertex positions
    """

    def __init__(self, shader_program, subdivisions=2, position=vec3(0.0, 0.0, 0.0), scale=1.0, projection=None):
        self.shader_program = shader_program
        self.position = position
        self.scale = float(scale)
        self.subdivisions = int(max(0, subdivisions))
        self.projection = projection if projection is not None else pyrr.matrix44.create_perspective_projection_matrix(45.0, 1.0, 0.1, 100.0, dtype=np.float32)

        # Build vertices and indices
        self.vertices = self.generate_vertices(position, self.scale, self.subdivisions)
        # setup GL buffers (parent's method uses self.vertices/self.indices)
        self.setup_buffers()

    def generate_vertices(self, position, scale, subdivisions):
        # Create base icosahedron
        t = (1.0 + np.sqrt(5.0)) / 2.0

        verts = [
            (-1,  t,  0),
            ( 1,  t,  0),
            (-1, -t,  0),
            ( 1, -t,  0),

            ( 0, -1,  t),
            ( 0,  1,  t),
            ( 0, -1, -t),
            ( 0,  1, -t),

            ( t,  0, -1),
            ( t,  0,  1),
            (-t,  0, -1),
            (-t,  0,  1),
        ]
        # normalize initial vertices
        vertices = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in verts]

        faces = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ]

        # midpoint cache to avoid duplicating vertices
        mid_cache = {}

        def midpoint(i1, i2):
            key = (i1, i2) if i1 < i2 else (i2, i1)
            if key in mid_cache:
                return mid_cache[key]
            v = vertices[i1] + vertices[i2]
            v = v / np.linalg.norm(v)
            idx = len(vertices)
            vertices.append(v)
            mid_cache[key] = idx
            return idx

        # subdivide faces
        for _ in range(subdivisions):
            new_faces = []
            for (a, b, c) in faces:
                ab = midpoint(a, b)
                bc = midpoint(b, c)
                ca = midpoint(c, a)
                new_faces.extend([
                    (a, ab, ca),
                    (b, bc, ab),
                    (c, ca, bc),
                    (ab, bc, ca),
                ])
            faces = new_faces

        # Build interleaved position + normal arrays
        positions = []
        normals = []
        # final vertex positions: normalized * scale + position offset
        pos_offset = np.array([position.x, position.y, position.z], dtype=np.float32)
        for v in vertices:
            n = np.array(v, dtype=np.float32)
            p = n * float(scale) + pos_offset
            positions.append(p)
            normals.append(n)  # normalized already

        # Flatten interleaved data
        positions_arr = np.vstack(positions).astype(np.float32)
        normals_arr = np.vstack(normals).astype(np.float32)
        interleaved = np.hstack((positions_arr, normals_arr)).flatten()

        # Build indices array (faces are triangles)
        idx_list = []
        for tri in faces:
            idx_list.extend([tri[0], tri[1], tri[2]])

        self.indices = np.array(idx_list, dtype=np.uint32)
        self.index_count = int(self.indices.size)

        return interleaved.astype(np.float32)


class TriangleSubdivided(TriangleIndexed):
    """
    Subclass of `TriangleIndexed` that supports recursive subdivision of the indexed
    triangle mesh. Call `subdivide()` to replace the current triangles with their
    4-way subdivision (per-triangle -> 4 triangles).
    """

    def __init__(self, shader_program, position=vec3(0.0, 0.0, 0.0), scale=1.0, projection=None):
        super().__init__(shader_program, position, scale, projection)

    def subdivide(self):
        # Parse current interleaved vertex buffer (N x 6 -> positions + normals)
        verts6 = self.vertices.reshape((-1, 6))
        positions = [v[:3].astype(np.float32) for v in verts6]
        normals = [v[3:].astype(np.float32) for v in verts6]
        indices = self.indices.flatten().tolist()

        # Helper to normalize a vector safely
        def _normalize(v):
            n = np.linalg.norm(v)
            return (v / n) if n > 0.0 else v

        # Edge midpoint cache: key = (min_idx, max_idx) -> midpoint_index
        mid_cache = {}
        def get_mid(i, j):
            key = (i, j) if i < j else (j, i)
            if key in mid_cache:
                return mid_cache[key]
            p_mid = (positions[i] + positions[j]) * 0.5
            n_mid = normals[i] + normals[j]
            n_mid = _normalize(n_mid.astype(np.float32))
            idx = len(positions)
            positions.append(p_mid.astype(np.float32))
            normals.append(n_mid.astype(np.float32))
            mid_cache[key] = idx
            return idx

        # Build new index list by subdividing each triangle into 4
        new_indices = []
        for t in range(0, len(indices), 3):
            a, b, c = indices[t], indices[t+1], indices[t+2]
            ab = get_mid(a, b)
            bc = get_mid(b, c)
            ca = get_mid(c, a)
            # four new triangles
            new_indices.extend([a, ab, ca])
            new_indices.extend([b, bc, ab])
            new_indices.extend([c, ca, bc])
            new_indices.extend([ab, bc, ca])

        # Convert back to interleaved arrays
        positions_arr = np.vstack(positions).astype(np.float32)
        normals_arr = np.vstack(normals).astype(np.float32)
        interleaved = np.hstack((positions_arr, normals_arr)).flatten().astype(np.float32)

        self.vertices = interleaved
        self.indices = np.array(new_indices, dtype=np.uint32)
        self.index_count = int(self.indices.size)

        # Delete old GL buffers if present and re-upload new buffers
        try:
            if hasattr(self, "VBO"):
                glDeleteBuffers(1, [self.VBO])
            if hasattr(self, "EBO"):
                glDeleteBuffers(1, [self.EBO])
            if hasattr(self, "VAO"):
                glDeleteVertexArrays(1, [self.VAO])
        except Exception:
            # If deletion fails (context lost or not created), continue to setup new buffers
            pass

        self.setup_buffers()


class IcosphereSubdivided(TriangleIndexed):
    """
    Icosphere with independent subdivision level per original icosahedron face.

    - face_subdivisions: int or sequence of 20 ints. If int, applied to all faces.
    - scale: radius of the sphere.
    - position: pyglm.vec3 translation applied to final vertex positions.

    Note: faces are subdivided independently and therefore vertices along shared
    edges are duplicated when adjacent faces have different subdivision levels.
    """

    def __init__(self, shader_program, face_subdivisions=2, position=vec3(0.0, 0.0, 0.0),
                 scale=1.0, projection=None):
        self.shader_program = shader_program
        self.position = position
        self.scale = float(scale)
        self.model_matrix = self.get_model_matrix()
        # normalize and coerce face_subdivisions to a list of length 20
        if isinstance(face_subdivisions, (int, np.integer)):
            self.face_subdivisions = [int(max(0, face_subdivisions))] * 20
        else:
            vals = list(face_subdivisions)
            if len(vals) != 20:
                raise ValueError("face_subdivisions must be an int or a sequence of length 20")
            self.face_subdivisions = [int(max(0, v)) for v in vals]

        self.projection = projection if projection is not None else pyrr.matrix44.create_perspective_projection_matrix(
            45.0, 1.0, 0.1, 100.0, dtype=np.float32
        )

        # Build vertices and indices
        self.vertices = self.generate_vertices(self.position, self.scale, self.face_subdivisions)
        self.setup_buffers()

    def generate_vertices(self, position, scale, face_subdivisions):
        # base icosahedron vertices (same layout as existing Icosphere)
        t = (1.0 + np.sqrt(5.0)) / 2.0
        base_verts = [
            (-1,  t,  0),
            ( 1,  t,  0),
            (-1, -t,  0),
            ( 1, -t,  0),

            ( 0, -1,  t),
            ( 0,  1,  t),
            ( 0, -1, -t),
            ( 0,  1, -t),

            ( t,  0, -1),
            ( t,  0,  1),
            (-t,  0, -1),
            (-t,  0,  1),
        ]
        vertices_unit = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in base_verts]

        faces = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ]

        # recursive subdivision of a single triangle (returns list of triangles, each tri is (v0,v1,v2) with unit-length positions)
        def subdivide_triangle(v0, v1, v2, level):
            if level == 0:
                return [(v0, v1, v2)]
            # midpoints on unit sphere
            a = v0 + v1
            a = a / np.linalg.norm(a)
            b = v1 + v2
            b = b / np.linalg.norm(b)
            c = v2 + v0
            c = c / np.linalg.norm(c)
            # recurse
            tris = []
            tris.extend(subdivide_triangle(v0, a, c, level - 1))
            tris.extend(subdivide_triangle(a, v1, b, level - 1))
            tris.extend(subdivide_triangle(c, b, v2, level - 1))
            tris.extend(subdivide_triangle(a, b, c, level - 1))
            return tris

        positions = []
        normals = []
        indices = []

        pos_offset = np.array([position.x, position.y, position.z], dtype=np.float32)
        next_index = 0

        # For each original face, subdivide according to the provided level
        for face_idx, tri in enumerate(faces):
            lvl = face_subdivisions[face_idx]
            v0 = vertices_unit[tri[0]]
            v1 = vertices_unit[tri[1]]
            v2 = vertices_unit[tri[2]]

            small_tris = subdivide_triangle(v0, v1, v2, lvl)
            for st in small_tris:
                # each small triangle contributes three vertices (duplicates allowed across faces)
                for v in st:
                    n = np.array(v, dtype=np.float32)                # normal = unit vector
                    p = n * float(scale) + pos_offset               # position scaled and offset
                    positions.append(p)
                    normals.append(n)
                    indices.append(next_index)
                    next_index += 1

        if len(positions) == 0:
            # fallback to a single face if something went wrong
            positions = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
            normals = [np.array([0.0, 0.0, 1.0], dtype=np.float32)]
            indices = [0]

        positions_arr = np.vstack(positions).astype(np.float32)
        normals_arr = np.vstack(normals).astype(np.float32)
        interleaved = np.hstack((positions_arr, normals_arr)).flatten().astype(np.float32)

        self.indices = np.array(indices, dtype=np.uint32)
        self.index_count = int(self.indices.size)

        return interleaved

    def update_subdivisions_by_player(self, player_position, min_level=0, max_level=None, regenerate=True):
        """
        Update `self.face_subdivisions` based on alignment between each original face normal
        and the vector from the icosphere center to `player_position`.

        - player_position: pyglm.vec3
        - min_level: minimum subdivision level (int)
        - max_level: maximum subdivision level (int). If None uses max(current face_subdivisions) or 3.
        - regenerate: if True, regenerate vertex/index data and re-upload GL buffers.
        """
        # determine max_level default
        if max_level is None:
            try:
                max_level = max(self.face_subdivisions)
            except Exception:
                max_level = max(3, int(max(0, getattr(self, "subdivisions", 3))))

        min_level = int(max(0, min_level))
        max_level = int(max(min_level, max_level))

        # direction from sphere center to player
        dir_vec = np.array([player_position.x - self.position.x,
                            player_position.y - self.position.y,
                            player_position.z - self.position.z], dtype=np.float32)
        norm = np.linalg.norm(dir_vec)
        if norm == 0.0:
            # player at center: set all to max_level
            new_levels = [max_level] * 20
        else:
            dir_norm = (dir_vec / norm).astype(np.float64)

            # recreate base icosahedron unit vertices and faces (same ordering as generate_vertices)
            t = (1.0 + np.sqrt(5.0)) / 2.0
            base_verts = [
                (-1, t, 0),
                (1, t, 0),
                (-1, -t, 0),
                (1, -t, 0),

                (0, -1, t),
                (0, 1, t),
                (0, -1, -t),
                (0, 1, -t),

                (t, 0, -1),
                (t, 0, 1),
                (-t, 0, -1),
                (-t, 0, 1),
            ]
            vertices_unit = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in base_verts]

            faces = [
                (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
                (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
                (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
                (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
            ]

            new_levels = []
            # sensitivity attribute (>=0): >1 makes selection more selective, <1 more permissive
            sensitivity = float(getattr(self, "sensitivity", 1.0))
            sensitivity = max(0.0, sensitivity)

            for (a, b, c) in faces:
                # compute face normal by averaging the unit vertex positions then normalizing
                fn = vertices_unit[a] + vertices_unit[b] + vertices_unit[c]
                fn_norm = fn / np.linalg.norm(fn)

                # dot in [-1,1], closer to 1 means face points toward player
                dot = float(np.dot(fn_norm, dir_norm))
                dot = float(np.clip(dot, -1.0, 1.0))
                tval = (dot + 1.0) * 0.5  # map to [0,1]

                # apply non-linear sensitivity mapping (power curve)
                if sensitivity != 1.0:
                    tval = float(np.clip(tval ** sensitivity, 0.0, 1.0))

                # map to integer subdivision level between min_level and max_level
                level = int(round(min_level + tval * (max_level - min_level)))
                level = int(np.clip(level, min_level, max_level))
                new_levels.append(level)
        # assign and optionally regenerate buffers
        self.face_subdivisions = list(new_levels)

        if regenerate:
            # regenerate vertex/index data
            self.vertices = self.generate_vertices(self.position, self.scale, self.face_subdivisions)
            # delete old GL buffers if present and re-upload new buffers
            try:
                if hasattr(self, "VBO"):
                    glDeleteBuffers(1, [self.VBO])
                if hasattr(self, "EBO"):
                    glDeleteBuffers(1, [self.EBO])
                if hasattr(self, "VAO"):
                    glDeleteVertexArrays(1, [self.VAO])
            except Exception:
                pass
            self.setup_buffers()


    def adaptive_subdivide(self, subdivide_condition, min_recursion_level=0, max_recursion_level=5, regenerate=True):
        """
        Optimized: Caches generated triangles to avoid redundant subdivision.
        """
        t = (1.0 + np.sqrt(5.0)) / 2.0
        base_verts = [
            (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
            (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
            (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
        ]
        vertices_unit = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in base_verts]
        faces = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ]

        triangle_cache = {}
        vertex_cache = {}

        def vertex_key(v):
            return tuple(np.round(v, 8))

        def triangle_key(v0, v1, v2, level):
            verts = tuple(sorted([vertex_key(v0), vertex_key(v1), vertex_key(v2)]))
            return (verts, level)

        def _normalize(v):
            n = np.linalg.norm(v)
            return v / n if n > 0.0 else v

        def recurse(v0, v1, v2, level):
            key = triangle_key(v0, v1, v2, level)
            if key in triangle_cache:
                return triangle_cache[key]
            if level < min_recursion_level or (level < max_recursion_level and subdivide_condition(v0, v1, v2, level)):
                a = vertex_cache.setdefault(vertex_key(v0 + v1), _normalize(v0 + v1))
                b = vertex_cache.setdefault(vertex_key(v1 + v2), _normalize(v1 + v2))
                c = vertex_cache.setdefault(vertex_key(v2 + v0), _normalize(v2 + v0))
                tris = []
                tris.extend(recurse(v0, a, c, level + 1))
                tris.extend(recurse(a, v1, b, level + 1))
                tris.extend(recurse(c, b, v2, level + 1))
                tris.extend(recurse(a, b, c, level + 1))
            else:
                tris = [(v0, v1, v2)]
            triangle_cache[key] = tris
            return tris

        final_tris = []
        for (ia, ib, ic) in faces:
            v0 = vertices_unit[ia]
            v1 = vertices_unit[ib]
            v2 = vertices_unit[ic]
            final_tris.extend(recurse(v0, v1, v2, 0))

        positions = []
        normals = []
        indices = []
        next_index = 0
        pos_offset = np.array([self.position.x, self.position.y, self.position.z], dtype=np.float32)

        for (v0, v1, v2) in final_tris:
            for v in (v0, v1, v2):
                n = np.array(v, dtype=np.float32)
                p = n * float(self.scale) + pos_offset
                positions.append(p)
                normals.append(n)
                indices.append(next_index)
                next_index += 1

        if len(positions) == 0:
            positions = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
            normals = [np.array([0.0, 0.0, 1.0], dtype=np.float32)]
            indices = [0]

        positions_arr = np.vstack(positions).astype(np.float32)
        normals_arr = np.vstack(normals).astype(np.float32)
        interleaved = np.hstack((positions_arr, normals_arr)).flatten().astype(np.float32)

        self.vertices = interleaved
        self.indices = np.array(indices, dtype=np.uint32)
        self.index_count = int(self.indices.size)

        if regenerate:
            try:
                if hasattr(self, "VBO"):
                    glDeleteBuffers(1, [self.VBO])
                if hasattr(self, "EBO"):
                    glDeleteBuffers(1, [self.EBO])
                if hasattr(self, "VAO"):
                    glDeleteVertexArrays(1, [self.VAO])
            except Exception:
                pass
            self.setup_buffers()


class PlanetMesh(IcosphereSubdivided):
    """
    Icosphere mesh with terrain displacement using fractal noise.
    - noise_scale: float scaling factor for noise input coordinates.
    - displacement_amplitude: float scaling factor for vertex displacement along normals.
    """

    planet_type_to_code = {
        "Earth": 2,
        "Moon": 0,
        "Mars": 1,
    }

    def __init__(self, shader_program, subdivisions=3, position=vec3(0.0, 0.0, 0.0),
                 scale=1.0, noise_scale=1.0, displacement_amplitude=0.1, projection=None,
                 lacunarity=2.0, gain=0.1, amplitude=2.5, frequency=0.06, seed=0, noise_method=fbm_noise, octaves=3,
                 planet_type="Earth"):
        self.noise_scale = float(noise_scale)
        self.displacement_amplitude = float(displacement_amplitude)
        self.lacunarity = lacunarity
        self.gain = gain
        self.amplitude = amplitude
        self.frequency = frequency
        self.seed = seed
        self.octaves = octaves
        self.noise_method = noise_method
        self.type = planet_type  # or "Moon", "Mars", etc. for different noise adjustments


        super().__init__(shader_program, subdivisions, position, scale, projection)
        glUseProgram(self.shader_program)
        glUniform1f(glGetUniformLocation(self.shader_program, "sphere_radius"), scale)
        glUniform1i(glGetUniformLocation(self.shader_program, "planet_type"), self.planet_type_to_code[self.type])
        glUseProgram(0)


    def generate_vertices(self, position, scale, subdivisions):
        base_vertices = super().generate_vertices(position, scale, subdivisions)
        verts6 = base_vertices.reshape((-1, 6))
        displaced_positions = []
        # first, displace vertices along their original normals using fbm
        for v in verts6:
            pos = v[:3].astype(np.float32)
            norm = v[3:].astype(np.float32)

            noise_val = self.noise_method(pos[0] * self.noise_scale,
                                  pos[1] * self.noise_scale,
                                  pos[2] * self.noise_scale,
                                  seed=self.seed,
                                  lacunarity=self.lacunarity,
                                  gain=self.gain,
                                  amplitude=self.amplitude,
                                  frequency=self.frequency,
                                  octaves=self.octaves,
                                  )
            if self.type == "Earth":
                noise_val = self._correct_noise_value_for_ocean_height(noise_val)
            displaced_pos = pos + norm * (noise_val * self.displacement_amplitude)
            displaced_positions.append(displaced_pos.astype(np.float32))

        positions_arr = np.vstack(displaced_positions).astype(np.float32)

        # recompute vertex normals from displaced geometry using triangle indices
        if hasattr(self, "indices") and self.indices is not None and self.indices.size >= 3:
            normals_accum = np.zeros_like(positions_arr, dtype=np.float32)
            tris = self.indices.reshape((-1, 3))
            for tri in tris:
                i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
                p0 = positions_arr[i0]
                p1 = positions_arr[i1]
                p2 = positions_arr[i2]
                e1 = p1 - p0
                e2 = p2 - p0
                face_n = np.cross(e1, e2).astype(np.float32)
                fn_len = np.linalg.norm(face_n)
                if fn_len > 0.0:
                    face_n = face_n / fn_len
                else:
                    face_n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                normals_accum[i0] += face_n
                normals_accum[i1] += face_n
                normals_accum[i2] += face_n

            norms = np.linalg.norm(normals_accum, axis=1)
            # avoid division by zero: fallback to normalized position
            fallback = positions_arr / np.linalg.norm(positions_arr, axis=1)[:, None]
            fallback[np.isnan(fallback)] = 0.0
            normals_arr = np.where(norms[:, None] > 0.0, normals_accum / norms[:, None], fallback).astype(np.float32)
        else:
            # no indices available - use position vectors as normals
            norms = np.linalg.norm(positions_arr, axis=1)
            normals_arr = (positions_arr / norms[:, None]).astype(np.float32)
            normals_arr[np.isnan(normals_arr)] = 0.0

        interleaved = np.hstack((positions_arr, normals_arr)).flatten().astype(np.float32)
        return interleaved

    def _correct_noise_value_for_ocean_height(self, noise_val: float) -> float:
        ocean_height = 0.01
        if noise_val < ocean_height:
            noise_val = 0.0
        return noise_val

    def adaptive_subdivide(self, subdivide_condition, min_recursion_level=0, max_recursion_level=5, regenerate=True):
        """
        Recursive subdivision that uses `subdivide_condition(v0,v1,v2,level)` to decide refinement.
        After triangles are finalized, vertices are displaced with `fbm_noise` using
        `self.noise_scale` and `self.displacement_amplitude`.
        """
        t = (1.0 + np.sqrt(5.0)) / 2.0
        base_verts = [
            (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
            (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
            (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
        ]
        vertices_unit = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in base_verts]
        faces = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ]

        triangle_cache = {}
        vertex_cache = {}

        def vertex_key(v):
            return tuple(np.round(v, 8))

        def triangle_key(v0, v1, v2, level):
            verts = tuple(sorted([vertex_key(v0), vertex_key(v1), vertex_key(v2)]))
            return (verts, level)

        def _normalize(v):
            n = np.linalg.norm(v)
            return v / n if n > 0.0 else v

        def recurse(v0, v1, v2, level):
            key = triangle_key(v0, v1, v2, level)
            if key in triangle_cache:
                return triangle_cache[key]
            # decide to subdivide: if level < min_recursion_level OR subdivide_condition true (and below max)
            if level < min_recursion_level or (level < max_recursion_level and subdivide_condition(v0, v1, v2, level)):
                a = vertex_cache.setdefault(vertex_key(v0 + v1), _normalize(v0 + v1))
                b = vertex_cache.setdefault(vertex_key(v1 + v2), _normalize(v1 + v2))
                c = vertex_cache.setdefault(vertex_key(v2 + v0), _normalize(v2 + v0))
                tris = []
                tris.extend(recurse(v0, a, c, level + 1))
                tris.extend(recurse(a, v1, b, level + 1))
                tris.extend(recurse(c, b, v2, level + 1))
                tris.extend(recurse(a, b, c, level + 1))
            else:
                tris = [(v0, v1, v2)]
            triangle_cache[key] = tris
            return tris

        final_tris = []
        for (ia, ib, ic) in faces:
            v0 = vertices_unit[ia]
            v1 = vertices_unit[ib]
            v2 = vertices_unit[ic]
            final_tris.extend(recurse(v0, v1, v2, 0))

        positions = []
        normals = []
        indices = []
        next_index = 0
        pos_offset = np.array([self.position.x, self.position.y, self.position.z], dtype=np.float32)

        for (v0, v1, v2) in final_tris:
            # unit normals (used as base directions for displacement)
            n0 = np.array(v0, dtype=np.float32)
            n1 = np.array(v1, dtype=np.float32)
            n2 = np.array(v2, dtype=np.float32)

            # base positions before displacement
            p0 = n0 * float(self.scale) + pos_offset
            p1 = n1 * float(self.scale) + pos_offset
            p2 = n2 * float(self.scale) + pos_offset

            # sample fbm for each vertex and displace along its original normal
            noise0 = self.noise_method(p0[0] * self.noise_scale,
                               p0[1] * self.noise_scale,
                               p0[2] * self.noise_scale,
                               seed=self.seed,
                               lacunarity=self.lacunarity,
                               gain=self.gain,
                               amplitude=self.amplitude,
                               frequency=self.frequency)
            noise1 = self.noise_method(p1[0] * self.noise_scale,
                               p1[1] * self.noise_scale,
                               p1[2] * self.noise_scale,
                               seed=self.seed,
                               lacunarity=self.lacunarity,
                               gain=self.gain,
                               amplitude=self.amplitude,
                               frequency=self.frequency)
            noise2 = self.noise_method(p2[0] * self.noise_scale,
                               p2[1] * self.noise_scale,
                               p2[2] * self.noise_scale,
                               seed=self.seed,
                               lacunarity=self.lacunarity,
                               gain=self.gain,
                               amplitude=self.amplitude,
                               frequency=self.frequency)
            if self.type == "Earth":
                noise0 = self._correct_noise_value_for_ocean_height(noise0)
                noise1 = self._correct_noise_value_for_ocean_height(noise1)
                noise2 = self._correct_noise_value_for_ocean_height(noise2)
            p0 = p0 + n0 * (noise0 * self.displacement_amplitude)
            p1 = p1 + n1 * (noise1 * self.displacement_amplitude)
            p2 = p2 + n2 * (noise2 * self.displacement_amplitude)

            # recompute face normal from displaced positions and normalize
            e1 = p1 - p0
            e2 = p2 - p0
            face_n = np.cross(e1, e2).astype(np.float32)
            fn_norm = np.linalg.norm(face_n)
            if fn_norm > 0.0:
                face_n = face_n / fn_norm
            else:
                # fallback to original averaged normal if degenerate
                face_n = _normalize((n0 + n1 + n2).astype(np.float32))

            # append three vertices with the recomputed face normal
            positions.append(p0)
            normals.append(face_n)
            indices.append(next_index)
            next_index += 1

            positions.append(p1)
            normals.append(face_n)
            indices.append(next_index)
            next_index += 1

            positions.append(p2)
            normals.append(face_n)
            indices.append(next_index)
            next_index += 1

        if len(positions) == 0:
            positions = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
            normals = [np.array([0.0, 0.0, 1.0], dtype=np.float32)]
            indices = [0]

        positions_arr = np.vstack(positions).astype(np.float32)
        normals_arr = np.vstack(normals).astype(np.float32)
        interleaved = np.hstack((positions_arr, normals_arr)).flatten().astype(np.float32)

        self.vertices = interleaved
        self.indices = np.array(indices, dtype=np.uint32)
        self.index_count = int(self.indices.size)

        if regenerate:
            try:
                if hasattr(self, "VBO"):
                    glDeleteBuffers(1, [self.VBO])
                if hasattr(self, "EBO"):
                    glDeleteBuffers(1, [self.EBO])
                if hasattr(self, "VAO"):
                    glDeleteVertexArrays(1, [self.VAO])
            except Exception:
                pass
            self.setup_buffers()


class PlanetMeshGPU(IcosphereSubdivided):
    """
    Icosphere planet mesh that performs terrain displacement on the GPU via a vertex shader.
    - noise_scale: float scaling factor for noise input coordinates.
    - displacement_amplitude: float scaling factor for vertex displacement along normals.
    """

    planet_type_to_code = {
        "Earth": 2,
        "Moon": 0,
        "Mars": 1,
    }

    def __init__(self, shader_program, subdivisions=0, position=vec3(0.0, 0.0, 0.0),
                 scale=1.0, projection=None,
                 planet_type="Earth", octaves=5, lacunarity=1.0, gain=1.0,amplitude=1.0, frequency=1.0, seed=1):
        self.type = planet_type  # or "Moon", "Mars", etc. for different noise adjustments

        super().__init__(shader_program, subdivisions, position, scale, projection)
        glUseProgram(self.shader_program)

        #Default uniform values for noise parameters
        glUniform1i(glGetUniformLocation(self.shader_program, "octaves"), octaves)
        glUniform1i(glGetUniformLocation(self.shader_program, "planet_type"), self.planet_type_to_code[planet_type])
        glUniform1f(glGetUniformLocation(self.shader_program, "lacunarity"), lacunarity)
        glUniform1f(glGetUniformLocation(self.shader_program, "gain"), gain)
        glUniform1f(glGetUniformLocation(self.shader_program, "amplitude"), amplitude)
        glUniform1f(glGetUniformLocation(self.shader_program, "frequency"), frequency)
        glUniform1f(glGetUniformLocation(self.shader_program, "seed"), seed)
        glUniform1f(glGetUniformLocation(self.shader_program, "sphere_radius"), self.scale)


        glUseProgram(0)

    def update_noise_parameter(self, parameter, value):
        glUseProgram(self.shader_program)
        if parameter == "octaves":
            glUniform1i(glGetUniformLocation(self.shader_program, "octaves"), value)
            return
        glUniform1f(glGetUniformLocation(self.shader_program, parameter), value)
        glUseProgram(0)

    def set_planet_type(self, planet_type):
        self.type = planet_type
        glUseProgram(self.shader_program)
        glUniform1i(glGetUniformLocation(self.shader_program, "planet_type"), self.planet_type_to_code[planet_type])
        glUseProgram(0)

    def rotate_planet(self, axis, angle_degrees):
        """
        Rotate the planet mesh around `axis` by `angle_degrees`.
        Uses proper matrix multiplication (matrix @ matrix) instead of element-wise '*'.
        """
        ax = axis

        angle_radians = np.radians(angle_degrees)

        # build pyglm rotation matrix
        rot_mat = rotate(mat4(1.0), float(angle_radians), ax)

        # convert pyglm.mat4 to a 4x4 numpy array (row-major)
        rot_np = np.array([[rot_mat[i][j] for j in range(4)] for i in range(4)], dtype=np.float32)

        # ensure model_matrix exists and is float32 4x4
        if not hasattr(self, "model_matrix") or self.model_matrix is None:
            self.model_matrix = np.identity(4, dtype=np.float32)
        else:
            self.model_matrix = np.array(self.model_matrix, dtype=np.float32).reshape((4, 4))

        # apply rotation (matrix multiplication) on the left so new transform = rotation * current
        self.model_matrix = rot_np @ self.model_matrix

    def reset_rotation(self):
        self.model_matrix = np.identity(4, dtype=np.float32)



class PlanetDiscreteLOD(PlanetMeshGPU):
    """
    Icosphere planet mesh with discrete LOD levels based on distance from a point (e.g., player position).
    """
    def __init__(self, shader_program, subdivisions=0, position=vec3(0.0, 0.0, 0.0),
                 scale=1.0, projection=None,
                 planet_type="Earth", octaves=5, lacunarity=1.0, gain=1.0,amplitude=1.0, frequency=1.0, seed=1):
        super().__init__(shader_program, subdivisions, position,
                 scale, projection,
                 planet_type, octaves, lacunarity, gain, amplitude,  frequency, seed)
        self.default_rotation_direction = vec3(1.0, 0.0, 0.0)
        self.current_rotation_direction = vec3(1.0, 0.0, 0.0)
        self.discrete_rotation_direction = vec3(1.0, 0.0, 0.0)
        self.lod = 0
        self.axis_vec = vec3(0.0, 1.0, 0.0)
        self.lod_meshes = {
            0: self.generate_vertices_directional(direction=(1.0, 0.0, 0.0), max_subdivisions=4, min_dot=0.0, scale=None, position=None),
            1: self.generate_vertices_directional(direction=(1.0, 0.0, 0.0), max_subdivisions=5, min_dot=0.9, scale=None, position=None),
            2: self.generate_vertices_directional(direction=(1.0, 0.0, 0.0), max_subdivisions=6, min_dot=0.9, scale=None, position=None),
            # 3: self.generate_vertices_directional(direction=(1.0, 0.0, 0.0), max_subdivisions=4, min_dot=0.9, scale=None, position=None),
            # 4: self.generate_vertices_directional(direction=(1.0, 0.0, 0.0), max_subdivisions=5, min_dot=0.9, scale=None, position=None),
            # 5: self.generate_vertices_directional(direction=(1.0, 0.0, 0.0), max_subdivisions=6, min_dot=0.9, scale=None, position=None),
            # 6: self.generate_vertices_directional(direction=(1.0, 0.0, 0.0), max_subdivisions=7, min_dot=0.9, scale=None, position=None),
            # 7: self.generate_vertices_directional(direction=(1.0, 0.0, 0.0), max_subdivisions=8, min_dot=0.9, scale=None, position=None),
            # 8: self.generate_vertices_directional(direction=(1.0, 0.0, 0.0), max_subdivisions=9, min_dot=0.9, scale=None, position=None),
            # 9: self.generate_vertices_directional(direction=(1.0, 0.0, 0.0), max_subdivisions=10, min_dot=0.9, scale=None, position=None),
            # 10: self.generate_vertices_directional(direction=(1.0, 0.0, 0.0), max_subdivisions=11, min_dot=0.9, scale=None, position=None),
            # 11: self.generate_vertices_directional(direction=(1.0, 0.0, 0.0), max_subdivisions=12, min_dot=0.9, scale=None, position=None),
            3: self.generate_vertices_directional(direction=(1.0, 0.0, 0.0), max_subdivisions=12, min_dot=0.9, scale=None, position=None),
        }
        self.vertices, self.indices, self.index_count = self.lod_meshes[0]
        self.setup_buffers()

    def lod_increase(self):
        print(f'LOD increased to {self.lod + 1}')
        self.lod += 1
        try:
            self.vertices, self.indices, self.index_count = self.lod_meshes[self.lod]
        except KeyError:
            self.lod -= 1  # revert if no higher LOD
            pass
        self.cleanup()
        self.setup_buffers()

    def lod_decrease(self):
        self.lod -= 1
        if self.lod < 0:
            self.lod = 0
        try:
            self.vertices, self.indices, self.index_count = self.lod_meshes[self.lod]
        except KeyError:
            self.lod += 1  # revert if no lower LOD
            pass
        print(f'LOD decreased to {self.lod}')
        self.cleanup()
        self.setup_buffers()

    def generate_vertices_directional(self, direction=(0.0, 1.0, 0.0), max_subdivisions=3, min_dot=0.9, scale=None, position=None):
        """
        Build an icosphere starting with 0 subdivisions and recursively subdivide
        triangles whose face normal aligns with `direction` above `min_dot`.
        - direction: tuple-like, the target direction (default up).
        - max_subdivisions: maximum recursion depth.
        - min_dot: minimum dot product between triangle normal and direction to trigger subdivision.
        - scale: optional override for sphere radius (defaults to self.scale).
        - position: optional override for sphere position (defaults to self.position).
        Returns interleaved positions+normals (flattened float32) and sets self.indices/self.index_count.
        """
        # use provided scale/position or fall back to instance values
        s = float(scale) if scale is not None else float(getattr(self, "scale", 1.0))
        pos = np.array([position.x, position.y, position.z], dtype=np.float32) if position is not None else np.array([self.position.x, self.position.y, self.position.z], dtype=np.float32)

        # normalize direction
        dir_v = np.array(direction, dtype=np.float64)
        dnorm = np.linalg.norm(dir_v)
        if dnorm == 0.0:
            dir_v = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        else:
            dir_v = dir_v / dnorm

        # base icosahedron (same ordering used elsewhere)
        t = (1.0 + np.sqrt(5.0)) / 2.0
        base_verts = [
            (-1,  t,  0),
            ( 1,  t,  0),
            (-1, -t,  0),
            ( 1, -t,  0),

            ( 0, -1,  t),
            ( 0,  1,  t),
            ( 0, -1, -t),
            ( 0,  1, -t),

            ( t,  0, -1),
            ( t,  0,  1),
            (-t,  0, -1),
            (-t,  0,  1),
        ]
        vertices_unit = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in base_verts]

        faces = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ]

        # recursive subdivision based on alignment with dir_v
        def subdivide_triangle(v0, v1, v2, level):
            # compute face normal (average of unit vertices) and normalize
            fn = v0 + v1 + v2
            fn_norm = fn / np.linalg.norm(fn)
            dot = float(np.dot(fn_norm, dir_v))

            level_to_min_dot_map = {
                # 0: 0,
                # 1: 0.25,
                # 2: 0.5,
                # 3: 0.75,
                # 4: 0.9,
                # 5: 0.95,
                # 6: 0.98,
                # 7: 0.99,
                # 8: 0.995,
                # 9: 0.999,
                # 10: 0.9995,
                # 11: 0.9999,
                # 12: 0.9999,
                # 13: 0.9999,
                0: -1.0,
                1: -1.0,
                2: -1.0,
                3: -1.0,
                4: -1.0,
                5: 0.25,
                6: 0.6,
                7: 0.85,
                8: 0.95,
                9: 0.99,
                10: 0.999,
                11: 0.999,
                12: 0.999,
                13: 0.999,
            }
            # decide to subdivide if aligned and not exceeded max depth
            if level < max_subdivisions and dot >= level_to_min_dot_map[level]:
                a = v0 + v1
                a = a / np.linalg.norm(a)
                b = v1 + v2
                b = b / np.linalg.norm(b)
                c = v2 + v0
                c = c / np.linalg.norm(c)
                tris = []
                tris.extend(subdivide_triangle(v0, a, c, level + 1))
                tris.extend(subdivide_triangle(a, v1, b, level + 1))
                tris.extend(subdivide_triangle(c, b, v2, level + 1))
                tris.extend(subdivide_triangle(a, b, c, level + 1))
                return tris
            else:
                return [(v0, v1, v2)]

        final_tris = []
        for (ia, ib, ic) in faces:
            v0 = vertices_unit[ia]
            v1 = vertices_unit[ib]
            v2 = vertices_unit[ic]
            final_tris.extend(subdivide_triangle(v0, v1, v2, 0))

        # build vertex lists (duplicate vertices per triangle as done in other methods)
        positions = []
        normals = []
        indices = []
        next_index = 0

        for (v0, v1, v2) in final_tris:
            for v in (v0, v1, v2):
                n = np.array(v, dtype=np.float32)               # unit normal
                p = n * s + pos                                # scaled position + offset
                positions.append(p)
                normals.append(n)
                indices.append(next_index)
                next_index += 1

        if len(positions) == 0:
            positions = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
            normals = [np.array([0.0, 0.0, 1.0], dtype=np.float32)]
            indices = [0]

        positions_arr = np.vstack(positions).astype(np.float32)
        normals_arr = np.vstack(normals).astype(np.float32)
        interleaved = np.hstack((positions_arr, normals_arr)).flatten().astype(np.float32)

        # set indices on self and return interleaved data
        indices = np.array(indices, dtype=np.uint32)
        index_count = int(indices.size)
        return interleaved, indices, index_count

    def _get_axis_of_rotation(self, position):
        """Given a position, compute the axis of rotation.  This will be the cross product of the current rotation direction and the vector from the planet center to the position."""
        to_position = vec3(position.x - self.position.x, position.y - self.position.y, position.z - self.position.z)
        to_position_norm = np.linalg.norm([to_position.x, to_position.y, to_position.z])
        if to_position_norm == 0.0:
            return vec3(0.0, 1.0, 0.0)  # default axis if position is exactly at planet center
        to_position_normalized = vec3(to_position.x / to_position_norm, to_position.y / to_position_norm, to_position.z / to_position_norm)
        axis = vec3(
            self.default_rotation_direction.y * to_position_normalized.z - self.default_rotation_direction.z * to_position_normalized.y,
            self.default_rotation_direction.z * to_position_normalized.x - self.default_rotation_direction.x * to_position_normalized.z,
            self.default_rotation_direction.x * to_position_normalized.y - self.default_rotation_direction.y * to_position_normalized.x,
        )
        axis_norm = np.linalg.norm([axis.x, axis.y, axis.z])
        if axis_norm == 0.0:
            return vec3(0.0, 1.0, 0.0)  # default axis if current rotation direction is parallel to vector to position
        return vec3(axis.x / axis_norm, axis.y / axis_norm, axis.z / axis_norm)

    def _update_rotation_direction(self, position):
        """Update the current rotation direction to point towards the given position."""
        to_position = vec3(position.x - self.position.x, position.y - self.position.y, position.z - self.position.z)
        to_position_norm = np.linalg.norm([to_position.x, to_position.y, to_position.z])
        if to_position_norm == 0.0:
            return  # do not update if position is exactly at planet center
        self.current_rotation_direction = vec3(to_position.x / to_position_norm, to_position.y / to_position_norm, to_position.z / to_position_norm)

    def _rotate_along_current_axis(self, angle_degrees):
        """Rotate the planet along the current axis of rotation by the given angle in degrees."""
        self.rotate_planet(self.axis_vec, angle_degrees)

    def _angle_between_vectors(self, v1, v2):
        """Compute the angle in degrees between two vectors."""
        v1_norm = np.linalg.norm([v1.x, v1.y, v1.z])
        v2_norm = np.linalg.norm([v2.x, v2.y, v2.z])
        if v1_norm == 0.0 or v2_norm == 0.0:
            return 0.0
        v1_normalized = vec3(v1.x / v1_norm, v1.y / v1_norm, v1.z / v1_norm)
        v2_normalized = vec3(v2.x / v2_norm, v2.y / v2_norm, v2.z / v2_norm)
        dot = v1_normalized.x * v2_normalized.x + v1_normalized.y * v2_normalized.y + v1_normalized.z * v2_normalized.z
        dot_clamped = max(min(dot, 1.0), -1.0)  # clamp for safety against numerical issues
        angle_radians = np.arccos(dot_clamped)
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees

    def _get_degrees_to_rotate(self, position):
        position_to_center = position - self.position
        direction_to_position = self.current_rotation_direction - self.position
        angle = self._angle_between_vectors(v1=position_to_center, v2=direction_to_position)
        return angle

    def rotate_towards_position(self, position):
        """
        working
        """
        self.reset_rotation()
        axis = self._get_axis_of_rotation(position)
        self.axis_vec = axis
        degrees = self._get_degrees_to_rotate(position)
        print(f'{degrees=}')
        self._rotate_along_current_axis(angle_degrees=degrees)
        # self._update_rotation_direction(position)


    def draw_lines(self, line_shader_program, view_matrix, camera_position):
        #create a line between the planet center and current rotation direction for debugging
        center_to_direction = Line(
            shader_program=line_shader_program,
            start=self.position,
            end=self.position + self.current_rotation_direction * self.scale * 1.5,
            color_start=vec3(1.0, 1.0, 1.0),
            color_end=vec3(1.0, 0.0, 1.0),
                projection=self.projection,
        )
        center_to_direction.draw(view_matrix=view_matrix)

        #draw line between planet center and camera position
        center_to_camera = Line(
            shader_program=line_shader_program,
            start=self.position,
            end=vec3(float(camera_position.x), float(camera_position.y) - 10.0, float(camera_position.z)),
            color_start=vec3(1.0, 1.0, 1.0),
            color_end=vec3(0.0, 1.0, 1.0),
            projection=self.projection,
        )
        center_to_camera.draw(view_matrix=view_matrix)

        #draw line between planet center and axis of rotation
        center_to_axis = Line(
            shader_program=line_shader_program,
            start=self.position,
            end=self.position + self.axis_vec * self.scale * 1.5,
            color_start=vec3(1.0, 1.0, 1.0),
            color_end=vec3(0.0, 1.0, 0.0),
            projection=self.projection,
        )
        center_to_axis.draw(view_matrix=view_matrix)

    def update_lod(self, target_position):
        """Update"""
        self.check_for_lod_update(target_position)
        self.check_for_rotation_update(target_position)

    def check_for_lod_update(self, target_position):
        """Checks distance between target_position and planet surface, updates LOD if thresholds are crossed."""
        #get the point of the planet surface between the target position and the planet center, assuming self.scale is the radius of the planet
        distance_to_surface = self._get_distance_to_surface(target_position)
        #map between LOD and min/max distance thresholds
        #each level should be 1/4 its previous
        lod_thresholds = {0: 2000, 1: 500, 2:125, 3: 32}
        #for current lod (self.lod) check if distance_to_surface is outside the thresholds for that LOD, if so update LOD
        if self.lod in lod_thresholds:
            threshold = lod_thresholds[self.lod]
            threshold_lower = lod_thresholds[self.lod - 1] if self.lod - 1 in lod_thresholds else float('inf')
            if distance_to_surface < threshold and self.lod < max(lod_thresholds.keys()):
                self.lod_increase()
            elif distance_to_surface >= threshold_lower and self.lod > 0:
                self.lod_decrease()

    def _get_distance_to_surface(self, target_position) -> float:
        to_target = target_position - self.position
        distance_to_center = np.linalg.norm([to_target.x, to_target.y, to_target.z])
        distance_to_surface = max(0.0, distance_to_center - self.scale)
        return distance_to_surface

    def check_for_rotation_update(self, target_position):
        """
        Checks angle between current rotation direction and vector to target position, updates rotation if angle exceeds threshold.

        status: essentially works, but we need to set threshold_degrees to increase sensitivity as target approachs planet
        """
        if self.lod < 3:
            return  # only update rotation for higher LODs for now

        #get distance between target position and self.position
        threshold_degrees = self._get_threshold_degrees_for_distance(
            distance=self._get_distance_to_surface(target_position=target_position)
        )
        to_target = target_position - self.position
        to_target_norm = np.linalg.norm([to_target.x, to_target.y, to_target.z])
        if to_target_norm == 0.0:
            return  # do not update if target is exactly at planet center
        to_target_normalized = to_target / to_target_norm
        angle = self._angle_between_vectors(v1=to_target_normalized, v2=self.discrete_rotation_direction)
        if angle > threshold_degrees:  # threshold in degrees for when to update rotation
            self.rotate_towards_position(position=target_position)
            self.discrete_rotation_direction = vec3(to_target_normalized)

    def _get_threshold_degrees_for_distance(self, distance):
        """Returns a threshold angle in degrees for updating rotation based on distance to target. Closer distance should have lower threshold for more responsive rotation."""
        if distance > 2000:
            return 20.0
        elif distance > 500:
            return 15.0
        elif distance > 125:
            return 10.0
        elif distance > 32:
            return 5.0
        else:
            return 2.0

