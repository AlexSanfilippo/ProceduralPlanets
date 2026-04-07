"""
Render an icosphere using OpenGL in Python.
"""


from icosphere import Icosphere
import sys
from ctypes import c_void_p
import glfw
from OpenGL.GL import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
from engine.shader_program import create_shader

#vertex shader that colors vertices of sphere for realistic heightmap
VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
out vec3 vertexColor;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);  
    vertexColor = color;
}
"""
FRAGMENT_SHADER_SRC = """
#version 330 core
in vec3 vertexColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vertexColor, 1.0);
}
"""

if not glfw.init():
    print("Failed to initialize GLFW", file=sys.stderr)

window = glfw.create_window(800, 600, "Icosphere", None, None)
if not window:
    glfw.terminate()
    print("Failed to create GLFW window", file=sys.stderr)

glfw.make_context_current(window)

shader_program = create_shader(vertex_file='engine/shaders/pos_only.vs', fragment_file='engine/shaders/pos_only.fs')
shader_program_3d = create_shader(vertex_file='engine/shaders/textured_lit.vs', fragment_file='engine/shaders/textured_lit.fs')

#DRAW IN WIREFRAME MODE
glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

# Create icosphere
icosphere = Icosphere(subdivisions=3, shader=shader_program)
VAO, VBO, EBO = icosphere.setup_buffers()

glEnable(GL_DEPTH_TEST)


while not glfw.window_should_close(window):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUseProgram(shader_program)
    # glUseProgram(shader_program_3d)

    # Set up model, view, projection matrices here as needed
    model = np.identity(4, dtype=np.float32)
    view = np.identity(4, dtype=np.float32)
    projection = np.identity(4, dtype=np.float32)

    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    proj_loc = glGetUniformLocation(shader_program, "projection")

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

    # Draw icosphere
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, len(icosphere.indices), GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()