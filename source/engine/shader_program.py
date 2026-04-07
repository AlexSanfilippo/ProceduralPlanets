"""
Tools for creating shaders
"""
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.raw.GL.ARB.compute_shader import GL_COMPUTE_SHADER
from OpenGL.raw.GL.VERSION.GL_2_0 import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER


def create_compute_shader(shader_file):
    with open(shader_file, 'r') as file:
        return compileProgram(compileShader(file.read(), GL_COMPUTE_SHADER))

def create_shader(vertex_file, fragment_file):
    with open(vertex_file, 'r') as file:
        vertex_src = file.read()
    with open(fragment_file, 'r') as file:
        fragment_src = file.read()
    return compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

