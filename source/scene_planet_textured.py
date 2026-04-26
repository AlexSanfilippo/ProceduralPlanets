"""
Scene for textured planet rendering
"""
from random import random

import glfw
import numpy as np
import pyglm
from OpenGL.GL import *
import pyrr

from engine.shader_program import create_shader
from engine.texture_loader import load_texture
from engine.camera import Camera, SimulationCamera, RollableCamera
from engine.skybox import Skybox
from math import sin, cos
from glm import cos, radians
import logging
import engine.point_light_cube as plc
from engine.procedural_mesh import CubeMeshStatic
from engine.gui import GUI
from engine.screen_capture import capture_screenshot, write_fbo_to_gif, save_to_gif
from planet_mesh import TexturedPlanetMeshGPU, Line
import ctypes
import numpy as np
from OpenGL.GL import *
from pyglm.glm import vec3

#todo: move to config file
config_scene = {
    "background_color": [0.0, 0.1, 0.1, 1.0],
    "initial_camera_pos": [0.0, 20.0, 20.0],
    "initial_camera_front": [0.0, -0.5, -1.0],
    "name": "Textured Planet Scene",
    "enable_backface_culling": True,
    "cull_face_mode": "BACK",  # options: BACK, FRONT
}


logger = logging.getLogger(name=__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


"""===============GLOBAL VARIABLES======================="""
WIDTH, HEIGHT = 1728, 972
WINDOW_POSITION = (40, 40)
WRITE_TO_GIF = False
lastX, lastY = WIDTH / 2, HEIGHT / 2
DRAW_DISTANCE = 10000

#key-input globals
first_mouse = True
left, right, forward, backward, make_new_surface = False, False, False, False, False
player_left, player_right, player_forward, player_backward = False, False, False, False
yaw_counterclockwise, yaw_clockwise = False, False
up, down = False, False
pause = False
switch_view_mode = False
test_subdivide = False
time_delta = 0.01

# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# creating the window
window = glfw.create_window(WIDTH, HEIGHT, config_scene['name'], None, None)

# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# set window position
glfw.set_window_pos(window, *WINDOW_POSITION)

# the keyboard input callback
def key_input_clb(window, key, scancode, action, mode):
    global left, right, forward, backward, make_new_surface, player_left, player_right, player_forward, \
        player_backward, yaw_counterclockwise, yaw_clockwise, \
        pause, up, down, wrote_to_gif, switch_view_mode, test_subdivide

    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
    if key == glfw.KEY_W and action == glfw.PRESS:
        forward = True
    elif key == glfw.KEY_W and action == glfw.RELEASE:
        forward = False
    if key == glfw.KEY_S and action == glfw.PRESS:
        backward = True
    elif key == glfw.KEY_S and action == glfw.RELEASE:
        backward = False
    if key == glfw.KEY_A and action == glfw.PRESS:
        left = True
    elif key == glfw.KEY_A and action == glfw.RELEASE:
        left = False
    if key == glfw.KEY_D and action == glfw.PRESS:
        right = True
    elif key == glfw.KEY_D and action == glfw.RELEASE:
        right = False
    if key == glfw.KEY_Q and action == glfw.PRESS:
        yaw_clockwise = True
    elif key == glfw.KEY_Q and action == glfw.RELEASE:
        yaw_clockwise = False
    if key == glfw.KEY_E and action == glfw.PRESS:
        yaw_counterclockwise = True
    elif key == glfw.KEY_E and action == glfw.RELEASE:
        yaw_counterclockwise = False
    if key == glfw.KEY_TAB and action == glfw.PRESS:
        up = True
    elif key == glfw.KEY_TAB and action == glfw.RELEASE:
        up = False
    if key == glfw.KEY_LEFT_SHIFT and action == glfw.PRESS:
        down = True
    elif key == glfw.KEY_LEFT_SHIFT and action == glfw.RELEASE:
        down = False
    if key == glfw.KEY_1 and action == glfw.PRESS:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    if key == glfw.KEY_2 and action == glfw.PRESS:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    if key == glfw.KEY_P and action == glfw.PRESS:
        pause = not pause
    if key == glfw.KEY_V and action == glfw.PRESS:
        switch_camera_mode()
    if key == glfw.KEY_F12 and action == glfw.PRESS:
        capture_screenshot(width=WIDTH, height=HEIGHT)
    if key == glfw.KEY_N and action == glfw.PRESS:
        print_camera_position()


def mouse_look_clb(window, xpos, ypos):
    global first_mouse, lastX, lastY
    if first_mouse:
        lastX = xpos
        lastY = ypos
        first_mouse = False
    xoffset = xpos - lastX
    yoffset = lastY - ypos
    lastX = xpos
    lastY = ypos
    cam.process_mouse_movement(xoffset, yoffset)


def scroll_callback(window, xoffset, yoffset):
    sim_cam.process_mouse_scroll(xoffset, yoffset)


def mouse_button_callback(window, button, action, mods):
    left_click = button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS
    right_click = button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS
    gui.button_update(position_mouse=glfw.get_cursor_pos(window), left_click=left_click, right_click=right_click)


def window_resize_clb(window, width, height):
    glViewport(0, 0, width, height)
    gui.set_screen_size(screen_size=(width, height))


def do_movement(speed=1.0):
    """
    do the camera movement, call this function in the main loop
    :param speed:
    """
    if left:
        active_camera.process_keyboard("LEFT", speed)
    if right:
        active_camera.process_keyboard("RIGHT", speed)
    if forward:
        active_camera.process_keyboard("FORWARD", speed)
    if backward:
        active_camera.process_keyboard("BACKWARD", speed)
    if yaw_clockwise:
        if use_sim_cam:
            sim_cam.process_keyboard("YAW_CLOCKWISE", speed * 2)
        else:
            cam.process_keyboard("ROLL_LEFT", speed * 30)
    if yaw_counterclockwise:
        if use_sim_cam:
            sim_cam.process_keyboard("YAW_COUNTERCLOCKWISE", speed * 2)
        else:
            cam.process_keyboard("ROLL_RIGHT", speed * 30)
    if up:
        active_camera.process_keyboard("UP", speed)
    if down:
        active_camera.process_keyboard("DOWN", speed)



# set the callback function for window resize
glfw.set_window_size_callback(window, window_resize_clb)
# set the mouse position callback
glfw.set_cursor_pos_callback(window, mouse_look_clb)
# set the keyboard input callback
glfw.set_key_callback(window, key_input_clb)
# set the scroll-wheel input callback
glfw.set_scroll_callback(window, scroll_callback)
glfw.set_mouse_button_callback(window, mouse_button_callback)


# make the context current
glfw.make_context_current(window)

"""CAMERA SETUP"""
sim_cam = SimulationCamera(camera_pos=[150.0, 20.0, 0.0])
cam = RollableCamera(camera_pos=[250.0, 20.0, 20.0], mouse_sensitivity=0.1)
use_sim_cam = True
active_camera = sim_cam

def switch_camera_mode():
    global use_sim_cam, active_camera, lastX, lastY
    use_sim_cam = not use_sim_cam
    if use_sim_cam:
        lastX, lastY = WIDTH / 2, HEIGHT / 2
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)
        active_camera = sim_cam
    else:
        lastX, lastY = WIDTH / 2, HEIGHT / 2
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        active_camera = cam



def print_camera_position():
    global active_cam
    print(f'Camera Position: {active_camera.camera_pos}')



if config_scene['enable_backface_culling']:
    glEnable(GL_CULL_FACE)
if config_scene['cull_face_mode'] == 'BACK':
    glCullFace(GL_BACK)
elif config_scene['cull_face_mode'] == 'FRONT':
    glCullFace(GL_FRONT)

glClearColor(0, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


"""direction light settings"""
# create the light cube
my_plc = plc.PointLightCube(pos=[1305.0, 0.0, 0.0], ambient=[0.0] * 3, diffuse=[1.0] * 3,
                            specular=[0.0] * 3, constant=1.0, linear=0.000014 , quadratic=0.000000007)
light_cubes = [my_plc]


# Load textures
textures = glGenTextures(12)
load_texture("engine/textures/button_atlas_gradient.png", textures[0])
load_texture("engine/fonts/my_font.png", textures[1])
load_texture("engine/textures/banana.png", textures[2])
load_texture("engine/textures/dirt.jpg", textures[3])  # Diffuse texture for planet
# For specular, we'll use dirt.jpg as well (or create a specular version)
load_texture("engine/textures/dirt.jpg", textures[4])  # Specular texture for planet

texture_dictionary = {
    "button_atlas": textures[0],
    "font_atlas": textures[1],
    "banana": textures[2],
    "dirt_diffuse": textures[3],
    "dirt_specular": textures[4],
}

"""Shader Compilation"""
shader_terrain_textured = create_shader(
    vertex_file='engine/shaders/terrain_planet_textured.vs',
    fragment_file='engine/shaders/terrain_textured.fs'
)
shader_line = create_shader(vertex_file='engine/shaders/line.vs', fragment_file='engine/shaders/line.fs')


projection = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH / HEIGHT, 0.1, DRAW_DISTANCE)

"""GUI CREATION"""
gui = GUI(screen_size=(WIDTH, HEIGHT))

"""Planet Controls"""
planet_settings = {
    "lacunarity": 2.700,
    "gain": 0.6,
    "amplitude": 1.25,
    "frequency": 0.09,
    "seed": int(random()*500),
    "subdivisions": 4,
    "octaves": 8,
    "shininess": 32.0,
}

def next_seed():
    global planet_settings
    planet_settings["seed"] = random() * 500
    print(f"New seed: {planet_settings['seed']:.3f}")
    regenerate_planet()

planet = TexturedPlanetMeshGPU(
    shader_program=shader_terrain_textured,
    subdivisions=planet_settings["subdivisions"],
    position=vec3(0.0, 0.0, 0.0),
    scale=160.0,
    projection=projection,
    diffuse_texture=texture_dictionary["dirt_diffuse"],
    specular_texture=texture_dictionary["dirt_specular"],
    octaves=planet_settings['octaves'],
    lacunarity=planet_settings['lacunarity'],
    gain=planet_settings['gain'],
    amplitude=planet_settings['amplitude'],
    frequency=planet_settings['frequency'],
    seed=planet_settings['seed'],
    shininess=planet_settings['shininess'],
)

# create a skybox using nebula textures
skybox_paths = ["engine/textures/nebula/skybox_left.png", "engine/textures/nebula/skybox_right.png", "engine/textures/nebula/skybox_up.png",] \
                + ["engine/textures/nebula/skybox_down.png", "engine/textures/nebula/skybox_front.png", "engine/textures/nebula/skybox_back.png"]
skybox = Skybox(skybox_paths, scale=2000)

def regenerate_planet():
    global planet, planet_settings
    planet.cleanup()
    planet = TexturedPlanetMeshGPU(
        shader_program=shader_terrain_textured,
        subdivisions=planet_settings["subdivisions"],
        position=vec3(0.0, 0.0, 0.0),
        scale=160.0,
        projection=projection,
        diffuse_texture=texture_dictionary["dirt_diffuse"],
        specular_texture=texture_dictionary["dirt_specular"],
        octaves=planet_settings['octaves'],
        lacunarity=planet_settings['lacunarity'],
        gain=planet_settings['gain'],
        amplitude=planet_settings['amplitude'],
        frequency=planet_settings['frequency'],
        seed=planet_settings['seed'],
        shininess=planet_settings['shininess'],
    )

def change_planet_setting(delta=0.05, setting="lacunarity"):
    global planet_settings
    planet_settings[setting] = delta + planet_settings[setting]
    planet.update_noise_parameter(parameter=setting, value=planet_settings[setting])
    print(f"Changed {setting} to {planet_settings[setting]:.3f}")

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(-0.55,0.85),
    text="Lacunarity +",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = 0.05,
    setting = "lacunarity",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(-0.55,0.75),
    text="Lacunarity -",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = -0.05,
    setting = "lacunarity",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(-0.25,0.85),
    text="frequency +",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = +0.01,
    setting = "frequency",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(-0.25,0.75),
    text="frequency -",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = -0.01,
    setting = "frequency",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.10,0.05),
    position=(-0.85,-0.95),
    text="Octaves +",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = +1,
    setting = "octaves",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.10,0.05),
    position=(-0.85,-0.85),
    text="Octaves -",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = -1,
    setting = "octaves",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.10,0.05),
    position=(0.85,-0.95),
    text="New Seed",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=next_seed,
)


def update_planet_lights(shader):
    glUseProgram(shader)
    for light in light_cubes:
        glUniform3f(glGetUniformLocation(shader, "point_light.position"), *light.pos)
        glUniform1f(glGetUniformLocation(shader, "point_light.constant"), light.constant)
        glUniform1f(glGetUniformLocation(shader, "point_light.linear"), light.linear)
        glUniform1f(glGetUniformLocation(shader, "point_light.quadratic"), light.quadratic)
        glUniform3f(glGetUniformLocation(shader, "point_light.ambient"), *light.ambient)
        glUniform3f(glGetUniformLocation(shader, "point_light.diffuse"), *light.diffuse)
        glUniform3f(glGetUniformLocation(shader, "point_light.specular"), *light.specular)
    glUniform3f(glGetUniformLocation(shader, "view_pos"), *active_camera.camera_pos)
    glUseProgram(0)


while not glfw.window_should_close(window):

    glfw.poll_events()
    time_start = glfw.get_time()
    do_movement(speed=25 * (time_delta))
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    view = active_camera.get_view_matrix()

    # draw skybox first (render as background)
    glDepthFunc(GL_LEQUAL)
    skybox.draw(view=view, projection=projection)
    glDepthFunc(GL_LESS)

    # Update lighting uniforms
    update_planet_lights(shader=shader_terrain_textured)

    # Draw the textured planet
    planet.draw(view_matrix=view)

    for light in light_cubes:
        light.draw(view=view)

    if use_sim_cam:
        gui.draw()

    if WRITE_TO_GIF:
        write_fbo_to_gif(width=WIDTH, height=HEIGHT)

    glfw.swap_buffers(window)
    time_end = glfw.get_time()
    time_delta = time_end - time_start


if WRITE_TO_GIF:
    save_to_gif()
glfw.terminate()

