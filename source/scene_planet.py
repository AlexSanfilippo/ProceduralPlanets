"""
Draw a triangular face
"""
from random import random

import glfw
import numpy as np
import pyglm
from OpenGL.GL import *
import pyrr

from engine.shader_program import create_shader
from engine.texture_loader import load_texture
from engine.camera import Camera, SimulationCamera
from math import sin, cos
from glm import cos, radians
import logging
import engine.point_light_cube as plc
from engine.procedural_mesh import CubeMeshStatic
from engine.gui import GUI
from engine.screen_capture import capture_screenshot, write_fbo_to_gif, save_to_gif
from noise_generators import fbm_noise, fbm_terrain_3d
from planet_mesh import TriangleIndexed, Cube, Icosphere, TriangleSubdivided, IcosphereSubdivided, PlanetMesh
import ctypes
import numpy as np
from OpenGL.GL import *
from pyglm.glm import vec3

#todo: move to config file
config_scene = {
    "background_color": [0.0, 0.1, 0.1, 1.0],
    "initial_camera_pos": [0.0, 20.0, 20.0],
    "initial_camera_front": [0.0, -0.5, -1.0],
    "name": "Minimal Scene",
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

#key-input globals
first_mouse = True
left, right, forward, backward, make_new_surface = False, False, False, False, False
player_left, player_right, player_forward, player_backward = False, False, False, False
yaw_counterclockwise, yaw_clockwise = False, False
up, down = False, False
pause = False
switch_view_mode = False
test_subdivide = False
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
    if key == glfw.KEY_Y and action == glfw.PRESS:
        test_subdivide = True


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
    # projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 2000)
    # glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
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
        sim_cam.process_keyboard("YAW_CLOCKWISE", speed*2)
    if yaw_counterclockwise:
        sim_cam.process_keyboard("YAW_COUNTERCLOCKWISE", speed*2)
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
# capture the mouse cursor
# glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
# glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_CAPTURED)



# make the context current
glfw.make_context_current(window)

"""CAMERA SETUP"""
sim_cam = SimulationCamera(camera_pos=[6.0, 60.0, 42.0])
cam = Camera(camera_pos=[0.0, 20.0, 20.0], mouse_sensitivity=0.1)
use_sim_cam = True
active_camera = sim_cam

def switch_camera_mode():
    global use_sim_cam, active_camera, lastX, lastY
    use_sim_cam = not use_sim_cam
    if use_sim_cam:
        lastX, lastY = WIDTH / 2, HEIGHT / 2
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)  # glfw.CURSOR_CAPTURED,
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
my_plc = plc.PointLightCube(pos=[1305.0, 0.0, 0.0], ambient=[0.0] * 3, diffuse=[1.5] * 3,
                            specular=[0.4] * 3, constant=1.0, linear=0.000014 , quadratic=0.000000007)
light_cubes = [my_plc]


textures = glGenTextures(12)
load_texture("engine/textures/button_atlas_gradient.png", textures[0])
load_texture("engine/fonts/my_font.png", textures[1])
load_texture("engine/textures/banana.png", textures[2])

texture_dictionary = {
    "button_atlas": textures[0],
    "font_atlas": textures[1],
    "banana": textures[2],
}

"""Shader Compilation"""
#New shader for position and normal only,with 3d camera
shader_program_pos_normal = create_shader(vertex_file='engine/shaders/pos_norm.vs', fragment_file='engine/shaders/pos_norm.fs')
# shader_point_light = create_shader(vertex_file='engine/shaders/pos_norm.vs', fragment_file='engine/shaders/pos_norm_lit.fs')
shader_point_light = create_shader(vertex_file='engine/shaders/pos_norm.vs', fragment_file='engine/shaders/terrain_height_coloring.fs')

"""Mesh Initialization"""
projection = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH / HEIGHT, 0.1, 2000)




"""GUI CREATION"""
gui = GUI(screen_size=(WIDTH, HEIGHT))

"""Planet Controls"""
planet_settings = {
    "lacunarity": 1.55,
    "gain": 0.800,
    "amplitude": 3.000,
    "frequency": 0.080,
    "seed": int(random()*500),
    "subdivisions": 5,
    "octaves": 8,
    "noise_method": fbm_terrain_3d,
    "displacement_amplitude": 10.0,
    "planet_type": "Earth"
}

def next_seed():
    global planet_settings
    planet_settings["seed"] = random() * 500
    print(f"New seed: {planet_settings['seed']:.3f}")
    regenerate_planet()

def cycle_planet_type():
    global planet_settings
    planet_types = ["Moon", "Earth", "Mars"]
    current_index = planet_types.index(planet_settings["planet_type"])
    new_index = (current_index + 1) % len(planet_types)
    planet_settings["planet_type"] = planet_types[new_index]
    print(f"New planet type: {planet_settings['planet_type']}")
    regenerate_planet()

planet = PlanetMesh(
        shader_program=shader_point_light,
        position=vec3(0.0, 0.0, 0.0),
        scale=40.0,
        projection=projection,
        noise_scale=0.25,
        displacement_amplitude=planet_settings["displacement_amplitude"],
        subdivisions=planet_settings["subdivisions"],
        lacunarity=planet_settings["lacunarity"],
        gain=planet_settings["gain"],
        amplitude=planet_settings["amplitude"],
        frequency=planet_settings["frequency"],
        seed=planet_settings["seed"],
        octaves=planet_settings["octaves"],
        noise_method=planet_settings["noise_method"],
        planet_type=planet_settings["planet_type"],
    )

def regenerate_planet():
    global planet, planet_settings
    planet.cleanup()
    planet = PlanetMesh(
        shader_program=shader_point_light,
        position=vec3(0.0, 0.0, 0.0),
        scale=40.0,
        projection=projection,
        noise_scale=0.25,
        displacement_amplitude=planet_settings["displacement_amplitude"],
        subdivisions=planet_settings["subdivisions"],
        lacunarity=planet_settings["lacunarity"],
        gain=planet_settings["gain"],
        amplitude=planet_settings["amplitude"],
        frequency=planet_settings["frequency"],
        seed=planet_settings["seed"],
        octaves=planet_settings["octaves"],
        noise_method=planet_settings["noise_method"],
        planet_type=planet_settings["planet_type"]
    )

def change_planet_setting(delta=0.05, setting="lacunarity"):
    global planet_settings
    planet_settings[setting] = delta + planet_settings[setting]
    print(f"Changed {setting} to {planet_settings[setting]:.3f}")

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.15,0.05),
    position=(-0.85,-0.75),
    text="displacement +",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = +0.5,
    setting = "displacement_amplitude",
)
gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.15,0.05),
    position=(-0.65,-0.75),
    text="displacement -",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = -0.5,
    setting = "displacement_amplitude",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.10,0.05),
    position=(.85,-0.95),
    text="Change Planet Type",
    font_size=0.2,
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=cycle_planet_type,
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
    scale=(0.20,0.05),
    position=(-0.05,0.85),
    text="amplitude +",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = +0.25,
    setting = "amplitude",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(-0.05,0.75),
    text="amplitude -",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = -0.25,
    setting = "amplitude",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(0.25,0.85),
    text="gain +",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = +0.05,
    setting = "gain",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(0.25,0.75),
    text="gain -",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta = -0.05,
    setting = "gain",
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.15,0.03),
    position=(0.85,0.95),
    text="Next Seed",
    context_id="button_1",
    atlas_size=2,
    font_size = 0.20,
    atlas_coordinate=(0,0),
    click_function=next_seed,
)


gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(-0.55,0.95),
    text="New Planet",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=regenerate_planet,
)

gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(-0.80,0.55),
    text="subdivisions +",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta= +1,
    setting="subdivisions",
)
gui.add_text_button(
    font_texture=texture_dictionary["font_atlas"],
    shader=None,
    texture=texture_dictionary["button_atlas"],
    scale=(0.20,0.05),
    position=(-0.80,0.45),
    text="subdivisions -",
    context_id="button_1",
    atlas_size=2,
    atlas_coordinate=(0,0),
    click_function=change_planet_setting,
    delta= -1,
    setting="subdivisions",
)

# must call as final setup of GUI
gui.build_elements_list()

# decouple fps from camera movement with time delta
time_start = 0
time_end = 1 / 60
time_delta = 1.0

def update_planet_lights(shader):
    glUniform1f(glGetUniformLocation(shader, "shininess"), 4.0)
    glUniform3fv(glGetUniformLocation(shader, "object_color"), 1, [0.15, 0.2, 0.25])
    glUniform3fv(glGetUniformLocation(shader, "view_pos"), 1, list(active_camera.camera_pos))

    glUniform3fv(glGetUniformLocation(shader, "point_light.position"), 1, my_plc.get_pos())

    glUniform3fv(glGetUniformLocation(shader, "point_light.ambient"), 1, my_plc.get_ambient())
    glUniform3fv(glGetUniformLocation(shader, "point_light.diffuse"), 1, my_plc.get_diffuse())
    glUniform3fv(glGetUniformLocation(shader, "point_light.specular"), 1, my_plc.get_specular())
    glUniform1fv(glGetUniformLocation(shader, "point_light.constant"), 1, my_plc.get_constant())
    glUniform1fv(glGetUniformLocation(shader, "point_light.linear"), 1, my_plc.get_linear())
    glUniform1fv(glGetUniformLocation(shader, "point_light.quadratic"), 1, my_plc.get_quadratic())


def update_lights(shader):
    glUseProgram(shader)
    view_pos_loc = glGetUniformLocation(shader, "view_pos")
    glUniform3fv(view_pos_loc, 1, list(active_camera.camera_pos))

    #TODO: loop over list of point lights
    glUniform1f(glGetUniformLocation(shader, "material.shininess"), 4.0)
    glUniform3fv(glGetUniformLocation(shader, "point_lights[0].position"), 1, light_cubes[0].get_pos())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[0].diffuse"), 1, light_cubes[0].get_diffuse())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[0].ambient"), 1, light_cubes[0].get_ambient())
    glUniform3fv(glGetUniformLocation(shader, "point_lights[0].specular"), 1, light_cubes[0].get_specular())
    glUniform1f(glGetUniformLocation(shader, "point_lights[0].constant"), light_cubes[0].get_constant())
    glUniform1f(glGetUniformLocation(shader, "point_lights[0].linear"), light_cubes[0].get_linear())
    glUniform1f(glGetUniformLocation(shader, "point_lights[0].quadratic"), light_cubes[0].get_quadratic())
    # second light for mesh viewing
    # glUniform3fv(glGetUniformLocation(shader, "point_lights[1].position"), 1, debug_plcs[1].get_pos())
    # glUniform3fv(glGetUniformLocation(shader, "point_lights[1].diffuse"), 1, debug_plcs[1].get_diffuse())
    # glUniform3fv(glGetUniformLocation(shader, "point_lights[1].ambient"), 1, debug_plcs[1].get_ambient())
    # glUniform3fv(glGetUniformLocation(shader, "point_lights[1].specular"), 1, debug_plcs[1].get_specular())
    # glUniform1f(glGetUniformLocation(shader, "point_lights[1].constant"), debug_plcs[1].get_constant())
    # glUniform1f(glGetUniformLocation(shader, "point_lights[1].linear"), debug_plcs[1].get_linear())
    # glUniform1f(glGetUniformLocation(shader, "point_lights[1].quadratic"), debug_plcs[1].get_quadratic())
    # spotlight
    glUniform3fv(glGetUniformLocation(shader, "spot_light.position"), 1, list(active_camera.camera_pos))
    glUniform3fv(glGetUniformLocation(shader, "spot_light.direction"), 1, list(active_camera.camera_front))
    glUniform3fv(glGetUniformLocation(shader, "spot_light.diffuse"), 1, [0.0] * 3)
    glUniform3fv(glGetUniformLocation(shader, "spot_light.ambient"), 1, [0.0] * 3)
    glUniform3fv(glGetUniformLocation(shader, "spot_light.specular"), 1, [0.0] * 3)
    glUniform1f(glGetUniformLocation(shader, "spot_light.cut_off"), cos(radians(12.5)))
    glUniform1f(glGetUniformLocation(shader, "spot_light.outer_cut_off"), cos(radians(45.0)))
    glUniform1f(glGetUniformLocation(shader, "spot_light.constant"), 1.0)
    glUniform1f(glGetUniformLocation(shader, "spot_light.linear"), 0.00003)
    glUniform1f(glGetUniformLocation(shader, "spot_light.quadratic"), 0.00007)

    #Direction
    glUniform3fv(glGetUniformLocation(shader, "dir_light.direction"), 1, [20.0, 50.0, 0.0])
    glUniform3fv(glGetUniformLocation(shader, "dir_light.ambient"), 1,   [0.0]*3)
    glUniform3fv(glGetUniformLocation(shader, "dir_light.diffuse"), 1,   [0.0]*3)
    glUniform3fv(glGetUniformLocation(shader, "dir_light.specular"), 1,  [0.0]*3)


#Experiment with new meshes/shaders








projection = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH / HEIGHT, 0.1, 2000)

shaders_lighting = [planet.shader_program]


def subdivide_method(v0, v1, v2, level):
    num = random()
    if num > 0.5:
        return False
    return True


def make_player_proximity_condition(player_position: vec3,
                                    center: vec3 = vec3(0.0, 0.0, 0.0),
                                    max_angle_deg: float = 15.0):
    """
    Returns a function `condition(v0, v1, v2, level) -> bool` suitable for
    `adaptive_subdivide`. Subdivides when the triangle normal is within
    `max_angle_deg` of the vector (center -> player_position).
    """
    dir_vec = np.array([player_position.x - center.x,
                        player_position.y - center.y,
                        player_position.z - center.z], dtype=np.float64)
    norm = np.linalg.norm(dir_vec)
    # if player sits at center, indicate "match all" (subdivide everything)
    dir_norm = None if norm == 0.0 else (dir_vec / norm)
    cos_thresh = np.cos(np.deg2rad(float(max_angle_deg)))

    def condition(v0, v1, v2, level):
        # v0, v1, v2 are unit-length numpy arrays (float64) provided by adaptive_subdivide
        if dir_norm is None:
            return True
        fn = v0 + v1 + v2
        fn_len = np.linalg.norm(fn)
        if fn_len == 0.0:
            return False
        fn_norm = fn / fn_len
        dot = float(np.dot(fn_norm, dir_norm))
        # True if angle between face normal and player direction <= max_angle_deg
        return dot >= cos_thresh

    return condition


def converging_sequence(n):
    """
    Returns the first n values of the sequence:
    0.5, 0.75, 0.875, ... converging to 1.

    Formula: a_k = 1 - 1/(2^k)
    """
    return [1 - 1 / (2 ** k) for k in range(1, n + 1)]



current_lod = 1

def update_lod(position_player, position_planet, radius_planet, max_lod):
    global current_lod
    distance_player_to_planet = np.linalg.norm(np.array([position_player.x, position_player.y, position_player.z]) -
                      np.array([position_planet.x, position_planet.y, position_planet.z]))
    surface_distance = max(0.0, distance_player_to_planet - radius_planet)

    new_lod = 1
    distance_to_lod = {20 * radius_planet * (0.5 ** k): k for k in range(max_lod, 0, -1)}
    for k, v in distance_to_lod.items():
        if surface_distance <= k:
            new_lod = v
            break
    # print(f'Surface Distance: {surface_distance:.3f}, New LOD: {new_lod}, Current LOD: {current_lod}')
    if new_lod != current_lod:
        return True
    return False



def make_player_level_condition(player_position: vec3,
                                center: vec3 = vec3(0.0, 0.0, 0.0),
                                max_sublevel: int = 5,
                                angle_sensitivity: float = 32.0
                                ):
    """
    Returns condition(v0, v1, v2, level) -> bool for adaptive_subdivide.

    - player_position: pyglm.vec3 world-space player position
    - center:   pyglm.vec3 center of the icosphere
    - max_sublevel: maximum desired subdivision level (int >= 0)

    The function computes the averaged triangle normal (normalized) and the
    normalized direction from center to player. The dot product is mapped to
    [0,1] and scaled to [0, max_sublevel] to produce a target subdivision
    depth. The predicate returns True when the current `level` is less than
    that target (i.e. keep subdividing until reaching target).
    """

    n = max_sublevel  # number of terms you want
    dot_to_lod = {1 - 1 / (2 ** k): k for k in range(1, n + 1)}

    max_sublevel = int(max(0, max_sublevel))
    dir_vec = np.array([player_position.x - center.x,
                        player_position.y - center.y,
                        player_position.z - center.z], dtype=np.float64)
    dir_norm_val = None
    dir_len = np.linalg.norm(dir_vec)
    if dir_len > 0.0:
        dir_norm_val = dir_vec / dir_len

    def condition(v0, v1, v2, level):
        # v0, v1, v2 are unit-length numpy arrays (float64)
        # If player at center -> subdivide up to max_sublevel
        if dir_norm_val is None:
            return int(level) < max_sublevel

        fn = v0 + v1 + v2
        fn_len = np.linalg.norm(fn)
        if fn_len == 0.0:
            return False

        fn_norm = fn / fn_len
        dot = float(np.dot(fn_norm, dir_norm_val))
        dot = float(np.clip(dot, -1.0, 1.0))


        # map dot from [-1,1] to [0,1], where 1 means perfectly aligned
        t = (dot + 1.0) * 0.5

        # t = np.sign(t) * (np.abs(t) ** angle_sensitivity)
        #
        # # desired target subdivision level (0..max_sublevel)
        # target_level = int(round(t * max_sublevel))
        # target_level = max(0, min(max_sublevel, target_level))

        def get_target_level():
            for k,v in dot_to_lod.items():
                if t < k:
                    return v
                else:
                    pass
            return max(dot_to_lod.values())

        target_level = get_target_level()

        return int(level) < int(target_level)

    return condition


def get_subdivision_level(player_pos, planet_pos, planet_radius, max_subdivision):
    """
    Returns subdivision level (0..max_subdivision) based on player distance to planet.
    Closer player yields higher level; farther yields lower.
    Each next level requires 1/4 the previous distance.
    """
    d = np.linalg.norm(np.array([player_pos.x, player_pos.y, player_pos.z]) -
                      np.array([planet_pos.x, planet_pos.y, planet_pos.z]))
    surface_dist = max(0.0, d - planet_radius)
    initial_threshold = planet_radius * 20.0  # adjust as needed

    level = 1
    threshold = initial_threshold
    for i in range(max_subdivision):
        if surface_dist <= threshold:
            level = i + 1
        threshold *= 0.5

    return min(level, max_subdivision)




while not glfw.window_should_close(window):

    glfw.poll_events()
    time_start = glfw.get_time()
    do_movement(speed=25 * (time_delta))
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)



    view = active_camera.get_view_matrix()

    planet.draw(view_matrix=view)


    if test_subdivide:
        planet.adaptive_subdivide(
            subdivide_condition=make_player_level_condition(
                player_position=active_camera.camera_pos,
                center=planet.position,
                max_sublevel=get_subdivision_level(
                    player_pos=active_camera.camera_pos,
                    planet_pos=planet.position,
                    planet_radius=planet.scale,
                    max_subdivision=12
                ),
                angle_sensitivity=1,
            ),
            regenerate=True,
            min_recursion_level=2,
            max_recursion_level=15,
        )
        test_subdivide = False


    # draw the scene
    for shader in shaders_lighting:
        update_lights(shader=shader)
    update_planet_lights(shader=shader_point_light)


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
